"""Run real model inference once, checkpoint it, and score without a GPU later.

Examples:
    python -m evaluation.run --topology-only
    python -m evaluation.run --resume
    python -m evaluation.run --score-only --checkpoint docs/evaluation/predictions.json
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
from datetime import datetime, timedelta, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import platform
import random
import sys
import time
import traceback

from .metrics import (aggregate_matching, connection_metrics, graph_structure,
                      hierarchy_structure, labelled_sample_metrics, manual_review_metrics, reference_degree_ceiling)

ROOT = Path(__file__).resolve().parent.parent
REVIEW_FIELDS = ["sample_id", "case_id", "pair_source", "pair_target", "description_state",
                 "description_error", "supported", "source", "predicate", "target", "explanation", "reference_direct_hypernym",
                 "factual_accuracy", "direction", "review_notes", "reviewer", "reviewed_at"]
HUMAN_FIELDS = ["factual_accuracy", "direction", "review_notes", "reviewer", "reviewed_at"]


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def load_benchmark(path: Path) -> dict:
    benchmark = read_json(path)
    ids = set()
    if not benchmark.get("cases") or not benchmark.get("benchmark_id"):
        raise ValueError("A benchmark needs benchmark_id and nonempty cases")
    for case in benchmark["cases"]:
        if case["id"] in ids:
            raise ValueError("Duplicate benchmark case ID")
        ids.add(case["id"])
        terms = case["terms"]
        if not 2 <= len(terms) <= 200 or len(set(terms)) != len(terms):
            raise ValueError("Each case needs 2 to 200 unique terms")
        for edge in case["ground_truth_edges"]:
            if edge["source"] not in terms or edge["target"] not in terms or edge["source"] == edge["target"]:
                raise ValueError("Invalid reference endpoint")
            if not isinstance(edge.get("predicate"), str):
                raise ValueError("Reference edges need predicates")
    return benchmark


def select_sample(benchmark: dict, cases: dict, sample_size: int, seed: int) -> list[dict]:
    """Select before looking at descriptions; abstentions and failures stay selected."""
    population = []
    for case in benchmark["cases"]:
        record = cases.get(case["id"], {})
        if record.get("state") != "done":
            continue
        for edge in record["topology"]["edges"]:
            population.append({"case_id": case["id"], "source": edge["source"], "target": edge["target"]})
    selected = random.Random(seed).sample(population, min(sample_size, len(population)))
    return [{"sample_id": f"pair-{index:03d}", **edge, "state": "pending"}
            for index, edge in enumerate(selected, 1)]


def review_rows(state: dict, benchmark: dict) -> list[dict]:
    cases = {case["id"]: case for case in benchmark["cases"]}
    rows = []
    for item in state.get("sample", []):
        description = item.get("description", {})
        reference = [edge for edge in cases[item["case_id"]]["ground_truth_edges"]
                     if {edge["source"], edge["target"]} == {item["source"], item["target"]}]
        rows.append({"sample_id": item["sample_id"], "case_id": item["case_id"],
                     "pair_source": item["source"], "pair_target": item["target"],
                     "description_state": item["state"], "description_error": item.get("error", ""), **{key: description.get(key, "")
                         for key in ("supported", "source", "predicate", "target", "explanation")},
                     "reference_direct_hypernym": "; ".join(f"{e['source']} -> {e['target']}" for e in reference),
                     **{key: "" for key in HUMAN_FIELDS}})
    return rows


def update_review_csv(path: Path, state: dict, benchmark: dict) -> list[dict]:
    previous = {}
    extra_fields = []
    if path.is_file():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            extra_fields = [name for name in (reader.fieldnames or []) if name not in REVIEW_FIELDS]
            previous = {row["sample_id"]: row for row in reader}
    rows = review_rows(state, benchmark)
    for row in rows:
        old = previous.get(row["sample_id"], {})
        # Never attach reviews from a different sample after changing a run.
        if (old.get("case_id"), old.get("pair_source"), old.get("pair_target")) == (row["case_id"], row["pair_source"], row["pair_target"]):
            row.update({key: old.get(key, "") for key in HUMAN_FIELDS + extra_fields})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REVIEW_FIELDS + extra_fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def score_run(benchmark: dict, state: dict, reviews: list[dict] | None = None) -> dict:
    reports = []
    for case in benchmark["cases"]:
        record = state["cases"].get(case["id"], {})
        topology = record.get("topology", {"nodes": case["terms"], "edges": []})
        sampled = [item for item in state.get("sample", []) if item["case_id"] == case["id"]]
        descriptions = [item["description"] for item in sampled if item.get("description")]
        report = {"id": case["id"], "state": record.get("state", "pending"),
                  "connection_matching": connection_metrics(topology["edges"], case["ground_truth_edges"]),
                  "reference_degree_ceiling": reference_degree_ceiling(case["terms"], case["ground_truth_edges"], state["settings"]["max_degree"]),
                  "structure": graph_structure(topology["nodes"], topology["edges"], state["settings"]["max_degree"]),
                  "sampled_label_matching": labelled_sample_metrics(case, sampled),
                  "sampled_hierarchy": hierarchy_structure(topology["nodes"], descriptions),
                  "topology_elapsed_seconds": record.get("elapsed_seconds"),
                  "topology_model_calls": topology.get("audit", {}).get("model_calls")}
        if record.get("error"):
            report["error"] = record["error"]
        reports.append(report)
    successful = [case for case in reports if case["state"] == "done"]
    sampled = state.get("sample", [])
    total_nodes = sum(case["structure"]["node_count"] for case in successful)
    total_isolated = sum(case["structure"]["isolated_count"] for case in successful)
    structure = {"successful_case_count": len(successful), "node_count": total_nodes,
                 "isolated_count": total_isolated, "isolated_rate": total_isolated / total_nodes if total_nodes else None,
                 "maximum_degree": max((case["structure"]["maximum_degree"] for case in successful), default=0)}
    for key in ("self_loop_count", "duplicate_or_reverse_pair_count", "unknown_endpoint_count", "undirected_cycle_rank"):
        structure[key] = sum(case["structure"][key] for case in successful)
    structure["degree_cap_violation_count"] = sum(len(case["structure"]["degree_cap_violations"]) for case in successful)
    structure["universal_hub_count"] = sum(len(case["structure"]["universal_hubs"]) for case in successful)
    structure["maximum_neighbor_fraction"] = max((case["structure"]["maximum_neighbor_fraction"] for case in successful), default=0.0)
    reference_total = sum(case["reference_degree_ceiling"]["reference_connections"] for case in reports)
    reference_feasible = sum(case["reference_degree_ceiling"]["maximum_reference_connections_under_degree_cap"] for case in reports)
    recognized_predicates, ignored_predicates = Counter(), Counter()
    for case in reports:
        recognized_predicates.update(case["sampled_hierarchy"]["recognized_predicate_counts"])
        ignored_predicates.update(case["sampled_hierarchy"]["ignored_supported_predicate_counts"])
    return {
        "schema_version": 1, "benchmark_id": benchmark["benchmark_id"], "source": benchmark.get("source", {}),
        "metadata": state.get("metadata", {}), "settings": state["settings"],
        "scoring_metadata": {"scored_at_utc": timestamp(), "method": "reference-pairs-v1; sampled-type-alias-cycles-v2",
                             "source_sha256": {f"evaluation/{name}": sha256_file(ROOT / "evaluation" / name)
                                               for name in ("run.py", "metrics.py")}},
        "scoring_scope": {
            "connections": "Unordered predicted pairs versus listed reference direct-hypernym pairs, across every case including inference failures as empty predictions.",
            "interpretation": "A reference mismatch is not proof that a conceptual connection is factually false. This selected benchmark is not an estimate for every domain.",
            "sampled_labels": "Exact predicate and direction only within the uniformly sampled endpoint-pair universe; this is conditional sample recall, not end-to-end full-graph label recall.",
            "cycles": "Undirected topology cycles and directed type-hierarchy cycles are separate. Only described sampled edges using the listed type aliases can be checked; reverse superclass aliases are reoriented for this diagnostic only. This does not establish absence of cycles in undescribed or unrecognized edges. Exact labelled matching does not normalize aliases.",
            "structure": "Structure summaries use successful cases; failed cases remain in connection recall and are listed explicitly.",
        },
        "coverage": {"case_count": len(reports), "successful_cases": len(successful),
                     "failed_cases": sum(case["state"] == "error" for case in reports),
                     "pending_cases": sum(case["state"] not in {"error", "done"} for case in reports),
                     "requested_sample_size": state["settings"]["sample_size"], "selected_sample_size": len(sampled),
                     "sample_shortfall": max(0, state["settings"]["sample_size"] - len(sampled)),
                     "completed_descriptions": sum(item["state"] == "done" for item in sampled),
                     "failed_descriptions": sum(item["state"] == "error" for item in sampled),
                     "pending_descriptions": sum(item["state"] not in {"done", "error"} for item in sampled)},
        "connection_matching": aggregate_matching([case["connection_matching"] for case in reports]),
        "reference_degree_ceiling": {"reference_connections": reference_total,
                                     "maximum_reference_connections_under_degree_cap": reference_feasible,
                                     "maximum_micro_recall_under_degree_cap": reference_feasible / reference_total if reference_total else 0.0},
        "structure": structure, "sampled_label_matching": aggregate_matching([case["sampled_label_matching"] for case in reports]),
        "sampled_hierarchy": {"evaluated_subclass_edge_count": sum(case["sampled_hierarchy"]["evaluated_subclass_edge_count"] for case in reports),
                              "recognized_predicate_counts": dict(sorted(recognized_predicates.items())),
                              "ignored_supported_predicate_counts": dict(sorted(ignored_predicates.items())),
                              "predicate_rules": reports[0]["sampled_hierarchy"]["predicate_rules"] if reports else {},
                              "directed_subclass_cycle_component_count": sum(case["sampled_hierarchy"]["directed_subclass_cycle_component_count"] for case in reports)},
        "manual_review": manual_review_metrics(reviews or []), "cases": reports,
    }


def runtime_metadata(generator, source: str) -> dict:
    from word_graph.config import MODEL_CACHE, MODEL_ID
    import torch
    config = generator.model.config
    source_path = Path(source) if Path(source).is_dir() else MODEL_CACHE
    return {"model_id": MODEL_ID, "model_revision": getattr(config, "_commit_hash", None),
            "model_config_sha256": sha256_file(source_path / "config.json") if (source_path / "config.json").is_file() else None,
            "weights_index_sha256": sha256_file(source_path / "model.safetensors.index.json") if (source_path / "model.safetensors.index.json").is_file() else None,
            "dtype": str(next(generator.model.parameters()).dtype), "gpu": torch.cuda.get_device_name(0),
            "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory, "torch_cuda_version": torch.version.cuda,
            "runtime_source_sha256": {f"word_graph/{name}": sha256_file(ROOT / "word_graph" / name)
                                      for name in ("inference.py", "topology.py", "constraints.py", "schemas.py", "config.py")},
            "evaluation_source_sha256": {f"evaluation/{name}": sha256_file(ROOT / "evaluation" / name)
                                         for name in ("run.py", "metrics.py")},
            "packages": {name: metadata.version(name) for name in ("torch", "transformers", "pydantic", "lm-format-enforcer")}}


def compact_checkpoint(state: dict) -> dict:
    """Keep all scored predictions, errors and provenance; omit raw decoding text."""
    value = json.loads(json.dumps(state))
    for case in value["cases"].values():
        case.pop("raw_outputs", None)
        case.pop("traceback", None)
    for item in value.get("sample", []):
        item.pop("raw_outputs", None)
        item.pop("traceback", None)
    return value


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=ROOT / "docs/evaluation/benchmark.json")
    parser.add_argument("--run-dir", type=Path, default=ROOT / ".cache/evaluation/run")
    parser.add_argument("--checkpoint", type=Path, help="Existing checkpoint for --resume or --score-only")
    parser.add_argument("--manual-review", type=Path, help="Review CSV; defaults to checkpoint directory/manual_review.csv")
    parser.add_argument("--export-checkpoint", type=Path, help="Write compact portable predictions without raw model decoding")
    parser.add_argument("--model")
    parser.add_argument("--quantization", choices=("bf16", "4bit"), default="bf16")
    parser.add_argument("--max-degree", type=int, choices=(3, 4), default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sample-size", type=int, default=50)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--topology-only", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    args = parser.parse_args(argv)
    if args.sample_size < 0:
        parser.error("--sample-size cannot be negative")
    benchmark = load_benchmark(args.benchmark)
    checkpoint = args.checkpoint or args.run_dir / "state.json"
    review_path = args.manual_review or checkpoint.parent / "manual_review.csv"
    if args.resume or args.score_only:
        state = read_json(checkpoint)
        if state["metadata"]["benchmark_sha256"] != sha256_file(args.benchmark):
            raise ValueError("Benchmark changed since this checkpoint; start a new run")
    else:
        if checkpoint.exists():
            parser.error(f"Checkpoint already exists: {checkpoint}; use --resume or choose a new --run-dir")
        state = {"schema_version": 1,
                 "metadata": {"started_at_utc": timestamp(),
                              "evaluation_date_asia_riyadh": datetime.now(timezone(timedelta(hours=3))).date().isoformat(),
                              "benchmark_sha256": sha256_file(args.benchmark), "python_version": platform.python_version(),
                              "platform": platform.platform()},
                 "settings": {"max_degree": args.max_degree, "sample_size": args.sample_size,
                              "seed": args.seed, "quantization": args.quantization,
                              "sampling": "Uniform without replacement over all successful topology predictions; selected before descriptions."},
                 "cases": {}, "sample": []}
        write_json(checkpoint, state)

    def save_reports():
        if not args.score_only:
            state["metadata"]["last_checkpoint_at_utc"] = timestamp()
            write_json(checkpoint, state)
        rows = update_review_csv(review_path, state, benchmark)
        report = score_run(benchmark, state, rows)
        write_json(args.run_dir / "results.json", report)
        if args.export_checkpoint:
            write_json(args.export_checkpoint, compact_checkpoint(state))
        return report

    if not args.score_only:
        case_ids = [case["id"] for case in benchmark["cases"]]
        need_topology = any(state["cases"].get(case_id, {}).get("state") not in {"done", "error"}
                            or (args.retry_failed and state["cases"].get(case_id, {}).get("state") == "error") for case_id in case_ids)
        need_descriptions = not args.topology_only and any(item["state"] not in {"done", "error"}
                            or (args.retry_failed and item["state"] == "error") for item in state.get("sample", []))
        generator = None
        if need_topology or need_descriptions:
            from word_graph.config import default_model_source
            from word_graph.inference import GraphGenerator
            source = args.model or default_model_source()
            print("Loading the local inference model...", flush=True)
            generator = GraphGenerator(source, state["settings"]["quantization"], state["settings"]["seed"])
            state["metadata"].update(runtime_metadata(generator, source))
            write_json(checkpoint, state)
        for index, case in enumerate(benchmark["cases"], 1):
            previous = state["cases"].get(case["id"], {})
            if previous.get("state") == "done" or (previous.get("state") == "error" and not args.retry_failed):
                continue
            print(f"Topology {index}/{len(case_ids)}: {case['id']} ({len(case['terms'])} terms)", flush=True)
            started = time.monotonic()
            record = state["cases"][case["id"]] = {"state": "running", "started_at_utc": timestamp()}
            write_json(checkpoint, state)
            try:
                topology, raw = generator.generate_topology(case["terms"], max_degree=state["settings"]["max_degree"])
                record.update(state="done", topology=topology.model_dump(), raw_outputs=raw)
                print(f"  {len(topology.edges)} connections", flush=True)
            except Exception as exc:
                record.update(state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                print(f"  FAILED: {record['error']}", flush=True)
            record["elapsed_seconds"] = round(time.monotonic() - started, 3)
            save_reports()
        if not state.get("sample"):
            state["sample"] = select_sample(benchmark, state["cases"], state["settings"]["sample_size"], state["settings"]["seed"])
            state["metadata"]["sample_selected_at_utc"] = timestamp()
            save_reports()
        if not args.topology_only:
            if generator is None and any(item["state"] not in {"done", "error"} or (args.retry_failed and item["state"] == "error") for item in state["sample"]):
                from word_graph.config import default_model_source
                from word_graph.inference import GraphGenerator
                source = args.model or default_model_source()
                generator = GraphGenerator(source, state["settings"]["quantization"], state["settings"]["seed"])
                state["metadata"].update(runtime_metadata(generator, source))
            case_lookup = {case["id"]: case for case in benchmark["cases"]}
            for index, item in enumerate(state["sample"], 1):
                if item["state"] == "done" or (item["state"] == "error" and not args.retry_failed):
                    continue
                print(f"Description {index}/{len(state['sample'])}: {item['source']} / {item['target']}", flush=True)
                item["state"] = "running"
                write_json(checkpoint, state)
                started = time.monotonic()
                topology = state["cases"][item["case_id"]]["topology"]
                clusters = {term: cluster["name"] for cluster in topology["clusters"] for term in cluster["terms"]}
                try:
                    descriptions, raw = generator.describe_relations(
                        [{"source": item["source"], "target": item["target"]}],
                        terms=case_lookup[item["case_id"]]["terms"], clusters=clusters)
                    item.update(state="done", description=descriptions[0].model_dump(), raw_outputs=raw)
                except Exception as exc:
                    item.update(state="error", error=f"{type(exc).__name__}: {exc}", traceback=traceback.format_exc())
                    print(f"  FAILED: {item['error']}", flush=True)
                item["elapsed_seconds"] = round(time.monotonic() - started, 3)
                save_reports()
        state["metadata"]["inference_finished_at_utc"] = timestamp()
    report = save_reports()
    print(json.dumps({"coverage": report["coverage"], "connection_matching": report["connection_matching"],
                      "structure": report["structure"], "manual_review": report["manual_review"]}, indent=2), flush=True)
    print(f"Results: {args.run_dir / 'results.json'}\nManual review: {review_path}", flush=True)
    return 1 if report["coverage"]["failed_cases"] or report["coverage"]["failed_descriptions"] else 0


if __name__ == "__main__":
    sys.exit(main())
