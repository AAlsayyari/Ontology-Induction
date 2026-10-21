"""Metrics and saved-prediction rescoring work without models, network, or CUDA."""
import contextlib
import csv
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from evaluation.metrics import (aggregate_matching, connection_metrics, graph_structure,
                                hierarchy_structure, labelled_sample_metrics, manual_review_metrics,
                                reference_degree_ceiling)
from evaluation.run import main, score_run, select_sample, sha256_file, update_review_csv, write_json


def edge(source, target, predicate="subClassOf"):
    return {"source": source, "target": target, "predicate": predicate}


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.benchmark = {"benchmark_id": "fixture", "source": {"name": "A test ontology"}, "cases": [
            {"id": "first", "terms": ["A", "B", "C"], "ground_truth_edges": [edge("A", "B"), edge("B", "C")]},
            {"id": "second", "terms": ["D", "E"], "ground_truth_edges": [edge("D", "E")]},
        ]}
        self.state = {"metadata": {}, "settings": {"max_degree": 3, "sample_size": 50, "seed": 42},
                      "cases": {"first": {"state": "done", "topology": {"nodes": ["A", "B", "C"], "edges": [edge("A", "B")] }},
                                "second": {"state": "error", "error": "Inference failed"}}, "sample": []}

    def test_connection_scores_ignore_orientation_and_duplicate_pairs(self):
        value = connection_metrics([edge("B", "A"), edge("A", "B"), edge("A", "C")],
                                   [edge("A", "B"), edge("B", "C")])
        self.assertEqual((value["true_positive"], value["false_positive"], value["false_negative"]), (1, 1, 1))
        self.assertEqual((value["precision"], value["recall"], value["f1"]), (.5, .5, .5))
        empty = connection_metrics([], [edge("A", "B")])
        self.assertEqual(empty["f1"], 0)
        self.assertEqual(empty["false_negative"], 1)

    def test_structural_diagnostics_separate_cycles_loops_and_reverse_duplicates(self):
        graph = graph_structure(["A", "B", "C", "D"], [edge("A", "B"), edge("B", "A"),
                                edge("B", "C"), edge("C", "A"), edge("A", "A"), edge("A", "missing")], 1)
        self.assertEqual(graph["undirected_cycle_rank"], 1)
        self.assertEqual(graph["component_count"], 2)
        self.assertEqual(graph["isolated_nodes"], ["D"])
        self.assertEqual(graph["self_loop_count"], 1)
        self.assertEqual(graph["duplicate_or_reverse_pair_count"], 1)
        self.assertEqual(graph["unknown_endpoint_count"], 1)
        self.assertEqual(graph["maximum_degree"], 2)
        self.assertEqual(graph["degree_cap_violations"], ["A", "B", "C"])

    def test_subclass_cycles_do_not_count_ordinary_relational_cycles(self):
        descriptions = [{**edge("A", "B"), "supported": True}, {**edge("B", "A"), "supported": True},
                        {**edge("B", "C", "influences"), "supported": True},
                        {**edge("C", "B"), "supported": False}]
        result = hierarchy_structure(["A", "B", "C"], descriptions)
        self.assertEqual(result["evaluated_subclass_edge_count"], 2)
        self.assertEqual(result["directed_subclass_cycle_components"], [["A", "B"]])

    def test_type_alias_cycles_recognize_direction_but_do_not_normalize_label_scores(self):
        descriptions = [{**edge("A", "B", "isA"), "supported": True},
                        {**edge("B", "C", "IS_A"), "supported": True},
                        {**edge("C", "A", "kindOf"), "supported": True},
                        {**edge("D", "A", "hasSubclass"), "supported": True},
                        {**edge("D", "A", "isTypeOf"), "supported": True},
                        {**edge("C", "D", "containsType"), "supported": True}]
        result = hierarchy_structure(["A", "B", "C", "D"], descriptions)
        self.assertEqual(result["evaluated_subclass_edge_count"], 5)
        self.assertEqual(result["directed_subclass_cycle_components"], [["A", "B", "C", "D"]])
        self.assertEqual(result["recognized_predicate_counts"]["IS_A"], 1)
        self.assertEqual(result["recognized_predicate_counts"]["hasSubclass"], 1)
        self.assertEqual(result["ignored_supported_predicate_counts"], {"containsType": 1})
        sample = [{"source": "A", "target": "B", "state": "done", "description": descriptions[0]}]
        exact = labelled_sample_metrics(self.benchmark["cases"][0], sample)
        self.assertEqual((exact["true_positive"], exact["false_positive"], exact["false_negative"]), (0, 1, 1))

    def test_degree_cap_reference_ceiling_is_exact_even_for_odd_cycles(self):
        star = [edge("A", value) for value in ["B", "C", "D", "E", "F"]]
        result = reference_degree_ceiling(["A", "B", "C", "D", "E", "F"], star, 3)
        self.assertEqual(result["maximum_reference_connections_under_degree_cap"], 3)
        self.assertEqual(result["maximum_recall_under_degree_cap"], .6)
        triangle = reference_degree_ceiling(["A", "B", "C"], [edge("A", "B"), edge("B", "C"), edge("C", "A")], 1)
        self.assertEqual(triangle["maximum_reference_connections_under_degree_cap"], 1)

    def test_failed_inference_keeps_reference_edges_in_recall_denominator(self):
        report = score_run(self.benchmark, self.state)
        micro = report["connection_matching"]["micro"]
        self.assertEqual((micro["true_positive"], micro["false_negative"]), (1, 2))
        self.assertAlmostEqual(micro["recall"], 1 / 3)
        self.assertEqual(report["coverage"]["failed_cases"], 1)
        self.assertEqual(report["structure"]["node_count"], 3)
        self.assertEqual(report["coverage"]["sample_shortfall"], 50)

    def test_sampling_is_deterministic_and_never_looks_at_descriptions(self):
        first = select_sample(self.benchmark, self.state["cases"], 50, 42)
        second = select_sample(self.benchmark, self.state["cases"], 50, 42)
        self.assertEqual(first, second)
        self.assertEqual(first, [{"sample_id": "pair-001", "case_id": "first", "source": "A", "target": "B", "state": "pending"}])

    def test_sampled_label_metric_requires_exact_predicate_and_direction(self):
        sample = [{"source": "A", "target": "B", "state": "done", "description": {**edge("B", "A"), "supported": True}}]
        score = labelled_sample_metrics(self.benchmark["cases"][0], sample)
        self.assertEqual((score["true_positive"], score["false_positive"], score["false_negative"]), (0, 1, 1))
        # B->C is outside the sampled pair universe and is not counted as a label FN.
        self.assertEqual(score["recall_denominator"], 1)
        sample[0]["description"] = {**edge("A", "B"), "supported": False, "predicate": None}
        self.assertEqual(labelled_sample_metrics(self.benchmark["cases"][0], sample)["abstentions"], 1)

    def test_manual_review_counts_uncertainty_and_symmetric_direction_explicitly(self):
        rows = [{"factual_accuracy": "correct", "direction": "correct"},
                {"factual_accuracy": "correct", "direction": "not_applicable"},
                {"factual_accuracy": "incorrect", "direction": "reversed"},
                {"factual_accuracy": "uncertain", "direction": "not_applicable"}]
        report = manual_review_metrics(rows)
        self.assertEqual(report["factual_accuracy_over_reviewed"], .5)
        self.assertEqual(report["direction_correct_over_applicable_reviewed"], .5)
        self.assertEqual(report["joint_factual_and_direction_correct_over_reviewed"], .5)

    def test_score_only_needs_no_torch_and_preserves_review_judgments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            benchmark_path, checkpoint = root / "benchmark.json", root / "predictions.json"
            write_json(benchmark_path, self.benchmark)
            self.state["metadata"]["benchmark_sha256"] = sha256_file(benchmark_path)
            self.state["sample"] = select_sample(self.benchmark, self.state["cases"], 50, 42)
            write_json(checkpoint, self.state)
            review_path = root / "manual_review.csv"
            rows = update_review_csv(review_path, self.state, self.benchmark)
            rows[0].update(factual_accuracy="uncertain", direction="not_applicable", review_notes="A deliberately preserved review.",
                           evidence_url="https://example.org/reference")
            with review_path.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            with patch.dict("sys.modules", {"torch": None}), contextlib.redirect_stdout(io.StringIO()):
                result = main(["--score-only", "--benchmark", str(benchmark_path), "--checkpoint", str(checkpoint), "--run-dir", str(root / "report")])
            self.assertEqual(result, 1)  # Inference failures stay visible even during rescoring.
            report = json.loads((root / "report/results.json").read_text())
            self.assertEqual(report["manual_review"]["reviewed_count"], 1)
            self.assertIn("scored_at_utc", report["scoring_metadata"])
            self.assertEqual(report["scoring_metadata"]["source_sha256"]["evaluation/metrics.py"],
                             sha256_file(Path(__file__).resolve().parent.parent / "evaluation/metrics.py"))
            with review_path.open(encoding="utf-8", newline="") as handle:
                review = next(csv.DictReader(handle))
                self.assertEqual(review["review_notes"], "A deliberately preserved review.")
                self.assertEqual(review["evidence_url"], "https://example.org/reference")


if __name__ == "__main__":
    unittest.main()
