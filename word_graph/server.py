"""Serve sparse graph structure first and describe individual connections on demand."""
from __future__ import annotations

import argparse
import csv
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import logging
import math
import queue
import threading
import time
from urllib.parse import urlsplit
import uuid
import webbrowser

from .config import ROOT, default_model_source, normalize
from .inference import GraphGenerator
from .constraints import MAX_INPUT_TERMS
from .topology import TOPOLOGY_BATCH_SIZE
from .schemas import ConnectedPair, RelationDescription, TopologyResponse, pair_key

LOG = logging.getLogger(__name__)
MAX_TERMS = MAX_INPUT_TERMS
MAX_BODY = 262144
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/index.html": ("index.html", "text/html; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/vendor/cytoscape.min.js": ("vendor/cytoscape.min.js", "text/javascript; charset=utf-8"),
}


def parse_words(text: str) -> list[str]:
    if not isinstance(text, str):
        raise ValueError("Enter your words as comma-separated text.")
    terms, seen = [], set()
    try:
        rows = csv.reader(io.StringIO(text), skipinitialspace=True, strict=True)
        for row in rows:
            for cell in row:
                term = " ".join(cell.split())
                if not term:
                    continue
                if len(term) > 200:
                    raise ValueError("Each term must be 200 characters or fewer.")
                if normalize(term) not in seen:
                    terms.append(term)
                    seen.add(normalize(term))
    except csv.Error as exc:
        raise ValueError("Check the commas and quotation marks in your words.") from exc
    if len(terms) < 2:
        raise ValueError("Enter at least two different terms, separated by commas.")
    if len(terms) > MAX_TERMS:
        raise ValueError(f"Use {MAX_TERMS} terms or fewer in one graph.")
    return terms


def parse_terms(values) -> list[str]:
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError("terms must be a list of strings")
    terms, seen = [], set()
    for value in values:
        term = " ".join(value.split())
        if len(term) > 200:
            raise ValueError("Each term must be 200 characters or fewer")
        if term and normalize(term) not in seen:
            seen.add(normalize(term))
            terms.append(term)
    if not 2 <= len(terms) <= MAX_TERMS:
        raise ValueError(f"Supply 2 to {MAX_TERMS} distinct terms")
    return terms


def topology_payload(topology: TopologyResponse) -> dict:
    ids = {term: f"n{i}" for i, term in enumerate(topology.nodes)}
    degree = dict.fromkeys(topology.nodes, 0)
    for pair in topology.edges:
        degree[pair.source] += 1
        degree[pair.target] += 1
    palette = [("#edf3ff", "#87a8ea"), ("#eef8f1", "#94bd9e"), ("#fff5e9", "#d5b180"),
               ("#f7f0fc", "#b6a0d0"), ("#edf8fc", "#8cbac9"), ("#fff0f2", "#d4a0aa")]
    group_cols = min(3, math.ceil(math.sqrt(len(topology.clusters))))
    widths = [max(1, math.ceil(math.sqrt(len(c.terms)))) for c in topology.clusters]
    cell_width = max(widths) * 235 + 100
    cell_height = max(math.ceil(len(c.terms) / widths[i]) for i, c in enumerate(topology.clusters)) * 125 + 100
    nodes = []
    for index, cluster in enumerate(topology.clusters):
        fill, border = palette[index % len(palette)]
        for position, term in enumerate(cluster.terms):
            nodes.append({"data": {"id": ids[term], "label": term, "cluster": cluster.name,
                "fill": fill, "border": border, "degree": degree[term], "isolated": degree[term] == 0},
                "position": {"x": (index % group_cols) * cell_width + position % widths[index] * 235,
                             "y": index // group_cols * cell_height + position // widths[index] * 125},
                "classes": "isolated" if degree[term] == 0 else ""})
    edges = [{"data": {"id": f"e{i}", "source": ids[p.source], "target": ids[p.target],
                       "term_source": p.source, "term_target": p.target, "label": "", "described": False}}
             for i, p in enumerate(topology.edges)]
    return {"nodes": nodes, "edges": edges, "terms": topology.nodes, "clusters": [c.model_dump() for c in topology.clusters],
            "node_count": len(nodes), "edge_count": len(edges), "max_degree": topology.max_degree,
            "maximum_degree": max(degree.values(), default=0), "audit": topology.audit}



class InferenceService:
    """One worker owns the model and serializes requests to the GPU."""

    def __init__(self, model: str, quantization: str = "bf16",
                 generator_factory=GraphGenerator):
        self.lock = threading.Lock()
        self.work = queue.Queue()
        self.jobs: dict[str, dict] = {}
        self.topologies: dict[str, dict] = {}
        self.standalone: dict[tuple, dict] = {}
        self.state = "loading"
        self.task_kind = None
        self.error = None
        self.model, self.quantization = model, quantization
        self.generator_factory = generator_factory
        self.thread = threading.Thread(target=self._run, name="ontology-model", daemon=True)
        self.thread.start()

    def status(self) -> dict:
        with self.lock:
            return {"state": self.state, "error": self.error, "task": self.task_kind}

    def submit_topology(self, terms: list[str], max_degree: int = 3) -> str:
        return self._submit("topology", {"terms": terms, "max_degree": max_degree})

    def _submit(self, kind, payload, allow_queue=False) -> str:
        with self.lock:
            if self.state in {"loading", "error"} or (self.state != "ready" and not allow_queue) or self.work.qsize() >= 64:
                raise RuntimeError("The model is busy or still starting. Please try again in a moment.")
            for key in list(self.jobs):
                if len(self.jobs) < 100:
                    break
                if self.jobs[key]["state"] in {"done", "error"}:
                    del self.jobs[key]
            job_id = uuid.uuid4().hex
            self.jobs[job_id] = {"state": "queued", "kind": kind}
            self.state = "running"
            self.work.put((job_id, kind, payload))
            return job_id

    def submit_descriptions(self, data: dict) -> dict:
        values = data.get("edges", [{"source": data.get("source"), "target": data.get("target")}])
        if not isinstance(values, list) or not 1 <= len(values) <= 32:
            raise ValueError("Supply one pair or a list of 1 to 32 active edges")
        pairs = [ConnectedPair.model_validate(value) for value in values]
        topology_id = data.get("topology_id")
        with self.lock:
            record = self.topologies.get(topology_id) if isinstance(topology_id, str) else None
        if topology_id is not None and record is None:
            raise LookupError("This topology was not found or has expired; generate it again")
        if record is None:
            terms = parse_terms(data["terms"]) if "terms" in data else parse_terms(
                list(dict.fromkeys(t for p in pairs for t in (p.source, p.target))))
            context_key = tuple(sorted(terms))
            with self.lock:
                if context_key not in self.standalone:
                    while len(self.standalone) >= 20:
                        del self.standalone[next(iter(self.standalone))]
                    self.standalone[context_key] = {"terms": terms, "clusters": {}, "descriptions": {}}
                record = self.standalone[context_key]
        canonical = {normalize(term): term for term in record["terms"]}
        unique = {}
        for pair in pairs:
            if normalize(pair.source) not in canonical or normalize(pair.target) not in canonical:
                raise ValueError("Requested endpoint outside this topology's terms")
            pair = ConnectedPair(source=canonical[normalize(pair.source)], target=canonical[normalize(pair.target)])
            key = pair_key(pair.source, pair.target)
            if topology_id is not None and key not in record["pairs"]:
                raise ValueError("Describe only an existing active edge in this topology")
            unique.setdefault(key, pair)
        pairs = list(unique.values())
        with self.lock:
            if all(key in record["descriptions"] for key in unique):
                return {"state": "done", "relations": [record["descriptions"][key] for key in unique], "cached": True}
        job_id = self._submit("describe", {"pairs": pairs, "record": record}, allow_queue=True)
        return {"job_id": job_id}

    def job(self, job_id: str) -> dict | None:
        with self.lock:
            return self.jobs.get(job_id)

    def close(self) -> None:
        self.work.put(None)

    def _run(self) -> None:
        try:
            generator = self.generator_factory(self.model, self.quantization)
        except Exception as exc:
            LOG.exception("Could not load the ontology model")
            with self.lock:
                self.state, self.error = "error", str(exc)
            return
        with self.lock:
            self.state = "ready"
        LOG.info("Model ready")
        while (item := self.work.get()) is not None:
            job_id, kind, data = item
            terms = data.get("terms", [])
            started = time.monotonic()
            with self.lock:
                self.task_kind = kind
                self.jobs[job_id] = {"state": "running", "kind": kind}
            def progress(completed: int, total: int) -> None:
                batch_size = TOPOLOGY_BATCH_SIZE
                message = ("Finalizing your graph…" if completed == total else
                           f"Processing terms {completed * batch_size + 1}–{min((completed + 1) * batch_size, len(terms))} of {len(terms)}…")
                with self.lock:
                    self.jobs[job_id] = {"state": "running", "progress": {"completed": completed, "total": total, "message": message}}
            def phase(message: str) -> None:
                with self.lock:
                    current = self.jobs[job_id].get("progress", {})
                    self.jobs[job_id] = {"state": "running", "progress": {**current, "message": message}}
            try:
                if kind == "topology":
                    response, _ = generator.generate_topology(terms, max_degree=data["max_degree"],
                        progress_callback=progress, phase_callback=phase)
                    response = TopologyResponse.model_validate(response)
                    record = {"terms": response.nodes, "clusters": {t: c.name for c in response.clusters for t in c.terms},
                              "pairs": {pair_key(p.source, p.target) for p in response.edges}, "descriptions": {}}
                    with self.lock:
                        while len(self.topologies) >= 20:
                            del self.topologies[next(iter(self.topologies))]
                        self.topologies[job_id] = record
                    result = {"state": "done", "topology_id": job_id, "topology": response.model_dump(),
                              "graph": topology_payload(response)}
                elif kind == "describe":
                    record, pairs = data["record"], data["pairs"]
                    with self.lock:
                        missing = [p for p in pairs if pair_key(p.source, p.target) not in record["descriptions"]]
                    if missing:
                        phase("Determining relation meanings...")
                        descriptions, _ = generator.describe_relations(missing, terms=record["terms"], clusters=record["clusters"])
                        if len(descriptions) != len(missing):
                            raise ValueError("Missing relation description")
                        validated = []
                        for pair, value in zip(missing, descriptions):
                            description = RelationDescription.model_validate(value)
                            if pair_key(description.source, description.target) != pair_key(pair.source, pair.target):
                                raise ValueError("Description changed the active edge endpoints")
                            validated.append(description.model_dump())
                        with self.lock:
                            for description in validated:
                                record["descriptions"][pair_key(description["source"], description["target"])] = description
                    with self.lock:
                        relations = [record["descriptions"][pair_key(p.source, p.target)] for p in pairs]
                    result = {"state": "done", "relations": relations, "cached": not missing}
                else:
                    raise ValueError("Unknown inference task")
                result.update(kind=kind, elapsed_seconds=round(time.monotonic() - started, 3))
            except ValueError:
                LOG.exception("Model answer failed validation for job %s", job_id)
                result = {"state": "error", "error": "The model could not finish a valid graph. Please try generating again."}
            except Exception:
                LOG.exception("Graph generation failed for job %s", job_id)
                result = {"state": "error", "error": "Graph generation failed. Check the server terminal and try again."}
            with self.lock:
                self.jobs[job_id] = result
                self.state = "ready" if self.work.empty() else "running"
                self.task_kind = None if self.state == "ready" else self.task_kind


class Handler(BaseHTTPRequestHandler):
    def __init__(self, *args, service: InferenceService, **kwargs):
        self.service = service
        super().__init__(*args, **kwargs)

    def send_bytes(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_json(self, code: int, value: dict) -> None:
        self.send_bytes(code, json.dumps(value, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self) -> None:
        path = urlsplit(self.path).path
        if path == "/api/status":
            self.send_json(200, self.service.status())
        elif path.startswith("/api/jobs/"):
            job = self.service.job(path.removeprefix("/api/jobs/"))
            self.send_json(200 if job else 404, job or {"error": "This graph request was not found. Please generate again."})
        elif path in ASSETS:
            filename, content_type = ASSETS[path]
            try:
                self.send_bytes(200, (ROOT / "web" / filename).read_bytes(), content_type)
            except FileNotFoundError:
                self.send_json(404, {"error": "Page asset not found."})
        elif path == "/favicon.ico":
            self.send_bytes(204, b"", "image/x-icon")
        else:
            self.send_json(404, {"error": "Not found."})

    def do_POST(self) -> None:
        path = urlsplit(self.path).path
        if path not in {"/api/generate-topology", "/api/describe-relation"}:
            self.send_json(404, {"error": "Not found."})
            return
        origin = self.headers.get("Origin")
        port = self.server.server_port
        if origin and origin not in {f"http://127.0.0.1:{port}", f"http://localhost:{port}"}:
            self.send_json(403, {"error": "Open the UI from its local URL."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= MAX_BODY:
                self.send_json(413, {"error": "The input is too large or empty."})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0] != "application/json":
                self.send_json(415, {"error": "Send JSON input."})
                return
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Enter your words as comma-separated text.")
            if path != "/api/describe-relation":
                terms = parse_terms(data["terms"]) if "terms" in data else parse_words(data.get("words"))
                max_degree = data.get("max_degree", 3)
                if type(max_degree) is not int or max_degree not in (3, 4):
                    raise ValueError("max_degree must be 3 or 4")
        except (UnicodeDecodeError, ValueError) as exc:
            self.send_json(400, {"error": str(exc)})
            return
        try:
            if path == "/api/describe-relation":
                result = self.service.submit_descriptions(data)
                self.send_json(202 if "job_id" in result else 200, result)
                return
            job_id = self.service.submit_topology(terms, max_degree)
        except LookupError as exc:
            self.send_json(404, {"error": str(exc)})
            return
        except ValueError as exc:
            self.send_json(400, {"error": str(exc)})
            return
        except RuntimeError as exc:
            self.send_json(409, {"error": str(exc)})
            return
        self.send_json(202, {"job_id": job_id})

    def log_message(self, format: str, *args) -> None:
        # Frequent browser polling does not need to fill the terminal.
        if not self.path.startswith(("/api/status", "/api/jobs/")):
            LOG.info("%s", format % args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default=default_model_source())
    parser.add_argument("--quantization", choices=("bf16", "4bit"), default="bf16")
    parser.add_argument("--open", action="store_true", help="Open the local UI in your browser")
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("Use a port between 1 and 65535")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    service = InferenceService(args.model, args.quantization)
    try:
        server = ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, service=service))
    except OSError:
        service.close()
        raise
    url = f"http://127.0.0.1:{args.port}"
    print(f"Words to graph: {url}\nLoading the model once. Press Ctrl+C to stop.", flush=True)
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping the web UI.", flush=True)
    finally:
        server.server_close()
        service.close()


if __name__ == "__main__":
    main()
