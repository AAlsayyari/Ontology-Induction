import json
from functools import partial
from http.server import ThreadingHTTPServer
import threading
import time
import unittest
import urllib.error
import urllib.request

from word_graph.server import Handler, InferenceService, MAX_TERMS, parse_terms, parse_words, topology_payload
from word_graph.schemas import ConnectedPair, RelationDescription, TermCluster, TopologyResponse


class InputAndGraphTests(unittest.TestCase):
    def test_comma_input_quotes_duplicates_and_whitespace(self):
        self.assertEqual(parse_words('  Child, Parent, child, , "Term, with comma"\n Other '),
                         ["Child", "Parent", "Term, with comma", "Other"])
        self.assertEqual(len(parse_words(",".join(f"Term{i}" for i in range(200)))), 200)
        for text in (None, "", "One, ONE", '"Unclosed quote', ",".join(f"Term{i}" for i in range(MAX_TERMS + 1))):
            with self.assertRaises(ValueError):
                parse_words(text)

    def test_topology_layout_keeps_isolates_and_unlabelled_connections(self):
        topology = TopologyResponse(nodes=["API", "Database", "Bread"],
            clusters=[TermCluster(name="Software", terms=["API", "Database"]), TermCluster(name="Food", terms=["Bread"])],
            edges=[ConnectedPair(source="API", target="Database")])
        payload = topology_payload(topology)
        self.assertEqual(payload["node_count"], 3)
        self.assertEqual(payload["edge_count"], 1)
        nodes = {node["data"]["label"]: node for node in payload["nodes"]}
        self.assertEqual(nodes["Bread"]["classes"], "isolated")
        self.assertEqual(nodes["Bread"]["data"]["degree"], 0)
        self.assertEqual(payload["edges"][0]["data"]["source"], nodes["API"]["data"]["id"])
        self.assertEqual(payload["edges"][0]["data"]["label"], "")
        self.assertFalse(payload["edges"][0]["data"]["described"])

    def test_list_input_normalizes_duplicates_and_rejects_invalid_values(self):
        self.assertEqual(parse_terms([" API ", "api", "Database"]), ["API", "Database"])
        for terms in ("A,B", ["A"], ["A", None], ["A", "x" * 201]):
            with self.assertRaises(ValueError):
                parse_terms(terms)


class FakeGenerator:
    loads = 0
    release = threading.Event()
    description_calls = 0

    def __init__(self, *args):
        type(self).loads += 1

    def generate_topology(self, terms, max_degree=3, **kwargs):
        if terms[0] == "Fail":
            raise ValueError("Invalid model response")
        if terms[0] == "Hold":
            type(self).release.wait(3)
        return TopologyResponse(nodes=terms, clusters=[TermCluster(name="Test", terms=terms)],
            edges=[ConnectedPair(source=terms[0], target=terms[1])], max_degree=max_degree), ["structure"]

    def describe_relations(self, pairs, terms=None, clusters=None):
        type(self).description_calls += 1
        return [RelationDescription(source=p.target, target=p.source, supported=True, predicate="serves",
            explanation=f"{p.target} supplies data to {p.source}.") for p in pairs], ["descriptions"]


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FakeGenerator.loads = 0
        cls.service = InferenceService("test", "bf16", generator_factory=FakeGenerator)
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, service=cls.service))
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.url = f"http://127.0.0.1:{cls.server.server_port}"
        deadline = time.monotonic() + 3
        while cls.service.status()["state"] != "ready" and time.monotonic() < deadline:
            time.sleep(.01)
        assert cls.service.status()["state"] == "ready"

    @classmethod
    def tearDownClass(cls):
        FakeGenerator.release.set()
        cls.service.close()
        cls.service.thread.join(3)
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(3)

    def request(self, path, body=None, origin=None):
        headers = {}
        if body is not None:
            body = json.dumps(body).encode()
            headers["Content-Type"] = "application/json"
        if origin:
            headers["Origin"] = origin
        request = urllib.request.Request(self.url + path, data=body, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as response:
            return response.code, response.read()

    def wait_job(self, job_id):
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            status, body = self.request("/api/jobs/" + job_id)
            self.assertEqual(status, 200)
            result = json.loads(body)
            if result["state"] in {"done", "error"}:
                return result
            time.sleep(.01)
        self.fail("Job did not finish")

    def test_static_page_and_input_validation(self):
        code, body = self.request("/")
        self.assertEqual(code, 200)
        self.assertIn(b'id="words"', body)
        self.assertEqual(self.request("/models/config.json")[0], 404)
        self.assertEqual(self.request("/api/generate-topology", {"words": "One"})[0], 400)
        self.assertEqual(self.request("/api/generate-topology", {"words": "A,B"}, origin="https://example.com")[0], 403)
        self.assertEqual(self.request("/api/generate", {"words": "A,B"})[0], 404)

    def test_cached_model_and_busy_request(self):
        FakeGenerator.release.clear()
        code, body = self.request("/api/generate-topology", {"words": "Hold, Parent"})
        self.assertEqual(code, 202)
        self.assertEqual(self.request("/api/generate-topology", {"words": "A,B"})[0], 409)
        FakeGenerator.release.set()
        result = self.wait_job(json.loads(body)["job_id"])
        self.assertEqual(result["state"], "done")
        self.assertEqual(result["graph"]["node_count"], 2)
        topology = TopologyResponse.model_validate(result["topology"])
        self.assertEqual(topology.nodes, ["Hold", "Parent"])
        code, body = self.request("/api/generate-topology", {"words": "Child, Parent"})
        self.assertEqual(self.wait_job(json.loads(body)["job_id"])["state"], "done")
        self.assertEqual(FakeGenerator.loads, 1)

    def test_model_error_is_visible_and_worker_recovers(self):
        with self.assertLogs("word_graph.server", level="ERROR"):
            code, body = self.request("/api/generate-topology", {"words": "Fail, Parent"})
            self.assertEqual(code, 202)
            result = self.wait_job(json.loads(body)["job_id"])
        self.assertEqual(result["state"], "error")
        self.assertIn("valid graph", result["error"])
        self.assertEqual(self.service.status()["state"], "ready")

    def test_topology_then_on_demand_description_cache_and_active_edge_validation(self):
        count = FakeGenerator.description_calls
        code, body = self.request("/api/generate-topology", {"terms": ["API", "Database", "Bread"]})
        self.assertEqual(code, 202)
        result = self.wait_job(json.loads(body)["job_id"])
        self.assertEqual(FakeGenerator.description_calls, count)
        self.assertEqual(result["graph"]["maximum_degree"], 1)
        self.assertEqual(result["graph"]["edges"][0]["data"]["label"], "")
        payload = {"topology_id": result["topology_id"], "source": "API", "target": "Database"}
        code, body = self.request("/api/describe-relation", payload)
        self.assertEqual(code, 202)
        described = self.wait_job(json.loads(body)["job_id"])
        self.assertEqual(described["relations"][0]["source"], "Database")
        self.assertEqual(described["relations"][0]["predicate"], "serves")
        code, body = self.request("/api/describe-relation", {**payload, "source": "Database", "target": "API"})
        self.assertEqual(code, 200)
        self.assertTrue(json.loads(body)["cached"])
        self.assertEqual(FakeGenerator.description_calls, count + 1)
        self.assertEqual(self.request("/api/describe-relation", {**payload, "target": "Bread"})[0], 400)
        self.assertEqual(self.request("/api/describe-relation", {**payload, "topology_id": "missing"})[0], 404)
        self.assertEqual(self.request("/api/generate-topology", {"terms": ["A", "B"], "max_degree": 20})[0], 400)

    def test_standalone_batch_relation_requests(self):
        code, body = self.request("/api/describe-relation", {"edges": [
            {"source": "API", "target": "Database"}, {"source": "Client", "target": "API"}]})
        self.assertEqual(code, 202)
        result = self.wait_job(json.loads(body)["job_id"])
        self.assertEqual(len(result["relations"]), 2)


if __name__ == "__main__":
    unittest.main()
