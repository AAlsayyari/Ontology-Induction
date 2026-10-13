"""Verify the multi-pass topology path with a real mixed-domain term list."""
import json
import argparse
from pathlib import Path
import sys
import time
import urllib.request

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from word_graph.schemas import TopologyResponse
from fixtures import LARGE_MIXED_TERMS


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    terms = LARGE_MIXED_TERMS
    base = args.url.rstrip("/")
    request = urllib.request.Request(base + "/api/generate-topology", json.dumps({"terms": terms}).encode(),
                                    headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(request, timeout=10) as response:
        job_id = json.load(response)["job_id"]
    while time.monotonic() - start < 600:
        with urllib.request.urlopen(base + "/api/jobs/" + job_id, timeout=10) as response:
            data = json.load(response)
        if data["state"] in ("done", "error"):
            break
        time.sleep(.5)
    assert data["state"] == "done", data
    topology = TopologyResponse.model_validate(data["topology"])
    assert topology.nodes == terms and topology.edges
    assert topology.audit["model_calls"] >= 3
    assert data["graph"]["maximum_degree"] <= 3
    assert all(not e["data"]["label"] for e in data["graph"]["edges"])
    output = Path(__file__).resolve().parents[1] / ".cache/validation/large.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"real_model": True, "elapsed_seconds": time.monotonic() - start, **data}, indent=2) + "\n", encoding="utf-8")
    print(f"Large topology passed: {len(terms)} nodes, {len(topology.edges)} pairs, {time.monotonic()-start:.1f}s.")


if __name__ == "__main__":
    main()
