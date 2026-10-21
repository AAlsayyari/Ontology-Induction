"""Replay saved, real model predictions through the UI without loading a GPU."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import hashlib
import json
from pathlib import Path
import sys
import threading

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from word_graph.schemas import TopologyResponse
from word_graph.server import Handler, topology_payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browser-channel", default=None)
    args = parser.parse_args()
    prediction_file = ROOT / "docs/evaluation/predictions.json"
    saved = json.loads(prediction_file.read_text(encoding="utf-8"))
    topology = TopologyResponse.model_validate(saved["cases"]["vehicles"]["topology"])
    sample = next(item for item in saved["sample"] if item["sample_id"] == "pair-014")
    assert sample["state"] == "done" and sample["case_id"] == "vehicles"
    description = sample["description"]
    expected_pair = {description["source"], description["target"]}
    payload = topology_payload(topology)
    output = ROOT / ".cache/validation"
    output.mkdir(parents=True, exist_ok=True)
    errors, description_requests = [], []
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, service=None)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(channel=args.browser_channel, headless=True)
                page = browser.new_page(viewport={"width": 1600, "height": 1100})
                page.on("pageerror", lambda error: errors.append(str(error)))
                page.route("**/api/status", lambda route: route.fulfill(json={"state": "ready"}))
                page.route("**/api/generate-topology", lambda route: route.fulfill(json={
                    "state": "done", "topology_id": "saved-vehicles", "graph": payload}))

                def describe(route):
                    request = route.request.post_data_json
                    assert len(request["edges"]) == 1
                    pair = request["edges"][0]
                    assert {pair["source"], pair["target"]} == expected_pair
                    description_requests.append(request)
                    route.fulfill(json={"state": "done", "relations": [description]})

                page.route("**/api/describe-relation", describe)
                page.goto(f"http://127.0.0.1:{server.server_port}", wait_until="networkidle")
                page.get_by_role("textbox").fill(", ".join(topology.nodes))
                page.get_by_role("button", name="Generate graph", exact=True).click()
                page.wait_for_function("() => graph !== null && !generating")
                assert not description_requests
                assert page.evaluate("() => graph.edges().every(e => e.data('label') === '')")
                page.screenshot(path=str(output / "structure.png"), full_page=True)
                page.evaluate("""pair => graph.edges().filter(edge =>
                    pair.includes(edge.data('term_source')) && pair.includes(edge.data('term_target'))
                ).first().emit('tap')""", list(expected_pair))
                page.wait_for_function("() => document.getElementById('relation-drawer').getAttribute('aria-busy') === 'false'")
                expect(page.locator("#relation-name")).to_have_text(description["predicate"])
                expect(page.locator("#relation-explanation")).to_have_text(description["explanation"])
                assert len(description_requests) == 1
                page.screenshot(path=str(output / "relation_drawer.png"), full_page=True)
                page.get_by_role("button", name="Close relation details").click()
                page.evaluate("() => graph.edges().filter(e => e.data('described')).first().emit('tap')")
                page.wait_for_function("() => document.getElementById('relation-drawer').getAttribute('aria-busy') === 'false'")
                assert len(description_requests) == 1, "A cached repeat click made another request"
                page.get_by_role("button", name="Fit graph", exact=True).click()
                assert not errors, errors
                browser.close()
        finally:
            server.shutdown()
            thread.join(timeout=3)
    report = {"prediction_source": "docs/evaluation/predictions.json",
              "prediction_sha256": hashlib.sha256(prediction_file.read_bytes()).hexdigest(),
              "case": "vehicles", "sample_id": "pair-014", "live_model_inference": False,
              "node_count": len(topology.nodes), "description_requests": len(description_requests),
              "checks": ["render actual saved model graph", "lazy edge inspection", "reversed endpoint routing",
                         "cached repeat click", "fit graph"], "javascript_errors": errors}
    (output / "saved_graph.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Saved model graph, relation drawer, cache, and Fit graph checks passed.")


if __name__ == "__main__":
    main()
