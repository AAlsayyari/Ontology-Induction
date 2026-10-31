"""Exercise edge selection, hidden labels and replacement graphs with delayed replies."""
import json
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path
import sys
import threading
from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from word_graph.schemas import ConnectedPair, TermCluster, TopologyResponse
from word_graph.server import Handler, topology_payload


def validate(base_url, browser_channel=None):
    terms = ["Alpha", "Beta", "Gamma"]
    payload = topology_payload(TopologyResponse(nodes=terms, clusters=[TermCluster(name="Test", terms=terms)],
        edges=[ConnectedPair(source="Alpha", target="Beta"), ConnectedPair(source="Beta", target="Gamma"),
               ConnectedPair(source="Alpha", target="Gamma")]))
    errors, pending = [], []
    def reply(route, source, target):
        route.fulfill(json={"state": "done", "relations": [{"source": source, "target": target,
            "supported": True, "predicate": "influences", "explanation": f"{source} influences {target}."}]})
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_channel, headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.route("**/api/status", lambda route: route.fulfill(json={"state": "ready"}))
        page.route("**/api/generate-topology", lambda route: route.fulfill(json={"state": "done", "topology_id": "mock", "graph": payload}))
        page.route("**/api/describe-relation", lambda route: pending.append(route))
        page.goto(base_url, wait_until="networkidle")
        page.get_by_role("textbox").fill(", ".join(terms))
        page.get_by_role("button", name="Generate graph", exact=True).click()
        page.wait_for_function("() => graph !== null && !generating")
        page.evaluate("() => graph.getElementById('e0').emit('tap')")
        page.wait_for_timeout(100)
        page.evaluate("() => graph.getElementById('e1').emit('tap')")
        page.wait_for_timeout(100)
        assert len(pending) == 2
        reply(pending[1], "Beta", "Gamma")
        page.wait_for_function("() => document.getElementById('relation-drawer').getAttribute('aria-busy') === 'false'")
        reply(pending[0], "Alpha", "Beta")
        page.wait_for_function("() => context.pending.size === 0")
        expect(page.locator("#relation-pair")).to_have_text("Beta → Gamma")
        page.get_by_role("button", name="Show relation labels", exact=True).click()
        page.wait_for_timeout(100)
        assert len(pending) == 3
        page.get_by_role("button", name="Hide relation labels", exact=True).click()
        reply(pending[2], "Alpha", "Gamma")
        page.wait_for_function("() => context.pending.size === 0")
        assert page.evaluate("() => graph.edges().every(e => e.style('label') === '')")
        page.evaluate("data => drawGraph(data, 'old')", payload)
        page.evaluate("() => { window.oldContext = context; graph.getElementById('e0').emit('tap'); }")
        page.wait_for_timeout(100)
        assert len(pending) == 4
        page.evaluate("data => drawGraph(data, 'replacement')", payload)
        reply(pending[3], "Alpha", "Beta")
        page.wait_for_function("() => window.oldContext.pending.size === 0")
        assert page.evaluate("() => graph.edges().every(e => !e.data('described'))")
        expect(page.locator("#relation-drawer")).to_be_hidden()
        failures = page.evaluate("""async () => {
          const originalRequest = request, failures = [];
          try {
            request = async () => { throw new Error('Offline'); };
            await checkModel();
            if (!button.disabled || modelReady) failures.push('offline state');
            request = async () => ({state: 'ready'});
            await checkModel();
            if (button.disabled || !modelReady || statusDot.classList.contains('error'))
              failures.push('connection recovery');
            setStatus('Enter at least two terms.', 'error');
            await checkModel();
            if (statusText.textContent !== 'Enter at least two terms.') failures.push('input error preservation');
            let finishOldCheck;
            request = () => new Promise(resolve => { finishOldCheck = resolve; });
            const oldCheck = checkModel();
            request = async () => ({state: 'ready'});
            await checkModel();
            finishOldCheck({state: 'loading'});
            await oldCheck;
            if (button.disabled || !modelReady || statusText.textContent !== 'Enter at least two terms.')
              failures.push('out-of-order status replies');
            for (const fail of [false, true]) {
              let finish;
              request = () => new Promise((resolve, reject) => { finish = fail ? reject : resolve; });
              const checking = checkModel();
              generating = true; button.disabled = true;
              setStatus('Generating the replacement graph…', 'loading');
              finish(fail ? new Error('Offline') : {state: 'ready'});
              await checking;
              if (!button.disabled || statusText.textContent !== 'Generating the replacement graph…')
                failures.push(fail ? 'late status failure' : 'late status success');
              generating = false;
            }
          } finally {
            request = originalRequest; generating = false; await checkModel();
          }
          return failures;
        }""")
        # A label request started on the old graph must not replace the status
        # while the user is generating its replacement.
        generations = []
        page.route("**/api/generate-topology", lambda route: generations.append(route))
        page.get_by_role("button", name="Show relation labels", exact=True).click()
        page.wait_for_function("() => context.pending.size === 3")
        page.get_by_role("button", name="Generate graph", exact=True).click()
        page.wait_for_function("() => generating")
        assert len(generations) == 1
        label_request = pending[4]
        label_request.fulfill(json={"state": "done", "relations": [
            {**pair, "supported": True, "predicate": "influences",
             "explanation": f"{pair['source']} influences {pair['target']}."}
            for pair in label_request.request.post_data_json["edges"]]})
        page.wait_for_function("() => context.pending.size === 0")
        if page.locator("#status").inner_text() != "Grouping your terms and finding connections…":
            failures.append("late label result")
        generations[0].fulfill(json={"state": "done", "topology_id": "new", "graph": payload})
        page.wait_for_function("() => !generating")
        assert not failures, failures
        assert not errors, errors
        output = Path(__file__).resolve().parents[1] / ".cache/validation/races.json"
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps({"stale_selection_guard": True, "hide_during_loading": True,
            "replacement_graph_guard": True, "status_recovery": True,
            "generation_status_guard": True, "javascript_errors": errors}, indent=2) + "\n", encoding="utf-8")
        browser.close()
        print("Selection races, hidden labels, replacement graphs, status recovery, and generation status checks passed.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", help="Use an existing server instead of the temporary static server")
    parser.add_argument("--browser-channel", default=None, help="Use an installed browser, e.g. chrome")
    args = parser.parse_args()
    if args.url:
        validate(args.url, args.browser_channel)
        return
    # API calls are mocked in the browser, so this check needs no model or GPU.
    with ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, service=None)) as server:
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            validate(f"http://127.0.0.1:{server.server_port}", args.browser_channel)
        finally:
            server.shutdown()
            thread.join(timeout=3)


if __name__ == "__main__":
    main()
