"""Desktop regression for structure-first generation and lazy relation descriptions."""
import argparse
import json
from pathlib import Path
import time
from playwright.sync_api import expect, sync_playwright

TERMS = ["Commercial Bank", "Payment", "Online Store", "Customer", "Order", "API", "Database",
         "Bakery", "Bread", "Wheat", "Farm", "Food"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--browser-channel", default=None, help="Use an installed browser, e.g. chrome")
    args = parser.parse_args()
    directory = Path(__file__).resolve().parents[1] / ".cache/validation"
    directory.mkdir(parents=True, exist_ok=True)
    errors, descriptions, topologies, posts = [], [], [], []
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=args.browser_channel, headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
        def capture(response):
            if "/api/jobs/" in response.url and response.ok:
                value = response.json()
                if value.get("state") == "done":
                    if "topology" in value: topologies.append(value)
                    if "relations" in value: descriptions.append(value)
        page.on("response", capture)
        page.goto(args.url, wait_until="networkidle")
        button = page.get_by_role("button", name="Generate graph", exact=True)
        expect(button).to_be_enabled(timeout=60000)
        page.get_by_role("textbox").fill(", ".join(TERMS))
        started = time.monotonic()
        button.click()
        page.wait_for_function("() => !generating", timeout=300000)
        assert topologies, page.locator("#status").inner_text()
        topology = topologies[-1]
        assert topology["topology"]["nodes"] == TERMS
        assert topology["graph"]["maximum_degree"] <= 3
        assert topology["graph"]["edge_count"] > 0
        assert not descriptions and not any("describe-relation" in url for url in posts)
        assert page.evaluate("() => graph.edges().every(e => e.style('label') === '' && e.data('label') === '')")
        page.screenshot(path=str(directory / "structure.png"), full_page=True)
        print(f"Sparse topology: {len(TERMS)} nodes, {topology['graph']['edge_count']} pairs, {time.monotonic()-started:.1f}s", flush=True)
        pair = page.evaluate("""() => {const e=graph.edges().first();e.emit('tap');
          return {source:e.data('term_source'),target:e.data('term_target')}}""")
        expect(page.locator("#relation-drawer")).to_be_visible()
        page.wait_for_function("() => document.getElementById('relation-drawer').getAttribute('aria-busy') === 'false'", timeout=180000)
        assert descriptions, page.locator("#relation-explanation").inner_text()
        assert len(page.locator("#relation-explanation").inner_text()) >= 8
        explanation = page.locator("#relation-explanation").inner_text()
        assert len(explanation) <= 200 and explanation.endswith(".") and explanation.count(".") == 1
        assert page.evaluate("() => graph.edges().every(e => e.style('label') === '')")
        page.screenshot(path=str(directory / "relation_drawer.png"), full_page=True)
        count = len(posts)
        page.get_by_role("button", name="Close relation details").click()
        page.evaluate("() => graph.edges().first().emit('tap')")
        page.wait_for_function("() => document.getElementById('relation-drawer').getAttribute('aria-busy') === 'false'")
        assert len(posts) == count, "Repeated click triggered model inference"
        cached = page.request.post(args.url + "/api/describe-relation", data={"topology_id": topology["topology_id"], **pair})
        assert cached.status == 200 and cached.json()["cached"]
        page.get_by_role("button", name="Show relation labels", exact=True).click()
        page.wait_for_function("() => graph.edges().every(e => e.data('described')) || document.getElementById('status-dot').classList.contains('error')", timeout=300000)
        assert not page.locator("#status-dot").evaluate("e => e.classList.contains('error')"), page.locator("#status").inner_text()
        assert page.evaluate("() => graph.edges().every(e => e.style('label') === (e.data('supported') ? e.data('label') : ''))")
        page.get_by_role("button", name="Hide relation labels", exact=True).click()
        assert page.evaluate("() => graph.edges().every(e => e.style('label') === '')")
        page.get_by_role("button", name="Fit graph", exact=True).click()
        assert not errors, errors
        (directory / "report.json").write_text(json.dumps({"real_model": True, "input_terms": TERMS,
            "topology": topology, "description_jobs": descriptions, "post_requests": posts,
            "javascript_errors": errors, "checks": ["degree cap", "exact nodes", "no initial predicates",
            "on-demand drawer", "cached repeat click", "label toggle", "fit graph"]}, indent=2) + "\n", encoding="utf-8")
        print("Drawer, cache, show/hide labels and Fit graph passed.", flush=True)
        browser.close()


if __name__ == "__main__":
    main()
