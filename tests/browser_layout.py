"""Measure automatic node/edge clearance without a model, GPU, or manual dragging."""
import argparse
from functools import partial
from http.server import ThreadingHTTPServer
import json
from pathlib import Path
import sys
import threading

from playwright.sync_api import expect, sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from word_graph.schemas import ConnectedPair, TermCluster, TopologyResponse
from word_graph.server import Handler, topology_payload


def fixture(count, long_labels=False, one_component=False):
    terms = []
    for index in range(count):
        label = f"Concept {index:03d}"
        if long_labels or index % 7 == 0:
            label += " with a lengthy technical description and " + "ExtremelyLongUnbrokenTerminology" * 3
        terms.append(label)
    pairs = []
    # Several connected components, cycles, cross-links, and isolated nodes.
    connected = count if count < 100 or one_component else count - 12
    group_size = count if one_component else 12
    for start in range(0, connected, group_size):
        end = min(start + group_size, connected)
        pairs.extend(ConnectedPair(source=terms[i], target=terms[i + 1]) for i in range(start, end - 1))
        if end - start > 3:
            pairs.append(ConnectedPair(source=terms[start], target=terms[end - 1]))
            pairs.append(ConnectedPair(source=terms[start + 1], target=terms[end - 2]))
    clusters = [TermCluster(name=f"Domain {start // 12 + 1}", terms=terms[start:start + 12]) for start in range(0, count, 12)]
    return topology_payload(TopologyResponse(nodes=terms, clusters=clusters, edges=pairs))


MEASURE = """() => {
  const nodes = graph.nodes().map(node => ({ id: node.id(), box: node.boundingBox({includeLabels: true, includeOverlays: false}),
    body: node.boundingBox({includeLabels: false, includeOverlays: false}), lines: node.data('display_label').split('\\n').length }));
  const overlaps = [], textOverflow = [], intersections = [];
  for (let i=0; i<nodes.length; i++) {
    const a=nodes[i];
    if(a.box.x1 < a.body.x1-2 || a.box.x2 > a.body.x2+2 || a.box.y1 < a.body.y1-2 || a.box.y2 > a.body.y2+2) textOverflow.push(a.id);
    for(let j=i+1;j<nodes.length;j++) {
      const b=nodes[j].box;
      if(Math.min(a.box.x2,b.x2)-Math.max(a.box.x1,b.x1) > .1 && Math.min(a.box.y2,b.y2)-Math.max(a.box.y1,b.y1) > .1) overlaps.push([a.id,nodes[j].id]);
    }
  }
  for (const edge of graph.edges()) {
    const points = [edge.sourceEndpoint(), ...(edge.segmentPoints() || []), edge.targetEndpoint()].filter(Boolean);
    for(const node of nodes) {
      if(node.id===edge.source().id() || node.id===edge.target().id()) continue;
      const box={...node.body, x1:node.body.x1+.5, x2:node.body.x2-.5, y1:node.body.y1+.5, y2:node.body.y2-.5};
      if(points.slice(1).some((point,index)=>segmentHitsBox(points[index],point,box))) intersections.push([edge.id(),node.id]);
    }
  }
  return { node_count:nodes.length, edge_count:graph.edges().length, node_overlaps:overlaps, label_overflow:textOverflow,
    edge_node_intersections:intersections, routed_edges:graph.edges('.routed').length,
    maximum_label_lines:Math.max(...nodes.map(n=>n.lines)), zoom:graph.zoom() };
}"""


def measure(page):
    page.wait_for_timeout(160)
    result = page.evaluate(MEASURE)
    assert not result["node_overlaps"], result
    assert not result["label_overflow"], result
    assert not result["edge_node_intersections"], result
    return result


def validate(base_url, browser_channel=None):
    report, errors, posts = {}, [], []
    directory = Path(__file__).resolve().parents[1] / ".cache/validation"
    directory.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(channel=browser_channel, headless=True)
        page = browser.new_page(viewport={"width": 1600, "height": 1100})
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.on("request", lambda request: posts.append(request.url) if request.method == "POST" else None)
        page.route("**/api/status", lambda route: route.fulfill(json={"state": "ready"}))
        small = fixture(24)
        page.route("**/api/generate-topology", lambda route: route.fulfill(json={"state": "done", "topology_id": "layout", "graph": small}))
        def describe(route):
            values = route.request.post_data_json["edges"]
            route.fulfill(json={"state": "done", "relations": [dict(source=v["source"], target=v["target"], supported=True,
                predicate="influencesDevelopmentOf", explanation="The first concept influences the development of the second concept.") for v in values]})
        page.route("**/api/describe-relation", describe)
        page.goto(base_url, wait_until="networkidle")
        page.get_by_role("textbox").fill(", ".join(small["terms"]))
        page.get_by_role("button", name="Generate graph", exact=True).click()
        page.wait_for_function("() => graph !== null && !generating")
        assert not any("describe-relation" in url for url in posts), "Layout must preserve lazy relation inference"
        report["generated_graph"] = measure(page)
        # The same long-label graph at the server's original preset positions
        # should expose the geometry problem this regression is intended to catch.
        page.evaluate("""data => {
          const automatic=arrangeGraph; try {arrangeGraph=()=>{}; drawGraph(data,'preset');}
          finally {arrangeGraph=automatic;}
        }""", fixture(36, True))
        page.wait_for_timeout(160)
        report["preset_positions_baseline"] = page.evaluate(MEASURE)
        assert report["preset_positions_baseline"]["node_overlaps"], "Fixture no longer exercises preset-position overlaps"
        for name, payload in [("long_labels", fixture(36, True)), ("maximum_200_nodes", fixture(200)),
                              ("connected_200_long_labels", fixture(200, True, True))]:
            elapsed = page.evaluate("data => { const started=performance.now(); drawGraph(data, 'layout'); return performance.now()-started; }", payload)
            report[name] = {**measure(page), "layout_ms": round(elapsed, 1)}
            assert elapsed < 10000, f"Layout took {elapsed:.0f}ms"
            page.set_viewport_size({"width": 1000, "height": 800})
            page.wait_for_timeout(350)
            report[name + "_resized"] = measure(page)
            page.set_viewport_size({"width": 1600, "height": 1100})
        isolated = fixture(200)
        isolated["edges"], isolated["edge_count"] = [], 0
        page.evaluate("data => drawGraph(data, 'isolated')", isolated)
        report["all_200_disconnected"] = measure(page)
        expect(page.get_by_role("button", name="Show relation labels", exact=True)).to_be_disabled()
        page.evaluate("data => drawGraph(data, 'minimum')", fixture(2))
        report["minimum_two_nodes"] = measure(page)
        page.evaluate("data => drawGraph(data, 'labels')", small)
        page.get_by_role("button", name="Show relation labels", exact=True).click()
        expect(page.locator("#status")).to_contain_text("Relation labels are shown")
        report["labels_visible"] = measure(page)
        page.screenshot(path=str(directory / "automatic_layout.png"), full_page=True)
        # Force an obstructed connection to exercise detours and reversal, even
        # when COSE finds a graph that happens to need no obstacle routing.
        page.evaluate("""() => {
          graph.nodes().forEach((node,index)=>node.position({x:index*450,y:600}));
          graph.getElementById('n0').position({x:0,y:0}); graph.getElementById('n1').position({x:650,y:0});
          graph.getElementById('n2').position({x:325,y:0}); routeConnections(); fitGraph();
        }""")
        assert page.evaluate("() => graph.getElementById('e0').hasClass('routed')")
        report["obstructed_edge"] = measure(page)
        page.evaluate("""() => {
          const edge=graph.getElementById('e0');
          applyRelations({relations:[{source:edge.target().data('label'),target:edge.source().data('label'),supported:true,
            predicate:'influences',explanation:'The second concept influences the first concept.'}]},context);
        }""")
        report["reversed_edge"] = measure(page)
        assert not errors, errors
        report["javascript_errors"] = errors
        (directory / "layout.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        browser.close()
    print(json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url")
    parser.add_argument("--browser-channel", default=None)
    args = parser.parse_args()
    if args.url:
        validate(args.url, args.browser_channel)
        return
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
