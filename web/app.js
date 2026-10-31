"use strict";
const form = document.getElementById("word-form"), input = document.getElementById("words");
const button = document.getElementById("generate"), buttonLabel = document.getElementById("button-label");
const statusText = document.getElementById("status"), statusDot = document.getElementById("status-dot");
const fitButton = document.getElementById("fit"), labelsButton = document.getElementById("toggle-labels");
const drawer = document.getElementById("relation-drawer");
let graph = null, generating = false, modelReady = false, labelsVisible = false;
let context = null, selectedEdge = null, selectionVersion = 0;
let modelCheckVersion = 0, modelStatusFailed = false;
function setStatus(message, state = "ready") { statusText.textContent = message; statusDot.className = "status-dot" + (state === "ready" ? "" : " " + state); }
async function request(path, options = {}) {
  const response = await fetch(path, { ...options, cache: "no-store" });
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || "The request failed. Please try again.");
  return data;
}
async function completeJob(path, body, progress) {
  let result = await request(path, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
  if (result.job_id) {
    const id = result.job_id;
    do {
      await new Promise(resolve => setTimeout(resolve, 500));
      result = await request(`/api/jobs/${encodeURIComponent(id)}`);
      if (result.progress && progress) progress(result.progress.message);
    } while (result.state === "running" || result.state === "queued");
  }
  if (result.state === "error") throw new Error(result.error);
  return result;
}
async function checkModel() {
  if (generating) return;
  const version = ++modelCheckVersion;
  try {
    const data = await request("/api/status");
    if (generating || version !== modelCheckVersion) return;
    modelReady = data.state === "ready";
    button.disabled = !modelReady;
    if (data.state === "error") { modelStatusFailed = true; setStatus("The model could not start. Check the server terminal.", "error"); }
    else if (data.state === "loading") setStatus("Starting the model…", "loading");
    else if (modelReady && (modelStatusFailed || statusDot.classList.contains("loading")) && !context?.pending.size) {
      modelStatusFailed = false;
      setStatus(graph ? "Your graph is ready. Click a connection to explore it." : "Ready when you are.");
    }
  } catch {
    if (generating || version !== modelCheckVersion) return;
    modelStatusFailed = true; button.disabled = true; modelReady = false;
    setStatus("Cannot reach the model. Make sure the web UI server is running.", "error");
  }
}
function fitGraph() { graph?.fit(undefined, 45); }
// Measure the text we actually render; character counts underestimate long words
// and make tall labels overlap their neighbours.
const textMeasure = document.createElement("canvas").getContext("2d");
function sizeNode(node) {
  const label = node.data("label"), width = label.length > 28 ? 220 : 190, lines = [];
  textMeasure.font = "500 14px Segoe UI, sans-serif";
  let line = "";
  for (const word of label.split(/\s+/)) {
    if (line && textMeasure.measureText(line + " " + word).width > width - 30) { lines.push(line); line = ""; }
    for (const character of (line ? " " : "") + word) {
      if (line && textMeasure.measureText(line + character).width > width - 30) { lines.push(line); line = ""; }
      line += character;
    }
  }
  if (line) lines.push(line);
  node.data({ display_label: lines.join("\n"), layout_width: width, layout_height: Math.max(60, 24 + lines.length * 19) });
}
function nodeBox(node, padding = 0) {
  const p = node.position(), w = node.data("layout_width") / 2 + padding, h = node.data("layout_height") / 2 + padding;
  return { x1: p.x - w, x2: p.x + w, y1: p.y - h, y2: p.y + h, id: node.id() };
}
function boxesOverlap(a, b) { return a.x1 < b.x2 && b.x1 < a.x2 && a.y1 < b.y2 && b.y1 < a.y2; }
function segmentHitsBox(a, b, box) {
  let lo = 0, hi = 1;
  for (const [start, delta, min, max] of [[a.x, b.x - a.x, box.x1, box.x2], [a.y, b.y - a.y, box.y1, box.y2]]) {
    if (Math.abs(delta) < .0001) { if (start <= min || start >= max) return false; }
    else { const t1 = (min - start) / delta, t2 = (max - start) / delta; lo = Math.max(lo, Math.min(t1, t2)); hi = Math.min(hi, Math.max(t1, t2)); }
  }
  return lo < hi && hi > 0 && lo < 1;
}
function separateNodes(nodes, gap) {
  // The force layout reduces crossings. This final placement pass makes node
  // clearance a guarantee even for very long labels or tightly connected graphs.
  const placed = [];
  for (const node of nodes) {
    const origin = { ...node.position() }, clear = () => !placed.some(box => boxesOverlap(nodeBox(node, gap / 2), box));
    if (!clear()) {
      let found = false;
      for (let ring = 1; !found; ring++) {
        const candidates = [];
        for (let step = -ring; step <= ring; step++) {
          candidates.push([step, -ring], [step, ring]);
          if (Math.abs(step) !== ring) candidates.push([-ring, step], [ring, step]);
        }
        candidates.sort((a, b) => a[0] ** 2 + a[1] ** 2 - b[0] ** 2 - b[1] ** 2);
        for (const [x, y] of candidates) {
          node.position({ x: origin.x + x * 28, y: origin.y + y * 28 });
          if (clear()) { found = true; break; }
        }
      }
    }
    placed.push(nodeBox(node, gap / 2));
  }
}
function packComponents(components) {
  const gap = labelsVisible ? 145 : 110, aspect = Math.max(1, graph.width() / graph.height());
  const items = components.map(component => ({ component, box: component.boundingBox({ includeLabels: false, includeOverlays: false, includeEdges: false }) }));
  items.sort((a, b) => b.box.h * b.box.w - a.box.h * a.box.w);
  const area = items.reduce((sum, item) => sum + (item.box.w + gap) * (item.box.h + gap), 0);
  const minimum = Math.max(...items.map(item => item.box.w)), maximum = items.reduce((sum, item) => sum + item.box.w + gap, 0);
  let best = null;
  // Try shelf widths instead of assuming a square packing. In particular, two
  // tall components usually read better side by side on a desktop screen.
  const widths = [Math.sqrt(area * aspect), ...Array.from({ length: 25 }, (_, i) => minimum + (maximum - minimum) * i / 24)];
  for (const rowWidth of widths) {
    let x = 0, y = 0, rowHeight = 0, width = 0; const positions = [];
    for (const { box } of items) {
      if (x && x + box.w > rowWidth) { x = 0; y += rowHeight + gap; rowHeight = 0; }
      positions.push({ x, y }); width = Math.max(width, x + box.w);
      x += box.w + gap; rowHeight = Math.max(rowHeight, box.h);
    }
    const scale = Math.min((graph.width() - 90) / width, (graph.height() - 90) / (y + rowHeight));
    if (!best || scale > best.scale) best = { scale, positions };
  }
  items.forEach(({ component, box }, index) => component.nodes().shift({ x: best.positions[index].x - box.x1, y: best.positions[index].y - box.y1 }));
}
function compactComponent(component, gap, aspect) {
  const nodes = component.nodes(), box = nodes.boundingBox({ includeLabels: false, includeOverlays: false });
  // COSE can stretch a sparse chain across a very tall canvas. Preserve its
  // relative ordering while bringing it into a compact, screen-shaped area.
  const area = nodes.reduce((sum, node) => sum + (node.data("layout_width") + gap) * (node.data("layout_height") + gap), 0) * 1.2;
  const sx = Math.min(1, Math.sqrt(area * aspect) / box.w), sy = Math.min(1, Math.sqrt(area / aspect) / box.h);
  nodes.positions(node => ({ x: (node.position().x - box.x1) * sx, y: (node.position().y - box.y1) * sy }));
}
function routeConnections() {
  // Keep clear straight connections; only detour edges that would cut through
  // another node. A shared occupancy grid also discourages stacked detours.
  const boxes = graph.nodes().map(node => nodeBox(node, 15)), bounds = graph.nodes().boundingBox({ includeLabels: false });
  const step = 28, left = Math.floor((bounds.x1 - 140) / step) * step, top = Math.floor((bounds.y1 - 140) / step) * step;
  const cols = Math.ceil((bounds.w + 280) / step) + 1, rows = Math.ceil((bounds.h + 280) / step) + 1;
  const blocked = new Map(), usage = new Map(), cellKey = (x, y) => y * cols + x;
  for (const box of boxes) {
    const x1 = Math.max(0, Math.ceil((box.x1 - left) / step)), x2 = Math.min(cols - 1, Math.floor((box.x2 - left) / step));
    const y1 = Math.max(0, Math.ceil((box.y1 - top) / step)), y2 = Math.min(rows - 1, Math.floor((box.y2 - top) / step));
    for (let y = y1; y <= y2; y++) for (let x = x1; x <= x2; x++) {
      const key = cellKey(x, y); if (!blocked.has(key)) blocked.set(key, []); blocked.get(key).push(box.id);
    }
  }
  for (const edge of graph.edges()) {
    edge.removeClass("routed");
    const source = edge.source(), target = edge.target(), a = source.position(), b = target.position();
    const obstacles = boxes.filter(box => box.id !== source.id() && box.id !== target.id());
    if (!obstacles.some(box => segmentHitsBox(a, b, box))) continue;
    const start = { x: Math.round((a.x - left) / step), y: Math.round((a.y - top) / step) };
    const end = { x: Math.round((b.x - left) / step), y: Math.round((b.y - top) / step) };
    const startKey = cellKey(start.x, start.y), endKey = cellKey(end.x, end.y);
    const distances = new Map([[startKey, 0]]), previous = new Map(), closed = new Set();
    // Binary heap: routing remains responsive with the supported 200 terms.
    const heap = [], push = value => {
      let i = heap.push(value) - 1;
      while (i) { const p = (i - 1) >> 1; if (heap[p].score <= value.score) break; heap[i] = heap[p]; i = p; } heap[i] = value;
    }, pop = () => {
      const first = heap[0], last = heap.pop();
      if (heap.length) { let i = 0; while (i * 2 + 1 < heap.length) { let child = i * 2 + 1; if (child + 1 < heap.length && heap[child + 1].score < heap[child].score) child++; if (heap[child].score >= last.score) break; heap[i] = heap[child]; i = child; } heap[i] = last; }
      return first;
    };
    push({ ...start, key: startKey, score: 0 });
    while (heap.length && !closed.has(endKey)) {
      const current = pop(); if (closed.has(current.key)) continue; closed.add(current.key);
      if (current.key === endKey) break;
      for (const [dx, dy] of [[1, 0], [0, 1], [-1, 0], [0, -1]]) {
        const x = current.x + dx, y = current.y + dy, key = cellKey(x, y);
        if (x < 0 || y < 0 || x >= cols || y >= rows || closed.has(key)) continue;
        if ((blocked.get(key) || []).some(id => id !== source.id() && id !== target.id())) continue;
        const cost = distances.get(current.key) + 1 + .35 * (usage.get(key) || 0);
        if (cost >= (distances.get(key) ?? Infinity)) continue;
        distances.set(key, cost); previous.set(key, current.key);
        push({ x, y, key, score: cost + Math.abs(x - end.x) + Math.abs(y - end.y) });
      }
    }
    if (!closed.has(endKey)) continue;
    const path = []; let key = endKey;
    while (key !== undefined) { path.push({ x: left + key % cols * step, y: top + Math.floor(key / cols) * step }); usage.set(key, (usage.get(key) || 0) + 1); key = previous.get(key); }
    path.reverse(); path.unshift({ ...a }); path.push({ ...b });
    // Remove bends whenever a direct segment has the same safe clearance.
    const simplified = [path[0]];
    for (let index = 0; index < path.length - 1;) {
      let next = path.length - 1;
      while (next > index + 1 && obstacles.some(box => segmentHitsBox(path[index], path[next], box))) next--;
      simplified.push(path[next]); index = next;
    }
    const dx = b.x - a.x, dy = b.y - a.y, length = Math.hypot(dx, dy), weights = [], offsets = [];
    for (const point of simplified.slice(1, -1)) {
      weights.push(((point.x - a.x) * dx + (point.y - a.y) * dy) / (length * length));
      offsets.push(((point.y - a.y) * dx - (point.x - a.x) * dy) / length);
    }
    if (weights.length) { edge.data({ route_weights: weights, route_offsets: offsets }); edge.addClass("routed"); }
  }
}
function arrangeGraph(recompute = true) {
  if (!graph || graph.destroyed()) return;
  const components = graph.elements().components();
  graph.batch(() => {
    for (const component of components) {
      if (recompute && component.nodes().length > 1) {
        component.layout({ name: "cose", animate: false, fit: false, randomize: false, nodeDimensionsIncludeLabels: true,
          nodeRepulsion: () => 125000, idealEdgeLength: () => labelsVisible ? 210 : 145, edgeElasticity: () => 100,
          nodeOverlap: 40, gravity: .25, numIter: 700, initialTemp: 150, coolingFactor: .95, minTemp: 1 }).run();
      }
      if (recompute) compactComponent(component, labelsVisible ? 90 : 58, components.length === 1 ? graph.width() / graph.height() : 1.25);
      separateNodes(component.nodes(), labelsVisible ? 90 : 58);
    }
    packComponents(components); routeConnections();
  });
  fitGraph();
}
function edgeKey(edge) { return JSON.stringify([edge.data("term_source"), edge.data("term_target")].sort()); }
function closeDrawer() { selectionVersion++; selectedEdge = null; drawer.hidden = true; graph?.elements().unselect(); }
function renderRelation(edge) {
  document.getElementById("relation-name").textContent = edge.data("supported") ? edge.data("label") : "Unconfirmed connection";
  document.getElementById("relation-pair").textContent = `${edge.source().data("label")} ${edge.data("supported") ? "→" : "↔"} ${edge.target().data("label")}`;
  document.getElementById("relation-explanation").textContent = edge.data("explanation");
  document.getElementById("relation-retry").hidden = true;
  drawer.setAttribute("aria-busy", "false");
}
function updateAccessibleEdges() {
  const list = document.getElementById("graph-description"); list.replaceChildren();
  for (const edge of graph.edges()) {
    const item = document.createElement("li"), control = document.createElement("button");
    control.type = "button";
    control.textContent = `Inspect ${edge.source().data("label")} and ${edge.target().data("label")}${edge.data("label") ? ": " + edge.data("label") : ""}`;
    control.addEventListener("click", () => inspectEdge(edge)); item.append(control); list.append(item);
  }
}
function applyRelations(result, targetContext) {
  if (context !== targetContext) return;
  const nodes = new Map(graph.nodes().map(node => [node.data("label"), node.id()]));
  const edges = new Map(graph.edges().map(edge => [edgeKey(edge), edge]));
  for (const relation of result.relations) {
    const edge = edges.get(JSON.stringify([relation.source, relation.target].sort()));
    if (!edge) continue;
    edge.move({ source: nodes.get(relation.source), target: nodes.get(relation.target) });
    edge.data({ label: relation.predicate || "", explanation: relation.explanation, supported: relation.supported, described: true });
    edge.addClass(relation.supported ? "described" : "unconfirmed");
  }
  graph.edges().updateStyle(); routeConnections(); updateAccessibleEdges();
}
async function ensureDescriptions(edges, targetContext) {
  const missing = edges.filter(edge => !edge.data("described") && !targetContext.pending.has(edgeKey(edge)));
  if (missing.length) {
    const keys = missing.map(edgeKey);
    const pending = completeJob("/api/describe-relation", { topology_id: targetContext.id,
      edges: missing.map(edge => ({ source: edge.data("term_source"), target: edge.data("term_target") })),
    }).then(result => applyRelations(result, targetContext)).finally(() => {
      for (const key of keys) if (targetContext.pending.get(key) === pending) targetContext.pending.delete(key);
    });
    for (const key of keys) targetContext.pending.set(key, pending);
  }
  await Promise.all([...new Set(edges.map(edge => targetContext.pending.get(edgeKey(edge))).filter(Boolean))]);
}
async function inspectEdge(edge) {
  if (generating || !context) return;
  const targetContext = context, version = ++selectionVersion; selectedEdge = edge.id();
  graph.elements().unselect(); edge.select(); edge.connectedNodes().select();
  drawer.hidden = false; drawer.setAttribute("aria-busy", "true");
  document.getElementById("relation-name").textContent = "Finding the relation…";
  document.getElementById("relation-pair").textContent = `${edge.source().data("label")} ↔ ${edge.target().data("label")}`;
  document.getElementById("relation-explanation").textContent = "Determining the predicate and its explanation.";
  document.getElementById("relation-retry").hidden = true;
  try {
    await ensureDescriptions([edge], targetContext);
    if (context === targetContext && selectionVersion === version) renderRelation(edge);
  } catch (error) {
    if (context === targetContext && selectionVersion === version) {
      document.getElementById("relation-name").textContent = "Could not describe this connection";
      document.getElementById("relation-explanation").textContent = error.message;
      document.getElementById("relation-retry").hidden = false; drawer.setAttribute("aria-busy", "false");
    }
  }
}
function drawGraph(data, topologyId) {
  closeDrawer(); if (graph) graph.destroy(); labelsVisible = false;
  labelsButton.textContent = "Show relation labels"; labelsButton.setAttribute("aria-pressed", "false");
  graph = cytoscape({ container: document.getElementById("graph"), elements: [...data.nodes, ...data.edges],
    layout: { name: "preset", fit: false }, minZoom: .005, maxZoom: 2.5, wheelSensitivity: .2,
    style: [
      { selector: "node", style: { shape: "roundrectangle", width: node => node.data("layout_width") || 190,
        height: node => node.data("layout_height") || 60,
        "background-color": "data(fill)", "border-color": "data(border)", "border-width": 1.4, label: node => node.data("display_label") || node.data("label"),
        "font-family": "Segoe UI, sans-serif", "font-size": 14, "font-weight": 500, color: "#34435a",
        "text-valign": "center", "text-halign": "center", "text-wrap": "wrap", "text-max-width": node => (node.data("layout_width") || 190) - 26,
        "line-height": 1.35, "text-overflow-wrap": "anywhere" } },
      { selector: "node.isolated", style: { "border-style": "dashed", color: "#8190a5" } },
      { selector: "edge", style: { width: 1.7, "line-color": "#9babc1", "target-arrow-color": "#9babc1", "target-arrow-shape": "none",
        "curve-style": "straight", "edge-distances": "node-position",
        label: edge => labelsVisible && edge.data("supported") ? edge.data("label") : "", "font-family": "Segoe UI, sans-serif",
        "font-size": 11, color: "#6c7f99", "text-background-color": "#fbfcfe", "text-background-opacity": 1, "text-background-padding": 4,
        "text-wrap": "wrap", "text-overflow-wrap": "anywhere", "text-max-width": 170 } },
      { selector: "edge.routed", style: { "curve-style": "segments", "segment-weights": edge => edge.data("route_weights"),
        "segment-distances": edge => edge.data("route_offsets"), "segment-radii": 14 } },
      { selector: "edge.described", style: { "target-arrow-shape": "triangle", "arrow-scale": .9 } },
      { selector: "edge.unconfirmed", style: { "line-style": "dashed", opacity: .65 } },
      { selector: "node:selected", style: { "border-color": "#376bd7", "border-width": 2.3 } },
      { selector: "edge:selected", style: { width: 3, "line-color": "#376bd7", "target-arrow-color": "#376bd7", color: "#2d58a5" } },
    ],
  });
  context = { id: topologyId, graph, pending: new Map() };
  graph.nodes().forEach(sizeNode); graph.nodes().updateStyle(); arrangeGraph();
  graph.on("tap", "edge", event => inspectEdge(event.target));
  graph.on("tap", "node", event => graph.animate({ center: { eles: event.target }, zoom: Math.max(graph.zoom(), 1), duration: 150 }));
  fitGraph(); document.getElementById("empty-state").hidden = true; fitButton.hidden = false;
  labelsButton.hidden = false; labelsButton.disabled = !data.edge_count;
  document.getElementById("interaction-hint").hidden = false;
  document.getElementById("graph-count").textContent = `${data.node_count} terms · ${data.edge_count} connections`;
  const legend = document.getElementById("cluster-legend"); legend.replaceChildren();
  for (const cluster of data.clusters) {
    const chip = document.createElement("span"), node = data.nodes.find(n => n.data.cluster === cluster.name);
    chip.textContent = cluster.name; chip.style.borderColor = node.data.border; chip.style.backgroundColor = node.data.fill; legend.append(chip);
  }
  legend.hidden = false;
  document.getElementById("graph-hint").textContent = "Click a connection for its predicate and explanation.";
  document.getElementById("graph").setAttribute("aria-label", `Interactive graph with ${data.node_count} terms and ${data.edge_count} connections. Click an edge for its meaning.`);
  updateAccessibleEdges();
}
form.addEventListener("submit", async event => {
  event.preventDefault(); if (generating || !modelReady) return;
  generating = true; modelCheckVersion++; closeDrawer(); button.disabled = true; labelsButton.disabled = true; input.readOnly = true;
  buttonLabel.textContent = "Generating…"; form.setAttribute("aria-busy", "true");
  setStatus("Grouping your terms and finding connections…", "loading");
  try {
    const result = await completeJob("/api/generate-topology", { words: input.value }, message => setStatus(message, "loading"));
    drawGraph(result.graph, result.topology_id);
    setStatus(result.graph.edge_count ? "Your graph is ready. Click a connection to explore it." : "Your terms are shown. No clear connections were found.");
  } catch (error) { setStatus(error.message, "error"); }
  finally {
    generating = false; input.readOnly = false; buttonLabel.textContent = "Generate graph"; form.removeAttribute("aria-busy");
    labelsButton.disabled = !graph?.edges().length; await checkModel();
  }
});
labelsButton.addEventListener("click", async () => {
  labelsVisible = !labelsVisible; labelsButton.textContent = labelsVisible ? "Hide relation labels" : "Show relation labels";
  labelsButton.setAttribute("aria-pressed", String(labelsVisible)); graph.edges().updateStyle();
  if (!labelsVisible) { setStatus("Relation labels are hidden."); return; }
  const targetContext = context; setStatus("Loading relation labels…", "loading");
  try {
    const edges = [...graph.edges()];
    for (let index = 0; index < edges.length && labelsVisible && context === targetContext && !generating; index += 12)
      await ensureDescriptions(edges.slice(index, index + 12), targetContext);
    if (context === targetContext && labelsVisible && !generating) {
      arrangeGraph(); setStatus("Relation labels are shown. Click a connection for its explanation.");
    }
  } catch (error) { if (context === targetContext && labelsVisible && !generating) setStatus(error.message, "error"); }
});
document.getElementById("relation-close").addEventListener("click", closeDrawer);
document.getElementById("relation-retry").addEventListener("click", () => { if (selectedEdge) inspectEdge(graph.getElementById(selectedEdge)); });
document.addEventListener("keydown", event => { if (event.key === "Escape") closeDrawer(); });
fitButton.addEventListener("click", fitGraph);
input.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); form.requestSubmit(); } });
let resizeTimer;
new ResizeObserver(() => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => { if (graph) { graph.resize(); arrangeGraph(false); } }, 120);
}).observe(document.getElementById("graph"));
checkModel(); setInterval(checkModel, 1500);
