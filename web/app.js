"use strict";
const form = document.getElementById("word-form"), input = document.getElementById("words");
const button = document.getElementById("generate"), buttonLabel = document.getElementById("button-label");
const statusText = document.getElementById("status"), statusDot = document.getElementById("status-dot");
const fitButton = document.getElementById("fit"), labelsButton = document.getElementById("toggle-labels");
const drawer = document.getElementById("relation-drawer");
let graph = null, generating = false, modelReady = false, labelsVisible = false;
let context = null, selectedEdge = null, selectionVersion = 0;
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
  try {
    const data = await request("/api/status");
    modelReady = data.state === "ready";
    button.disabled = !modelReady;
    if (data.state === "error") setStatus("The model could not start. Check the server terminal.", "error");
    else if (data.state === "loading") setStatus("Starting the model…", "loading");
    else if (modelReady && statusDot.classList.contains("loading") && !context?.pending.size)
      setStatus(graph ? "Your graph is ready. Click a connection to explore it." : "Ready when you are.");
  } catch { button.disabled = true; modelReady = false; setStatus("Cannot reach the model. Make sure the web UI server is running.", "error"); }
}
function fitGraph() { graph?.fit(undefined, 45); }
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
  graph.edges().updateStyle(); updateAccessibleEdges();
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
    layout: { name: "preset", fit: true, padding: 45 }, minZoom: .02, maxZoom: 2.5, wheelSensitivity: .2,
    style: [
      { selector: "node", style: { shape: "roundrectangle", width: 180,
        height: node => Math.max(60, 22 + 18 * Math.ceil(node.data("label").length / 22)),
        "background-color": "data(fill)", "border-color": "data(border)", "border-width": 1.4, label: "data(label)",
        "font-family": "Segoe UI, sans-serif", "font-size": 14, "font-weight": 500, color: "#34435a",
        "text-valign": "center", "text-halign": "center", "text-wrap": "wrap", "text-max-width": 160 } },
      { selector: "node.isolated", style: { "border-style": "dashed", color: "#8190a5" } },
      { selector: "edge", style: { width: 1.7, "line-color": "#9babc1", "target-arrow-color": "#9babc1", "target-arrow-shape": "none",
        "curve-style": "unbundled-bezier", "control-point-distances": 45, "control-point-weights": .5,
        label: edge => labelsVisible && edge.data("supported") ? edge.data("label") : "", "font-family": "Segoe UI, sans-serif",
        "font-size": 11, color: "#6c7f99", "text-background-color": "#fbfcfe", "text-background-opacity": 1, "text-background-padding": 4 } },
      { selector: "edge.described", style: { "target-arrow-shape": "triangle", "arrow-scale": .9 } },
      { selector: "edge.unconfirmed", style: { "line-style": "dashed", opacity: .65 } },
      { selector: "node:selected", style: { "border-color": "#376bd7", "border-width": 2.3 } },
      { selector: "edge:selected", style: { width: 3, "line-color": "#376bd7", "target-arrow-color": "#376bd7", color: "#2d58a5" } },
    ],
  });
  context = { id: topologyId, graph, pending: new Map() };
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
  generating = true; closeDrawer(); button.disabled = true; labelsButton.disabled = true; input.readOnly = true;
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
    if (context === targetContext && labelsVisible) setStatus("Relation labels are shown. Click a connection for its explanation.");
  } catch (error) { if (context === targetContext && labelsVisible) setStatus(error.message, "error"); }
});
document.getElementById("relation-close").addEventListener("click", closeDrawer);
document.getElementById("relation-retry").addEventListener("click", () => { if (selectedEdge) inspectEdge(graph.getElementById(selectedEdge)); });
document.addEventListener("keydown", event => { if (event.key === "Escape") closeDrawer(); });
fitButton.addEventListener("click", fitGraph);
input.addEventListener("keydown", event => { if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); form.requestSubmit(); } });
new ResizeObserver(() => { if (graph) { graph.resize(); fitGraph(); } }).observe(document.getElementById("graph"));
checkModel(); setInterval(checkModel, 1500);
