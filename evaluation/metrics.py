"""Dependency-free graph and reference-matching metrics.

Reference negatives mean "not a listed direct hypernym", not "factually false".
An undirected cycle is reported separately from a directed subclass cycle.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean

# These aliases apply only to the hierarchy-cycle diagnostic. Exact reference
# predicate matching below deliberately retains the model's original spelling.
CHILD_TO_PARENT_TYPE_PREDICATES = ("subClassOf", "isA", "is_a", "typeOf", "kindOf", "isTypeOf", "isKindOf")
PARENT_TO_CHILD_TYPE_PREDICATES = ("superClassOf", "hasSubclass", "hasSubtype")


def pair_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def precision_recall_f1(true_positive: int, false_positive: int, false_negative: int) -> dict:
    precision_denominator = true_positive + false_positive
    recall_denominator = true_positive + false_negative
    precision = true_positive / precision_denominator if precision_denominator else 0.0
    recall = true_positive / recall_denominator if recall_denominator else 0.0
    return {
        "true_positive": true_positive, "false_positive": false_positive, "false_negative": false_negative,
        "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "precision_denominator": precision_denominator, "recall_denominator": recall_denominator,
    }


def matching_metrics(predictions: set, reference: set) -> dict:
    return precision_recall_f1(len(predictions & reference), len(predictions - reference), len(reference - predictions))


def connection_metrics(predictions: list[dict], reference: list[dict]) -> dict:
    """Compare deduplicated unordered endpoint pairs, ignoring predicate labels."""
    predicted = {pair_key(edge["source"], edge["target"]) for edge in predictions}
    expected = {pair_key(edge["source"], edge["target"]) for edge in reference}
    result = matching_metrics(predicted, expected)
    result.update(predicted_connections=len(predicted), reference_connections=len(expected))
    return result


def aggregate_matching(cases: list[dict]) -> dict:
    micro = precision_recall_f1(*(sum(case[key] for case in cases)
                                  for key in ("true_positive", "false_positive", "false_negative")))
    return {"micro": micro, "macro": {key: mean(case[key] for case in cases) if cases else 0.0
                                       for key in ("precision", "recall", "f1")}, "case_count": len(cases)}


def reference_degree_ceiling(nodes: list[str], reference: list[dict], max_degree: int) -> dict:
    """Exact maximum reference edges feasible under the graph's degree cap.

    Small selected fixtures permit branch-and-bound, without another solver.
    This measures a structural recall ceiling, not a correction to reported recall.
    """
    pairs = sorted({pair_key(edge["source"], edge["target"]) for edge in reference})
    reference_degree = Counter(node for pair in pairs for node in pair)
    ordered = sorted(pairs, key=lambda pair: -sum(reference_degree[node] for node in pair))
    degree = Counter()
    best = 0
    def search(index, selected):
        nonlocal best
        if selected + len(ordered) - index <= best:
            return
        if index == len(ordered):
            best = max(best, selected)
            return
        source, target = ordered[index]
        if degree[source] < max_degree and degree[target] < max_degree:
            degree.update((source, target))
            search(index + 1, selected + 1)
            degree.subtract((source, target))
        search(index + 1, selected)
    search(0, 0)
    return {"reference_connections": len(pairs), "reference_maximum_degree": max(reference_degree.values(), default=0),
            "reference_nodes_above_degree_cap": [node for node in nodes if reference_degree[node] > max_degree],
            "maximum_reference_connections_under_degree_cap": best,
            "maximum_recall_under_degree_cap": best / len(pairs) if pairs else 0.0}


def graph_structure(nodes: list[str], edges: list[dict], max_degree: int = 3) -> dict:
    """Report diagnostics without modifying or repairing the evaluated graph."""
    node_set = set(nodes)
    adjacency = {node: set() for node in node_set}
    seen, duplicates, self_loops, unknown = set(), [], [], []
    for edge in edges:
        source, target = edge["source"], edge["target"]
        if source not in node_set or target not in node_set:
            unknown.append(edge)
            continue
        if source == target:
            self_loops.append(edge)
            continue
        key = pair_key(source, target)
        if key in seen:
            duplicates.append(edge)
            continue
        seen.add(key)
        adjacency[source].add(target)
        adjacency[target].add(source)

    parent = {node: node for node in node_set}
    def find(node):
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    cycle_closing_edges = []
    for source, target in sorted(seen):
        root_source, root_target = find(source), find(target)
        if root_source == root_target:
            cycle_closing_edges.append({"source": source, "target": target})
        else:
            parent[root_source] = root_target
    components = defaultdict(list)
    for node in nodes:
        components[find(node)].append(node)
    component_values = sorted(components.values(), key=lambda values: (-len(values), values))
    degree = {node: len(adjacency[node]) for node in nodes}
    maximum_degree = max(degree.values(), default=0)
    isolated = [node for node in nodes if degree[node] == 0]
    return {
        "node_count": len(nodes), "unique_connection_count": len(seen),
        "self_loop_count": len(self_loops), "self_loops": self_loops,
        "duplicate_or_reverse_pair_count": len(duplicates), "duplicate_or_reverse_pairs": duplicates,
        "unknown_endpoint_count": len(unknown), "unknown_endpoints": unknown,
        "isolated_count": len(isolated), "isolated_rate": len(isolated) / len(nodes) if nodes else 0.0,
        "isolated_nodes": isolated, "component_count": len(component_values), "components": component_values,
        "degree": degree, "maximum_degree": maximum_degree, "degree_cap": max_degree,
        "degree_cap_violations": [node for node in nodes if degree[node] > max_degree],
        "maximum_neighbor_fraction": maximum_degree / (len(nodes) - 1) if len(nodes) > 1 else 0.0,
        "universal_hubs": [node for node in nodes if len(nodes) > 2 and degree[node] == len(nodes) - 1],
        "undirected_cycle_rank": len(seen) - len(node_set) + len(component_values),
        "undirected_cycle_closing_edges": cycle_closing_edges,
    }


def hierarchy_structure(nodes: list[str], descriptions: list[dict]) -> dict:
    """Check sampled type-hierarchy cycles, including explicit universal aliases.

    The alias list does not include composition, instance membership, or arbitrary
    predicates containing 'type'. Parent-to-child aliases are reversed only for
    this diagnostic, without changing saved predictions or exact-label scores.
    """
    adjacency = {node: set() for node in nodes}
    normalize_predicate = lambda name: name.casefold().replace("_", "")
    child_to_parent = {normalize_predicate(name) for name in CHILD_TO_PARENT_TYPE_PREDICATES}
    parent_to_child = {normalize_predicate(name) for name in PARENT_TO_CHILD_TYPE_PREDICATES}
    recognized, ignored = Counter(), Counter()
    for edge in descriptions:
        if not edge.get("supported") or not edge.get("predicate"):
            continue
        predicate = normalize_predicate(edge["predicate"])
        if predicate not in child_to_parent | parent_to_child:
            ignored[edge["predicate"]] += 1
            continue
        source, target = edge["source"], edge["target"]
        if predicate in parent_to_child:
            source, target = target, source
        if source in adjacency and target in adjacency:
            recognized[edge["predicate"]] += 1
            adjacency[source].add(target)
    index, lowlink, stack, active, components = {}, {}, [], set(), []
    def visit(node):
        index[node] = lowlink[node] = len(index)
        stack.append(node)
        active.add(node)
        for target in sorted(adjacency[node]):
            if target not in index:
                visit(target)
                lowlink[node] = min(lowlink[node], lowlink[target])
            elif target in active:
                lowlink[node] = min(lowlink[node], index[target])
        if lowlink[node] == index[node]:
            component = []
            while True:
                member = stack.pop()
                active.remove(member)
                component.append(member)
                if member == node:
                    break
            if len(component) > 1 or node in adjacency[node]:
                components.append(sorted(component))
    for node in nodes:
        if node not in index:
            visit(node)
    return {"evaluated_subclass_edge_count": sum(len(values) for values in adjacency.values()),
            "recognized_predicate_counts": dict(sorted(recognized.items())),
            "ignored_supported_predicate_counts": dict(sorted(ignored.items())),
            "predicate_rules": {"normalization": "Case-insensitive; underscores removed for this cycle diagnostic only.",
                                "child_to_parent": list(CHILD_TO_PARENT_TYPE_PREDICATES),
                                "parent_to_child_reversed": list(PARENT_TO_CHILD_TYPE_PREDICATES)},
            "directed_subclass_cycle_component_count": len(components),
            "directed_subclass_cycle_components": sorted(components)}


def labelled_sample_metrics(case: dict, sample: list[dict]) -> dict:
    """Exact predicate+direction matching within the sampled endpoint-pair universe."""
    universe = {pair_key(item["source"], item["target"]) for item in sample}
    expected = {(edge["source"], edge["predicate"], edge["target"])
                for edge in case["ground_truth_edges"] if pair_key(edge["source"], edge["target"]) in universe}
    predictions = set()
    for item in sample:
        description = item.get("description")
        if description and description.get("supported"):
            predictions.add((description["source"], description["predicate"], description["target"]))
    return {**matching_metrics(predictions, expected), "sampled_connections": len(sample),
            "supported_descriptions": len(predictions),
            "description_failures": sum(item.get("state") == "error" for item in sample),
            "abstentions": sum(bool(item.get("description")) and not item["description"].get("supported") for item in sample)}


def manual_review_metrics(rows: list[dict]) -> dict:
    invalid = {row.get("factual_accuracy", "").strip() for row in rows} - {"", "correct", "incorrect", "uncertain"}
    if invalid:
        raise ValueError(f"Unrecognized factual review labels: {sorted(invalid)}")
    factual = Counter(row.get("factual_accuracy", "").strip() for row in rows)
    directions = Counter(row.get("direction", "").strip() for row in rows)
    reviewed = [row for row in rows if row.get("factual_accuracy") in {"correct", "incorrect", "uncertain"}]
    applicable = [row for row in reviewed if row.get("direction") != "not_applicable"]
    correct_joint = sum(row.get("factual_accuracy") == "correct" and row.get("direction") in {"correct", "not_applicable"} for row in reviewed)
    return {
        "sample_size": len(rows), "reviewed_count": len(reviewed),
        "factual_accuracy_counts": dict(factual), "direction_counts": dict(directions),
        "factual_accuracy_over_reviewed": factual["correct"] / len(reviewed) if reviewed else None,
        "direction_correct_over_applicable_reviewed": sum(row.get("direction") == "correct" for row in applicable) / len(applicable) if applicable else None,
        "joint_factual_and_direction_correct_over_reviewed": correct_joint / len(reviewed) if reviewed else None,
    }
