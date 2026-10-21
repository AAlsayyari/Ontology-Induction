import json
from collections import Counter
import unittest

from word_graph.schemas import ConnectedPair, RelationDescription, TermCluster, TopologyResponse, sparse_pairs
from word_graph.topology import cluster_schema, description_schema, pairs_schema
from word_graph.inference import GraphGenerator
from helpers import accepts_json


class Tokenizer:
    def encode(self, text, **kwargs):
        return text.split()


class TopologyTests(unittest.TestCase):
    def test_degree_cap_handles_stars_dense_graphs_and_reverse_duplicates(self):
        terms = [f"Term{i}" for i in range(12)]
        candidates = [{"source": s, "target": t} for s in terms for t in terms if s != t]
        for cap in (3, 4):
            pairs, pruned = sparse_pairs(terms, candidates, cap)
            degree = Counter(t for p in pairs for t in (p.source, p.target))
            self.assertLessEqual(max(degree.values()), cap)
            self.assertEqual(len(degree), len(terms))
            self.assertEqual(len({frozenset((p.source, p.target)) for p in pairs}), len(pairs))
            self.assertTrue(pruned)
        star = [{"source": terms[0], "target": t} for t in terms[1:]]
        selected, _ = sparse_pairs(terms, star)
        self.assertEqual(len(selected), 3)

    def test_response_rejects_missing_clusters_unknown_nodes_and_high_degree(self):
        terms = ["A", "B", "C", "D", "E"]
        data = {"nodes": terms, "clusters": [{"name": "Test", "terms": terms}], "edges": []}
        TopologyResponse.model_validate(data)
        for update in ({"clusters": [{"name": "Test", "terms": terms[:2]}]},
                       {"edges": [{"source": "A", "target": "Unknown"}]},
                       {"edges": [{"source": "A", "target": t} for t in terms[1:]]},
                       {"edges": [{"source": "A", "target": "B"}, {"source": "B", "target": "A"}]}):
            with self.assertRaises(ValueError):
                TopologyResponse.model_validate({**data, **update})

    def test_initial_generation_never_induces_predicates_or_descriptions(self):
        generator = object.__new__(GraphGenerator)
        generator.tokenizer = Tokenizer()
        generator._induce_domain = lambda *args: self.fail("Predicate induction ran before topology")
        generator.describe_relations = lambda *args: self.fail("Description ran before an edge request")
        calls = []
        def decode(messages, schema, budget, *args, **kwargs):
            calls.append(schema)
            self.assertEqual(set(schema["properties"]), {"clusters", "edges"})
            return json.dumps({"clusters": {"API": "Software", "Database": "Software", "Bread": "Food"},
                               "edges": [{"source": "API", "target": "Database"}]})
        generator._decode = decode
        response, raw = generator.generate_topology(["API", "Database", "Bread"])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(response.clusters), 2)
        self.assertNotIn("predicate", response.edges[0].model_dump())
        self.assertEqual(response.audit["isolated_terms"], ["Bread"])

    def test_large_passes_keep_full_target_context_and_cross_batch_connections(self):
        generator = object.__new__(GraphGenerator)
        generator.tokenizer = Tokenizer()
        terms = [f"Term{i}" for i in range(40)]
        sources = []
        def decode(messages, schema, budget, *args, **kwargs):
            if schema["type"] == "object":
                self.assertEqual(schema["required"], terms)
                return json.dumps(dict.fromkeys(terms, "Test domain"))
            branches = schema["items"]["anyOf"]
            focus = [branch["properties"]["source"]["enum"][0] for branch in branches]
            for source, branch in zip(focus, branches):
                self.assertEqual(branch["properties"]["target"]["enum"], [term for term in terms if term != source])
            sources.extend(focus)
            return json.dumps([{"source": focus[0], "target": terms[-1]}])
        generator._decode = decode
        response, raw = generator.generate_topology(terms)
        self.assertEqual(sources, terms)
        self.assertEqual(len(raw), 3)
        self.assertIn((terms[0], terms[-1]), [(p.source, p.target) for p in response.edges])

    def test_decoder_enforces_cluster_coverage_pair_endpoints_and_description_direction(self):
        self.assertTrue(accepts_json({"A": "Food", "B": "Food"}, cluster_schema(["A", "B"])))
        self.assertFalse(accepts_json({"A": "Food"}, cluster_schema(["A", "B"])))
        self.assertFalse(accepts_json({"A": "{'A','B'}", "B": "Food"}, cluster_schema(["A", "B"])))
        self.assertFalse(accepts_json([{"source": "A", "target": "Unknown"}], pairs_schema(["A", "B"], ["A"], 2)))
        self.assertFalse(accepts_json([{"source": "A", "target": "A"}], pairs_schema(["A", "B"], ["A"], 2)))
        self.assertTrue(accepts_json([{"source": "A", "target": "B"}], pairs_schema(["A", "B"], ["A"], 2)))
        pair = ConnectedPair(source="API", target="Database")
        valid = {"explanation": "The database serves data used by the API.", "source": "Database", "target": "API",
                 "predicate": "serves", "supported": "yes"}
        self.assertTrue(accepts_json({"0": valid}, description_schema([pair])))
        self.assertFalse(accepts_json({"0": {**valid, "target": "Unknown"}}, description_schema([pair])))
        self.assertFalse(accepts_json({"0": {**valid, "source": "API"}}, description_schema([pair])))
        self.assertFalse(accepts_json({"0": {**valid, "predicate": "refersTo"}}, description_schema([pair])))
        self.assertFalse(accepts_json({"0": {**valid, "predicate": "refers_to"}}, description_schema([pair])))
        self.assertFalse(accepts_json({"0": {**valid, "explanation": "It serves data. It also does other things."}}, description_schema([pair])))
        unsupported = {**valid, "supported": "no", "predicate": None}
        self.assertTrue(accepts_json({"0": unsupported}, description_schema([pair])))

    def test_batch_descriptions_evaluate_each_pair_independently(self):
        generator = object.__new__(GraphGenerator)
        generator.tokenizer = Tokenizer()
        calls = []
        def decode(messages, schema, *args, **kwargs):
            edges = json.loads(messages[1]["content"])["active_edges"]
            self.assertEqual(len(edges), 1)
            self.assertEqual(schema["required"], ["0"])
            calls.append(edges[0])
            return json.dumps({"0": {"explanation": "The source uses the target resource.", "source": edges[0]["term_a"],
                "target": edges[0]["term_b"], "predicate": "uses", "supported": "yes"}})
        generator._decode = decode
        result, _ = generator.describe_relations([ConnectedPair(source="API", target="Database"),
                                                  ConnectedPair(source="Application", target="Cache")])
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result), 2)

    def test_description_uses_context_and_cannot_add_or_change_pairs(self):
        generator = object.__new__(GraphGenerator)
        generator.tokenizer = Tokenizer()
        value = {"supported": "yes", "predicate": "serves", "source": "Database", "target": "API",
                 "explanation": "The database serves data used by the API."}
        def decode(messages, *args, **kwargs):
            data = json.loads(messages[1]["content"])
            self.assertEqual(data["clusters"], {"API": "Software", "Database": "Software"})
            return json.dumps({"0": value})
        generator._decode = decode
        result, _ = generator.describe_relations([ConnectedPair(source="API", target="Database")],
             ["API", "Database", "Food"], {"API": "Software", "Database": "Software"})
        self.assertEqual(result[0].source, "Database")
        value["target"] = "Food"
        with self.assertRaises(ValueError):
            generator.describe_relations([ConnectedPair(source="API", target="Database")], ["API", "Database", "Food"],
                                        {"API": "Software", "Database": "Software"})

    def test_unsupported_relation_has_no_fabricated_predicate(self):
        RelationDescription(source="A", target="B", supported=False, predicate=None,
                            explanation="No clear factual relationship could be confirmed.")
        with self.assertRaises(ValueError):
            RelationDescription(source="A", target="B", supported=False, predicate="relatedTo",
                                explanation="No clear factual relationship could be confirmed.")
