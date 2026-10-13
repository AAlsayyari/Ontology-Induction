"""Contracts for sparse graph structure and independently described relations."""
from __future__ import annotations

from collections import Counter
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, StrictStr, model_validator

from .config import normalize

# This is a denylist of empty/filler predicates, not a domain vocabulary.
FILLER_PREDICATES = {"relatedto", "refersto", "associatedwith", "linkedto", "connectedto", "appliesto"}
RESERVED_NAMES = {"label", "rdfs", "owl", "ontology"}


def validate_predicate(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,63}", name):
        raise ValueError("Predicate names must be simple identifiers of at most 64 characters")
    if normalize(name).replace("_", "") in FILLER_PREDICATES or name in RESERVED_NAMES:
        raise ValueError(f"Predicate {name!r} is a generic filler or reserved name")
    key = normalize(name).replace("_", "")
    for canonical in ("subClassOf", "subFieldOf", "disjointWith"):
        marker = normalize(canonical)
        if marker in key and name != canonical:
            raise ValueError(f"Use the reusable predicate {canonical}, without attaching concept names or aliases")
    return name



class ConnectedPair(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: StrictStr = Field(min_length=1)
    target: StrictStr = Field(min_length=1)

    @model_validator(mode="after")
    def distinct_endpoints(self):
        if normalize(self.source) == normalize(self.target):
            raise ValueError("A connection requires two different terms")
        return self


class TermCluster(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: StrictStr = Field(min_length=2, max_length=64)
    terms: list[StrictStr] = Field(min_length=1)


class TopologyResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    nodes: list[StrictStr] = Field(min_length=2)
    clusters: list[TermCluster] = Field(min_length=1)
    edges: list[ConnectedPair]
    max_degree: Literal[3, 4] = 3
    audit: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_sparse_topology(self):
        nodes = set(self.nodes)
        if any(not n.strip() for n in self.nodes) or len({normalize(n) for n in self.nodes}) != len(self.nodes):
            raise ValueError("Topology nodes must be nonempty and unique")
        members = [term for cluster in self.clusters for term in cluster.terms]
        if len(members) != len(self.nodes) or set(members) != nodes:
            raise ValueError("Every term must belong to exactly one cluster")
        if len({normalize(c.name) for c in self.clusters}) != len(self.clusters):
            raise ValueError("Cluster names must be unique")
        degree, seen = Counter(), set()
        for edge in self.edges:
            if edge.source not in nodes or edge.target not in nodes:
                raise ValueError("Topology endpoints must be exact input terms")
            key = frozenset((edge.source, edge.target))
            if key in seen:
                raise ValueError("Duplicate or reverse duplicate connection")
            seen.add(key)
            degree.update((edge.source, edge.target))
        if degree and max(degree.values()) > self.max_degree:
            raise ValueError("Topology exceeds the node degree limit")
        return self


class RelationDescription(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: StrictStr = Field(min_length=1)
    target: StrictStr = Field(min_length=1)
    supported: bool
    predicate: StrictStr | None
    explanation: StrictStr = Field(min_length=13, max_length=200)

    @model_validator(mode="after")
    def valid_description(self):
        if normalize(self.source) == normalize(self.target):
            raise ValueError("A relation requires distinct endpoints")
        if self.supported:
            if self.predicate is None:
                raise ValueError("A supported relation needs a predicate")
            validate_predicate(self.predicate)
        elif self.predicate is not None:
            raise ValueError("An unconfirmed connection must not invent a predicate")
        if not re.fullmatch(r"[A-Za-z0-9 ,:'()&/+_-]{12,199}[.]", self.explanation):
            raise ValueError("Use a short single-sentence explanation")
        return self


def pair_key(source: str, target: str) -> tuple[str, str]:
    return tuple(sorted((source, target)))


def sparse_pairs(terms: list[str], candidates: list[dict], max_degree: int = 3):
    """Prefer coverage, then low-degree endpoints; never invent replacement links."""
    if max_degree not in (3, 4):
        raise ValueError("max_degree must be 3 or 4")
    order = {term: i for i, term in enumerate(terms)}
    unique = {}
    for value in candidates:
        pair = ConnectedPair.model_validate(value)
        if pair.source not in order or pair.target not in order:
            raise ValueError("Connection endpoint outside the input")
        source, target = sorted((pair.source, pair.target), key=order.get)
        unique.setdefault((source, target), ConnectedPair(source=source, target=target))
    pending = list(unique.values())
    degree, chosen = Counter(), []
    while pending:
        feasible = [p for p in pending if degree[p.source] < max_degree and degree[p.target] < max_degree]
        if not feasible:
            break
        best = min(feasible, key=lambda p: (-(int(degree[p.source] == 0) + int(degree[p.target] == 0)),
                                            max(degree[p.source], degree[p.target]),
                                            degree[p.source] + degree[p.target], order[p.source], order[p.target]))
        chosen.append(best)
        degree.update((best.source, best.target))
        pending.remove(best)
    return chosen, [p.model_dump() for p in pending]
