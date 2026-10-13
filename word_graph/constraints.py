"""Batch limits and domain-independent predicate identifier constraints."""
from __future__ import annotations

import re
import string

PROTECTED_RELATIONS = {"subClassOf", "subFieldOf", "disjointWith"}

MAX_INPUT_TERMS = 200
SUBJECT_BATCH_SIZE = 10


def subject_batches(terms: list[str], size: int = SUBJECT_BATCH_SIZE) -> list[list[str]]:
    if size < 1:
        raise ValueError("Batch size must be positive")
    return [terms[start:start + size] for start in range(0, len(terms), size)]


def predicate_name_pattern(terms: list[str], allow_underscores: bool = True) -> str:
    """Allow arbitrary identifiers while preventing protected-prefix aliases.

    Build a prefix trie rather than unsupported regex lookarounds. This does not
    supply domain predicates: the unrestricted branches still accept any other
    relation name, and the model chooses the vocabulary.
    """
    canonical = sorted(PROTECTED_RELATIONS)
    blocked = set()
    for name in canonical:
        blocked.update((name, name.lower(), name[0].upper() + name[1:]))
    for name in ("refersTo", "relatedTo", "associatedWith", "linkedTo", "connectedTo", "appliesTo"):
        blocked.update((name, name.lower(), name[0].upper() + name[1:]))
    for term in terms:
        words = re.findall(r"[A-Za-z0-9]+", term)
        if len(words) > 1:
            camel = words[0].lower() + "".join(word.capitalize() for word in words[1:])
            blocked.update((camel, camel.lower(), camel[0].upper() + camel[1:]))
    trie = {}
    for prefix in blocked:
        cursor = trie
        for character in prefix:
            cursor = cursor.setdefault(character, {})
        cursor[None] = True
    alphabet = string.ascii_letters + string.digits + ("_" if allow_underscores else "")
    continuation = "[A-Za-z0-9_]*" if allow_underscores else "[A-Za-z0-9]*"
    def suffix(node, depth):
        if None in node:
            return None
        parts = [""] if depth else []
        if depth < 64:
            candidates = (string.ascii_letters if allow_underscores else string.ascii_lowercase) if depth == 0 else alphabet
            diverging = "".join(c for c in candidates if c not in node)
            if diverging:
                # Identifier length is checked by Pydantic. An unbounded suffix
                # keeps this automaton small instead of expanding a length
                # counter at every excluded-prefix branch.
                parts.append("[" + diverging + "]" + continuation)
            for character, child in sorted(node.items()):
                remainder = suffix(child, depth + 1)
                if remainder is not None:
                    parts.append(re.escape(character) + remainder)
        return "(" + "|".join(parts) + ")"
    return "(" + "|".join(canonical) + "|" + suffix(trie, 0) + ")"
