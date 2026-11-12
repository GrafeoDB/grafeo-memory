"""The deterministic entity-normalization ladder.

1. Exact prefLabel match (string equality, typed by entity_type).
2. altLabel / notation table (deterministic map: "JS" -> "JavaScript").
3. Type-constrained embedding similarity above threshold.
4. Only the genuinely ambiguous residue reaches a model: we return the
   candidate canonical IDs so the caller can ask one CategoricalQuestion.

Rungs 1-2 cost zero model calls.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field

from grafeo_memory.core.ontology.scheme import ConceptScheme
from grafeo_memory.core.ontology.types import EntityType, coerce_entity_type

EmbedFn = Callable[[str], list[float]]


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Resolution:
    mention: str
    entity_type: str
    canonical_id: str | None
    rung: int  # 1..4; 4 = unresolved residue for a categorical question
    confidence: float
    candidates: list[str] = field(default_factory=list)  # rung 4: canonical IDs to choose from

    @property
    def resolved(self) -> bool:
        return self.canonical_id is not None


def normalize(
    mention: str,
    entity_type: EntityType | str,
    scheme: ConceptScheme,
    embed: EmbedFn | None = None,
    threshold: float = 0.92,
    max_candidates: int = 8,
    concept_vectors: dict[str, list[float]] | None = None,
) -> Resolution:
    """Resolve a surface mention to a canonical concept. Deterministic rungs first."""
    coerced = coerce_entity_type(entity_type)
    et = coerced.value if coerced is not None else EntityType(entity_type).value  # unknown type raises
    # Rung 1: exact prefLabel, typed.
    hit = scheme.lookup_pref(mention, et)
    if hit is not None:
        return Resolution(mention, et, hit, rung=1, confidence=1.0)
    # Rung 2: alias table.
    hit = scheme.lookup_alias(mention, et)
    if hit is not None:
        return Resolution(mention, et, hit, rung=2, confidence=1.0)
    # Rung 3: type-constrained embedding similarity (only if an embedder is wired).
    same_type = [c for c in scheme.concepts.values() if c.entity_type == et]
    if embed is not None and same_type:
        qv = embed(mention)
        scored: list[tuple[float, str]] = []
        for c in same_type:
            cv = (concept_vectors or {}).get(c.id)
            if cv is None:
                cv = embed(c.pref_label)
                if concept_vectors is not None:
                    concept_vectors[c.id] = cv
            scored.append((cosine(qv, cv), c.id))
        scored.sort(reverse=True)
        best_score, best_id = scored[0]
        if best_score >= threshold:
            return Resolution(mention, et, best_id, rung=3, confidence=best_score)
        # Rung 4: residue, hand the ranked candidates to a categorical question.
        candidates = [cid for _, cid in scored[:max_candidates]]
        return Resolution(mention, et, None, rung=4, confidence=best_score, candidates=candidates)
    # No embedder: everything unresolved is residue.
    candidates = [c.id for c in same_type[:max_candidates]]
    return Resolution(mention, et, None, rung=4, confidence=0.0, candidates=candidates)
