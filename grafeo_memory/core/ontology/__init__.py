"""Layer D ontological core: SKOS-style concept scheme, controlled vocabularies,
the deterministic entity-normalization ladder, and pre-model temporal rules.

Principle: models propose, standards validate. The store holds
only canonical content; decision-model choices are generated from the scheme,
never free text.
"""

from __future__ import annotations

from grafeo_memory.core.ontology.normalize import Resolution, cosine, normalize
from grafeo_memory.core.ontology.scheme import Concept, ConceptScheme, normalize_key
from grafeo_memory.core.ontology.temporal_rules import point_in_time_filter, pre_model_conflict
from grafeo_memory.core.ontology.types import EntityType, RelationType, is_violation, relation_spec, validate_relation

__all__ = [
    "Concept",
    "ConceptScheme",
    "EntityType",
    "RelationType",
    "Resolution",
    "cosine",
    "is_violation",
    "normalize",
    "normalize_key",
    "point_in_time_filter",
    "pre_model_conflict",
    "relation_spec",
    "validate_relation",
]
