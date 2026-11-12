"""Controlled vocabularies for entity and relation types.

``entity_type`` is a closed enum (user-extensible as a concept scheme, never
free text). ``relation_type`` is a controlled vocabulary with domain/range
constraints; the executor validates before writing and violations are
deterministic rejections, not silent graph pollution.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class EntityType(StrEnum):
    PERSON = "Person"
    ORGANIZATION = "Organization"
    PROJECT = "Project"
    TECHNOLOGY = "Technology"
    SKILL = "Skill"
    PREFERENCE = "Preference"
    LOCATION = "Location"
    EVENT = "Event"
    CONCEPT = "Concept"
    OTHER = "Other"


_ANY: frozenset[EntityType] = frozenset(EntityType)
_PERSON = frozenset({EntityType.PERSON})
_ORG = frozenset({EntityType.ORGANIZATION})
_PROJECT = frozenset({EntityType.PROJECT})
_TECH = frozenset({EntityType.TECHNOLOGY, EntityType.SKILL})
_LOCATION = frozenset({EntityType.LOCATION})


@dataclass(frozen=True)
class RelationSpec:
    type: RelationType
    domain: frozenset[EntityType]
    range: frozenset[EntityType]
    label: str


class RelationType(StrEnum):
    WORKS_AT = "WORKS_AT"  # Person -> Organization
    WORKED_AT = "WORKED_AT"  # Person -> Organization (past; interval-closed)
    LIVES_IN = "LIVES_IN"  # Person -> Location
    BORN_IN = "BORN_IN"  # Person -> Location
    MANAGES = "MANAGES"  # Person -> Organization | Project
    MEMBER_OF = "MEMBER_OF"  # Person -> Organization
    CREATED = "CREATED"  # Person | Organization -> Project | Technology
    USES = "USES"  # Person | Organization -> Technology | Skill
    PREFERS = "PREFERS"  # Person -> any
    KNOWS = "KNOWS"  # Person -> Person
    ATTENDED = "ATTENDED"  # Person -> Event
    LOCATED_IN = "LOCATED_IN"  # Organization | Project -> Location
    PART_OF = "PART_OF"  # any -> Project | Organization
    RELATED_TO = "RELATED_TO"  # any -> any
    SUPERSEDES = "SUPERSEDES"  # version chain on facts/edges


_RELATIONS: dict[RelationType, RelationSpec] = {
    RelationType.WORKS_AT: RelationSpec(RelationType.WORKS_AT, _PERSON, _ORG, "works at"),
    RelationType.WORKED_AT: RelationSpec(RelationType.WORKED_AT, _PERSON, _ORG, "worked at"),
    RelationType.LIVES_IN: RelationSpec(RelationType.LIVES_IN, _PERSON, _LOCATION, "lives in"),
    RelationType.BORN_IN: RelationSpec(RelationType.BORN_IN, _PERSON, _LOCATION, "born in"),
    RelationType.MANAGES: RelationSpec(RelationType.MANAGES, _PERSON, _ORG | _PROJECT, "manages"),
    RelationType.MEMBER_OF: RelationSpec(RelationType.MEMBER_OF, _PERSON, _ORG, "member of"),
    RelationType.CREATED: RelationSpec(RelationType.CREATED, _PERSON | _ORG, _PROJECT | _TECH, "created"),
    RelationType.USES: RelationSpec(RelationType.USES, _PERSON | _ORG, _TECH, "uses"),
    RelationType.PREFERS: RelationSpec(RelationType.PREFERS, _PERSON, _ANY, "prefers"),
    RelationType.KNOWS: RelationSpec(RelationType.KNOWS, _PERSON, _PERSON, "knows"),
    RelationType.ATTENDED: RelationSpec(RelationType.ATTENDED, _PERSON, frozenset({EntityType.EVENT}), "attended"),
    RelationType.LOCATED_IN: RelationSpec(RelationType.LOCATED_IN, _ORG | _PROJECT, _LOCATION, "located in"),
    RelationType.PART_OF: RelationSpec(RelationType.PART_OF, _ANY, _PROJECT | _ORG, "part of"),
    RelationType.RELATED_TO: RelationSpec(RelationType.RELATED_TO, _ANY, _ANY, "related to"),
    RelationType.SUPERSEDES: RelationSpec(RelationType.SUPERSEDES, _ANY, _ANY, "supersedes"),
}


def relation_spec(rel: RelationType | str) -> RelationSpec:
    return _RELATIONS[RelationType(rel)]


_ENTITY_BY_KEY = {e.value.lower(): e for e in EntityType}
_RELATION_BY_KEY = {r.value: r for r in RelationType}


def coerce_entity_type(value: EntityType | str | None) -> EntityType | None:
    """The controlled entity type for an LLM-extracted type, case-insensitively
    ("person", "PERSON" and "Person" are all Person); None when unknown."""
    if isinstance(value, EntityType):
        return value
    if not value:
        return None
    return _ENTITY_BY_KEY.get(str(value).strip().lower())


def coerce_relation_type(value: RelationType | str | None) -> RelationType | None:
    """The controlled relation type, ignoring case and spaces ("works at", "works_at"
    and "WORKS_AT" are all WORKS_AT); None when not in the vocabulary."""
    if isinstance(value, RelationType):
        return value
    if not value:
        return None
    return _RELATION_BY_KEY.get("_".join(str(value).strip().upper().replace("-", " ").split()))


def validate_relation(rel: RelationType | str, subject_type: EntityType | str, object_type: EntityType | str) -> bool:
    """Domain/range check for the executor. Unknown types fail closed."""
    rt = coerce_relation_type(rel)
    s, o = coerce_entity_type(subject_type), coerce_entity_type(object_type)
    if rt is None or s is None or o is None:
        return False
    spec = relation_spec(rt)
    return s in spec.domain and o in spec.range


def is_violation(rel: str, subject_type: str, object_type: str) -> bool:
    """True only when ``rel`` is in the controlled vocabulary AND both endpoint
    types are known AND the domain/range constraint is violated. Unknown
    vocabulary (free-text relation types from extraction, unmapped entity types)
    is NOT a violation, pass-through baseline behavior, never a crash."""
    rt = coerce_relation_type(rel)
    s, o = coerce_entity_type(subject_type), coerce_entity_type(object_type)
    if rt is None or s is None or o is None:
        return False
    spec = relation_spec(rt)
    return not (s in spec.domain and o in spec.range)
