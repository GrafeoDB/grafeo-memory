"""Pydantic output schemas for LLM structured output.

These BaseModel classes are used as `output_type` for pydantic-ai Agents.
They define the expected JSON structure the LLM should return.
"""

from __future__ import annotations

from pydantic import BaseModel

# --- Fact extraction ---


class FactsOutput(BaseModel):
    """Output schema for fact extraction."""

    facts: list[str]


# --- Entity / relation extraction ---


class EntityItem(BaseModel):
    """A single entity in extraction output."""

    name: str
    entity_type: str


class RelationItem(BaseModel):
    """A single relation in extraction output."""

    source: str
    target: str
    relation_type: str


class EntitiesOutput(BaseModel):
    """Output schema for entity/relation extraction."""

    entities: list[EntityItem]
    relations: list[RelationItem] = []


class ExtractionOutput(BaseModel):
    """Combined output for fact + entity extraction in a single LLM call."""

    facts: list[str]
    entities: list[EntityItem] = []
    relations: list[RelationItem] = []


# --- Memory reconciliation ---


class ReconciliationItem(BaseModel):
    """A single reconciliation decision in output."""

    action: str
    text: str = ""
    target_memory_id: str | None = None


class ReconciliationOutput(BaseModel):
    """Output schema for memory reconciliation."""

    decisions: list[ReconciliationItem]


# --- Relation reconciliation ---


class RelationDeleteItem(BaseModel):
    """A single relation to delete."""

    source: str
    target: str
    relation_type: str


class RelationReconciliationOutput(BaseModel):
    """Output schema for relation reconciliation."""

    delete: list[RelationDeleteItem]


# --- Summarization / consolidation ---


class SummarizeOutput(BaseModel):
    """Output schema for memory consolidation."""

    memories: list[str]
