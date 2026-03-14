"""Core types for grafeo-memory."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.usage import RunUsage

# Re-export Pydantic schemas from their new home for backward compatibility
from .schemas import (
    EntitiesOutput,
    EntityItem,
    FactsOutput,
    ReconciliationItem,
    ReconciliationOutput,
    RelationDeleteItem,
    RelationItem,
    RelationReconciliationOutput,
)


class MemoryAction(Enum):
    """Decision action for a memory reconciliation step."""

    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NONE = "none"


class MemoryType(StrEnum):
    """Type classification for memories."""

    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    EPISODIC = "episodic"


@dataclass
class MemoryConfig:
    """Configuration for a MemoryManager instance."""

    db_path: str | None = None
    user_id: str = "default"
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    reconciliation_threshold: float = 0.3
    search_min_score: float = 0.0
    agreement_bonus: float = 0.1
    embedding_dimensions: int = 1536
    vector_property: str = "embedding"
    text_property: str = "text"
    custom_fact_prompt: str | None = None
    custom_update_prompt: str | None = None
    custom_procedural_prompt: str | None = None
    # Importance scoring (opt-in)
    enable_importance: bool = False
    decay_rate: float = 0.1
    weight_similarity: float = 0.4
    weight_recency: float = 0.3
    weight_frequency: float = 0.15
    weight_importance: float = 0.15
    # LLM usage tracking (opt-in)
    usage_callback: Callable[[str, RunUsage], None] | None = None
    # Topology-aware scoring (opt-in, requires enable_importance)
    weight_topology: float = 0.0
    # Structural decay modulation (opt-in, requires enable_importance)
    enable_structural_decay: bool = False
    structural_feedback_gamma: float = 0.3
    # Topology boost in search pipeline (opt-in, no LLM call)
    enable_topology_boost: bool = False
    topology_boost_factor: float = 0.2
    # Topology-aware consolidation: protect well-connected memories from summarize()
    consolidation_protect_threshold: float = 0.0  # 0.0 = consolidate everything (backward compat)
    # OpenTelemetry instrumentation (opt-in)
    instrument: InstrumentationSettings | bool = False
    # Vision / multimodal (opt-in)
    enable_vision: bool = False
    vision_model: object | None = None

    def __post_init__(self) -> None:
        if self.embedding_dimensions <= 0:
            raise ValueError(f"embedding_dimensions must be positive, got {self.embedding_dimensions}")
        if not (0.0 <= self.reconciliation_threshold <= 1.0):
            raise ValueError(f"reconciliation_threshold must be in [0.0, 1.0], got {self.reconciliation_threshold}")
        if not (0.0 <= self.search_min_score <= 1.0):
            raise ValueError(f"search_min_score must be in [0.0, 1.0], got {self.search_min_score}")
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")

        for name in (
            "weight_similarity",
            "weight_recency",
            "weight_frequency",
            "weight_importance",
            "weight_topology",
            "topology_boost_factor",
            "structural_feedback_gamma",
            "consolidation_protect_threshold",
            "agreement_bonus",
        ):
            val = getattr(self, name)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")

        weight_sum = self.weight_similarity + self.weight_recency + self.weight_frequency + self.weight_importance
        if abs(weight_sum - 1.0) > 0.05:
            warnings.warn(
                f"Core importance weights sum to {weight_sum:.3f}, expected ~1.0. "
                f"Composite scores may not behave as expected.",
                UserWarning,
                stacklevel=2,
            )

    @classmethod
    def yolo(cls, **kwargs) -> MemoryConfig:
        """Create a config with all optional features enabled.

        Turns on importance scoring, vision, and (if no usage_callback
        is supplied) a default stderr logger for LLM token usage.
        """
        import sys

        defaults: dict = {
            "enable_importance": True,
            "enable_vision": True,
            "instrument": True,
        }
        defaults.update(kwargs)

        if "usage_callback" not in defaults:

            def _stderr_usage(operation: str, usage: RunUsage) -> None:
                print(f"[usage] {operation}: {usage}", file=sys.stderr)

            defaults["usage_callback"] = _stderr_usage

        return cls(**defaults)


@dataclass
class MemoryEvent:
    """A single action taken during a memory add operation."""

    action: MemoryAction
    memory_id: str | None = None
    text: str = ""
    old_text: str | None = None
    actor_id: str | None = None
    role: str | None = None
    memory_type: str = "semantic"


@dataclass
class SearchResult:
    """A single search result from memory."""

    memory_id: str
    text: str
    score: float
    user_id: str
    metadata: dict | None = None
    relations: list[dict] | None = None
    actor_id: str | None = None
    role: str | None = None
    importance: float | None = None
    access_count: int | None = None
    memory_type: str | None = None
    source: str | None = None


@dataclass
class Fact:
    """A discrete fact extracted from conversation text."""

    text: str


@dataclass
class Entity:
    """An entity extracted from conversation text."""

    name: str
    entity_type: str


@dataclass
class Relation:
    """A relationship between two entities."""

    source: str
    target: str
    relation_type: str


@dataclass
class ReconciliationDecision:
    """A reconciliation decision for a single fact against existing memories."""

    action: MemoryAction
    text: str
    target_memory_id: str | None = None


@dataclass
class ExtractionResult:
    """Combined output from the extraction phase."""

    facts: list[Fact] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)


@dataclass
class MemoryStats:
    """Database introspection snapshot."""

    total_memories: int = 0
    semantic_count: int = 0
    procedural_count: int = 0
    episodic_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    db_info: dict = field(default_factory=dict)


@dataclass
class ExplainStep:
    """A single stage in the search pipeline trace."""

    name: str
    detail: dict = field(default_factory=dict)


@dataclass
class ExplainResult:
    """Full trace of a search pipeline execution."""

    query: str
    steps: list[ExplainStep] = field(default_factory=list)
    results: list[SearchResult] = field(default_factory=list)


# Graph schema constants
MEMORY_LABEL = "Memory"
ENTITY_LABEL = "Entity"
HAS_ENTITY_EDGE = "HAS_ENTITY"
RELATION_EDGE = "RELATION"
DERIVED_FROM_EDGE = "DERIVED_FROM"


class AddResult(list):
    """List of MemoryEvents with aggregated LLM usage."""

    def __init__(self, events=(), usage=None):
        super().__init__(events)
        from pydantic_ai.usage import RunUsage

        self.usage: RunUsage = usage or RunUsage()


class SearchResponse(list):
    """List of SearchResults with aggregated LLM usage."""

    def __init__(self, results=(), usage=None):
        super().__init__(results)
        from pydantic_ai.usage import RunUsage

        self.usage: RunUsage = usage or RunUsage()


__all__ = [
    "DERIVED_FROM_EDGE",
    "ENTITY_LABEL",
    "HAS_ENTITY_EDGE",
    "MEMORY_LABEL",
    "RELATION_EDGE",
    "AddResult",
    "EntitiesOutput",
    "Entity",
    "EntityItem",
    "ExplainResult",
    "ExplainStep",
    "ExtractionResult",
    "Fact",
    "FactsOutput",
    "MemoryAction",
    "MemoryConfig",
    "MemoryEvent",
    "MemoryStats",
    "MemoryType",
    "ReconciliationDecision",
    "ReconciliationItem",
    "ReconciliationOutput",
    "Relation",
    "RelationDeleteItem",
    "RelationItem",
    "RelationReconciliationOutput",
    "SearchResponse",
    "SearchResult",
]
