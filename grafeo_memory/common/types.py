"""Core types for grafeo-memory."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from pydantic_ai.models import KnownModelName, Model
    from pydantic_ai.models.instrumented import InstrumentationSettings
    from pydantic_ai.usage import RunUsage


# Re-export Pydantic schemas from their new home for backward compatibility
from grafeo_memory.common.schemas import (
    EntitiesOutput,
    EntityItem,
    FactsOutput,
    ReconciliationItem,
    ReconciliationOutput,
    RelationDeleteItem,
    RelationItem,
    RelationReconciliationOutput,
)

#: Model parameter type accepted by pydantic-ai Agent.
type ModelType = Model | KnownModelName | str | None


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


def _normalize_history_retention(value: str | timedelta | float) -> str | timedelta:
    """Validate and normalize ``history_retention``: "off", "forever", a timedelta, or seconds."""
    if isinstance(value, str):
        low = value.lower()
        if low not in ("off", "forever"):
            raise ValueError(f"history_retention must be 'off', 'forever', a timedelta or seconds, got {value!r}")
        return low
    if isinstance(value, bool):
        raise ValueError(f"history_retention must be 'off', 'forever', a timedelta or seconds, got {value!r}")
    if isinstance(value, int | float):
        if value <= 0:
            raise ValueError(f"history_retention in seconds must be positive, got {value!r}")
        return timedelta(seconds=value)
    if isinstance(value, timedelta):
        if value < timedelta(0):
            raise ValueError(f"history_retention must not be negative, got {value!r}")
        return value
    raise ValueError(f"history_retention must be 'off', 'forever', a timedelta or seconds, got {value!r}")


@dataclass
class CacheConfig:
    """In-memory LRU cache for LLM / embedding / decision-model calls.

    Enabled via ``MemoryConfig(cache=CacheConfig())`` or ``cache=True``
    (defaults below), or process-wide with ``GRAFEO_MEMORY_CACHE_ENABLED=true``.

    - ``max_entries`` / ``max_memory_bytes`` bound the in-memory hot cache
      (LRU eviction).
    - ``max_storage_bytes`` bounds the optional JSONL persistence file
      (``persist_path``); when the file grows past it, it is compacted to the
      live entries. ``persist_path=None`` means no persistence unless the
      manager derives one from ``db_path`` (``<db_path>.llmcache.jsonl``).
    - ``ttl_seconds`` expires entries; None (default) keeps them forever so
      benchmark reruns stay deterministic.
    """

    enabled: bool = True
    max_entries: int = 10_000
    max_memory_bytes: int = 128 * 1024 * 1024
    max_storage_bytes: int = 512 * 1024 * 1024
    persist_path: str | None = None
    ttl_seconds: float | None = None


class Engine(StrEnum):
    """Which engine runs a pipeline stage."""

    LLM = "llm"  # generative LLM (pydantic-ai agent), today's behavior, the default
    DECISION = "decision"  # decision model only (typed questions, no LLM call)
    HYBRID = "hybrid"  # decision model first, low-confidence items escalate to one LLM call


class Granularity(StrEnum):
    """Question-pack granularity for decision-model reconciliation."""

    PER_FACT = "per_fact"  # one categorical per fact, choices carry candidate memory IDs
    PAIRWISE = "pairwise"  # one question per (fact, candidate), argmax over candidates after


class EscalationSignal(StrEnum):
    """Which signal HYBRID uses to decide a decision is too uncertain for the LLM to skip."""

    CONFIDENCE = "confidence"  # provider's calibrated confidence below threshold -> escalate
    PROB_GAP = "prob_gap"  # p(chosen) - p(runner-up) below threshold -> escalate
    COMBINED = "combined"  # either signal below threshold -> escalate


@dataclass
class StageConfig:
    """Per-stage engine configuration.

    ``facts`` / ``entities`` / ``temporal`` are generation-only and always LLM
    (enforced in ``MemoryConfig.__post_init__``). ``memories``, ``relations``
    and ``search`` can run on a decision model.
    """

    engine: Engine = Engine.LLM
    confidence_threshold: float = 0.75  # HYBRID escalation cut-off, in [0, 1]
    granularity: Granularity = Granularity.PER_FACT  # memories/relations question pack
    escalation_signal: EscalationSignal = EscalationSignal.CONFIDENCE
    # Search stage only: blend weight of the decision-model score vs the retrieval
    # score. 0.0 = pure retrieval score, 1.0 = pure decision score.
    rerank_blend_weight: float = 0.5


@dataclass
class PipelineConfig:
    """Engine selection per pipeline stage. Defaults are LLM everywhere, exactly
    today's behavior. Any non-LLM engine requires ``MemoryConfig.decision_model``."""

    facts: StageConfig = field(default_factory=StageConfig)  # LLM-only, enforced
    entities: StageConfig = field(default_factory=StageConfig)  # LLM-only, enforced
    temporal: StageConfig = field(default_factory=StageConfig)  # LLM-only for now
    memories: StageConfig = field(default_factory=StageConfig)  # decision-able
    relations: StageConfig = field(default_factory=StageConfig)  # decision-able
    search: StageConfig = field(default_factory=StageConfig)  # rerank, decision-able
    # One global question-pack override for the decision side: "per_fact",
    # "pairwise", or a callable (facts, candidates) -> list[Question]. None = each
    # stage's own granularity.
    question_pack: str | Callable[..., Any] | None = None
    # Layer D pre-model switches (pure code, zero model calls; False = exactly
    # today's behavior). temporal_precheck: a new fact whose validity interval is
    # disjoint from every reconciliation candidate is deterministically ADD
    # ("both stand") and never reaches the engine. ontology: deterministic entity
    # normalization (ladder rungs 1-2) + controlled relation-type validation
    # before graph writes.
    temporal_precheck: bool = False
    ontology: bool = False
    # --- Composable layer toggles ---------------------------------------------
    # Every layer is an independent switch; all default False = exactly today's
    # behavior. Users compose the set that works for them (bench flags map 1:1).
    # Layer D storage: validity intervals on facts/edges (sugar over
    # MemoryConfig.enable_bitemporal; either flag turns it on, see MemoryConfig.bitemporal_on).
    bitemporal: bool = False
    # Layer A: episodic tier (either this or MemoryConfig.enable_episodes turns it on).
    episodes: bool = False
    # Layer E: the bench harnesses call memory.compile() after ingest when set. The
    # manager does not read it; applications call compile() themselves.
    compile: bool = False
    # Layer B: faceted-taxonomy scoping/routing. Not implemented yet, no-op + warning.
    taxonomy: bool = False
    # Layer F: policy cascade (rule packs over the canonical store). Not yet, no-op.
    policy: bool = False
    # Layer G: open-decisions bucket. Not implemented yet, no-op + warning.
    bucket: bool = False


@dataclass
class MemoryConfig:
    """Configuration for a MemoryManager instance."""

    db_path: str | None = None
    user_id: str = "default"
    session_id: str | None = None
    agent_id: str | None = None
    run_id: str | None = None
    graph_name: str | None = None
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
    # Retries for transient LLM provider errors (timeouts, connection errors, 429, 5xx), with
    # exponential backoff; 0 disables. Other errors fail at once, as before.
    llm_retries: int = 3
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
    vision_model: ModelType | None = None
    # Multi-hop graph search depth (1 = current behavior, 2 = entity→relation→entity→memory)
    graph_search_depth: int = 1
    # Re-sort results chronologically when temporal keywords ("when did", "latest", ...)
    # are detected in the query. Off keeps relevance (score) order, which is what
    # QA-style retrieval wants; the answerer can order events itself.
    chronological_sort: bool = True
    # RRF constant for hybrid (BM25 + vector) search. 1 compresses ranks and is tuned
    # for small collections (10-100 memories); larger collections want ~60.
    hybrid_rrf_k: int = 1
    # MMR search for diversity (opt-in, uses engine's mmr_search)
    mmr_lambda: float = 0.5
    # Cross-session entity reinforcement boost factor
    cross_session_factor: float = 0.0
    # Graph algorithm-based scoring (opt-in, replaces hand-coded topology)
    enable_graph_algorithms: bool = False
    # Bi-temporal model: extract real-world validity times from facts (opt-in)
    enable_bitemporal: bool = False
    # LLM/embedding/decision call cache (opt-in; False = off, True = CacheConfig() defaults)
    cache: CacheConfig | bool = False
    # GrafeoDB engine limits (bytes). None = grafeo's detected machine/container
    # default. memory_limit is enforced per write (cold chunks spill to disk);
    # spill_limit caps the on-disk spill/WAL side.
    db_memory_limit_bytes: int | None = None
    db_spill_limit_bytes: int | None = None
    # Episode provenance: store raw messages as Episode nodes with provenance edges (opt-in)
    enable_episodes: bool = False
    # Community summaries: LLM-generated summaries for entity clusters (opt-in, requires enable_graph_algorithms)
    enable_community_summaries: bool = False
    # Time travel (grafeo 0.6.2 history retention): how far back the database keeps its
    # history for as_of reads: "off", "forever", a timedelta, or seconds as a number.
    # None keeps the engine default ("off" for a new database, the stored value for a file).
    history_retention: str | timedelta | float | None = None
    # Decision model for non-LLM pipeline stages (see PipelineConfig): a registered
    # name ("jev", "fake", or an entry point in "grafeo_memory.decision_models"),
    # or a DecisionModel instance. None = all stages run on the LLM (default).
    decision_model: str | Any | None = None  # Any = DecisionModel (grafeo_memory.providers.decision.protocol)
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)

    @property
    def bitemporal_on(self) -> bool:
        """Bi-temporal storage is on when ``enable_bitemporal`` or ``pipeline.bitemporal`` is set.
        Read at use, never written back, so changing either flag later takes effect."""
        return bool(self.enable_bitemporal or self.pipeline.bitemporal)

    @property
    def episodes_on(self) -> bool:
        """Episodes are on when ``enable_episodes`` or ``pipeline.episodes`` is set."""
        return bool(self.enable_episodes or self.pipeline.episodes)

    def __post_init__(self) -> None:
        if self.enable_community_summaries and not self.enable_graph_algorithms:
            raise ValueError("enable_community_summaries requires enable_graph_algorithms=True")
        if self.history_retention is not None:
            self.history_retention = _normalize_history_retention(self.history_retention)
        if self.embedding_dimensions <= 0:
            raise ValueError(f"embedding_dimensions must be positive, got {self.embedding_dimensions}")
        if not (0.0 <= self.reconciliation_threshold <= 1.0):
            raise ValueError(f"reconciliation_threshold must be in [0.0, 1.0], got {self.reconciliation_threshold}")
        if not (0.0 <= self.search_min_score <= 1.0):
            raise ValueError(f"search_min_score must be in [0.0, 1.0], got {self.search_min_score}")
        if self.decay_rate <= 0:
            raise ValueError(f"decay_rate must be positive, got {self.decay_rate}")

        if isinstance(self.cache, CacheConfig):
            if self.cache.max_entries <= 0:
                raise ValueError(f"cache.max_entries must be positive, got {self.cache.max_entries}")
            if self.cache.max_memory_bytes <= 0:
                raise ValueError(f"cache.max_memory_bytes must be positive, got {self.cache.max_memory_bytes}")
            if self.cache.max_storage_bytes <= 0:
                raise ValueError(f"cache.max_storage_bytes must be positive, got {self.cache.max_storage_bytes}")
        if self.llm_retries < 0:
            raise ValueError(f"llm_retries must be >= 0, got {self.llm_retries}")
        if self.db_memory_limit_bytes is not None and self.db_memory_limit_bytes <= 0:
            raise ValueError(f"db_memory_limit_bytes must be positive, got {self.db_memory_limit_bytes}")
        if self.db_spill_limit_bytes is not None and self.db_spill_limit_bytes <= 0:
            raise ValueError(f"db_spill_limit_bytes must be positive, got {self.db_spill_limit_bytes}")

        # Pipeline engine validation: generation-only stages stay LLM; any non-LLM
        # engine needs a decision model. Enforced here so misconfiguration fails at
        # construction, not mid-write.
        pipeline = self.pipeline
        for stage_name in ("facts", "entities", "temporal", "memories", "relations", "search"):
            stage = getattr(pipeline, stage_name)
            stage.engine = Engine(stage.engine)
            stage.granularity = Granularity(stage.granularity)
            stage.escalation_signal = EscalationSignal(stage.escalation_signal)
            if not (0.0 <= stage.confidence_threshold <= 1.0):
                raise ValueError(
                    f"pipeline.{stage_name}.confidence_threshold must be in [0.0, 1.0], "
                    f"got {stage.confidence_threshold}"
                )
            if not (0.0 <= stage.rerank_blend_weight <= 1.0):
                raise ValueError(
                    f"pipeline.{stage_name}.rerank_blend_weight must be in [0.0, 1.0], got {stage.rerank_blend_weight}"
                )
            if stage_name in ("facts", "entities", "temporal") and stage.engine != Engine.LLM:
                raise ValueError(
                    f"pipeline.{stage_name} is a generation-only stage and must use Engine.LLM, "
                    f"got {stage.engine.value!r}"
                )
            if stage.engine != Engine.LLM and self.decision_model is None:
                raise ValueError(
                    f"pipeline.{stage_name}.engine={stage.engine.value!r} requires MemoryConfig.decision_model "
                    "to be set (a registered name like 'jev' or a DecisionModel instance)"
                )
        if (
            pipeline.question_pack is not None
            and not isinstance(pipeline.question_pack, str)
            and not callable(pipeline.question_pack)
        ):
            raise ValueError("pipeline.question_pack must be a pack name or a callable")
        # pack-name membership is validated by the registry in providers.decision

        if self.graph_search_depth not in (1, 2):
            raise ValueError(f"graph_search_depth must be 1 or 2, got {self.graph_search_depth}")

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
            "mmr_lambda",
            "cross_session_factor",
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
    valid_at: int | None = None
    # Which engine produced this event ("llm", "decision", "fast") and how
    # confident it was (None when the provider gave no calibrated confidence).
    engine: str = "llm"
    confidence: float | None = None


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
    created_at: int | None = None
    learned_at: int | None = None
    session_id: str | None = None
    expired_at: int | None = None
    valid_at: int | None = None
    invalid_at: int | None = None


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
    # Which engine made this decision ("llm", "decision", "fast") and its
    # calibrated confidence, when the engine provides one.
    engine: str = "llm"
    confidence: float | None = None


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for text embedding generation (implemented in providers.embedding)."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Embedding vector size."""
        ...


@dataclass
class ExtractionResult:
    """Combined output from the extraction phase."""

    facts: list[Fact] = field(default_factory=list)
    entities: list[Entity] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    temporal_annotations: dict[int, tuple[int | None, int | None]] = field(default_factory=dict)


@dataclass
class MemoryStats:
    """Database introspection snapshot."""

    total_memories: int = 0
    semantic_count: int = 0
    procedural_count: int = 0
    episodic_count: int = 0
    entity_count: int = 0
    relation_count: int = 0
    episode_count: int = 0
    community_count: int = 0
    db_info: dict = field(default_factory=dict)
    # LLM cache stats (empty when the cache is disabled)
    cache: dict = field(default_factory=dict)
    # Engine limits vs usage: memory_bytes/memory_limit_bytes, disk_bytes, spill_limit_bytes
    db_limits: dict = field(default_factory=dict)


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
EPISODE_LABEL = "Episode"
HAS_ENTITY_EDGE = "HAS_ENTITY"
RELATION_EDGE = "RELATION"
DERIVED_FROM_EDGE = "DERIVED_FROM"
SUPERSEDES_EDGE = "SUPERSEDES"
LEADS_TO_EDGE = "LEADS_TO"
PRODUCED_EDGE = "PRODUCED"
MENTIONS_EDGE = "MENTIONS"
NEXT_EPISODE_EDGE = "NEXT_EPISODE"
COMMUNITY_LABEL = "Community"
HAS_MEMBER_EDGE = "HAS_MEMBER"


@dataclass
class EpisodeResult:
    """A raw episode with provenance information."""

    episode_id: str
    content: str
    source: str
    user_id: str
    session_id: str | None = None
    run_id: str | None = None
    created_at: int | None = None
    valid_at: int | None = None
    produced_memories: list[str] = field(default_factory=list)
    mentioned_entities: list[str] = field(default_factory=list)


@dataclass
class CommunityInfo:
    """A detected community with its LLM summary."""

    community_id: str
    name: str
    summary: str
    member_count: int
    member_entities: list[str] = field(default_factory=list)


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
    "COMMUNITY_LABEL",
    "DERIVED_FROM_EDGE",
    "ENTITY_LABEL",
    "EPISODE_LABEL",
    "HAS_ENTITY_EDGE",
    "HAS_MEMBER_EDGE",
    "LEADS_TO_EDGE",
    "MEMORY_LABEL",
    "MENTIONS_EDGE",
    "NEXT_EPISODE_EDGE",
    "PRODUCED_EDGE",
    "RELATION_EDGE",
    "SUPERSEDES_EDGE",
    "AddResult",
    "CommunityInfo",
    "EntitiesOutput",
    "Entity",
    "EntityItem",
    "EpisodeResult",
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
