"""Core types for grafeo-memory."""

from __future__ import annotations

import warnings
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import timedelta
from enum import Enum, StrEnum
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

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
    # Layer B: faceted-taxonomy scoping (sugar over MemoryConfig.taxonomy.enabled).
    taxonomy: bool = False
    # Layer F: policy cascade, rule packs over the canonical store (sugar over MemoryConfig.policy.enabled).
    policy: bool = False
    # Layer G: open-decisions bucket (sugar over MemoryConfig.bucket.enabled).
    bucket: bool = False


# --- Layer configs (all default off = exactly the behavior without them) ---------


@dataclass
class OntologyConfig:
    """Layer D knobs beyond the ``pipeline.ontology`` switch (rungs 1-2 + controlled types)."""

    # Rung 3 at write time: type-constrained embedding similarity of a new mention
    # against the existing entities (embeddings are content-cached).
    write_embedding_rung: bool = False
    embed_threshold: float = 0.92
    # Rung 4: the ambiguous residue (best similarity in [rung4_floor, embed_threshold))
    # becomes one categorical question to the decision model. Below the floor the
    # mention is a new entity; an answer below the bucket floor parks.
    rung4: bool = False
    rung4_floor: float = 0.75
    # UPDATE closes the superseded fact's validity interval at the new fact's start
    # (parsed from its text when the temporal annotator gave none). Needs bitemporal.
    interval_closing: bool = False


@dataclass
class RetrievalConfig:
    """Read-path knobs (Layer D retrieval order, Layer E certification ranking)."""

    # "vector": today's hybrid + graph search. "deterministic": query mentions ->
    # entities (ladder rungs 1-2) -> relation expansion -> memories of those entities,
    # boosted and filled in by vector similarity within that set.
    order: str = "vector"
    expansion_depth: int = 1  # relation hops from the mentioned entities (0 = only them)
    deterministic_boost: float = 0.2  # score multiplier bonus for memories in the deterministic set
    deterministic_cap: int = 200  # max memories pulled from the deterministic set
    # Allen filter: when the query names a time, memories whose interval overlaps it
    # get temporal_boost, disjoint ones lose it.
    temporal_filter: bool = False
    temporal_boost: float = 0.2
    certified_boost: float = 0.0  # score multiplier bonus for certified memories


@dataclass
class CompileConfig:
    """Layer E compile and certification knobs."""

    merge_threshold: float = 0.92  # rung-3 entity merges at or above this similarity
    propose_threshold: float = 0.85  # pairs in [propose, merge) become merge proposals
    every_n_adds: int = 0  # 0 = manual compile(); N = compile after every N infer adds
    grow_taxonomy: bool = True  # generated subject children (when the taxonomy is on)
    promote_after: int = 3  # stable observations (re-stated facts) that promote a memory to certified


@dataclass
class BucketConfig:
    """Layer G: park low-certainty judgments as typed open decisions instead of forcing them."""

    enabled: bool = False
    # A decision-model answer with confidence below the floor parks (DECISION and
    # HYBRID; in HYBRID it parks instead of escalating). The fact is kept, flagged
    # provisional; the contested memory is left as is.
    floor: float = 0.5
    park_actions: tuple[str, ...] = ("update", "delete", "none")  # low-confidence ADD always stands
    max_open: int = 100  # oldest open entries beyond this expire to an explicit unknown
    ttl_days: float = 30.0  # open entries older than this expire (0 = never)
    self_resolution: bool = True  # re-judge on new evidence and in compile()
    rejudge_per_add: int = 3  # max parked questions re-judged per write that touches their evidence
    resolve_engine: str = "llm"  # rung that re-judges parked items: "llm" or "decision"
    provisional_penalty: float = 0.0  # search score reduction for provisional memories, in [0, 1]
    ask_user: bool = False  # never prompt by default; the app decides when to surface the queue
    ask_importance_floor: float = 0.6


@dataclass
class PolicyConfig:
    """Layer F: rule packs resolve judgments before any model (resolve or abstain)."""

    enabled: bool = False
    packs: tuple[str, ...] = ("compliance", "temporal", "entrenchment", "lifecycle")
    # Conservative mode: a model decision that would change certified content parks
    # (or stands down) instead of applying.
    conservative: bool = True
    # SHACL-style shape severities: "violation" rejects, "warning" writes with lowered
    # confidence, "info" only logs.
    shape_severity: dict[str, str] = field(
        default_factory=lambda: {"relation_domain_range": "violation", "memory_provenance": "info"}
    )
    retention_days: dict[str, float] = field(default_factory=dict)  # memory_type -> TTL, applied in compile()
    extra_secret_patterns: tuple[str, ...] = ()  # regexes added to the compliance pack


@dataclass(frozen=True)
class RootSpec:
    """One designed subject root; ``keywords`` route queries and facts to it lexically."""

    id: str
    label: str
    status: str = "shipped"  # shipped | user | certified
    keywords: tuple[str, ...] = ()


# The designed subject roots: id -> (label, routing keywords). Shipped and versioned, never inferred.
_ROOT_KEYWORDS: dict[str, tuple[str, str]] = {
    "general": ("General", ""),
    "career": (
        "Career & work",
        "job work career interview employer colleague coworker boss "
        "manager salary promotion office hired resume company client meeting",
    ),
    "technology": (
        "Technology & tools",
        "software code coding programming computer laptop phone app "
        "python javascript tool tools stack database server ai",
    ),
    "skills": (
        "Skills & learning",
        "learn learning course class study studying school university degree skill skills practice lesson teacher exam",
    ),
    "projects": (
        "Projects & goals",
        "project projects goal goals plan plans deadline launch milestone startup business",
    ),
    "health": (
        "Health & wellness",
        "health doctor hospital sick illness injury therapy medication "
        "exercise gym workout running diet sleep yoga mental",
    ),
    "finance": (
        "Finance & legal",
        "money bank budget loan mortgage tax taxes invest investment savings rent lawyer contract insurance debt",
    ),
    "relationships": (
        "Relationships & social",
        "friend friends family mother father mom dad sister brother wife "
        "husband partner girlfriend boyfriend kids children son daughter wedding dating party",
    ),
    "home": (
        "Home & daily life",
        "home house apartment kitchen cooking cook cleaning garden "
        "pet dog cat car commute shopping groceries neighborhood moved",
    ),
    "interests": (
        "Interests & leisure",
        "hobby hobbies music movie movies book books reading painting art "
        "game games travel trip vacation concert sports photography hiking camping festival",
    ),
}
DEFAULT_ROOTS: tuple[RootSpec, ...] = tuple(
    RootSpec(rid, label, keywords=tuple(words.split())) for rid, (label, words) in _ROOT_KEYWORDS.items()
)


@dataclass
class GrowthConfig:
    min_cluster_size: int = 8
    depth_cap: int = 4
    merge_similarity: float = 0.9  # siblings above this similarity merge
    cluster_similarity: float = 0.8  # cosine to a cluster centroid to join it (generated children)
    demote_ttl_days: int = 30
    min_fact_count: int = 3


@dataclass
class InstanceScopeConfig:
    prominence_threshold: int = 5  # memories linked to an entity before it becomes a scope
    hard_default: bool = False


@dataclass
class TaxonomyConfig:
    """Layer B: subject facet (designed roots + generated children) x instance facet (prominent entities)."""

    enabled: bool = False  # default off: single general scope = today's behavior
    roots: tuple[RootSpec, ...] = DEFAULT_ROOTS
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    instance_scopes: InstanceScopeConfig = field(default_factory=InstanceScopeConfig)
    router_engine: str = "lexical"  # "lexical" | "decision" (ScoreQuestions for facts and queries with no lexical hit)
    router_threshold: float = 0.5
    max_scopes_per_query: int = 4
    # "soft": in-scope memories get boosts, nothing is dropped. "filter": only
    # in-scope memories and the general scope survive when a scope is active.
    mode: str = "soft"
    subject_boost: float = 0.15
    instance_boost: float = 0.15
    combination_boost: float = 0.1  # extra when a memory holds both an active subject and an active instance
    hard_roots: tuple[str, ...] = ()  # memories under these roots surface only when the root is active


@dataclass
class WorkingConfig:
    """Tier 0: the session window that ``assemble()`` puts first."""

    enabled: bool = False
    token_budget: int = 2000
    window_messages: int = 20
    summarize_on_overflow: bool = True
    promote_on_session_end: bool = True
    # True: add() only buffers the session and end_session() runs extraction once
    # over the window. False: add() extracts as today and the window is context only.
    defer_extraction: bool = False


@dataclass
class EpisodicConfig:
    """Tier 1: wraps episodes + recency-weighted retrieval."""

    enabled: bool = False
    recency_weight: float = 0.0  # in [0, 1]: blend of the score with recency (0 = off)
    recency_half_life_days: float = 30.0


@dataclass
class ProceduralConfig:
    """Tier 3: standing instructions and corrections."""

    enabled: bool = False
    boost_weight: float = 0.2  # score multiplier bonus for procedural memories in search
    auto_triggers: bool = False  # store "always/never/from now on/no, use X" user turns as instructions


@dataclass
class TiersConfig:
    working: WorkingConfig = field(default_factory=WorkingConfig)
    episodic: EpisodicConfig = field(default_factory=EpisodicConfig)
    semantic: bool = True  # always on
    procedural: ProceduralConfig = field(default_factory=ProceduralConfig)
    # ContextBlock budget shares per section (assemble()); renormalized over the sections present.
    section_shares: dict[str, float] = field(
        default_factory=lambda: {"working": 0.4, "semantic": 0.3, "episodic": 0.15, "procedural": 0.15}
    )


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
    # Layer configs; every one defaults to off. The flat pipeline toggles
    # (pipeline.taxonomy / .policy / .bucket) are sugar that switch the matching layer on.
    ontology: OntologyConfig = field(default_factory=OntologyConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    compilation: CompileConfig = field(default_factory=CompileConfig)
    taxonomy: TaxonomyConfig = field(default_factory=TaxonomyConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    bucket: BucketConfig = field(default_factory=BucketConfig)
    tiers: TiersConfig = field(default_factory=TiersConfig)

    @property
    def taxonomy_on(self) -> bool:
        return bool(self.taxonomy.enabled or self.pipeline.taxonomy)

    @property
    def policy_on(self) -> bool:
        return bool(self.policy.enabled or self.pipeline.policy)

    @property
    def bucket_on(self) -> bool:
        return bool(self.bucket.enabled or self.pipeline.bucket)

    @property
    def bitemporal_on(self) -> bool:
        """Bi-temporal storage is on when ``enable_bitemporal`` or ``pipeline.bitemporal`` is set.
        Read at use, never written back, so changing either flag later takes effect."""
        return bool(self.enable_bitemporal or self.pipeline.bitemporal)

    @property
    def episodes_on(self) -> bool:
        """Episodes are on when ``enable_episodes``, ``pipeline.episodes`` or ``tiers.episodic.enabled`` is set."""
        return bool(self.enable_episodes or self.pipeline.episodes or self.tiers.episodic.enabled)

    def _validate_layers(self) -> None:
        unit = {
            "ontology.embed_threshold": self.ontology.embed_threshold,
            "ontology.rung4_floor": self.ontology.rung4_floor,
            "retrieval.deterministic_boost": self.retrieval.deterministic_boost,
            "retrieval.temporal_boost": self.retrieval.temporal_boost,
            "retrieval.certified_boost": self.retrieval.certified_boost,
            "compilation.merge_threshold": self.compilation.merge_threshold,
            "compilation.propose_threshold": self.compilation.propose_threshold,
            "bucket.floor": self.bucket.floor,
            "bucket.provisional_penalty": self.bucket.provisional_penalty,
            "bucket.ask_importance_floor": self.bucket.ask_importance_floor,
            "taxonomy.router_threshold": self.taxonomy.router_threshold,
            "taxonomy.subject_boost": self.taxonomy.subject_boost,
            "taxonomy.instance_boost": self.taxonomy.instance_boost,
            "taxonomy.combination_boost": self.taxonomy.combination_boost,
            "tiers.episodic.recency_weight": self.tiers.episodic.recency_weight,
            "tiers.procedural.boost_weight": self.tiers.procedural.boost_weight,
        }
        for name, val in unit.items():
            if not (0.0 <= float(val) <= 1.0):
                raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        choices = {
            "retrieval.order": (self.retrieval.order, ("vector", "deterministic")),
            "bucket.resolve_engine": (self.bucket.resolve_engine, ("llm", "decision")),
            "taxonomy.mode": (self.taxonomy.mode, ("soft", "filter")),
            "taxonomy.router_engine": (self.taxonomy.router_engine, ("lexical", "decision")),
        }
        for name, (val, allowed) in choices.items():
            if val not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, got {val!r}")
        for a in self.bucket.park_actions:
            if a not in ("update", "delete", "none"):
                raise ValueError(f"bucket.park_actions holds {a!r}; allowed: update, delete, none")
        for sev in self.policy.shape_severity.values():
            if sev not in ("violation", "warning", "info"):
                raise ValueError(f"policy.shape_severity values must be violation/warning/info, got {sev!r}")
        if self.retrieval.expansion_depth < 0:
            raise ValueError("retrieval.expansion_depth must be >= 0")
        if self.compilation.every_n_adds < 0:
            raise ValueError("compilation.every_n_adds must be >= 0")
        if self.tiers.working.token_budget <= 0 or self.tiers.working.window_messages <= 0:
            raise ValueError("tiers.working.token_budget and window_messages must be positive")
        uses_decision = (self.ontology.rung4 or self.taxonomy.router_engine == "decision") or (
            self.bucket_on and self.bucket.resolve_engine == "decision"
        )
        if uses_decision and self.decision_model is None:
            raise ValueError(
                "ontology.rung4, taxonomy.router_engine='decision' and bucket.resolve_engine='decision' "
                "need MemoryConfig.decision_model"
            )

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

        self._validate_layers()

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
    # Layer F trace: the rung that resolved it ("rule:<id>", "decision", "llm", "fast", "temporal").
    rung: str | None = None
    # Layer G: the open decision this event parked (the memory is provisional), when any.
    open_decision_id: str | None = None


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
    # Layer F trace: which rung resolved the decision (None = the engine above).
    rung: str | None = None
    # Layer G: confidence stayed below the bucket floor; the executor parks it.
    parked: bool = False
    # The decision model's runner-up choice when it parked (for the open question).
    alternatives: list[str] = field(default_factory=list)


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
    # Layer metrics: open decisions by status/kind, resolution rungs (deterministic share),
    # certified/provisional counts. Empty when no layer is on.
    layers: dict = field(default_factory=dict)


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
OPEN_DECISION_LABEL = "OpenDecision"
META_LABEL = "GrafeoMemoryMeta"
TAXONOMY_LABEL = "TaxonomyState"
SCOPE_OF_EDGE = "SCOPE_OF"
SCHEMA_VERSION = 2


@dataclass
class OpenDecision:
    """A parked judgment (Layer G): typed question, evidence, status and resolution."""

    id: str
    kind: str  # contradiction | duplicate | entity_identity | entity_merge | root_migration | ...
    question: str
    options: list[str]
    evidence: list[str]  # memory ids (or entity names for entity kinds)
    status: str = "open"  # open -> asked -> resolved | expired
    proposed: str | None = None  # the engine's low-confidence answer
    confidence: float | None = None
    abstained_at_rung: str | None = None
    created_at: int | None = None
    updated_at: int | None = None
    resolution: str | None = None  # the chosen option, or "unknown" on expiry
    resolved_by: str | None = None  # user | self | expiry
    resolved_at: int | None = None
    importance: float = 0.5
    payload: dict = field(default_factory=dict)


@dataclass
class ContextItem:
    """One line of an assembled context, with its provenance."""

    text: str
    section: str  # working | semantic | episodic | procedural
    memory_id: str | None = None
    score: float | None = None
    confidence: float | None = None
    engine: str | None = None
    status: str | None = None  # certified | inferred | provisional
    session_id: str | None = None
    created_at: int | None = None

    def provenance(self) -> str:
        parts = [f"id={self.memory_id}" if self.memory_id else "", f"engine={self.engine or 'llm'}"]
        if self.confidence is not None:
            parts.append(f"confidence={self.confidence:.2f}")
        parts.append(f"status={self.status or 'inferred'}")
        if self.session_id:
            parts.append(f"session={self.session_id}")
        return ", ".join(p for p in parts if p)


@dataclass
class ContextBlock:
    """The working-tier output of ``assemble()``: sections in a stable order within a token budget."""

    query: str
    budget: int
    sections: dict[str, list[ContextItem]] = field(default_factory=dict)
    summary: str | None = None  # rolling summary of the overflowed window
    tokens: int = 0
    open_questions: list[OpenDecision] = field(default_factory=list)

    ORDER: ClassVar[tuple[str, ...]] = ("working", "procedural", "semantic", "episodic")

    def items(self) -> list[ContextItem]:
        return [i for name in self.ORDER for i in self.sections.get(name, [])]

    def render(self, provenance: bool = True) -> str:
        out: list[str] = []
        titles = {
            "working": "Current session",
            "procedural": "Instructions",
            "semantic": "Known facts",
            "episodic": "Past events",
        }
        if self.summary:
            out.append("## Session summary\n" + self.summary)
        for name in self.ORDER:
            items = self.sections.get(name) or []
            if not items:
                continue
            out.append(f"## {titles[name]}")
            for it in items:
                line = f"- {it.text}"
                if provenance and name != "working":
                    line += f" [{it.provenance()}]"
                out.append(line)
        if self.open_questions:
            out.append("## Open questions")
            out.extend(f"- {q.question}" for q in self.open_questions)
        return "\n".join(out)

    def to_dict(self) -> dict:
        from dataclasses import asdict

        return asdict(self)


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
    "DEFAULT_ROOTS",
    "DERIVED_FROM_EDGE",
    "ENTITY_LABEL",
    "EPISODE_LABEL",
    "HAS_ENTITY_EDGE",
    "HAS_MEMBER_EDGE",
    "LEADS_TO_EDGE",
    "MEMORY_LABEL",
    "MENTIONS_EDGE",
    "META_LABEL",
    "NEXT_EPISODE_EDGE",
    "OPEN_DECISION_LABEL",
    "PRODUCED_EDGE",
    "RELATION_EDGE",
    "SCHEMA_VERSION",
    "SCOPE_OF_EDGE",
    "SUPERSEDES_EDGE",
    "TAXONOMY_LABEL",
    "AddResult",
    "BucketConfig",
    "CommunityInfo",
    "CompileConfig",
    "ContextBlock",
    "ContextItem",
    "EntitiesOutput",
    "Entity",
    "EntityItem",
    "EpisodeResult",
    "EpisodicConfig",
    "ExplainResult",
    "ExplainStep",
    "ExtractionResult",
    "Fact",
    "FactsOutput",
    "GrowthConfig",
    "InstanceScopeConfig",
    "MemoryAction",
    "MemoryConfig",
    "MemoryEvent",
    "MemoryStats",
    "MemoryType",
    "OntologyConfig",
    "OpenDecision",
    "PolicyConfig",
    "ProceduralConfig",
    "ReconciliationDecision",
    "ReconciliationItem",
    "ReconciliationOutput",
    "Relation",
    "RelationDeleteItem",
    "RelationItem",
    "RelationReconciliationOutput",
    "RetrievalConfig",
    "RootSpec",
    "SearchResponse",
    "SearchResult",
    "TaxonomyConfig",
    "TiersConfig",
    "WorkingConfig",
]
