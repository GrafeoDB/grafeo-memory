"""grafeo-memory — AI memory layer powered by GrafeoDB."""

from pydantic_ai.models.instrumented import InstrumentationSettings

from .embedding import EmbeddingClient, MistralEmbedder, OpenAIEmbedder
from .history import HistoryEntry
from .manager import AsyncMemoryManager, MemoryManager
from .messages import ImageContent, Message
from .reranker import LLMReranker, Reranker
from .scoring import apply_importance_scoring, apply_topology_boost, compute_composite_score
from .types import (
    AddResult,
    EntitiesOutput,
    Entity,
    ExplainResult,
    ExplainStep,
    ExtractionResult,
    Fact,
    FactsOutput,
    MemoryAction,
    MemoryConfig,
    MemoryEvent,
    MemoryStats,
    MemoryType,
    ReconciliationOutput,
    Relation,
    RelationReconciliationOutput,
    SearchResponse,
    SearchResult,
)

__all__ = [
    "AddResult",
    "AsyncMemoryManager",
    "EmbeddingClient",
    "EntitiesOutput",
    "Entity",
    "ExplainResult",
    "ExplainStep",
    "ExtractionResult",
    "Fact",
    "FactsOutput",
    "HistoryEntry",
    "ImageContent",
    "InstrumentationSettings",
    "LLMReranker",
    "MemoryAction",
    "MemoryConfig",
    "MemoryEvent",
    "MemoryManager",
    "MemoryStats",
    "MemoryType",
    "Message",
    "MistralEmbedder",
    "OpenAIEmbedder",
    "ReconciliationOutput",
    "Relation",
    "RelationReconciliationOutput",
    "Reranker",
    "SearchResponse",
    "SearchResult",
    "apply_importance_scoring",
    "apply_topology_boost",
    "compute_composite_score",
]

__version__ = "0.1.5"
