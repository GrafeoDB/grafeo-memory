"""Decision-model integration: structured, calibrated decisions instead of raw LLM calls.

Two engines per pipeline stage (see :class:`~grafeo_memory.common.types.PipelineConfig`):

- ``llm``, the existing free-form extraction/reconciliation prompts.
- ``decision``, a :class:`DecisionModel` (fake for tests, Jev for production)
  answering typed question packs, with confidence scores for escalation.
- ``hybrid``, decision model first, low-confidence items escalated to one LLM batch.

All decision calls are cached through :class:`~grafeo_memory.common.cache.MemoryCache`
when a cache is configured, keyed by content hash, reruns never re-pay.
"""

from grafeo_memory.providers.decision.fake import FakeDecisionModel
from grafeo_memory.providers.decision.packs import (
    PACKS,
    build_relation_state,
    build_state,
    decisions_to_actions_pairwise,
    decisions_to_actions_per_fact,
    decisions_to_deletes,
    pairwise_pack,
    per_fact_pack,
    relations_pack,
    should_escalate,
)
from grafeo_memory.providers.decision.protocol import DecisionModel, decide_async
from grafeo_memory.providers.decision.registry import get_decision_model, register_decision_model
from grafeo_memory.providers.decision.types import CategoricalQuestion, Decision, Question, ScoreQuestion

__all__ = [
    "PACKS",
    "CategoricalQuestion",
    "Decision",
    "DecisionModel",
    "FakeDecisionModel",
    "Question",
    "ScoreQuestion",
    "build_relation_state",
    "build_state",
    "decide_async",
    "decisions_to_actions_pairwise",
    "decisions_to_actions_per_fact",
    "decisions_to_deletes",
    "get_decision_model",
    "pairwise_pack",
    "per_fact_pack",
    "register_decision_model",
    "relations_pack",
    "should_escalate",
]
