"""Relation reconciliation: detect contradictions in graph edges.

Two engines per call (see ``MemoryConfig.pipeline.relations``):

- ``Engine.LLM``, the pydantic-ai agent with the existing prompt (default).
- ``Engine.DECISION`` / ``Engine.HYBRID``, a decision model answers one
  KEEP/DELETE question per (new relation, existing relation) pair; an existing
  relation is deleted when any pair answers DELETE for it. HYBRID uses the same
  pack and escalates the existing relations whose verdict is below the stage's
  confidence threshold to one LLM call. A provider error makes both fall back
  to the LLM path.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from grafeo_memory.common.schemas import RelationReconciliationOutput
from grafeo_memory.common.types import Engine, ModelType, Relation, StageConfig
from grafeo_memory.core._compat import run_sync
from grafeo_memory.core.prompts import RELATION_RECONCILE_SYSTEM, RELATION_RECONCILE_USER

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

    from grafeo_memory.providers.decision.protocol import DecisionModel

logger = logging.getLogger(__name__)


def _make_agent(model: ModelType) -> Agent[object, RelationReconciliationOutput]:
    return Agent(model, system_prompt=RELATION_RECONCILE_SYSTEM, output_type=RelationReconciliationOutput)


def filter_relation_candidates(
    new_relations: list[Relation], existing_relations: list[dict], cap: int = 50
) -> list[dict]:
    """Existing relations that could conflict with a new one, most likely first, at most ``cap``.

    A new relation can only invalidate a relation that shares an entity with it, so the
    others are dropped. The rest are ranked: (0) same source and relation type ("Alice
    WORKS_AT"), (1) same target and relation type, or the same two entities, (2) any shared
    entity; newest first within a rank. Both engines (LLM and decision model) judge the same
    bounded evidence, and the pairwise decision pack stays O(new x related)."""
    src_type = {(r.source, r.relation_type) for r in new_relations}
    tgt_type = {(r.target, r.relation_type) for r in new_relations}
    pairs = {frozenset((r.source, r.target)) for r in new_relations}
    ents = {r.source for r in new_relations} | {r.target for r in new_relations}

    def rank(r: dict) -> int:
        s, t, rt = r.get("source"), r.get("target"), r.get("relation_type")
        if (s, rt) in src_type:
            return 0
        if (t, rt) in tgt_type or frozenset((s, t)) in pairs:
            return 1
        return 2

    related = [r for r in existing_relations if r.get("source") in ents or r.get("target") in ents]
    related.sort(key=lambda r: (rank(r), -int(r.get("edge_id", 0) or 0)))
    return related[:cap]


async def _llm_reconcile_relations(
    model: ModelType,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[dict]:
    """The LLM path: existing prompt + schema. On provider error, keep everything."""
    existing_text = "\n".join(f"- {r['source']} -- {r['relation_type']} -- {r['target']}" for r in existing_relations)
    new_text = "\n".join(f"- {r.source} -- {r.relation_type} -- {r.target}" for r in new_relations)

    agent = _make_agent(model)
    try:
        result = await agent.run(
            RELATION_RECONCILE_USER.format(existing_relations=existing_text, new_relations=new_text),
        )
    except Exception:
        logger.warning("Failed relation reconciliation", exc_info=True)
        return []

    if _on_usage is not None:
        _on_usage("reconcile_relations", result.usage)
    output: RelationReconciliationOutput = result.output
    return [{"source": d.source, "target": d.target, "relation_type": d.relation_type} for d in output.delete]


async def _decision_reconcile_relations(
    decision_model: DecisionModel,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list:
    """KEEP/DELETE per (new, existing) relation pair via the decision model; returns
    the raw decisions (map them with ``relation_verdicts``)."""
    from grafeo_memory.providers.decision.packs import build_relation_state, relations_pack

    state = build_relation_state(new_relations, existing_relations)
    questions = relations_pack(new_relations, existing_relations)
    decisions = await decision_model.decide_async(state, questions)

    if _on_usage is not None:
        from pydantic_ai.usage import RunUsage

        usage = RunUsage()
        last = getattr(decision_model, "last_usage", None)
        if isinstance(last, dict):
            usage = RunUsage(
                input_tokens=int(last.get("prompt_tokens", 0) or 0),
                output_tokens=int(last.get("completion_tokens", 0) or 0),
            )
        _on_usage("reconcile_relations_decision", usage)

    return list(decisions)


async def reconcile_relations_async(
    model: ModelType,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    engine: Engine | str = Engine.LLM,
    decision_model: DecisionModel | None = None,
    stage: StageConfig | None = None,
) -> list[dict]:
    """Identify existing graph relations that should be deleted (async).

    Args:
        model: The pydantic-ai model (LLM path / fallback).
        new_relations: Relations extracted from new text.
        existing_relations: Existing relations from graph.
            Each dict has keys: source, target, relation_type, edge_id.
        engine: LLM (default), DECISION, or HYBRID.
        decision_model: The DecisionModel for DECISION/HYBRID.
        stage: The relations StageConfig: HYBRID escalates the existing relations whose
            KEEP/DELETE verdict falls below ``confidence_threshold`` (by
            ``escalation_signal``) to one LLM call. The pack is always pairwise.

    Returns:
        List of dicts identifying relations to delete (source, target, relation_type).
    """
    if not new_relations or not existing_relations:
        return []

    eng = Engine(engine)
    if eng == Engine.LLM or decision_model is None:
        if eng != Engine.LLM and decision_model is None:
            logger.warning("engine=%s but no decision_model supplied; falling back to LLM", eng.value)
        return await _llm_reconcile_relations(model, new_relations, existing_relations, _on_usage=_on_usage)

    from grafeo_memory.providers.decision.packs import relation_key, relation_verdicts, should_escalate

    try:
        decisions = await _decision_reconcile_relations(
            decision_model, new_relations, existing_relations, _on_usage=_on_usage
        )
    except Exception:
        logger.warning("Decision-model relation reconciliation failed, falling back to LLM", exc_info=True)
        return await _llm_reconcile_relations(model, new_relations, existing_relations, _on_usage=_on_usage)

    verdicts = relation_verdicts(new_relations, existing_relations, decisions)
    if eng != Engine.HYBRID:
        return [relation_key(r) for r, (delete, _d) in zip(existing_relations, verdicts, strict=True) if delete]

    st = stage or StageConfig(engine=Engine.HYBRID)
    deletes: list[dict] = []
    unsure: list[dict] = []
    for r, (delete, d) in zip(existing_relations, verdicts, strict=True):
        if d is None or should_escalate(d, st.confidence_threshold, st.escalation_signal):
            unsure.append(r)
        elif delete:
            deletes.append(relation_key(r))
    if unsure:
        deletes.extend(await _llm_reconcile_relations(model, new_relations, unsure, _on_usage=_on_usage))
    return deletes


def reconcile_relations(
    model: ModelType,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    engine: Engine | str = Engine.LLM,
    decision_model: DecisionModel | None = None,
    stage: StageConfig | None = None,
) -> list[dict]:
    """Identify existing graph relations that should be deleted (sync)."""
    return run_sync(
        reconcile_relations_async(
            model,
            new_relations,
            existing_relations,
            _on_usage=_on_usage,
            engine=engine,
            decision_model=decision_model,
            stage=stage,
        )
    )
