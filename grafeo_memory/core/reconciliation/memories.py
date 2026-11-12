"""Memory reconciliation: compare new facts against existing memories.

Three engines per call (see ``MemoryConfig.pipeline``):

- ``Engine.LLM``, the pydantic-ai agent with the existing prompts (default,
  byte-for-byte today's behavior).
- ``Engine.DECISION``, a decision model answers a typed question pack
  (per-fact or pairwise) built from the same state text the LLM prompt uses.
- ``Engine.HYBRID``, decision model first; facts whose decision is too
  uncertain (confidence / prob-gap / combined below the threshold) are
  escalated to a single LLM batch call.

The no-candidate fast path (all ADD, zero calls of any kind) and the
LLM-failure fallback to ADD are unchanged. A decision-model provider error
falls back to the LLM path, which itself falls back to ADD.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from grafeo_memory.common.schemas import ReconciliationOutput
from grafeo_memory.common.types import Engine, Fact, MemoryAction, ModelType, ReconciliationDecision, StageConfig
from grafeo_memory.core._compat import run_sync
from grafeo_memory.core.prompts import RECONCILIATION_SYSTEM, RECONCILIATION_USER
from grafeo_memory.core.reconciliation.temporal import extract_interval, partition_candidates

if TYPE_CHECKING:
    from pydantic_ai import AgentRunResult
    from pydantic_ai.usage import RunUsage

    from grafeo_memory.providers.decision.protocol import DecisionModel
    from grafeo_memory.providers.decision.types import Decision

logger = logging.getLogger(__name__)


def _make_agent(model: ModelType) -> Agent[object, ReconciliationOutput]:
    return Agent(model, system_prompt=RECONCILIATION_SYSTEM, output_type=ReconciliationOutput)


def _parse(result: AgentRunResult[ReconciliationOutput]) -> list[ReconciliationDecision]:
    decisions: list[ReconciliationDecision] = []
    for raw in result.output.decisions:
        action_str = raw.action.upper()
        try:
            action = MemoryAction(action_str.lower())
        except ValueError:
            action = MemoryAction.NONE

        target_id = raw.target_memory_id

        # Validate: UPDATE/DELETE require target_memory_id
        if action in (MemoryAction.UPDATE, MemoryAction.DELETE) and not target_id:
            fallback = "ADD" if action == MemoryAction.UPDATE else "NONE"
            logger.warning(
                "%s missing target_memory_id, downgrading to %s: %r",
                action.value.upper(),
                fallback,
                raw.text,
            )
            action = MemoryAction.ADD if action == MemoryAction.UPDATE else MemoryAction.NONE

        decisions.append(
            ReconciliationDecision(
                action=action,
                text=raw.text,
                target_memory_id=target_id,
            )
        )
    return decisions


def _fast_path_add(new_facts: list[Fact]) -> list[ReconciliationDecision]:
    return [ReconciliationDecision(action=MemoryAction.ADD, text=f.text, engine="fast") for f in new_facts]


async def _llm_reconcile_or_none(
    model: ModelType,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    op: str = "reconcile",
) -> list[ReconciliationDecision] | None:
    """The LLM path: existing prompts + schema. None on provider error."""
    facts_text = "\n".join(f"{i + 1}. {f.text}" for i, f in enumerate(new_facts))
    memories_text = "\n".join(
        f'- ID {m["id"]}: "{m["text"]}" (similarity: {m["score"]:.2f})' for m in existing_memories
    )

    agent = _make_agent(model)
    try:
        result = await agent.run(
            RECONCILIATION_USER.format(new_facts=facts_text, existing_memories=memories_text),
        )
    except Exception:
        logger.warning("Failed reconciliation (%s)", op, exc_info=True)
        return None

    if _on_usage is not None:
        _on_usage(op, result.usage)
    return _parse(result)


async def _llm_reconcile(
    model: ModelType,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[ReconciliationDecision]:
    """The LLM path; on provider error, degrade to the deterministic ADD fast path
    (tagged engine="fast")."""
    out = await _llm_reconcile_or_none(model, new_facts, existing_memories, _on_usage=_on_usage)
    if out is None:
        logger.warning("Falling back to ADD for %d facts", len(new_facts))
        return _fast_path_add(new_facts)
    return out


def _resolve_pack(stage: StageConfig, question_pack) -> tuple[Callable[..., list], bool]:
    """Return (pack_fn, use_pairwise_mapper). A global question_pack override wins
    over the stage's granularity."""
    from grafeo_memory.providers.decision.packs import PACKS

    if question_pack is not None:
        if callable(question_pack):
            # A pack declares its mapper with a ``granularity`` attribute; functools.partial
            # wrappers inherit it from the wrapped function.
            from grafeo_memory.providers.decision.packs import PACK_GRANULARITY

            fn = getattr(question_pack, "func", question_pack)
            gran = (
                getattr(question_pack, "granularity", None)
                or getattr(fn, "granularity", None)
                or PACK_GRANULARITY.get(fn)
            )
            return question_pack, gran == "pairwise"
        return PACKS[str(question_pack)], str(question_pack) == "pairwise"
    return PACKS[stage.granularity.value], stage.granularity.value == "pairwise"


async def _decision_reconcile(
    decision_model: DecisionModel,
    new_facts: list[Fact],
    existing_memories: list[dict],
    stage: StageConfig,
    question_pack,
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[tuple[ReconciliationDecision, Decision | None]]:
    """Run the question pack through the decision model and map decisions to actions."""
    from grafeo_memory.providers.decision.packs import (
        build_state,
        decisions_to_actions_pairwise,
        decisions_to_actions_per_fact,
    )

    state = build_state(new_facts, existing_memories)
    pack_fn, use_pairwise = _resolve_pack(stage, question_pack)
    questions = pack_fn(new_facts, existing_memories)
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
        _on_usage("reconcile_decision", usage)

    if use_pairwise:
        return decisions_to_actions_pairwise(new_facts, existing_memories, decisions)
    return decisions_to_actions_per_fact(new_facts, existing_memories, decisions)


async def _temporal_precheck(
    model: ModelType,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    engine: Engine | str = Engine.LLM,
    decision_model: DecisionModel | None = None,
    stage: StageConfig | None = None,
    question_pack=None,
) -> list[ReconciliationDecision]:
    """Layer D pre-model temporal rules: partition candidates per new fact by
    validity-interval overlap. A fact with no contested candidate is
    deterministically ADD (engine="temporal") with zero model calls, disjoint
    validities both stand. The rest reconcile against only their contested
    candidates, so cleared pairs never reach the engine."""
    by_id = {str(m["id"]): m for m in existing_memories}
    cand_ivs = [(cid, extract_interval(str(m.get("text", "")))) for cid, m in by_id.items()]
    out: list[ReconciliationDecision | None] = [None] * len(new_facts)
    pending: list[int] = []
    contested: set[str] = set()
    for i, f in enumerate(new_facts):
        part = partition_candidates(extract_interval(f.text), cand_ivs)
        if part.contested:
            pending.append(i)
            contested.update(part.contested)
        else:
            out[i] = ReconciliationDecision(action=MemoryAction.ADD, text=f.text, engine="temporal")
    if pending:
        filtered = [by_id[c] for c in contested if c in by_id]
        sub_facts = [new_facts[i] for i in pending]
        sub = await reconcile_async(
            model,
            sub_facts,
            filtered,
            _on_usage=_on_usage,
            engine=engine,
            decision_model=decision_model,
            stage=stage,
            question_pack=question_pack,
        )
        if len(sub) == len(pending):
            for j, i in enumerate(pending):
                out[i] = sub[j]
        else:
            # The engine answered a different number of decisions than facts (an LLM may
            # add an UPDATE plus a DELETE for one fact, or skip one): keep every answer as
            # given, like the plain LLM path, rather than padding or truncating by position.
            return [d for d in out if d is not None] + list(sub)
    return [d for d in out if d is not None]


async def reconcile_async(
    model: ModelType,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    engine: Engine | str = Engine.LLM,
    decision_model: DecisionModel | None = None,
    stage: StageConfig | None = None,
    question_pack=None,
    temporal_precheck: bool = False,
) -> list[ReconciliationDecision]:
    """Reconcile new facts against existing memories (async).

    Args:
        model: The pydantic-ai model to use for the LLM path / escalation.
        new_facts: Facts extracted from the new text.
        existing_memories: List of dicts with keys: id, text, score.
        engine: Which engine decides, LLM (default), DECISION, or HYBRID.
        decision_model: The DecisionModel for DECISION/HYBRID (ignored for LLM).
        stage: StageConfig carrying threshold/granularity/escalation signal.
        question_pack: Optional global pack override ("per_fact"/"pairwise"/callable).
        temporal_precheck: Layer D pre-model temporal rules:
            facts with validity disjoint from every candidate skip the engine.

    Returns:
        A list of ReconciliationDecision objects. Each carries ``engine``
        ("llm"/"decision"/"fast"/"temporal") and ``confidence`` when the engine
        provides one.
    """
    if not new_facts:
        return []

    eng = Engine(engine)
    stage = stage or StageConfig()

    if not existing_memories:
        logger.debug("No existing memories found: all %d facts will be ADD (fast path)", len(new_facts))
        return _fast_path_add(new_facts)

    if temporal_precheck:
        return await _temporal_precheck(
            model,
            new_facts,
            existing_memories,
            _on_usage=_on_usage,
            engine=eng,
            decision_model=decision_model,
            stage=stage,
            question_pack=question_pack,
        )

    if eng == Engine.LLM:
        return await _llm_reconcile(model, new_facts, existing_memories, _on_usage=_on_usage)

    if decision_model is None:
        logger.warning("engine=%s but no decision_model supplied; falling back to LLM", eng.value)
        return await _llm_reconcile(model, new_facts, existing_memories, _on_usage=_on_usage)

    from grafeo_memory.providers.decision.packs import should_escalate

    try:
        pairs = await _decision_reconcile(
            decision_model, new_facts, existing_memories, stage, question_pack, _on_usage=_on_usage
        )
    except Exception:
        logger.warning("Decision-model reconciliation failed, falling back to LLM", exc_info=True)
        return await _llm_reconcile(model, new_facts, existing_memories, _on_usage=_on_usage)

    if eng == Engine.DECISION:
        return [action for action, _d in pairs]

    # HYBRID: escalate uncertain facts to one LLM batch call. Confident
    # decisions stand as-is; the LLM only sees the uncertain subset.
    out = [action for action, _d in pairs]
    esc_idx = [
        i
        for i, (_action, d) in enumerate(pairs)
        if d is None or should_escalate(d, stage.confidence_threshold, stage.escalation_signal)
    ]
    if esc_idx:
        sub_facts = [new_facts[i] for i in esc_idx]
        llm_decisions = await _llm_reconcile_or_none(
            model, sub_facts, existing_memories, _on_usage=_on_usage, op="reconcile_escalated"
        )
        if llm_decisions is None:  # escalation failed: the decision model's answers stand
            return out
        if len(llm_decisions) == len(esc_idx):
            for j, i in enumerate(esc_idx):
                out[i] = llm_decisions[j]
        else:  # answer count differs from fact count: take the LLM's answers as given
            esc = set(esc_idx)
            out = [d for i, d in enumerate(out) if i not in esc] + llm_decisions
    return out


def reconcile(
    model: ModelType,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    engine: Engine | str = Engine.LLM,
    decision_model: DecisionModel | None = None,
    stage: StageConfig | None = None,
    question_pack=None,
    temporal_precheck: bool = False,
) -> list[ReconciliationDecision]:
    """Reconcile new facts against existing memories (sync)."""
    return run_sync(
        reconcile_async(
            model,
            new_facts,
            existing_memories,
            _on_usage=_on_usage,
            engine=engine,
            decision_model=decision_model,
            stage=stage,
            question_pack=question_pack,
            temporal_precheck=temporal_precheck,
        )
    )
