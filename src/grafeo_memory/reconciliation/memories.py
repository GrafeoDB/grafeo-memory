"""Memory reconciliation: compare new facts against existing memories."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .._compat import run_sync
from ..prompts import RECONCILIATION_SYSTEM, RECONCILIATION_USER
from ..schemas import ReconciliationOutput
from ..types import (
    Fact,
    MemoryAction,
    ReconciliationDecision,
)

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)


def _make_agent(model: object) -> Agent[None, ReconciliationOutput]:
    return Agent(model, system_prompt=RECONCILIATION_SYSTEM, output_type=ReconciliationOutput)


def _parse(result: object, new_facts: list[Fact]) -> list[ReconciliationDecision]:
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
    return [ReconciliationDecision(action=MemoryAction.ADD, text=f.text) for f in new_facts]


async def reconcile_async(
    model: object,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[ReconciliationDecision]:
    """Reconcile new facts against existing memories via LLM (async).

    Args:
        model: The pydantic-ai model to use for reconciliation.
        new_facts: Facts extracted from the new text.
        existing_memories: List of dicts with keys: id, text, score.

    Returns:
        A list of ReconciliationDecision objects.
    """
    if not new_facts:
        return []

    if not existing_memories:
        logger.debug("No existing memories found — all %d facts will be ADD (fast-path)", len(new_facts))
        return _fast_path_add(new_facts)

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
        logger.warning("Failed reconciliation, falling back to ADD", exc_info=True)
        return _fast_path_add(new_facts)

    if _on_usage is not None:
        _on_usage("reconcile", result.usage())
    return _parse(result, new_facts)


def reconcile(
    model: object,
    new_facts: list[Fact],
    existing_memories: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[ReconciliationDecision]:
    """Reconcile new facts against existing memories via LLM (sync)."""
    return run_sync(reconcile_async(model, new_facts, existing_memories, _on_usage=_on_usage))
