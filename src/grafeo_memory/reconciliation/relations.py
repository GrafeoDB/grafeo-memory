"""Relation reconciliation: detect contradictions in graph edges."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .._compat import run_sync
from ..prompts import RELATION_RECONCILE_SYSTEM, RELATION_RECONCILE_USER
from ..schemas import RelationReconciliationOutput
from ..types import Relation

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)


def _make_agent(model: object) -> Agent[None, RelationReconciliationOutput]:
    return Agent(model, system_prompt=RELATION_RECONCILE_SYSTEM, output_type=RelationReconciliationOutput)


async def reconcile_relations_async(
    model: object,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[dict]:
    """Identify existing graph relations that should be deleted (async).

    Args:
        model: The pydantic-ai model.
        new_relations: Relations extracted from new text.
        existing_relations: Existing relations from graph.
            Each dict has keys: source, target, relation_type, edge_id.

    Returns:
        List of dicts identifying relations to delete (source, target, relation_type).
    """
    if not new_relations or not existing_relations:
        return []

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
        _on_usage("reconcile_relations", result.usage())
    return [{"source": d.source, "target": d.target, "relation_type": d.relation_type} for d in result.output.delete]


def reconcile_relations(
    model: object,
    new_relations: list[Relation],
    existing_relations: list[dict],
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[dict]:
    """Identify existing graph relations that should be deleted (sync)."""
    return run_sync(reconcile_relations_async(model, new_relations, existing_relations, _on_usage=_on_usage))
