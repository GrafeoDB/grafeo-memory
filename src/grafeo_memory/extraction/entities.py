"""Entity and relation extraction from facts via LLM."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .._compat import run_sync
from ..prompts import ENTITY_EXTRACTION_SYSTEM, ENTITY_EXTRACTION_USER
from ..schemas import EntitiesOutput
from ..types import Entity, ExtractionResult, Fact, Relation

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)


def _make_agent(model: object) -> Agent[None, EntitiesOutput]:
    return Agent(model, system_prompt=ENTITY_EXTRACTION_SYSTEM, output_type=EntitiesOutput)


def _parse(result: object) -> tuple[list[Entity], list[Relation]]:
    entities = [Entity(name=e.name, entity_type=e.entity_type) for e in result.output.entities]
    relations = [
        Relation(source=r.source, target=r.target, relation_type=r.relation_type) for r in result.output.relations
    ]
    return entities, relations


async def extract_entities_async(
    model: object,
    facts: list[Fact],
    user_id: str,
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from facts via LLM (async)."""
    if not facts:
        return [], []

    facts_text = "\n".join(f"- {f.text}" for f in facts)
    agent = _make_agent(model)
    try:
        result = await agent.run(ENTITY_EXTRACTION_USER.format(user_id=user_id, facts=facts_text))
    except Exception:
        logger.warning("Entity extraction failed", exc_info=True)
        return [], []
    if _on_usage is not None:
        _on_usage("extract_entities", result.usage())
    return _parse(result)


def extract_entities(
    model: object,
    facts: list[Fact],
    user_id: str,
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> tuple[list[Entity], list[Relation]]:
    """Extract entities and relations from facts via LLM (sync)."""
    return run_sync(extract_entities_async(model, facts, user_id, _on_usage=_on_usage))


async def _extract_combined_async(
    model: object,
    text: str,
    user_id: str,
    *,
    custom_prompt: str | None = None,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> ExtractionResult:
    """Extract facts, entities, and relations in a single LLM call (async)."""
    from ..prompts import COMBINED_EXTRACTION_SYSTEM, COMBINED_EXTRACTION_USER
    from ..schemas import ExtractionOutput

    agent = Agent(model, system_prompt=custom_prompt or COMBINED_EXTRACTION_SYSTEM, output_type=ExtractionOutput)
    try:
        result = await agent.run(COMBINED_EXTRACTION_USER.format(user_id=user_id, text=text))
    except Exception:
        logger.warning("Combined extraction failed", exc_info=True)
        return ExtractionResult()
    if _on_usage is not None:
        _on_usage("extract", result.usage())
    facts = [Fact(text=f) for f in result.output.facts if f]
    entities = [Entity(name=e.name, entity_type=e.entity_type) for e in result.output.entities]
    relations = [
        Relation(source=r.source, target=r.target, relation_type=r.relation_type) for r in result.output.relations
    ]
    return ExtractionResult(facts=facts, entities=entities, relations=relations)


async def extract_async(
    model: object,
    text: str,
    user_id: str,
    *,
    custom_fact_prompt: str | None = None,
    memory_type: str = "semantic",
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> ExtractionResult:
    """Run extraction pipeline (async): facts + entities + relations in 1 LLM call.

    For procedural memory_type, uses the procedural extraction prompt unless custom_fact_prompt is set.
    """
    effective_prompt = custom_fact_prompt
    if effective_prompt is None and memory_type == "procedural":
        from ..prompts import COMBINED_PROCEDURAL_EXTRACTION_SYSTEM

        effective_prompt = COMBINED_PROCEDURAL_EXTRACTION_SYSTEM

    return await _extract_combined_async(model, text, user_id, custom_prompt=effective_prompt, _on_usage=_on_usage)


def extract(
    model: object,
    text: str,
    user_id: str,
    *,
    custom_fact_prompt: str | None = None,
    memory_type: str = "semantic",
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> ExtractionResult:
    """Run the full extraction pipeline (sync)."""
    return run_sync(
        extract_async(
            model, text, user_id, custom_fact_prompt=custom_fact_prompt, memory_type=memory_type, _on_usage=_on_usage
        )
    )
