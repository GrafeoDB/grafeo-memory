"""Fact extraction from conversation text via LLM."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .._compat import run_sync
from ..prompts import FACT_EXTRACTION_SYSTEM, FACT_EXTRACTION_USER
from ..schemas import FactsOutput
from ..types import Fact

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)


def _make_agent(model: object, system_prompt: str | None = None) -> Agent[None, FactsOutput]:
    return Agent(model, system_prompt=system_prompt or FACT_EXTRACTION_SYSTEM, output_type=FactsOutput)


def _parse(result: object) -> list[Fact]:
    return [Fact(text=f) for f in result.output.facts if f]


async def extract_facts_async(
    model: object,
    text: str,
    user_id: str,
    *,
    custom_prompt: str | None = None,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[Fact]:
    """Extract discrete facts from conversation text via LLM (async).

    Returns a list of Fact objects. Returns an empty list if extraction fails
    or no facts are found.
    """
    agent = _make_agent(model, custom_prompt)
    try:
        result = await agent.run(FACT_EXTRACTION_USER.format(user_id=user_id, text=text))
    except Exception:
        logger.warning("Fact extraction failed", exc_info=True)
        return []
    if _on_usage is not None:
        _on_usage("extract_facts", result.usage())
    return _parse(result)


def extract_facts(
    model: object,
    text: str,
    user_id: str,
    *,
    custom_prompt: str | None = None,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[Fact]:
    """Extract discrete facts from conversation text via LLM (sync)."""
    return run_sync(extract_facts_async(model, text, user_id, custom_prompt=custom_prompt, _on_usage=_on_usage))
