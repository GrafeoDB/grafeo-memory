"""Temporal annotation extraction: identify when facts became true/ceased being true."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from datetime import UTC, date, datetime
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .._compat import run_sync
from ..prompts import TEMPORAL_ANNOTATION_SYSTEM, TEMPORAL_ANNOTATION_USER
from ..schemas import TemporalAnnotationOutput
from ..types import Fact, ModelType

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)

_YEAR_ONLY_RE = re.compile(r"^\d{4}$")


def _parse_date_to_epoch_ms(date_str: str | None) -> int | None:
    """Parse an ISO-8601 date string (or year-only) to epoch milliseconds.

    Returns None for unparseable or null input.
    """
    if not date_str:
        return None

    date_str = date_str.strip()

    # Year-only: "2024" -> Jan 1 of that year
    if _YEAR_ONLY_RE.match(date_str):
        try:
            dt = datetime(int(date_str), 1, 1, tzinfo=UTC)
            return int(dt.timestamp() * 1000)
        except (ValueError, OverflowError):
            return None

    # ISO-8601 date: "2024-01-15"
    try:
        d = date.fromisoformat(date_str)
        dt = datetime(d.year, d.month, d.day, tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    # ISO-8601 datetime: "2024-01-15T10:30:00"
    try:
        dt = datetime.fromisoformat(date_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        return int(dt.timestamp() * 1000)
    except ValueError:
        pass

    logger.debug("Could not parse date string: %r", date_str)
    return None


async def annotate_temporal_async(
    model: ModelType,
    facts: list[Fact],
    original_text: str,
    *,
    today: str | None = None,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> dict[int, tuple[int | None, int | None]]:
    """Annotate extracted facts with real-world validity timestamps.

    Returns a mapping of fact_index -> (valid_at_ms, invalid_at_ms).
    Only facts with detected temporal information are included.
    """
    if not facts:
        return {}

    today_str = today or date.today().isoformat()
    facts_text = "\n".join(f"{i}. {f.text}" for i, f in enumerate(facts))

    agent = Agent(model, system_prompt=TEMPORAL_ANNOTATION_SYSTEM, output_type=TemporalAnnotationOutput)
    try:
        result = await agent.run(TEMPORAL_ANNOTATION_USER.format(today=today_str, text=original_text, facts=facts_text))
    except Exception:
        logger.warning("Temporal annotation failed", exc_info=True)
        return {}

    if _on_usage is not None:
        _on_usage("annotate_temporal", result.usage())

    annotations: dict[int, tuple[int | None, int | None]] = {}
    for ann in result.output.annotations:
        valid_ms = _parse_date_to_epoch_ms(ann.valid_at)
        invalid_ms = _parse_date_to_epoch_ms(ann.invalid_at)
        if valid_ms is not None or invalid_ms is not None:
            annotations[ann.fact_index] = (valid_ms, invalid_ms)

    return annotations


def annotate_temporal(
    model: ModelType,
    facts: list[Fact],
    original_text: str,
    *,
    today: str | None = None,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> dict[int, tuple[int | None, int | None]]:
    """Annotate extracted facts with real-world validity timestamps (sync)."""
    return run_sync(annotate_temporal_async(model, facts, original_text, today=today, _on_usage=_on_usage))
