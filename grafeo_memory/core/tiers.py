"""Layer A, pure parts: token budgets for ``assemble()`` and procedural write triggers."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from grafeo_memory.common.types import ContextItem

# User turns that state a standing instruction or correct the assistant.
_TRIGGERS = re.compile(
    r"^\s*(?:please\s+)?(?:always|never|from now on|going forward|in future|in the future|"
    r"remember to|don'?t ever|do not ever|stop\s+\w+ing)\b"
    r"|^\s*no[,.!]?\s+(?:use|it'?s|it is|i meant|i said|call me|please use)\b",
    re.IGNORECASE,
)


def estimate_tokens(text: str) -> int:
    """About four characters per token; enough for budgeting, no tokenizer needed."""
    return max(1, (len(text) + 3) // 4)


def detect_instructions(messages: Sequence[Mapping]) -> list[str]:
    """User-role turns that read as standing instructions or corrections."""
    out: list[str] = []
    for m in messages:
        if (m.get("role") or "user") != "user":
            continue
        text = str(m.get("content") or "").strip()
        for sentence in re.split(r"(?<=[.!?])\s+", text):
            if sentence and _TRIGGERS.search(sentence):
                out.append(sentence.strip())
    return out


def section_budgets(total: int, shares: dict[str, float], present: Sequence[str]) -> dict[str, int]:
    """Split ``total`` tokens over the sections that have content, by renormalized share."""
    weights = {s: max(0.0, float(shares.get(s, 0.0))) for s in present}
    norm = sum(weights.values())
    if norm <= 0:
        return dict.fromkeys(present, total // max(1, len(present)))
    return {s: int(total * w / norm) for s, w in weights.items()}


def fill(items: Sequence[ContextItem], budget: int) -> tuple[list[ContextItem], int]:
    """Take items in order until the budget is spent (at least one when any fit nowhere)."""
    out: list[ContextItem] = []
    used = 0
    for it in items:
        cost = estimate_tokens(it.text) + 8  # provenance line
        if used + cost > budget and out:
            break
        out.append(it)
        used += cost
    return out, used


def rank_key(item: ContextItem) -> tuple[int, float, float]:
    """Certified before inferred before provisional, then confidence, then score."""
    status_rank = {"certified": 0, "inferred": 1, None: 1, "provisional": 2}.get(item.status, 1)
    return (status_rank, -(item.confidence if item.confidence is not None else 0.5), -(item.score or 0.0))
