"""Temporal keyword detection for search queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# Keywords that suggest the query is about expired/changed facts
_EXPIRED_KEYWORDS = re.compile(
    r"\b(used to|previously|formerly|no longer|changed|before .+ switched|was .+ but now)\b",
    re.IGNORECASE,
)

# Keywords that suggest chronological sorting
_CHRONO_KEYWORDS = re.compile(
    r"\b(when did|first|earliest|latest|last time|most recent|in what order|timeline|chronolog)\b",
    re.IGNORECASE,
)

# Keywords that suggest a time-difference calculation
_TIMEDIFF_KEYWORDS = re.compile(
    r"\b(how many days|how long|how much time|duration|elapsed)\b",
    re.IGNORECASE,
)

# Keywords that suggest a point-in-time query (bi-temporal)
_POINT_IN_TIME_KEYWORDS = re.compile(
    r"\b(as of|at that time|back then|at the time|in \d{4})\b",
    re.IGNORECASE,
)

# Any temporal signal at all (superset)
_ANY_TEMPORAL = re.compile(
    r"\b(when|before|after|during|since|until|first|last|earliest|latest|"
    r"used to|previously|changed|how many days|how long|timeline|recent)\b",
    re.IGNORECASE,
)


@dataclass
class TemporalHints:
    """Hints derived from temporal keywords in a query."""

    include_expired: bool = False
    sort_chronologically: bool = False
    is_temporal: bool = False
    expand_limit: bool = False
    point_in_time_hint: bool = False
    signals: list[str] = field(default_factory=list)


def detect_temporal_hints(query: str) -> TemporalHints:
    """Scan a query for temporal keywords and return search hints.

    This is a fast, rule-based heuristic (no LLM call).
    """
    hints = TemporalHints()

    if _EXPIRED_KEYWORDS.search(query):
        hints.include_expired = True
        hints.is_temporal = True
        hints.signals.append("expired")

    if _CHRONO_KEYWORDS.search(query):
        hints.sort_chronologically = True
        hints.is_temporal = True
        hints.expand_limit = True
        hints.signals.append("chronological")

    if _TIMEDIFF_KEYWORDS.search(query):
        hints.include_expired = True
        hints.is_temporal = True
        hints.expand_limit = True
        hints.signals.append("timediff")

    if _POINT_IN_TIME_KEYWORDS.search(query):
        hints.point_in_time_hint = True
        hints.is_temporal = True
        hints.expand_limit = True
        hints.include_expired = True
        hints.signals.append("point_in_time")

    if not hints.is_temporal and _ANY_TEMPORAL.search(query):
        hints.is_temporal = True
        hints.expand_limit = True
        hints.signals.append("general")

    return hints
