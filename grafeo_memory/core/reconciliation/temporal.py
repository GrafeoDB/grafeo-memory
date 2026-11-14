"""Layer D temporal wiring for reconciliation: pure code, no model calls.

- extract validity intervals from fact text deterministically (ISO 8601, month
  names, fuzzy dates via ``grafeo_memory.common.intervals.parse_fuzzy``),
- partition candidates into cleared (disjoint validity, never a conflict)
  and contested (overlapping or unknown, reaches the engine),
- interval-closing on UPDATE (the superseded fact's validity ends at the new
  fact's boundary).

Only explicit validity counts: "since 2019" opens an interval with no end,
"until 2022" closes one, "2020-2022" / "from X to Y" / "between X and Y" are
ranges, and a dated mention ("in 2024", "on 7 May 2023", an ISO date) is a
point. A bare four-digit number without a time word ("2048 followers") is not a
date, and text without any validity stays unknown, so it is never cleared.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime

from grafeo_memory.common.intervals import Bound, Interval, Precision, parse_fuzzy
from grafeo_memory.core.ontology.temporal_rules import pre_model_conflict

_MONTH_NAMES = [
    ("january", "jan"),
    ("february", "feb"),
    ("march", "mar"),
    ("april", "apr"),
    ("may",),
    ("june", "jun"),
    ("july", "jul"),
    ("august", "aug"),
    ("september", "sept", "sep"),
    ("october", "oct"),
    ("november", "nov"),
    ("december", "dec"),
]
_MONTHS = {name: i + 1 for i, names in enumerate(_MONTH_NAMES) for name in names}
_MONTH_ALT = "|".join(sorted(_MONTHS, key=len, reverse=True))
_YEAR = r"(?:19|20)\d{2}"
_ORD = r"(?:st|and|rd|th)?"

# Longest forms first. The ISO month form must not swallow a year range ("2020-2022").
_DATE = re.compile(
    r"(?P<iso>\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?(?:Z|[+-]\d{2}:?\d{2})?|\d{4}-\d{2}(?!\d))"
    rf"|(?P<dmy>\b\d{{1,2}}{_ORD}\s+(?:{_MONTH_ALT})\.?,?\s+{_YEAR}\b)"
    rf"|(?P<mdy>\b(?:{_MONTH_ALT})\.?\s+\d{{1,2}}{_ORD},?\s+{_YEAR}\b)"
    rf"|(?P<my>\b(?:{_MONTH_ALT})\.?,?\s+{_YEAR}\b)"
    rf"|(?P<fuzzy>\b(?:early|mid|late)\s+{_YEAR}s\b|\b{_YEAR}s\b|(?:circa|~|c\.)\s*{_YEAR}\b|\b{_YEAR}\?)"
    rf"|(?P<year>\b{_YEAR}\b)",
    re.IGNORECASE,
)
# "after X" opens an interval at X and "before X" closes one there: neither is a point in X.
_START_CUE = re.compile(r"\b(?:since|from|after|starting(?:\s+in|\s+on)?|beginning(?:\s+in|\s+on)?|as\s+of)\s*$", re.I)
_END_CUE = re.compile(r"\b(?:until|till|til|through|up\s+to|before)\s*$", re.I)
_POINT_CUE = re.compile(r"\b(?:in|on|during|of|at|by|around|about|circa|year)\s*$", re.I)
_RANGE_JOIN = re.compile(r"^\s*(?:-|\u2013|\u2014|to|until|till|through)\s*$", re.I)  # hyphen, en or em dash
_AND_JOIN = re.compile(r"^\s*and\s*$", re.I)
_BETWEEN = re.compile(r"\bbetween\s*$", re.I)


def _parse_token(kind: str, text: str) -> Interval | None:
    if kind in ("iso", "fuzzy", "year"):
        return parse_fuzzy(text)
    words = re.findall(r"[a-z]+|\d+", text.lower())
    nums = [int(w) for w in words if w.isdigit()]
    month = next((_MONTHS[w] for w in words if w in _MONTHS), None)
    if month is None or not nums:
        return None
    try:
        if kind == "my":
            b = Bound(datetime(nums[-1], month, 1, tzinfo=UTC), Precision.MONTH)
        else:
            b = Bound(datetime(nums[-1], month, nums[0], tzinfo=UTC), Precision.DAY)
    except ValueError:
        return None
    return Interval(start=b, end=b)


def extract_interval(text: str) -> Interval | None:
    """Deterministically pull a validity interval out of fact text, or None when the
    text states no validity (see the module docstring for what counts)."""
    tokens: list[tuple[str, Interval, re.Match[str]]] = []
    for m in _DATE.finditer(text):
        kind = m.lastgroup or ""
        iv = _parse_token(kind, m.group(0))
        if iv is not None:
            tokens.append((kind, iv, m))
    if not tokens:
        return None

    parts: list[tuple[str, Interval]] = []  # (role, interval)
    i = 0
    while i < len(tokens):
        kind, iv, m = tokens[i]
        before = text[max(0, m.start() - 24) : m.start()]
        if i + 1 < len(tokens):
            _kind2, iv2, m2 = tokens[i + 1]
            join = text[m.end() : m2.start()]
            is_range = _RANGE_JOIN.match(join) or (_AND_JOIN.match(join) and _BETWEEN.search(before))
            if is_range and iv.start is not None and iv2.start is not None:
                parts.append(("range", Interval(start=iv.start, end=iv2.start, closed=True)))
                i += 2
                continue
        if _START_CUE.search(before) and iv.start is not None:
            parts.append(("start", Interval(start=iv.start, end=None)))
        elif _END_CUE.search(before) and iv.start is not None:
            parts.append(("end", Interval(start=None, end=iv.start, closed=True)))
        elif kind != "year" or _POINT_CUE.search(before):
            parts.append(("point", iv))
        # else: a bare four-digit number with no time word is not a date
        i += 1

    if not parts:
        return None
    if len(parts) == 1:
        return parts[0][1]
    ivs = [iv for _role, iv in parts]
    starts = [iv.start for iv in ivs if iv.start is not None]
    start = min(starts, key=lambda b: b.lo) if starts else None
    open_end = any(iv.end is None for iv in ivs) and not any(role == "end" for role, _iv in parts)
    if open_end:
        return Interval(start=start, end=None)
    hi = max(iv.hi for iv in ivs if iv.end is not None)
    return Interval(start=start, end=Bound(hi, Precision.DAY), closed=True)


@dataclass
class TemporalPartition:
    cleared: list[str]  # candidate ids that can never conflict (disjoint validity)
    contested: list[str]  # candidate ids whose validity overlaps or is unknown: they reach the engine


def partition_candidates(
    new_interval: Interval | None,
    existing: list[tuple[str, Interval | None]],
) -> TemporalPartition:
    """Pre-model split: non-overlapping valid intervals are never conflicts and skip
    the engine entirely."""
    cleared: list[str] = []
    contested: list[str] = []
    for cid, iv in existing:
        if pre_model_conflict(new_interval, iv):
            contested.append(cid)
        else:
            cleared.append(cid)
    return TemporalPartition(cleared=cleared, contested=contested)


def close_interval_on_supersede(old: Interval, new_start: Bound) -> Interval:
    """UPDATE = interval-closing: the superseded fact's validity ends at the
    new fact's validity boundary (half-open, so adjacent intervals MEET). Closing
    never extends it: a fact that already ended by then keeps its own end."""
    if old.end is not None and old.hi <= new_start.lo:
        return old
    return Interval(start=old.start, end=new_start, closed=True)
