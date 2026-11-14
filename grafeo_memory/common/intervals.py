"""Layer D temporal core: ISO 8601 / EDTF-style intervals and Allen algebra in pure code.

No model calls, no I/O. Facts and edges carry ``time:Interval`` validity
(``hasBeginning`` / ``hasEnd``) with ISO 8601 values; fuzzy human dates
("1990", "early 2020s", "circa 1995") become precision-annotated bounds.
Allen's 13 interval relations are computed here so temporal questions
("while Alice was at Acme") are interval arithmetic, not model calls.

Intervals are half-open: [start.lo, end.lo), an end bound is exclusive at the
start of its precision unit, so adjacent intervals chain as MEETS without a
spurious overlap. A point interval (one parsed instant) spans its unit. An open
end is unbounded.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta, timezone
from enum import StrEnum


class Precision(StrEnum):
    YEAR = "year"
    MONTH = "month"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"


_UNIT_DELTA = {
    Precision.DAY: timedelta(days=1),
    Precision.HOUR: timedelta(hours=1),
    Precision.MINUTE: timedelta(minutes=1),
    Precision.SECOND: timedelta(seconds=1),
}


def _add_unit(dt: datetime, precision: Precision) -> datetime:
    """dt + one unit of the given precision (calendar-correct), clamped at
    ``datetime.max`` so sentinel dates such as 9999-12-31 never overflow."""
    try:
        if precision == Precision.YEAR:
            return dt.replace(year=dt.year + 1)
        if precision == Precision.MONTH:
            return dt.replace(year=dt.year + 1) if dt.month == 12 else dt.replace(month=dt.month + 1)
        return dt + _UNIT_DELTA[precision]
    except (OverflowError, ValueError):
        return datetime.max.replace(tzinfo=dt.tzinfo)


@dataclass(frozen=True)
class Bound:
    """A time point with precision. ``lo`` fills missing parts with the unit's
    start, ``hi`` with the unit's end (exclusive), so a year-precision bound
    spans the whole year."""

    dt: datetime
    precision: Precision = Precision.DAY

    @property
    def lo(self) -> datetime:
        return self.dt

    @property
    def hi(self) -> datetime:
        return _add_unit(self.dt, self.precision)

    def to_iso(self) -> str:
        """Serialize at the bound's own precision so parse_iso round-trips it."""
        d = self.dt
        if self.precision == Precision.YEAR:
            return f"{d.year:04d}"
        if self.precision == Precision.MONTH:
            return f"{d.year:04d}-{d.month:02d}"
        if self.precision == Precision.DAY:
            return f"{d.year:04d}-{d.month:02d}-{d.day:02d}"
        if self.precision == Precision.HOUR:
            return d.isoformat(timespec="hours")  # parse_iso reads a bare HH back as HOUR
        if self.precision == Precision.MINUTE:
            return d.isoformat(timespec="minutes")
        return d.isoformat()


@dataclass(frozen=True)
class Interval:
    """Half-open validity interval [start.lo, end.lo); None bound = unbounded.

    A point interval (both bounds the same parsed instant) spans its precision
    unit: [b.lo, b.hi), "on 2023-05-30" covers the whole day."""

    start: Bound | None = None
    end: Bound | None = None
    # True when ``end`` is an explicit boundary (a range, or closed by close_at), so an
    # interval closed at its own start is empty instead of reading as a point.
    closed: bool = False

    @property
    def lo(self) -> datetime:
        return self.start.lo if self.start is not None else datetime.min.replace(tzinfo=UTC)

    @property
    def hi(self) -> datetime:
        if self.end is None:
            return datetime.max.replace(tzinfo=UTC)
        if self.is_point:
            return self.end.hi
        return self.end.lo

    @property
    def is_point(self) -> bool:
        """A point interval: both bounds are the same parsed instant (from parse_iso)."""
        return (
            not self.closed
            and self.start is not None
            and self.end is not None
            and self.start.dt == self.end.dt
            and self.start.precision == self.end.precision
        )

    def contains_point(self, point: datetime) -> bool:
        p = point if point.tzinfo is not None else point.replace(tzinfo=UTC)
        return self.lo <= p < self.hi

    def to_iso_pair(self) -> tuple[str | None, str | None]:
        return (self.start.to_iso() if self.start else None, self.end.to_iso() if self.end else None)


class Allen(StrEnum):
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    STARTS = "starts"
    STARTED_BY = "started_by"
    DURING = "during"
    CONTAINS = "contains"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


_ALLEN_INVERSE: dict[Allen, Allen] = {
    Allen.BEFORE: Allen.AFTER,
    Allen.AFTER: Allen.BEFORE,
    Allen.MEETS: Allen.MET_BY,
    Allen.MET_BY: Allen.MEETS,
    Allen.OVERLAPS: Allen.OVERLAPPED_BY,
    Allen.OVERLAPPED_BY: Allen.OVERLAPS,
    Allen.STARTS: Allen.STARTED_BY,
    Allen.STARTED_BY: Allen.STARTS,
    Allen.DURING: Allen.CONTAINS,
    Allen.CONTAINS: Allen.DURING,
    Allen.FINISHES: Allen.FINISHED_BY,
    Allen.FINISHED_BY: Allen.FINISHES,
    Allen.EQUALS: Allen.EQUALS,
}


def allen(a: Interval, b: Interval) -> Allen:
    """The single Allen relation holding between half-open intervals a and b."""
    also, ahi, blo, bhi = a.lo, a.hi, b.lo, b.hi
    if also == blo and ahi == bhi:
        return Allen.EQUALS
    if ahi <= blo:
        return Allen.MEETS if ahi == blo else Allen.BEFORE
    if bhi <= also:
        return Allen.MET_BY if bhi == also else Allen.AFTER
    if also < blo < ahi < bhi:
        return Allen.OVERLAPS
    if blo < also < bhi < ahi:
        return Allen.OVERLAPPED_BY
    if also == blo and ahi < bhi:
        return Allen.STARTS
    if also == blo and bhi < ahi:
        return Allen.STARTED_BY
    if blo < also and ahi < bhi:
        return Allen.DURING
    if also < blo and bhi < ahi:
        return Allen.CONTAINS
    if ahi == bhi and blo < also:
        return Allen.FINISHES
    if ahi == bhi and also < blo:
        return Allen.FINISHED_BY
    raise AssertionError(f"uncovered interval pair: {a} vs {b}")  # pragma: no cover


def allen_inverse(r: Allen) -> Allen:
    return _ALLEN_INVERSE[r]


def intervals_overlap(a: Interval, b: Interval) -> bool:
    """True when the intervals share at least one point (the five overlap-class
    Allen relations). Non-overlapping intervals are never temporal conflicts: the
    pre-model reconciliation rule. An empty interval (closed at its own start) shares
    no point with anything."""
    if a.lo >= a.hi or b.lo >= b.hi:
        return False
    return allen(a, b) in (
        Allen.OVERLAPS,
        Allen.OVERLAPPED_BY,
        Allen.STARTS,
        Allen.STARTED_BY,
        Allen.DURING,
        Allen.CONTAINS,
        Allen.FINISHES,
        Allen.FINISHED_BY,
        Allen.EQUALS,
    )


def close_at(interval: Interval, bound: Bound) -> Interval:
    """Interval-closing on UPDATE: the superseded fact's validity ends at ``bound``
    (closing at its own start leaves an empty interval, not a point)."""
    return Interval(start=interval.start, end=bound, closed=True)


# ---------------------------------------------------------------------------
# Parsing: ISO 8601 with precision + a pragmatic EDTF subset
# ---------------------------------------------------------------------------

_ISO_RE = re.compile(
    r"^(\d{4})(?:-(\d{2})(?:-(\d{2})(?:[T ](\d{2})(?::(\d{2})(?::(\d{2}))?)?)?)?)?\s*(Z|[+-]\d{2}:?\d{2})?$"
)


def parse_iso(text: str) -> Bound | None:
    """Parse an ISO 8601 date/datetime with implicit precision.

    "2023" -> year precision, "2023-05" -> month, "2023-05-30" -> day,
    "2023-05-30T14" -> hour, "2023-05-30T14:00" -> minute, with seconds -> second. Returns None when
    the text is not a date.
    """
    m = _ISO_RE.match(text.strip())
    if not m:
        return None
    year, month, day, hour, minute, second, tz = m.groups()
    try:
        if tz == "Z":
            tzinfo = UTC
        elif tz:
            sign = 1 if tz[0] == "+" else -1
            tz = tz[1:].replace(":", "")
            tzinfo = timezone(sign * timedelta(hours=int(tz[:2]), minutes=int(tz[2:])))
        else:
            tzinfo = UTC
        if month is None:
            return Bound(datetime(int(year), 1, 1, tzinfo=tzinfo), Precision.YEAR)
        if day is None:
            return Bound(datetime(int(year), int(month), 1, tzinfo=tzinfo), Precision.MONTH)
        if hour is None:
            return Bound(datetime(int(year), int(month), int(day), tzinfo=tzinfo), Precision.DAY)
        if minute is None:
            return Bound(datetime(int(year), int(month), int(day), int(hour), tzinfo=tzinfo), Precision.HOUR)
        if second is None:
            return Bound(
                datetime(int(year), int(month), int(day), int(hour), int(minute), tzinfo=tzinfo),
                Precision.MINUTE,
            )
        return Bound(
            datetime(int(year), int(month), int(day), int(hour), int(minute), int(second), tzinfo=tzinfo),
            Precision.SECOND,
        )
    except ValueError:
        return None


_DECADE_RE = re.compile(r"^(early|mid|late)?\s*(\d{4})s$", re.IGNORECASE)
_CIRCA_RE = re.compile(r"^(?:circa|~|c\.)\s*(\d{4})$", re.IGNORECASE)
_UNCERTAIN_RE = re.compile(r"^(\d{4})\?$")


def parse_fuzzy(text: str) -> Interval | None:
    """Parse a fuzzy human date into an Interval with precision-annotated bounds.

    Handles: "2020s", "early/mid/late 2020s", "circa 2020" / "~2020" / "2020?"
    (year-precision interval), and plain ISO via :func:`parse_iso` (point interval).
    Returns None when nothing matches.
    """
    t = text.strip()
    tl = t.lower()
    m = _DECADE_RE.match(tl)
    if m:
        qualifier, decade = m.groups()
        start_year = int(decade)
        if qualifier == "early":
            lo, hi = start_year, start_year + 3
        elif qualifier == "mid":
            lo, hi = start_year + 3, start_year + 7
        elif qualifier == "late":
            lo, hi = start_year + 7, start_year + 10
        else:
            lo, hi = start_year, start_year + 10
        if lo < 1:  # year 0 does not exist in datetime
            return None
        # "9990s" ends at year 10000, past datetime's range: clamp to the open future
        end = datetime(hi, 1, 1, tzinfo=UTC) if hi <= 9999 else datetime.max.replace(tzinfo=UTC)
        return Interval(
            start=Bound(datetime(lo, 1, 1, tzinfo=UTC), Precision.YEAR),
            end=Bound(end, Precision.YEAR),
        )
    m = _CIRCA_RE.match(tl) or _UNCERTAIN_RE.match(tl)
    if m:
        y = int(m.group(1))
        if y < 1:
            return None
        bnd = Bound(datetime(y, 1, 1, tzinfo=UTC), Precision.YEAR)
        return Interval(start=bnd, end=bnd)
    b = parse_iso(t)
    if b is not None:
        return Interval(start=b, end=b)
    return None


def parse_interval(start_iso: str | None, end_iso: str | None, *, closed: bool = False) -> Interval | None:
    """Build an Interval from stored ISO strings (None = unbounded side).

    ``to_iso_pair`` writes a point and an empty interval alike as (s, s); pass the
    stored ``closed`` flag to read the latter back as empty instead of as a point."""
    s = parse_iso(start_iso) if start_iso else None
    e = parse_iso(end_iso) if end_iso else None
    if s is None and start_iso:
        return None
    if e is None and end_iso:
        return None
    return Interval(start=s, end=e, closed=closed and e is not None)
