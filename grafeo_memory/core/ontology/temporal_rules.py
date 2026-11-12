"""Pre-model temporal reconciliation rules, in pure code.

Non-overlapping valid intervals are never conflicts (Acme 2020-2022, Beta
2022-now both stand); only overlapping + contradictory facts reach a model.
A missing interval means unknown time, so it cannot be ruled out and the pair
still reaches the model.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime

from grafeo_memory.common.intervals import Interval, intervals_overlap


def pre_model_conflict(a: Interval | None, b: Interval | None) -> bool:
    """True when the pair may conflict and must go to a model; False when the
    pre-model rule already clears it (disjoint validity)."""
    if a is None or b is None:
        return True
    return intervals_overlap(a, b)


def point_in_time_filter[T](items: Iterable[tuple[T, Interval | None]], point: datetime) -> list[T]:
    """Bi-temporal point-in-time query: keep items whose validity covers ``point``.
    Unknown interval (None) is kept, cannot exclude deterministically."""
    return [item for item, iv in items if iv is None or iv.contains_point(point)]
