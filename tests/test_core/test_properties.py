"""Property checks over seeded random inputs: Allen algebra, interval closing, the ladder."""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta

from grafeo_memory.common.intervals import Allen, Bound, Interval, Precision, allen, allen_inverse, close_at
from grafeo_memory.core.ontology.normalize import normalize
from grafeo_memory.core.ontology.scheme import ConceptScheme

RNG = random.Random(20261112)
BASE = datetime(2020, 1, 1, tzinfo=UTC)


def _iv(rng: random.Random) -> Interval:
    a, b = sorted(rng.sample(range(0, 40), 2))
    return Interval(start=Bound(BASE + timedelta(days=a)), end=Bound(BASE + timedelta(days=b)))


def test_allen_is_total_and_inverse_symmetric():
    seen: set[Allen] = set()
    for _ in range(3000):
        a, b = _iv(RNG), _iv(RNG)
        r = allen(a, b)
        seen.add(r)
        assert allen(b, a) == allen_inverse(r)
        assert allen_inverse(allen_inverse(r)) == r
        assert (r == Allen.EQUALS) == (a.lo == b.lo and a.hi == b.hi)
    assert seen == set(Allen)  # every one of the 13 relations occurs


def test_interval_closing_leaves_no_overlap_or_gap():
    for _ in range(500):
        start = RNG.randrange(0, 30)
        cut = start + RNG.randrange(1, 10)
        old = Interval(start=Bound(BASE + timedelta(days=start)), end=None)
        new_start = Bound(BASE + timedelta(days=cut))
        closed = close_at(old, new_start)
        new = Interval(start=new_start, end=None)
        assert allen(closed, new) == Allen.MEETS  # back to back: no gap, no overlap


def test_ladder_is_idempotent():
    scheme = ConceptScheme()
    scheme.add_concept("JavaScript", "JavaScript", "Technology", alt_labels=["JS", "ECMAScript"])
    scheme.add_concept("Acme", "Acme Corp", "Organization", alt_labels=["ACME"])
    mentions = [
        ("JS", "Technology"),
        ("ecmascript", "Technology"),
        ("ACME", "Organization"),
        ("Acme Corp", "Organization"),
    ]
    for mention, etype in mentions:
        first = normalize(mention, etype, scheme)
        assert first.resolved
        again = normalize(scheme.concepts[first.canonical_id].pref_label, etype, scheme)
        assert again.canonical_id == first.canonical_id and again.rung == 1


def test_year_precision_bounds_contain_their_days():
    for _ in range(200):
        year = RNG.randrange(1990, 2030)
        y = Interval(start=Bound(datetime(year, 1, 1, tzinfo=UTC), Precision.YEAR), end=None)
        day = BASE.replace(year=year) + timedelta(days=RNG.randrange(0, 365))
        assert y.contains_point(day)
