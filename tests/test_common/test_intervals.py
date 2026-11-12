"""Tests for the temporal core (grafeo_memory.intervals): Allen algebra + parsing."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from grafeo_memory.common.intervals import (
    Allen,
    Bound,
    Interval,
    Precision,
    allen,
    allen_inverse,
    close_at,
    intervals_overlap,
    parse_fuzzy,
    parse_interval,
    parse_iso,
)


def b(text: str) -> Bound:
    out = parse_iso(text)
    assert out is not None, text
    return out


def iv(start: str | None, end: str | None) -> Interval:
    return Interval(start=b(start) if start else None, end=b(end) if end else None)


class TestBound:
    def test_year_precision_spans_whole_year(self):
        y = b("2023")
        assert y.precision == Precision.YEAR
        assert y.lo == datetime(2023, 1, 1, tzinfo=UTC)
        assert y.hi == datetime(2024, 1, 1, tzinfo=UTC)

    def test_month_precision(self):
        m = b("2023-02")
        assert m.precision == Precision.MONTH
        assert m.hi == datetime(2023, 3, 1, tzinfo=UTC)

    def test_datetime_with_z(self):
        d = b("2023-05-30T14:00:00Z")
        assert d.precision == Precision.SECOND
        assert d.dt == datetime(2023, 5, 30, 14, 0, tzinfo=UTC)

    def test_datetime_with_offset(self):
        d = b("2023-05-30T14:00+02:00")
        assert d.dt.utcoffset().total_seconds() == 7200  # type: ignore[attr-defined]

    def test_non_date_returns_none(self):
        assert parse_iso("last tuesday-ish") is None
        assert parse_iso("2023-13-40") is None


class TestAllen:
    def test_equals_is_reflexive(self):
        a = iv("2020", "2021")
        assert allen(a, a) == Allen.EQUALS
        assert allen_inverse(Allen.EQUALS) == Allen.EQUALS

    def test_all_thirteen_relations(self):
        base = iv("2020-01-01", "2021-01-01")
        assert allen(base, iv("2021-06-01", "2022-01-01")) == Allen.BEFORE
        assert allen(base, iv("2019-01-01", "2019-06-01")) == Allen.AFTER
        assert allen(base, iv("2021-01-01", "2022-01-01")) == Allen.MEETS
        assert allen(base, iv("2019-01-01", "2020-01-01")) == Allen.MET_BY
        assert allen(base, iv("2020-06-01", "2021-06-01")) == Allen.OVERLAPS
        assert allen(base, iv("2019-06-01", "2020-06-01")) == Allen.OVERLAPPED_BY
        assert allen(base, iv("2020-01-01", "2022-01-01")) == Allen.STARTS
        assert allen(base, iv("2020-01-01", "2020-06-01")) == Allen.STARTED_BY
        assert allen(base, iv("2019-01-01", "2022-01-01")) == Allen.DURING
        assert allen(base, iv("2020-03-01", "2020-09-01")) == Allen.CONTAINS
        assert allen(base, iv("2019-06-01", "2021-01-01")) == Allen.FINISHES
        assert allen(base, iv("2020-06-01", "2021-01-01")) == Allen.FINISHED_BY
        assert allen(base, iv("2020-01-01", "2021-01-01")) == Allen.EQUALS

    @pytest.mark.parametrize(
        "a_start,a_end,b_start,b_end",
        [
            ("2019-01-01", "2020-01-01", "2020-01-01", "2021-01-01"),
            ("2020-01-01", "2021-01-01", "2019-01-01", "2020-01-01"),
            ("2020-01-01", "2021-01-01", "2020-06-01", "2021-06-01"),
            ("2020-01-01", "2021-01-01", "2020-03-01", "2020-09-01"),
            ("2020-01-01", "2021-01-01", "2020-01-01", "2021-01-01"),
            ("2020", "2022", "2021", "2023"),
            ("2020", "2020", "2020", "2021"),
        ],
    )
    def test_inverse_symmetry(self, a_start, a_end, b_start, b_end):
        a, c = iv(a_start, a_end), iv(b_start, b_end)
        r = allen(a, c)
        assert allen(c, a) == allen_inverse(r)

    def test_before_composes(self):
        a = iv("2018", "2019")
        c = iv("2019", "2020")
        d = iv("2020", "2021")
        assert allen(a, c) == Allen.MEETS and allen(c, d) == Allen.MEETS
        assert allen(a, d) == Allen.BEFORE

    def test_year_precision_meets_day_precision(self):
        # Year-precision 2020..2021 ends exactly where 2021-01-01 begins.
        assert allen(iv("2020", "2021"), iv("2021-01-01", None)) == Allen.MEETS


class TestOverlapAndPreModelRule:
    def test_overlapping_classes(self):
        a = iv("2020-01-01", "2021-01-01")
        assert intervals_overlap(a, iv("2020-06-01", "2021-06-01"))
        assert intervals_overlap(a, iv("2020-03-01", "2020-09-01"))
        assert intervals_overlap(a, a)

    def test_non_overlap_means_never_conflict(self):
        # The doc §7 pre-model rule: disjoint validity never conflicts.
        acme = iv("2020-01-01", "2022-01-01")
        beta = iv("2022-01-01", None)
        assert not intervals_overlap(acme, beta)
        assert allen(acme, beta) == Allen.MEETS
        past = iv(None, "2019-12-31")
        assert not intervals_overlap(acme, past)

    def test_unbounded_overlaps_everything_around_it(self):
        now = iv("2023-06-01", None)
        assert intervals_overlap(now, iv("2024-01-01", "2025-01-01"))
        assert not intervals_overlap(now, iv("2020-01-01", "2023-01-01"))


class TestHelpers:
    def test_close_at(self):
        job = iv("2020-01-01", None)
        closed = close_at(job, b("2022-06-01"))
        assert closed.start is not None and closed.start.dt == datetime(2020, 1, 1, tzinfo=UTC)
        assert closed.end is not None and closed.end.dt == datetime(2022, 6, 1, tzinfo=UTC)
        assert allen(closed, iv("2023-01-01", None)) == Allen.BEFORE

    def test_contains_point(self):
        job = iv("2020-01-01", "2022-01-01")
        assert job.contains_point(datetime(2021, 5, 5, tzinfo=UTC))
        assert not job.contains_point(datetime(2022, 1, 1, tzinfo=UTC))  # half-open end
        assert not job.contains_point(datetime(2019, 12, 31, tzinfo=UTC))

    def test_contains_point_point_interval_spans_unit(self):
        p = parse_fuzzy("2023-05-30T12:00:00Z")
        assert p is not None
        assert p.contains_point(datetime(2023, 5, 30, 12, 0, tzinfo=UTC))
        assert not p.contains_point(datetime(2023, 5, 30, 12, 0, 1, tzinfo=UTC))

    def test_is_point(self):
        assert parse_fuzzy("2023-05-30").is_point
        assert not iv("2023", "2024").is_point
        assert not iv(None, None).is_point

    def test_iso_pair_roundtrip(self):
        i = iv("2020-01-01", "2021-06-01")
        s, e = i.to_iso_pair()
        back = parse_interval(s, e)
        assert back is not None and back.lo == i.lo and back.hi == i.hi


class TestFuzzyParsing:
    def test_decade(self):
        i = parse_fuzzy("2020s")
        assert i is not None and i.lo == datetime(2020, 1, 1, tzinfo=UTC) and i.hi == datetime(2030, 1, 1, tzinfo=UTC)

    def test_early_mid_late_decade(self):
        early = parse_fuzzy("early 2020s")
        mid = parse_fuzzy("mid 2020s")
        late = parse_fuzzy("late 2020s")
        assert early is not None and early.hi == datetime(2023, 1, 1, tzinfo=UTC)
        assert mid is not None
        assert mid.lo == datetime(2023, 1, 1, tzinfo=UTC)
        assert mid.hi == datetime(2027, 1, 1, tzinfo=UTC)
        assert late is not None and late.lo == datetime(2027, 1, 1, tzinfo=UTC)

    def test_circa_and_uncertain(self):
        for text in ("circa 1995", "~1995", "c. 1995", "1995?"):
            i = parse_fuzzy(text)
            assert i is not None, text
            assert i.lo == datetime(1995, 1, 1, tzinfo=UTC)

    def test_plain_year(self):
        i = parse_fuzzy("1990")
        assert i is not None and i.lo == datetime(1990, 1, 1, tzinfo=UTC) and i.hi == datetime(1991, 1, 1, tzinfo=UTC)

    def test_garbage_returns_none(self):
        assert parse_fuzzy("sometime in college") is None


def test_sentinel_far_future_date_does_not_overflow():
    """0.3.0 review: "valid until 9999-12-31" crashed add() under temporal_precheck."""
    from grafeo_memory.common.intervals import allen
    from grafeo_memory.core.reconciliation.temporal import extract_interval

    far = extract_interval("valid until 9999-12-31")
    near = extract_interval("since 2020")
    assert far is not None and near is not None
    assert allen(far, near) is not None


def test_minute_precision_round_trips():
    from grafeo_memory.common.intervals import Precision, parse_iso

    b = parse_iso("2023-05-30T14:05Z")
    assert b is not None and b.precision == Precision.MINUTE
    again = parse_iso(b.to_iso())
    assert again is not None and again.precision == Precision.MINUTE and again.dt == b.dt
