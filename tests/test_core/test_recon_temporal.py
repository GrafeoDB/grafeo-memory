"""Tests for reconciliation/temporal.py: deterministic interval extraction,
pre-model candidate partition, interval-closing on UPDATE."""

from __future__ import annotations

from grafeo_memory.common.intervals import Interval, parse_iso
from grafeo_memory.core.reconciliation.temporal import (
    close_interval_on_supersede,
    extract_interval,
    partition_candidates,
)


def _iv(s: str, e: str | None = None) -> Interval:
    b = parse_iso(s)
    assert b is not None
    end = parse_iso(e) if e else None
    return Interval(start=b, end=end)


class TestExtractInterval:
    def test_no_date_returns_none(self):
        assert extract_interval("Alice likes hiking") is None

    def test_iso_date(self):
        iv = extract_interval("Alice started at Acme on 2022-06-01.")
        assert iv is not None and iv.lo.year == 2022 and iv.lo.month == 6 and iv.lo.day == 1

    def test_bare_year_is_year_point(self):
        iv = extract_interval("She was born in 1995.")
        assert iv is not None and iv.lo.year == 1995 and iv.hi.year == 1996

    def test_decade_and_fuzzy(self):
        assert extract_interval("grew up in the 1990s") is not None
        assert extract_interval("in the early 2020s") is not None
        assert extract_interval("circa 1995") is not None
        assert extract_interval("around 1995?") is not None

    def test_multiple_dates_span(self):
        iv = extract_interval("worked at Acme from 2020-01-01 to 2022-06-01")
        assert iv is not None
        assert (iv.lo.year, iv.lo.month, iv.lo.day) == (2020, 1, 1)
        assert (iv.hi.year, iv.hi.month, iv.hi.day) == (2022, 6, 1)

    def test_datetime_with_time(self):
        iv = extract_interval("meeting on 2023-05-30T14:00:00Z")
        assert iv is not None and iv.lo.hour == 14 and iv.is_point


class TestPartition:
    def test_disjoint_cleared_overlapping_contested(self):
        new = _iv("2022-01-01", None)
        existing = [
            ("acme", _iv("2020-01-01", "2022-01-01")),  # meets new -> cleared
            ("beta", _iv("2021-06-01", None)),  # overlaps -> contested
            ("unknown", None),  # unknown time -> contested
        ]
        p = partition_candidates(new, existing)
        assert p.cleared == ["acme"]
        assert set(p.contested) == {"beta", "unknown"}

    def test_all_cleared_when_new_has_no_time(self):
        # No interval on the new fact: cannot rule anything out deterministically.
        p = partition_candidates(None, [("a", _iv("2020", "2021")), ("b", _iv("2022", None))])
        assert p.cleared == [] and set(p.contested) == {"a", "b"}


class TestCloseOnSupersede:
    def test_interval_closing(self):
        old = _iv("2020-01-01", None)
        new_start = parse_iso("2022-06-01")
        assert new_start is not None
        closed = close_interval_on_supersede(old, new_start)
        assert (closed.lo.year, closed.lo.month) == (2020, 1)
        assert (closed.hi.year, closed.hi.month, closed.hi.day) == (2022, 6, 1)
        # Adjacent, not overlapping: the closed fact and the new one MEET.
        from grafeo_memory.common.intervals import Allen, allen

        assert allen(closed, _iv("2022-06-01", None)) == Allen.MEETS


class TestTemporalPrecheckWiring:
    """reconcile_async(..., temporal_precheck=True): disjoint-validity facts are
    deterministically ADD with zero model calls; only contested pairs reach the
    engine, and cleared candidates never appear in the prompt."""

    def test_disjoint_fact_skips_model_entirely(self):
        from mock_llm import make_error_model

        from grafeo_memory.common.types import Fact, MemoryAction
        from grafeo_memory.core.reconciliation import reconcile

        model = make_error_model()  # any LLM call raises
        facts = [Fact("Alice worked at Acme from 2020-01-01 to 2022-01-01.")]
        existing = [{"id": "7", "text": "Alice works at Beta since 2022-06-01.", "score": 0.9}]
        decisions = reconcile(model, facts, existing, temporal_precheck=True)
        assert len(decisions) == 1
        assert decisions[0].action == MemoryAction.ADD
        assert decisions[0].engine == "temporal"

    def test_all_cleared_no_call_at_all(self):
        from mock_llm import make_error_model

        from grafeo_memory.common.types import Fact, MemoryAction
        from grafeo_memory.core.reconciliation import reconcile

        model = make_error_model()
        facts = [
            Fact("Alice born 1990-05-05."),
            Fact("Alice hired by Acme 2020-01-01."),
        ]
        existing = [{"id": "7", "text": "Alice works at Beta since 2022-06-01.", "score": 0.9}]
        decisions = reconcile(model, facts, existing, temporal_precheck=True)
        assert [d.action for d in decisions] == [MemoryAction.ADD, MemoryAction.ADD]
        assert all(d.engine == "temporal" for d in decisions)

    def test_contested_fact_reaches_model_with_only_contested_candidates(self):
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import FunctionModel

        from grafeo_memory.common.types import Fact, MemoryAction
        from grafeo_memory.core.reconciliation import reconcile

        seen_prompts: list[str] = []

        def handler(messages, info):
            seen_prompts.append(
                " ".join(str(part.content) for m in messages for part in m.parts if hasattr(part, "content"))
            )
            tool_name = info.output_tools[0].name
            return ModelResponse(
                parts=[
                    ToolCallPart(
                        tool_name=tool_name,
                        args={
                            "decisions": [
                                {
                                    "action": "UPDATE",
                                    "target_memory_id": "7",
                                    "text": "Alice works at Beta since 2022-01-01.",
                                }
                            ]
                        },
                    )
                ]
            )

        model = FunctionModel(function=handler)
        facts = [Fact("Alice works at Beta since 2022-01-01.")]
        existing = [
            {"id": "7", "text": "Alice works at Acme from 2020-01-01 to 2022-06-01.", "score": 0.9},
            {"id": "9", "text": "Alice was born 1990-05-05.", "score": 0.4},
        ]
        decisions = reconcile(model, facts, existing, temporal_precheck=True)
        assert len(decisions) == 1
        assert decisions[0].action == MemoryAction.UPDATE
        assert decisions[0].target_memory_id == "7"
        assert seen_prompts, "model should have been called once"
        prompt = seen_prompts[0]
        assert "ID 7" in prompt  # contested candidate present
        assert "ID 9" not in prompt  # cleared candidate filtered out

    def test_off_by_default_preserves_baseline(self):
        from mock_llm import make_test_model

        from grafeo_memory.common.types import Fact
        from grafeo_memory.core.reconciliation import reconcile

        model = make_test_model([{"decisions": [{"action": "ADD", "text": "x"}]}])
        facts = [Fact("Alice worked at Acme from 2020-01-01 to 2022-01-01.")]
        existing = [{"id": "7", "text": "Alice works at Beta since 2022-06-01.", "score": 0.9}]
        decisions = reconcile(model, facts, existing)  # no temporal_precheck
        assert decisions[0].engine != "temporal"


class TestValidityCues:
    """0.3.0 review: only explicit validity clears a pair; a bare number or an
    open "since" never makes two state facts disjoint."""

    def test_since_is_open_ended(self):
        iv = extract_interval("Lives in Paris since 2019")
        assert iv is not None and iv.end is None and iv.lo.year == 2019

    def test_until_closes(self):
        iv = extract_interval("Lived in Paris until 2022")
        assert iv is not None and iv.start is None and iv.hi.year == 2022

    def test_number_without_time_word_is_not_a_year(self):
        assert extract_interval("Has 2048 followers") is None

    def test_year_range(self):
        iv = extract_interval("Worked at Acme 2020-2022")
        assert iv is not None and (iv.lo.year, iv.hi.year) == (2020, 2022) and not iv.is_point

    def test_between_and(self):
        iv = extract_interval("Studied in Leiden between 2018 and 2020")
        assert iv is not None and (iv.lo.year, iv.hi.year) == (2018, 2020)

    def test_month_names(self):
        day = extract_interval("went to a support group on 7 May 2023")
        assert day is not None and (day.lo.month, day.lo.day, day.hi.day) == (5, 7, 8)
        assert extract_interval("on May 20, 2023").lo.day == 20
        month = extract_interval("visited in May 2023")
        assert month is not None and (month.lo.month, month.hi.month) == (5, 6)

    def test_move_against_since_is_contested(self):
        p = partition_candidates(
            extract_interval("Moved to Berlin in 2024"), [("paris", extract_interval("Lives in Paris since 2019"))]
        )
        assert p.contested == ["paris"]

    def test_counts_are_contested(self):
        p = partition_candidates(
            extract_interval("Has 2048 followers"), [("f", extract_interval("Has 2010 followers"))]
        )
        assert p.contested == ["f"]

    def test_meeting_ranges_clear(self):
        p = partition_candidates(
            extract_interval("Works at Beta since 2022"), [("acme", extract_interval("Worked at Acme 2020-2022"))]
        )
        assert p.cleared == ["acme"]

    def test_close_at_own_start_is_empty_not_a_point(self):
        from grafeo_memory.common.intervals import close_at

        iv = extract_interval("since 2020")
        assert iv is not None and iv.start is not None
        closed = close_at(iv, iv.start)
        assert not closed.is_point and closed.lo == closed.hi
