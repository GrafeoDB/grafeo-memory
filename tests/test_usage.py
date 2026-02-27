"""Tests for LLM usage tracking (callbacks + aggregated usage on results)."""

from __future__ import annotations

from pydantic_ai.usage import RunUsage

from grafeo_memory import AddResult, MemoryConfig, MemoryManager, SearchResponse
from tests.mock_llm import MockEmbedder, make_test_model


def _make_manager(model, **config_kwargs):
    embedder = MockEmbedder(dims=16)
    config = MemoryConfig(**config_kwargs)
    return MemoryManager(model, config, embedder=embedder)


# --- AddResult / SearchResponse basic behavior ---


class TestResultTypes:
    def test_add_result_is_list(self):
        r = AddResult([1, 2, 3])
        assert list(r) == [1, 2, 3]
        assert len(r) == 3
        assert r[0] == 1

    def test_add_result_default_usage(self):
        r = AddResult()
        assert isinstance(r.usage, RunUsage)
        assert r.usage.requests == 0
        assert r.usage.input_tokens == 0

    def test_add_result_with_usage(self):
        u = RunUsage(requests=2, input_tokens=100, output_tokens=50)
        r = AddResult([1], usage=u)
        assert r.usage.requests == 2
        assert r.usage.input_tokens == 100

    def test_search_response_is_list(self):
        r = SearchResponse(["a", "b"])
        assert list(r) == ["a", "b"]

    def test_search_response_default_usage(self):
        r = SearchResponse()
        assert isinstance(r.usage, RunUsage)
        assert r.usage.requests == 0

    def test_search_response_with_usage(self):
        u = RunUsage(requests=1, output_tokens=30)
        r = SearchResponse(["x"], usage=u)
        assert r.usage.output_tokens == 30

    def test_add_result_backward_compat_iteration(self):
        """Existing code that iterates over events should still work."""
        r = AddResult(["event1", "event2"])
        collected = [e for e in r]
        assert collected == ["event1", "event2"]

    def test_add_result_backward_compat_bool(self):
        """Empty result is falsy, non-empty is truthy."""
        assert not AddResult()
        assert AddResult(["x"])


# --- Usage callback on add() ---


class TestUsageOnAdd:
    def test_add_returns_add_result_with_usage(self):
        model = make_test_model(
            [
                # combined extraction
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "Alice", "entity_type": "PERSON"}],
                    "relations": [],
                },
                {"decisions": [{"action": "ADD", "text": "alice works at acme", "target_memory_id": None}]},
            ]
        )
        manager = _make_manager(model)
        result = manager.add("Alice works at Acme", user_id="u1")
        assert isinstance(result, AddResult)
        assert result.usage.requests > 0
        assert result.usage.input_tokens > 0
        manager.close()

    def test_add_usage_callback_fires(self):
        model = make_test_model(
            [
                # combined extraction
                {"facts": ["bob likes hiking"], "entities": [], "relations": []},
                {"decisions": [{"action": "ADD", "text": "bob likes hiking", "target_memory_id": None}]},
            ]
        )
        calls: list[tuple[str, RunUsage]] = []
        manager = _make_manager(model, usage_callback=lambda op, u: calls.append((op, u)))
        manager.add("Bob likes hiking", user_id="u1")
        ops = [c[0] for c in calls]
        assert "extract" in ops
        assert all(isinstance(c[1], RunUsage) for c in calls)
        manager.close()

    def test_add_raw_mode_returns_add_result(self):
        """infer=False doesn't call LLM, but still returns AddResult."""
        model = make_test_model([])
        manager = _make_manager(model)
        result = manager.add("raw text", user_id="u1", infer=False)
        assert isinstance(result, AddResult)
        assert len(result) == 1
        assert result.usage.requests == 0  # no LLM calls in raw mode
        manager.close()

    def test_add_empty_extraction_returns_add_result(self):
        model = make_test_model([{"facts": [], "entities": [], "relations": []}])
        manager = _make_manager(model)
        result = manager.add("nothing here", user_id="u1")
        assert isinstance(result, AddResult)
        assert len(result) == 0
        assert result.usage.requests > 0  # extraction LLM call was made
        manager.close()

    def test_usage_callback_error_does_not_break_add(self):
        """A failing usage callback should not prevent add() from succeeding."""
        model = make_test_model(
            [
                # combined extraction
                {"facts": ["test fact"], "entities": [], "relations": []},
                {"decisions": [{"action": "ADD", "text": "test fact", "target_memory_id": None}]},
            ]
        )

        def bad_callback(op, u):
            raise RuntimeError("callback boom")

        manager = _make_manager(model, usage_callback=bad_callback)
        result = manager.add("test", user_id="u1")
        assert isinstance(result, AddResult)
        assert len(result) == 1
        manager.close()

    def test_reconcile_usage_callback_unit(self):
        """Test reconciliation _on_usage callback fires (unit-level test).

        In-memory GrafeoDB vector search doesn't return matches in tests, so
        reconciliation fast-paths in integration tests. Test the callback directly.
        """
        from grafeo_memory.reconciliation.memories import reconcile
        from grafeo_memory.types import Fact

        model = make_test_model(
            [
                {"decisions": [{"action": "NONE", "text": "alice works at acme", "target_memory_id": "1"}]},
            ]
        )
        calls: list[tuple[str, RunUsage]] = []
        facts = [Fact(text="alice works at acme")]
        existing = [{"id": "1", "text": "alice works at acme", "score": 0.95}]
        reconcile(model, facts, existing, _on_usage=lambda op, u: calls.append((op, u)))
        assert len(calls) == 1
        assert calls[0][0] == "reconcile"
        assert isinstance(calls[0][1], RunUsage)
        assert calls[0][1].requests > 0


# --- Usage callback on search() ---


class TestUsageOnSearch:
    def test_search_returns_search_response(self):
        model = make_test_model(
            [
                # add: combined extraction
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "Alice", "entity_type": "PERSON"}],
                    "relations": [],
                },
                {"decisions": [{"action": "ADD", "text": "alice works at acme", "target_memory_id": None}]},
                # For graph search entity extraction during search
                {"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relations": []},
            ]
        )
        manager = _make_manager(model)
        manager.add("Alice works at Acme", user_id="u1")
        result = manager.search("Alice", user_id="u1")
        assert isinstance(result, SearchResponse)
        assert isinstance(result.usage, RunUsage)
        manager.close()

    def test_search_usage_callback_fires(self):
        model = make_test_model(
            [
                # add: combined extraction
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "Alice", "entity_type": "PERSON"}],
                    "relations": [],
                },
                {"decisions": [{"action": "ADD", "text": "alice works at acme", "target_memory_id": None}]},
                # search: graph search entity extraction
                {"entities": [{"name": "Alice", "entity_type": "PERSON"}], "relations": []},
            ]
        )
        calls: list[tuple[str, RunUsage]] = []
        manager = _make_manager(model, usage_callback=lambda op, u: calls.append((op, u)))
        manager.add("Alice works at Acme", user_id="u1")
        calls.clear()

        manager.search("Alice", user_id="u1")
        ops = [c[0] for c in calls]
        # Graph search calls extract_entities
        assert "extract_entities" in ops
        manager.close()


# --- Usage on batch ---


class TestUsageOnBatch:
    def test_add_batch_returns_add_result(self):
        model = make_test_model([])
        manager = _make_manager(model)
        result = manager.add_batch(["text1", "text2"], user_id="u1", infer=False)
        assert isinstance(result, AddResult)
        assert len(result) == 2
        assert result.usage.requests == 0
        manager.close()

    def test_add_batch_infer_accumulates_usage(self):
        model = make_test_model(
            [
                # First message: combined extraction
                {"facts": ["fact one"], "entities": [], "relations": []},
                {"decisions": [{"action": "ADD", "text": "fact one", "target_memory_id": None}]},
                # Second message: combined extraction
                {"facts": ["fact two"], "entities": [], "relations": []},
                {"decisions": [{"action": "ADD", "text": "fact two", "target_memory_id": None}]},
            ]
        )
        manager = _make_manager(model)
        result = manager.add_batch(["msg1", "msg2"], user_id="u1", infer=True)
        assert isinstance(result, AddResult)
        assert len(result) == 2
        # Usage from both messages combined
        assert result.usage.requests >= 2  # at least 2 combined extraction calls
        manager.close()


# --- Usage on summarize() ---


class TestUsageOnSummarize:
    def test_summarize_returns_add_result_with_usage(self):
        model = make_test_model(
            [
                # Add 7 raw memories, then summarize will consolidate first 2
                # Summarize LLM output
                {"memories": ["consolidated memory"]},
            ]
        )
        manager = _make_manager(model)
        for i in range(7):
            manager.add(f"memory {i}", user_id="u1", infer=False)

        result = manager.summarize(user_id="u1", preserve_recent=5, batch_size=20)
        assert isinstance(result, AddResult)
        assert result.usage.requests > 0
        ops_in_usage = result.usage.requests
        assert ops_in_usage >= 1  # at least the summarize LLM call
        manager.close()

    def test_summarize_no_memories_returns_empty_add_result(self):
        model = make_test_model([])
        manager = _make_manager(model)
        result = manager.summarize(user_id="u1")
        assert isinstance(result, AddResult)
        assert len(result) == 0
        assert result.usage.requests == 0
        manager.close()
