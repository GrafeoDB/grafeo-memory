"""Tests for v0.2.0 search quality improvements."""

from __future__ import annotations

import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryConfig, MemoryManager


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder)


def _make_async_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return AsyncMemoryManager(model, config, embedder=embedder)


# -- LLM output fixtures --

_ADD_OUTPUTS = [
    {"facts": ["alice works at acme"], "entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
    {"decisions": [{"action": "ADD", "text": "alice works at acme", "target_memory_id": None}]},
    {"delete": []},
]

_SEARCH_ENTITY_OUTPUT = {"entities": [{"name": "alice", "entity_type": "person"}], "relations": []}


class TestSearchResultSource:
    """Test that search results include source provenance."""

    def test_search_result_has_source(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs)
        mgr.add("alice works at acme")
        results = mgr.search("alice")
        assert len(results) > 0
        for r in results:
            assert r.source in ("vector", "graph", "both"), f"Unexpected source: {r.source}"

    def test_get_all_source_is_none(self):
        outputs = _ADD_OUTPUTS
        mgr = _make_manager(outputs)
        mgr.add("alice works at acme")
        results = mgr.get_all()
        assert len(results) > 0
        for r in results:
            assert r.source is None


class TestSearchMinScore:
    """Test minimum score filtering on search results."""

    def test_min_score_default_returns_all(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs)
        mgr.add("alice works at acme")
        results = mgr.search("alice")
        assert len(results) > 0

    def test_min_score_filters_low(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs)
        mgr.add("alice works at acme")
        # Set impossibly high threshold: should filter everything
        results = mgr.search("alice", min_score=0.99)
        # With mock embeddings, scores are unlikely to be >= 0.99
        # Either empty or all results have score >= 0.99
        for r in results:
            assert r.score >= 0.99

    def test_min_score_from_config(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs, search_min_score=0.99)
        mgr.add("alice works at acme")
        results = mgr.search("alice")
        for r in results:
            assert r.score >= 0.99

    def test_min_score_param_overrides_config(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs, search_min_score=0.99)
        mgr.add("alice works at acme")
        # Per-call min_score=0.0 overrides config's 0.99
        results = mgr.search("alice", min_score=0.0)
        # Should return results since 0.0 means no filtering
        assert len(results) > 0


class TestAgreementBonus:
    """Test cross-source agreement bonus in result merging."""

    def test_agreement_bonus_default(self):
        config = MemoryConfig()
        assert config.agreement_bonus == 0.1

    def test_agreement_bonus_zero_disables(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs, agreement_bonus=0.0)
        mgr.add("alice works at acme")
        results = mgr.search("alice")
        # Should still work, just no bonus applied
        assert len(results) > 0


class TestReconciliationThreshold:
    """Test renamed reconciliation_threshold (was similarity_threshold)."""

    def test_default_value(self):
        config = MemoryConfig()
        assert config.reconciliation_threshold == 0.3

    def test_high_threshold_reduces_matches(self):
        # With threshold=0.99, almost nothing will be similar enough to reconcile
        add1 = [
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
            {"decisions": [{"action": "ADD", "text": "alice works at acme", "target_memory_id": None}]},
            {"delete": []},
        ]
        add2 = [
            {"facts": ["alice works at acme inc"], "entities": [], "relations": []},
            {"decisions": [{"action": "ADD", "text": "alice works at acme inc", "target_memory_id": None}]},
            {"delete": []},
        ]
        mgr = _make_manager(add1 + add2, reconciliation_threshold=0.99)
        mgr.add("alice works at acme")
        mgr.add("alice works at acme inc")
        # Both should be added (no reconciliation due to high threshold)
        all_mems = mgr.get_all()
        assert len(all_mems) == 2


class TestEmbeddingDimensionValidation:
    """Test that mismatched embedding dimensions raise ValueError."""

    def test_dimension_mismatch_raises(self):
        # Create manager with dims=16 but use an embedder that returns wrong dims
        class WrongDimEmbedder:
            @property
            def dimensions(self):
                return 16  # lie about dims

            def embed(self, texts):
                # Return 8-dim vectors instead of 16
                return [[0.1] * 8 for _ in texts]

        model = make_test_model(
            [
                {"facts": ["test fact"], "entities": [], "relations": []},
                {"decisions": [{"action": "ADD", "text": "test fact", "target_memory_id": None}]},
                {"delete": []},
            ]
        )
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        mgr = MemoryManager(model, config, embedder=WrongDimEmbedder())
        with pytest.raises(ValueError, match="Embedding dimension mismatch"):
            mgr.add("test fact")

    def test_dimension_match_succeeds(self):
        outputs = _ADD_OUTPUTS
        mgr = _make_manager(outputs, embedding_dimensions=16)
        events = mgr.add("alice works at acme")
        assert len(events) > 0


class TestConcurrentAddLocking:
    """Test that per-user locking exists and serializes concurrent add() calls."""

    def test_user_lock_created(self):
        """Verify that _add() creates a per-user asyncio.Lock."""
        outputs = _ADD_OUTPUTS
        mgr = _make_manager(outputs)
        mgr.add("alice works at acme")
        # After add(), the user_id should have a lock entry
        assert "test_user" in mgr._user_locks

    def test_different_users_get_separate_locks(self):
        add1 = [
            {"facts": ["fact one"], "entities": [], "relations": []},
            {"decisions": [{"action": "ADD", "text": "fact one", "target_memory_id": None}]},
            {"delete": []},
        ]
        add2 = [
            {"facts": ["fact two"], "entities": [], "relations": []},
            {"decisions": [{"action": "ADD", "text": "fact two", "target_memory_id": None}]},
            {"delete": []},
        ]
        mgr = _make_manager(add1 + add2)
        mgr.add("fact one", user_id="user_a")
        mgr.add("fact two", user_id="user_b")
        assert "user_a" in mgr._user_locks
        assert "user_b" in mgr._user_locks
        assert mgr._user_locks["user_a"] is not mgr._user_locks["user_b"]


class TestExplainMinScoreStep:
    """Test that explain() includes min_score_filter step when active."""

    def test_explain_includes_min_score_step(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs, search_min_score=0.5)
        mgr.add("alice works at acme")
        result = mgr.explain("alice")
        step_names = [s.name for s in result.steps]
        assert "min_score_filter" in step_names

    def test_explain_no_min_score_step_when_zero(self):
        outputs = [*_ADD_OUTPUTS, _SEARCH_ENTITY_OUTPUT]
        mgr = _make_manager(outputs, search_min_score=0.0)
        mgr.add("alice works at acme")
        result = mgr.explain("alice")
        step_names = [s.name for s in result.steps]
        assert "min_score_filter" not in step_names
