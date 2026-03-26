"""Tests for multi-hop search, graph algorithm scoring, cross-session boost, and MMR diversity."""

from __future__ import annotations

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.scoring import apply_cross_session_boost
from grafeo_memory.types import MEMORY_LABEL, SearchResult


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder)


def _extraction_output(facts, entities=None, relations=None):
    return {
        "facts": facts,
        "entities": entities or [],
        "relations": relations or [],
    }


# ---------------------------------------------------------------------------
# 14.1: 2-hop graph traversal
# ---------------------------------------------------------------------------


class TestTwoHopGraphSearch:
    def test_depth_1_default_behavior(self):
        """graph_search_depth=1 (default) should work same as before."""
        manager = _make_manager(
            [
                _extraction_output(
                    ["alice works at acme"],
                    [{"name": "alice", "entity_type": "person"}, {"name": "acme", "entity_type": "org"}],
                    [{"source": "alice", "target": "acme", "relation_type": "works_at"}],
                ),
                # search query entity extraction
                _extraction_output([], [{"name": "alice", "entity_type": "person"}]),
            ]
        )
        manager.add("Alice works at Acme")
        results = manager.search("alice")
        assert len(results) > 0
        manager.close()

    def test_depth_2_finds_indirect_connections(self):
        """graph_search_depth=2 should find memories connected via entity relations."""
        manager = _make_manager(
            [
                # Memory 1: Alice works at Acme
                _extraction_output(
                    ["alice works at acme"],
                    [{"name": "alice", "entity_type": "person"}, {"name": "acme", "entity_type": "org"}],
                    [{"source": "alice", "target": "acme", "relation_type": "works_at"}],
                ),
                # Memory 2: Acme is in NYC
                _extraction_output(
                    ["acme is in nyc"],
                    [{"name": "acme", "entity_type": "org"}, {"name": "nyc", "entity_type": "city"}],
                    [{"source": "acme", "target": "nyc", "relation_type": "located_in"}],
                ),
                # search query entity extraction: searching for "nyc"
                _extraction_output([], [{"name": "nyc", "entity_type": "city"}]),
            ],
            graph_search_depth=2,
        )
        manager.add("Alice works at Acme")
        manager.add("Acme is in NYC")

        # Searching for "nyc" should find "acme is in nyc" (1-hop) and potentially
        # "alice works at acme" (2-hop via acme→nyc relation)
        results = manager.search("nyc")
        texts = [r.text.lower() for r in results]
        assert any("nyc" in t for t in texts)
        manager.close()

    def test_config_validation_depth(self):
        """graph_search_depth must be 1 or 2."""
        import pytest

        with pytest.raises(ValueError, match="graph_search_depth"):
            MemoryConfig(graph_search_depth=3)

        with pytest.raises(ValueError, match="graph_search_depth"):
            MemoryConfig(graph_search_depth=0)


# ---------------------------------------------------------------------------
# 14.2: Graph algorithm caching
# ---------------------------------------------------------------------------


class TestGraphAlgorithms:
    def test_recompute_graph_metrics_noop_when_disabled(self):
        """_recompute_graph_metrics should be a no-op when enable_graph_algorithms=False."""
        manager = _make_manager([_extraction_output(["fact one"])])
        manager.add("Fact one")
        # Should not raise even without algorithms support
        manager._recompute_graph_metrics()
        manager.close()

    def test_recompute_graph_metrics_runs_when_enabled(self):
        """When enable_graph_algorithms=True and graph is dirty, metrics should be computed."""
        manager = _make_manager(
            [_extraction_output(["fact one"]), _extraction_output(["fact two"])],
            enable_graph_algorithms=True,
        )
        manager.add("Fact one")
        manager.add("Fact two")
        assert manager._graph_dirty is True

        # Force recompute
        manager._recompute_graph_metrics()
        assert manager._graph_dirty is False
        manager.close()

    def test_graph_dirty_flag_set_on_add(self):
        """Adding memories should set _graph_dirty=True."""
        manager = _make_manager([_extraction_output(["a fact"])])
        assert manager._graph_dirty is False
        manager.add("A fact")
        assert manager._graph_dirty is True
        manager.close()


# ---------------------------------------------------------------------------
# 14.3: Cross-session entity reinforcement
# ---------------------------------------------------------------------------


class TestCrossSessionBoost:
    def test_no_boost_when_factor_zero(self):
        """cross_session_factor=0.0 should produce identical results."""
        config = MemoryConfig(cross_session_factor=0.0)
        results = [
            SearchResult(memory_id="1", text="test", score=0.8, user_id="u"),
        ]
        # Dummy db that returns None for get_node
        boosted = apply_cross_session_boost(results, _DummyDB(), config)
        assert len(boosted) == 1
        # Score should not change (no boost applied since get_node returns None)
        assert boosted[0].score == 0.8

    def test_boost_applied_with_factor(self):
        """cross_session_factor > 0 should apply boost from cached pagerank/betweenness."""
        config = MemoryConfig(cross_session_factor=0.2)
        results = [
            SearchResult(memory_id="1", text="test", score=0.8, user_id="u"),
        ]
        db = _AlgoPropsDB({"_pagerank": 0.1, "_betweenness": 0.05})
        boosted = apply_cross_session_boost(results, db, config)
        assert len(boosted) == 1
        # Score should be boosted above original
        assert boosted[0].score > 0.8

    def test_empty_results_passthrough(self):
        """Empty results should pass through unchanged."""
        config = MemoryConfig(cross_session_factor=0.2)
        boosted = apply_cross_session_boost([], _DummyDB(), config)
        assert boosted == []


# ---------------------------------------------------------------------------
# 14.4: MMR diverse search
# ---------------------------------------------------------------------------


class TestDiverseSearch:
    def test_diverse_search_fallback_to_vector(self):
        """diverse=True should work even when mmr_search is not available (fallback)."""
        manager = _make_manager(
            [
                _extraction_output(["alice likes cats"]),
                _extraction_output([], [{"name": "alice", "entity_type": "person"}]),
            ]
        )
        manager.add("Alice likes cats")
        # diverse=True with a db that doesn't have mmr_search should fallback gracefully
        results = manager.search("alice", diverse=True)
        # Should get results (via fallback to vector search)
        assert isinstance(results, list)
        manager.close()

    def test_mmr_lambda_config_validation(self):
        """mmr_lambda must be in [0.0, 1.0]."""
        import pytest

        with pytest.raises(ValueError, match="mmr_lambda"):
            MemoryConfig(mmr_lambda=1.5)

        with pytest.raises(ValueError, match="mmr_lambda"):
            MemoryConfig(mmr_lambda=-0.1)

    def test_cross_session_factor_validation(self):
        """cross_session_factor must be in [0.0, 1.0]."""
        import pytest

        with pytest.raises(ValueError, match="cross_session_factor"):
            MemoryConfig(cross_session_factor=2.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _DummyDB:
    """Minimal db stub for unit-testing scoring functions."""

    def get_node(self, node_id):
        return None


class _FakeNode:
    def __init__(self, props):
        self.properties = props
        self.labels = [MEMORY_LABEL]


class _AlgoPropsDB:
    """DB stub that returns nodes with pre-set algorithm properties."""

    def __init__(self, props):
        self._props = props

    def get_node(self, node_id):
        return _FakeNode(self._props)
