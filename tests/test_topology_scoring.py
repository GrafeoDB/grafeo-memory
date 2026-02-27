"""Tests for topology-aware scoring and structural decay modulation.

Inspired by VimRAG (arXiv:2602.12735v1) — graph connectivity as a scoring signal.
"""

from __future__ import annotations

import time

import grafeo

from grafeo_memory.scoring import (
    _compute_reinforcement,
    _modulated_recency_score,
    _recency_score,
    _topology_score,
    apply_importance_scoring,
    compute_composite_score,
)
from grafeo_memory.types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL, MemoryConfig, SearchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph(
    memories: list[dict],
    entities: list[dict],
    edges: list[tuple[int, int]],
) -> tuple[object, list[str]]:
    """Create a GrafeoDB with Memory nodes, Entity nodes, and HAS_ENTITY edges.

    memories: [{"text": ..., "created_at": ..., "importance": ..., ...}]
    entities: [{"name": ..., "entity_type": ...}]
    edges: [(memory_index, entity_index)] -- indexes into the lists above

    Returns (db, memory_ids).
    """
    db = grafeo.GrafeoDB()
    now_ms = int(time.time() * 1000)
    mem_ids: list[str] = []
    entity_nodes: list[int] = []

    for m in memories:
        props = {
            "text": m["text"],
            "user_id": m.get("user_id", "test_user"),
            "created_at": m.get("created_at", now_ms),
            "importance": m.get("importance", 1.0),
            "access_count": m.get("access_count", 0),
        }
        node = db.create_node([MEMORY_LABEL], props)
        mem_ids.append(str(node.id))

    for e in entities:
        node = db.create_node(
            [ENTITY_LABEL],
            {"name": e["name"], "entity_type": e.get("entity_type", "thing"), "user_id": "test_user"},
        )
        entity_nodes.append(node.id)

    for mem_idx, ent_idx in edges:
        db.create_edge(int(mem_ids[mem_idx]), entity_nodes[ent_idx], HAS_ENTITY_EDGE)

    return db, mem_ids


# ---------------------------------------------------------------------------
# _topology_score unit tests
# ---------------------------------------------------------------------------


class TestTopologyScore:
    def test_no_entities(self):
        assert _topology_score(0, 0.0) == 0.0

    def test_one_entity_not_shared(self):
        score = _topology_score(1, 0.0)
        assert 0.0 < score < 0.5  # low but nonzero (degree only)

    def test_many_entities_all_shared(self):
        score = _topology_score(10, 1.0)
        assert score > 0.9

    def test_shared_ratio_boosts_score(self):
        not_shared = _topology_score(3, 0.0)
        all_shared = _topology_score(3, 1.0)
        assert all_shared > not_shared

    def test_more_entities_higher_score(self):
        few = _topology_score(1, 0.5)
        many = _topology_score(8, 0.5)
        assert many > few

    def test_score_bounded_zero_one(self):
        assert 0.0 <= _topology_score(0, 0.0) <= 1.0
        assert 0.0 <= _topology_score(100, 1.0) <= 1.0
        assert 0.0 <= _topology_score(1, 0.5) <= 1.0


# ---------------------------------------------------------------------------
# Structural reinforcement unit tests
# ---------------------------------------------------------------------------


class TestComputeReinforcement:
    def test_no_children(self):
        """Memory with no shared entities has zero reinforcement."""
        db, mem_ids = _make_graph(
            memories=[{"text": "lone fact"}],
            entities=[{"name": "alpha"}],
            edges=[(0, 0)],
        )
        reinf = _compute_reinforcement(db, int(mem_ids[0]), int(time.time() * 1000), gamma=0.3)
        assert reinf == 0.0

    def test_with_younger_child(self):
        """Memory linked to entities shared with a newer memory gets reinforcement."""
        now_ms = int(time.time() * 1000)
        old_time = now_ms - (10 * 24 * 60 * 60 * 1000)  # 10 days ago

        db, mem_ids = _make_graph(
            memories=[
                {"text": "original fact", "created_at": old_time, "importance": 0.8},
                {"text": "newer fact", "created_at": now_ms, "importance": 1.0},
            ],
            entities=[{"name": "shared_entity"}],
            edges=[(0, 0), (1, 0)],  # both memories link to same entity
        )

        reinf = _compute_reinforcement(db, int(mem_ids[0]), old_time, gamma=0.3)
        assert reinf > 0.0
        assert reinf <= 1.0

    def test_older_sibling_not_counted(self):
        """Memories created BEFORE this one are not children."""
        now_ms = int(time.time() * 1000)
        old_time = now_ms - (10 * 24 * 60 * 60 * 1000)

        db, mem_ids = _make_graph(
            memories=[
                {"text": "older fact", "created_at": old_time},
                {"text": "this fact", "created_at": now_ms},
            ],
            entities=[{"name": "shared_entity"}],
            edges=[(0, 0), (1, 0)],
        )

        # mem_ids[1] is the newer one; older fact was created before it → no children
        reinf = _compute_reinforcement(db, int(mem_ids[1]), now_ms, gamma=0.3)
        assert reinf == 0.0

    def test_gamma_zero_disables(self):
        """gamma=0 means no reinforcement regardless of graph structure."""
        now_ms = int(time.time() * 1000)
        old_time = now_ms - (10 * 24 * 60 * 60 * 1000)

        db, mem_ids = _make_graph(
            memories=[
                {"text": "original", "created_at": old_time},
                {"text": "newer", "created_at": now_ms, "importance": 1.0},
            ],
            entities=[{"name": "shared"}],
            edges=[(0, 0), (1, 0)],
        )

        reinf = _compute_reinforcement(db, int(mem_ids[0]), old_time, gamma=0.0)
        assert reinf == 0.0


# ---------------------------------------------------------------------------
# Modulated recency score
# ---------------------------------------------------------------------------


class TestModulatedRecencyScore:
    def test_zero_reinforcement_matches_base(self):
        """With reinforcement=0, result matches base _recency_score."""
        now_ms = int(time.time() * 1000)
        one_day_ago = now_ms - (24 * 60 * 60 * 1000)
        base = _recency_score(one_day_ago, decay_rate=0.1)
        modulated = _modulated_recency_score(one_day_ago, decay_rate=0.1, reinforcement=0.0)
        assert abs(base - modulated) < 1e-10

    def test_reinforcement_slows_decay(self):
        """Higher reinforcement should produce higher recency score (slower decay)."""
        now_ms = int(time.time() * 1000)
        ten_days_ago = now_ms - (10 * 24 * 60 * 60 * 1000)
        normal = _modulated_recency_score(ten_days_ago, decay_rate=0.1, reinforcement=0.0)
        reinforced = _modulated_recency_score(ten_days_ago, decay_rate=0.1, reinforcement=1.0)
        assert reinforced > normal

    def test_max_reinforcement_halves_decay(self):
        """reinforcement=1.0 should halve the decay rate."""
        now_ms = int(time.time() * 1000)
        ten_days_ago = now_ms - (10 * 24 * 60 * 60 * 1000)
        normal = _recency_score(ten_days_ago, decay_rate=0.1)
        half_decay = _recency_score(ten_days_ago, decay_rate=0.05)
        modulated = _modulated_recency_score(ten_days_ago, decay_rate=0.1, reinforcement=1.0)
        assert abs(modulated - half_decay) < 1e-10
        assert modulated > normal

    def test_zero_timestamp(self):
        assert _modulated_recency_score(0, 0.1, 0.5) == 0.0


# ---------------------------------------------------------------------------
# composite_score with topology
# ---------------------------------------------------------------------------


class TestCompositeScoreWithTopology:
    def test_weight_topology_zero_no_effect(self):
        """weight_topology=0.0 makes topology irrelevant (backward compat)."""
        config = MemoryConfig(enable_importance=True, weight_topology=0.0)
        now_ms = int(time.time() * 1000)
        without = compute_composite_score(0.8, now_ms, 5, 0.5, config, topology=0.0)
        with_topo = compute_composite_score(0.8, now_ms, 5, 0.5, config, topology=1.0)
        assert abs(without - with_topo) < 1e-10

    def test_topology_adds_to_score(self):
        """Positive weight_topology should boost score when topology > 0."""
        config = MemoryConfig(enable_importance=True, weight_topology=0.2)
        now_ms = int(time.time() * 1000)
        without = compute_composite_score(0.8, now_ms, 5, 0.5, config, topology=0.0)
        with_topo = compute_composite_score(0.8, now_ms, 5, 0.5, config, topology=0.8)
        assert with_topo > without
        assert abs(with_topo - without - 0.2 * 0.8) < 1e-10

    def test_structural_decay_modulation(self):
        """enable_structural_decay with reinforcement should change recency score."""
        config = MemoryConfig(
            enable_importance=True,
            enable_structural_decay=True,
            structural_feedback_gamma=0.3,
        )
        now_ms = int(time.time() * 1000)
        ten_days_ago = now_ms - (10 * 24 * 60 * 60 * 1000)

        without_reinf = compute_composite_score(0.8, ten_days_ago, 5, 0.5, config, reinforcement=0.0)
        with_reinf = compute_composite_score(0.8, ten_days_ago, 5, 0.5, config, reinforcement=0.8)
        assert with_reinf > without_reinf


# ---------------------------------------------------------------------------
# Integration: apply_importance_scoring with topology
# ---------------------------------------------------------------------------


class TestApplyImportanceScoringWithTopology:
    def test_topology_boosts_connected_memory(self):
        """A well-connected memory should score higher than an isolated one."""
        now_ms = int(time.time() * 1000)

        # Memory 0: connected to 3 shared entities
        # Memory 1: connected to 0 entities (isolated)
        db, mem_ids = _make_graph(
            memories=[
                {"text": "connected fact", "created_at": now_ms, "importance": 0.5},
                {"text": "isolated fact", "created_at": now_ms, "importance": 0.5},
            ],
            entities=[
                {"name": "entity_a"},
                {"name": "entity_b"},
                {"name": "entity_c"},
            ],
            edges=[(0, 0), (0, 1), (0, 2)],  # only memory 0 has edges
        )

        results = [
            SearchResult(memory_id=mem_ids[0], text="connected fact", score=0.8, user_id="test_user"),
            SearchResult(memory_id=mem_ids[1], text="isolated fact", score=0.8, user_id="test_user"),
        ]

        config = MemoryConfig(enable_importance=True, weight_topology=0.2)
        scored = apply_importance_scoring(results, db, config)

        # Connected memory should rank first
        assert scored[0].text == "connected fact"
        assert scored[0].score > scored[1].score

    def test_topology_weight_zero_preserves_order(self):
        """weight_topology=0.0 should not change relative ordering from base scoring."""
        now_ms = int(time.time() * 1000)

        db, mem_ids = _make_graph(
            memories=[
                {"text": "fact a", "created_at": now_ms, "importance": 1.0},
                {"text": "fact b", "created_at": now_ms, "importance": 0.5},
            ],
            entities=[{"name": "entity_x"}],
            edges=[(0, 0), (1, 0)],
        )

        results = [
            SearchResult(memory_id=mem_ids[0], text="fact a", score=0.8, user_id="test_user"),
            SearchResult(memory_id=mem_ids[1], text="fact b", score=0.8, user_id="test_user"),
        ]

        config = MemoryConfig(enable_importance=True, weight_topology=0.0)
        scored = apply_importance_scoring(results, db, config)

        # fact a has higher importance → should still rank first
        assert scored[0].text == "fact a"

    def test_structural_decay_protects_foundational_memory(self):
        """An old memory with newer 'children' should decay slower than an isolated old memory."""
        now_ms = int(time.time() * 1000)
        old_time = now_ms - (15 * 24 * 60 * 60 * 1000)  # 15 days ago

        db, mem_ids = _make_graph(
            memories=[
                {"text": "foundational fact", "created_at": old_time, "importance": 0.8},
                {"text": "isolated old fact", "created_at": old_time, "importance": 0.8},
                {"text": "newer child fact", "created_at": now_ms, "importance": 1.0},
            ],
            entities=[
                {"name": "shared_entity"},
                {"name": "lone_entity"},
            ],
            # Memory 0 and 2 share an entity; Memory 1 has its own entity
            edges=[(0, 0), (2, 0), (1, 1)],
        )

        results = [
            SearchResult(memory_id=mem_ids[0], text="foundational fact", score=0.8, user_id="test_user"),
            SearchResult(memory_id=mem_ids[1], text="isolated old fact", score=0.8, user_id="test_user"),
        ]

        config = MemoryConfig(
            enable_importance=True,
            enable_structural_decay=True,
            structural_feedback_gamma=0.5,
        )
        scored = apply_importance_scoring(results, db, config)

        # Foundational memory should score higher due to structural reinforcement
        assert scored[0].text == "foundational fact"
        assert scored[0].score > scored[1].score
