"""Tests for grafeo_memory.scoring — pure scoring functions."""

from __future__ import annotations

import time

import grafeo

from grafeo_memory.scoring import (
    _frequency_score,
    _recency_score,
    apply_importance_scoring,
    compute_composite_score,
)
from grafeo_memory.types import MEMORY_LABEL, MemoryConfig, SearchResult


class TestRecencyScore:
    def test_brand_new_memory(self):
        """A just-created memory should have recency ~1.0."""
        now_ms = int(time.time() * 1000)
        score = _recency_score(now_ms, decay_rate=0.1)
        assert score > 0.99

    def test_one_day_old(self):
        """One day old with decay_rate=0.1 -> ~0.905."""
        now_ms = int(time.time() * 1000)
        one_day_ago = now_ms - (24 * 60 * 60 * 1000)
        score = _recency_score(one_day_ago, decay_rate=0.1)
        assert 0.89 < score < 0.92

    def test_ten_days_old(self):
        """Ten days old with decay_rate=0.1 -> ~0.368."""
        now_ms = int(time.time() * 1000)
        ten_days_ago = now_ms - (10 * 24 * 60 * 60 * 1000)
        score = _recency_score(ten_days_ago, decay_rate=0.1)
        assert 0.35 < score < 0.40

    def test_very_old(self):
        """100 days old -> near 0."""
        now_ms = int(time.time() * 1000)
        old = now_ms - (100 * 24 * 60 * 60 * 1000)
        score = _recency_score(old, decay_rate=0.1)
        assert score < 0.001

    def test_zero_timestamp(self):
        """created_at=0 -> 0.0."""
        assert _recency_score(0, decay_rate=0.1) == 0.0

    def test_custom_decay_rate(self):
        """Faster decay rate produces lower score for same age."""
        now_ms = int(time.time() * 1000)
        one_day_ago = now_ms - (24 * 60 * 60 * 1000)
        slow = _recency_score(one_day_ago, decay_rate=0.1)
        fast = _recency_score(one_day_ago, decay_rate=0.5)
        assert fast < slow


class TestFrequencyScore:
    def test_zero_accesses(self):
        assert _frequency_score(0) == 0.0

    def test_one_access(self):
        score = _frequency_score(1)
        assert 0.1 < score < 0.2

    def test_ten_accesses(self):
        score = _frequency_score(10)
        assert 0.4 < score < 0.6

    def test_hundred_accesses(self):
        score = _frequency_score(100)
        assert score == 1.0

    def test_above_soft_cap(self):
        """Above soft cap should be capped at 1.0."""
        assert _frequency_score(200) == 1.0


class TestCompositeScore:
    def test_all_maximum(self):
        """All factors at max with default weights -> 1.0."""
        config = MemoryConfig(enable_importance=True)
        now_ms = int(time.time() * 1000)
        score = compute_composite_score(
            similarity=1.0,
            created_at=now_ms,
            access_count=100,
            importance=1.0,
            config=config,
        )
        # recency ~1.0, frequency = 1.0, so should be very close to 1.0
        assert score > 0.98

    def test_custom_weights(self):
        """Custom weights produce expected result."""
        config = MemoryConfig(
            enable_importance=True,
            weight_similarity=1.0,
            weight_recency=0.0,
            weight_frequency=0.0,
            weight_importance=0.0,
        )
        score = compute_composite_score(
            similarity=0.75,
            created_at=0,
            access_count=0,
            importance=0.0,
            config=config,
        )
        assert abs(score - 0.75) < 0.001

    def test_similarity_dominates(self):
        """With default weights, similarity has the highest weight (0.4)."""
        config = MemoryConfig(enable_importance=True)
        now_ms = int(time.time() * 1000)
        high_sim = compute_composite_score(1.0, now_ms, 0, 0.5, config)
        low_sim = compute_composite_score(0.0, now_ms, 0, 0.5, config)
        assert high_sim > low_sim
        assert high_sim - low_sim > 0.3  # similarity weight = 0.4


class TestApplyImportanceScoring:
    def _make_db_with_memories(self, memories: list[dict]) -> tuple[object, list[SearchResult]]:
        """Create a GrafeoDB with Memory nodes and return (db, search_results)."""
        db = grafeo.GrafeoDB()
        results = []
        for m in memories:
            node = db.create_node(
                [MEMORY_LABEL],
                {
                    "text": m["text"],
                    "user_id": "test_user",
                    "created_at": m.get("created_at", int(time.time() * 1000)),
                    "importance": m.get("importance", 1.0),
                    "access_count": m.get("access_count", 0),
                },
            )
            node_id = node.id if hasattr(node, "id") else node
            results.append(
                SearchResult(
                    memory_id=str(node_id),
                    text=m["text"],
                    score=m.get("score", 0.8),
                    user_id="test_user",
                )
            )
        return db, results

    def test_empty_results(self):
        """Empty results -> empty list."""
        config = MemoryConfig(enable_importance=True)
        db = grafeo.GrafeoDB()
        assert apply_importance_scoring([], db, config) == []

    def test_rescores_and_reorders(self):
        """Results with different properties get reordered by composite score."""
        now_ms = int(time.time() * 1000)
        old_time = now_ms - (30 * 24 * 60 * 60 * 1000)  # 30 days ago

        db, results = self._make_db_with_memories(
            [
                {"text": "old fact", "created_at": old_time, "score": 0.9, "importance": 0.5, "access_count": 0},
                {"text": "new fact", "created_at": now_ms, "score": 0.7, "importance": 1.0, "access_count": 5},
            ]
        )

        config = MemoryConfig(enable_importance=True)
        scored = apply_importance_scoring(results, db, config)

        # New fact should rank higher due to recency + importance + frequency
        assert scored[0].text == "new fact"

    def test_updates_access_stats(self):
        """access_count should be incremented after scoring."""
        db, results = self._make_db_with_memories(
            [
                {"text": "fact one", "access_count": 3},
            ]
        )

        config = MemoryConfig(enable_importance=True)
        apply_importance_scoring(results, db, config)

        # Check that access_count was incremented
        from grafeo_memory.search.vector import _get_props

        node_id = int(results[0].memory_id)
        node = db.get_node(node_id)
        props = _get_props(node)
        assert int(props.get("access_count", 0)) == 4

    def test_populates_importance_fields(self):
        """Scored results should have importance and access_count populated."""
        db, results = self._make_db_with_memories(
            [
                {"text": "fact", "importance": 0.7, "access_count": 5},
            ]
        )

        config = MemoryConfig(enable_importance=True)
        scored = apply_importance_scoring(results, db, config)

        assert scored[0].importance == 0.7
        assert scored[0].access_count == 5

    def test_backward_compat_missing_props(self):
        """Nodes without importance/access_count should default to 1.0 and 0."""
        db = grafeo.GrafeoDB()
        node = db.create_node(
            [MEMORY_LABEL],
            {
                "text": "old memory",
                "user_id": "test_user",
                "created_at": int(time.time() * 1000),
            },
        )
        node_id = node.id if hasattr(node, "id") else node

        results = [
            SearchResult(
                memory_id=str(node_id),
                text="old memory",
                score=0.8,
                user_id="test_user",
            )
        ]

        config = MemoryConfig(enable_importance=True)
        scored = apply_importance_scoring(results, db, config)

        assert scored[0].importance == 1.0
        assert scored[0].access_count == 0
