"""Memory importance scoring and recency decay."""

from __future__ import annotations

import math
import time

from .types import MemoryConfig, SearchResult


def compute_composite_score(
    similarity: float,
    created_at: int,
    access_count: int,
    importance: float,
    config: MemoryConfig,
) -> float:
    """Compute a weighted composite score from multiple factors.

    All factor scores are in [0, 1]. The composite is a weighted sum
    of similarity, recency, frequency, and importance.
    """
    recency = _recency_score(created_at, config.decay_rate)
    frequency = _frequency_score(access_count)

    return (
        config.weight_similarity * similarity
        + config.weight_recency * recency
        + config.weight_frequency * frequency
        + config.weight_importance * importance
    )


def apply_importance_scoring(
    results: list[SearchResult],
    db: object,
    config: MemoryConfig,
) -> list[SearchResult]:
    """Re-score results with composite importance scoring and update access stats.

    Reads node properties (created_at, access_count, importance) from the db,
    computes composite scores, updates access stats, returns re-scored + sorted results.
    """
    from .search.vector import _get_props

    if not results:
        return results

    now_ms = int(time.time() * 1000)
    scored: list[SearchResult] = []

    for r in results:
        try:
            node_id = int(r.memory_id)
        except ValueError:
            scored.append(r)
            continue

        node = db.get_node(node_id)
        if node is None:
            scored.append(r)
            continue

        props = _get_props(node)
        created_at = int(props.get("created_at", 0))
        access_count = int(props.get("access_count", 0))
        importance = float(props.get("importance", 1.0))

        composite = compute_composite_score(r.score, created_at, access_count, importance, config)

        scored.append(
            SearchResult(
                memory_id=r.memory_id,
                text=r.text,
                score=composite,
                user_id=r.user_id,
                metadata=r.metadata,
                relations=r.relations,
                actor_id=r.actor_id,
                role=r.role,
                importance=importance,
                access_count=access_count,
                memory_type=r.memory_type,
            )
        )

        # Update access stats (best-effort)
        try:
            db.set_node_property(node_id, "access_count", access_count + 1)
            db.set_node_property(node_id, "last_accessed", now_ms)
        except Exception:
            pass

    scored.sort(key=lambda r: r.score, reverse=True)
    return scored


def _recency_score(created_at: int, decay_rate: float) -> float:
    """Exponential decay based on age in days. Returns value in [0, 1].

    A decay_rate of 0.1 means ~90% after 1 day, ~37% after 10 days.
    """
    if created_at <= 0:
        return 0.0
    now_ms = int(time.time() * 1000)
    age_ms = max(0, now_ms - created_at)
    age_days = age_ms / (1000 * 60 * 60 * 24)
    return math.exp(-decay_rate * age_days)


def _frequency_score(access_count: int) -> float:
    """Log-scaled frequency score in [0, 1] with soft cap at 100 accesses."""
    if access_count <= 0:
        return 0.0
    soft_cap = 100
    return min(1.0, math.log(1 + access_count) / math.log(1 + soft_cap))
