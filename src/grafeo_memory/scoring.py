"""Memory importance scoring, recency decay, and topology-aware ranking."""

from __future__ import annotations

import logging
import math
import time

from .types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL, MemoryConfig, SearchResult

logger = logging.getLogger(__name__)


def compute_composite_score(
    similarity: float,
    created_at: int,
    access_count: int,
    importance: float,
    config: MemoryConfig,
    *,
    topology: float = 0.0,
    reinforcement: float = 0.0,
) -> float:
    """Compute a weighted composite score from multiple factors.

    All factor scores are in [0, 1]. The composite is a weighted sum
    of similarity, recency, frequency, importance, and optionally topology.
    """
    if config.enable_structural_decay and reinforcement > 0:
        recency = _modulated_recency_score(created_at, config.decay_rate, reinforcement)
    else:
        recency = _recency_score(created_at, config.decay_rate)
    frequency = _frequency_score(access_count)

    return (
        config.weight_similarity * similarity
        + config.weight_recency * recency
        + config.weight_frequency * frequency
        + config.weight_importance * importance
        + config.weight_topology * topology
    )


def apply_importance_scoring(
    results: list[SearchResult],
    db: object,
    config: MemoryConfig,
) -> list[SearchResult]:
    """Re-score results with composite importance scoring and update access stats.

    Reads node properties (created_at, access_count, importance) from the db,
    computes composite scores, updates access stats, returns re-scored + sorted results.
    When weight_topology > 0, also computes graph-topology scores.
    When enable_structural_decay is True, modulates recency decay by structural reinforcement.
    """
    from .search.vector import _get_props

    if not results:
        return results

    need_topology = config.weight_topology > 0 or config.enable_structural_decay
    topo_cache: dict[str, tuple[float, float]] = {}  # memory_id -> (topology, reinforcement)

    if need_topology:
        topo_cache = _batch_topology_scores(db, [r.memory_id for r in results], config)

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

        topology, reinforcement = topo_cache.get(r.memory_id, (0.0, 0.0))

        composite = compute_composite_score(
            r.score,
            created_at,
            access_count,
            importance,
            config,
            topology=topology,
            reinforcement=reinforcement,
        )

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


def apply_topology_boost(
    results: list[SearchResult],
    db: object,
    config: MemoryConfig,
) -> list[SearchResult]:
    """Boost search result scores based on graph-topology connectivity.

    Multiplicative boost: score *= (1 + boost_factor * topology_score).
    Well-connected memories rise in rankings. No LLM call needed.
    """
    if not results or config.topology_boost_factor <= 0:
        return results

    topo_cache = _batch_topology_scores(db, [r.memory_id for r in results], config)

    boosted: list[SearchResult] = []
    for r in results:
        topo, _ = topo_cache.get(r.memory_id, (0.0, 0.0))
        boost = 1.0 + config.topology_boost_factor * topo
        boosted.append(
            SearchResult(
                memory_id=r.memory_id,
                text=r.text,
                score=r.score * boost,
                user_id=r.user_id,
                metadata=r.metadata,
                relations=r.relations,
                actor_id=r.actor_id,
                role=r.role,
                importance=r.importance,
                access_count=r.access_count,
                memory_type=r.memory_type,
            )
        )

    boosted.sort(key=lambda r: r.score, reverse=True)
    return boosted


# ---------------------------------------------------------------------------
# Topology scoring (VimRAG-inspired)
# ---------------------------------------------------------------------------


def _topology_score(entity_count: int, shared_entity_ratio: float) -> float:
    """Score in [0, 1] based on graph connectivity.

    entity_count: number of HAS_ENTITY edges from this Memory node.
    shared_entity_ratio: fraction of those entities that also link to other Memory nodes.
    """
    if entity_count <= 0:
        return 0.0
    # Degree component: log-scaled, soft cap at 10 entities
    degree = min(1.0, math.log(1 + entity_count) / math.log(1 + 10))
    # Connectivity component: ratio of shared entities
    connectivity = max(0.0, min(1.0, shared_entity_ratio))
    return 0.6 * degree + 0.4 * connectivity


def _compute_reinforcement(
    db: object,
    memory_id: int,
    created_at: int,
    gamma: float,
) -> float:
    """Compute structural reinforcement from 'child' memories.

    Children = memories sharing entities with this memory that were created AFTER it.
    Returns a value in [0, 1].
    """
    if gamma <= 0:
        return 0.0

    from .search.vector import _get_props

    try:
        # Find entities linked to this memory, then other memories linked to those entities
        query = (
            f"MATCH (m:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL})"
            f"<-[:{HAS_ENTITY_EDGE}]-(child:{MEMORY_LABEL}) "
            f"WHERE id(m) = $mid AND id(child) <> id(m) "
            f"RETURN DISTINCT id(child)"
        )
        rows = db.execute(query, {"mid": memory_id})
    except Exception:
        logger.debug("reinforcement: graph query failed for memory %s", memory_id, exc_info=True)
        return 0.0

    if not rows:
        return 0.0

    child_importances: list[float] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        vals = list(row.values())
        if not vals:
            continue
        child_id = int(vals[0])
        child_node = db.get_node(child_id)
        if child_node is None:
            continue
        child_props = _get_props(child_node)
        child_created = int(child_props.get("created_at", 0))
        # Only count memories created AFTER this one
        if child_created > created_at:
            child_importances.append(float(child_props.get("importance", 1.0)))

    if not child_importances:
        return 0.0

    avg_importance = sum(child_importances) / len(child_importances)
    # Clamp to [0, 1]
    return max(0.0, min(1.0, gamma * avg_importance))


def _batch_topology_scores(
    db: object,
    memory_ids: list[str],
    config: MemoryConfig,
) -> dict[str, tuple[float, float]]:
    """Compute topology score and reinforcement for a batch of memory IDs.

    Returns {memory_id: (topology_score, reinforcement)}.
    """
    from .search.vector import _get_props

    result: dict[str, tuple[float, float]] = {}
    for mid_str in memory_ids:
        try:
            mid = int(mid_str)
        except ValueError:
            result[mid_str] = (0.0, 0.0)
            continue

        # --- Entity count and shared ratio ---
        entity_count = 0
        shared_count = 0
        try:
            # Count entities linked to this memory
            q_entities = (
                f"MATCH (m:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) WHERE id(m) = $mid RETURN id(e)"
            )
            entity_rows = db.execute(q_entities, {"mid": mid})
            entity_ids = []
            if entity_rows:
                for row in entity_rows:
                    if isinstance(row, dict):
                        vals = list(row.values())
                        if vals:
                            entity_ids.append(int(vals[0]))
            entity_count = len(entity_ids)

            # Check which entities are shared with other memories
            if entity_ids:
                for eid in entity_ids:
                    try:
                        q_shared = (
                            f"MATCH (other:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) "
                            f"WHERE id(e) = $eid AND id(other) <> $mid "
                            f"RETURN id(other) LIMIT 1"
                        )
                        shared_rows = db.execute(q_shared, {"eid": eid, "mid": mid})
                        if shared_rows:
                            shared_count += 1
                    except Exception:
                        pass
        except Exception:
            logger.debug("topology: entity query failed for memory %s", mid_str, exc_info=True)

        shared_ratio = shared_count / entity_count if entity_count > 0 else 0.0
        topo = _topology_score(entity_count, shared_ratio)

        # --- Structural reinforcement ---
        reinf = 0.0
        if config.enable_structural_decay:
            node = db.get_node(mid)
            if node is not None:
                props = _get_props(node)
                created_at = int(props.get("created_at", 0))
                reinf = _compute_reinforcement(db, mid, created_at, config.structural_feedback_gamma)

        result[mid_str] = (topo, reinf)

    return result


# ---------------------------------------------------------------------------
# Core scoring factors
# ---------------------------------------------------------------------------


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


def _modulated_recency_score(created_at: int, decay_rate: float, reinforcement: float) -> float:
    """Recency with structural reinforcement slowing decay.

    reinforcement in [0, 1]. Higher reinforcement = slower decay (at most halved).
    Inspired by VimRAG Eq. 7: foundational memories resist temporal decay.
    """
    if created_at <= 0:
        return 0.0
    modulated_rate = decay_rate * (1.0 - 0.5 * max(0.0, min(1.0, reinforcement)))
    now_ms = int(time.time() * 1000)
    age_ms = max(0, now_ms - created_at)
    age_days = age_ms / (1000 * 60 * 60 * 24)
    return math.exp(-modulated_rate * age_days)


def _frequency_score(access_count: int) -> float:
    """Log-scaled frequency score in [0, 1] with soft cap at 100 accesses."""
    if access_count <= 0:
        return 0.0
    soft_cap = 100
    return min(1.0, math.log(1 + access_count) / math.log(1 + soft_cap))
