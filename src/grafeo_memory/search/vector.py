"""Vector similarity search against GrafeoDB."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from ..types import MEMORY_LABEL, SearchResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..embedding import EmbeddingClient


def vector_search(
    db: object,
    embedder: EmbeddingClient,
    query: str,
    *,
    user_id: str,
    k: int = 10,
    filters: dict | None = None,
    vector_property: str = "embedding",
) -> list[SearchResult]:
    """Search memories by vector similarity.

    Embeds the query, runs cosine similarity search, and converts
    distances to similarity scores (higher = more relevant).
    """
    query_embedding = embedder.embed([query])[0]

    search_filters = {"user_id": user_id}
    if filters:
        search_filters.update(filters)

    try:
        results = db.vector_search(MEMORY_LABEL, vector_property, query_embedding, k, filters=search_filters)
    except RuntimeError:
        logger.warning("vector_search failed for label=%s", MEMORY_LABEL, exc_info=True)
        results = []

    search_results: list[SearchResult] = []
    for node_id, distance in results:
        node = db.get_node(node_id)
        if node is None:
            continue
        props = _get_props(node)
        relations = _get_node_relations(db, node_id)
        similarity = max(0.0, 1.0 - float(distance))
        search_results.append(
            SearchResult(
                memory_id=str(node_id),
                text=props.get("text", ""),
                score=similarity,
                user_id=user_id,
                metadata=_parse_metadata(props.get("metadata")),
                relations=relations if relations else None,
                memory_type=props.get("memory_type", "semantic"),
            )
        )

    return search_results


def hybrid_search(
    db: object,
    embedder: EmbeddingClient,
    query: str,
    *,
    user_id: str,
    k: int = 10,
    filters: dict | None = None,
    vector_property: str = "embedding",
    text_property: str = "text",
    fusion: str | None = None,
) -> list[SearchResult]:
    """Search memories using hybrid BM25 + vector search with RRF fusion.

    Falls back to vector-only search if hybrid_search is not available on the db.
    Filters are applied post-hoc on node properties since the db hybrid_search
    may not support native filtering.

    Default fusion is RRF with k=1, tuned for small memory collections
    (10-100 memories) where score differentiation matters.
    """
    if not hasattr(db, "hybrid_search"):
        return vector_search(
            db, embedder, query, user_id=user_id, k=k, filters=filters, vector_property=vector_property
        )

    query_embedding = embedder.embed([query])[0]

    # Build combined filters for post-hoc application
    all_filters: dict = {"user_id": user_id}
    if filters:
        all_filters.update(filters)

    try:
        results = db.hybrid_search(
            MEMORY_LABEL,
            text_property,
            vector_property,
            query,
            k,
            query_vector=query_embedding,
            fusion=fusion or "rrf",
            rrf_k=1,
        )
    except Exception:
        logger.warning("hybrid_search failed, falling back to vector_search", exc_info=True)
        return vector_search(
            db, embedder, query, user_id=user_id, k=k, filters=filters, vector_property=vector_property
        )

    search_results: list[SearchResult] = []
    for node_id, score in results:
        node = db.get_node(node_id)
        if node is None:
            continue
        props = _get_props(node)
        # Apply all scope filters post-hoc (supports operator-based filters)
        if not _matches_filters(props, all_filters):
            continue
        relations = _get_node_relations(db, node_id)
        search_results.append(
            SearchResult(
                memory_id=str(node_id),
                text=props.get("text", ""),
                score=float(score),
                user_id=user_id,
                metadata=_parse_metadata(props.get("metadata")),
                relations=relations if relations else None,
                actor_id=props.get("actor_id"),
                role=props.get("role"),
                memory_type=props.get("memory_type", "semantic"),
            )
        )

    return search_results


def search_similar(
    db: object,
    embeddings: list[list[float]],
    *,
    user_id: str,
    threshold: float = 0.7,
    vector_property: str = "embedding",
    filters: dict | None = None,
) -> list[dict]:
    """Search for existing memories similar to any of the given embeddings.

    Returns a deduplicated list of dicts: {id, text, score}.
    Used during add() to find candidates for reconciliation.
    """
    seen: set[int] = set()
    results: list[dict] = []

    search_filters: dict = {"user_id": user_id}
    if filters:
        search_filters.update(filters)

    for emb in embeddings:
        try:
            hits = db.vector_search(MEMORY_LABEL, vector_property, emb, 10, filters=search_filters)
        except Exception:
            logger.warning("search_similar: vector_search failed for embedding", exc_info=True)
            continue

        logger.debug("search_similar: got %d hits for embedding", len(hits))
        for node_id, distance in hits:
            if node_id in seen:
                continue
            if float(distance) > threshold:
                continue  # distance — lower is more similar
            seen.add(node_id)

            node = db.get_node(node_id)
            if node is None:
                continue
            props = _get_props(node)
            similarity = max(0.0, 1.0 - float(distance))
            results.append(
                {
                    "id": str(node_id),
                    "text": props.get("text", ""),
                    "score": similarity,
                }
            )

    logger.debug("search_similar: returning %d total results", len(results))
    return results


def _matches_filters(props: dict, filters: dict) -> bool:
    """Check if node properties match all filters, including operator-based filters.

    Supports equality (scalar values) and operators: $gt, $gte, $lt, $lte, $ne, $in, $nin, $contains.
    """
    for key, expected in filters.items():
        actual = props.get(key)
        # Treat missing memory_type as "semantic" for backward compat with pre-v0.6 nodes
        if actual is None and key == "memory_type" and expected == "semantic":
            continue
        if isinstance(expected, dict):
            for op, val in expected.items():
                if op == "$gt" and not (actual is not None and actual > val):
                    return False
                if op == "$gte" and not (actual is not None and actual >= val):
                    return False
                if op == "$lt" and not (actual is not None and actual < val):
                    return False
                if op == "$lte" and not (actual is not None and actual <= val):
                    return False
                if op == "$ne" and actual == val:
                    return False
                if op == "$in" and actual not in val:
                    return False
                if op == "$nin" and actual in val:
                    return False
                if op == "$contains" and (actual is None or val not in str(actual)):
                    return False
        elif actual != expected:
            return False
    return True


def _get_node_relations(db: object, node_id: int) -> list[dict]:
    """Get relations for a memory node by traversing HAS_ENTITY and RELATION edges."""
    from ..types import ENTITY_LABEL, HAS_ENTITY_EDGE, RELATION_EDGE

    relations: list[dict] = []
    try:
        query = (
            f"MATCH (m)-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL})"
            f"-[r:{RELATION_EDGE}]->(t:{ENTITY_LABEL}) "
            f"WHERE id(m) = $nid RETURN e.name, r.relation_type, t.name"
        )
        result = db.execute(query, {"nid": node_id})
        for row in result:
            if isinstance(row, dict):
                vals = list(row.values())
                if len(vals) >= 3:
                    relations.append(
                        {
                            "source": vals[0],
                            "relation": vals[1],
                            "target": vals[2],
                        }
                    )
    except Exception:
        logger.warning("_get_node_relations failed for node %s", node_id, exc_info=True)
    return relations


def _get_props(node: object) -> dict:
    """Extract properties dict from a Grafeo Node object."""
    if hasattr(node, "properties"):
        p = node.properties
        if callable(p):
            return p()
        return p
    return {}


def _parse_metadata(value: object) -> dict | None:
    """Parse a JSON-encoded metadata string back to a dict."""
    if value is None:
        return None
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return None
    return None
