"""Vector similarity search against GrafeoDB."""

from __future__ import annotations

import json
import logging
from typing import TYPE_CHECKING

from grafeo_memory.common.types import MEMORY_LABEL, SearchResult

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from grafeo_memory.core.protocol import GrafeoDBProtocol
    from grafeo_memory.providers.embedding import EmbeddingClient


def vector_search(
    db: GrafeoDBProtocol,
    embedder: EmbeddingClient,
    query: str,
    *,
    user_id: str,
    k: int = 10,
    filters: dict | None = None,
    vector_property: str = "embedding",
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Search memories by vector similarity.

    Embeds the query, runs cosine similarity search, and converts
    distances to similarity scores (higher = more relevant).
    """
    if query_embedding is None:
        query_embedding = embedder.embed([query])[0]

    search_filters = {"user_id": user_id}
    if filters:
        search_filters.update(filters)
    engine_filters, fetch_k = _engine_filters(search_filters, k)

    try:
        results = db.vector_search(MEMORY_LABEL, vector_property, query_embedding, fetch_k, filters=engine_filters)
    except RuntimeError:
        logger.warning("vector_search failed for label=%s", MEMORY_LABEL, exc_info=True)
        results = []

    search_results: list[SearchResult] = []
    for node_id, distance in results:
        if len(search_results) >= k:
            break
        node = db.get_node(node_id)
        if node is None:
            continue
        props = _get_props(node)
        # Skip expired memories (soft-deleted)
        if props.get("expired_at") is not None:
            continue
        if engine_filters is not search_filters and not _matches_filters(props, search_filters):
            continue
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
                source="vector",
                created_at=props.get("created_at"),
                learned_at=props.get("learned_at"),
                session_id=props.get("session_id"),
                expired_at=props.get("expired_at"),
                valid_at=props.get("valid_at"),
                invalid_at=props.get("invalid_at"),
            )
        )

    return search_results


def hybrid_search(
    db: GrafeoDBProtocol,
    embedder: EmbeddingClient,
    query: str,
    *,
    user_id: str,
    k: int = 10,
    filters: dict | None = None,
    vector_property: str = "embedding",
    text_property: str = "text",
    fusion: str | None = None,
    rrf_k: int = 1,
    query_embedding: list[float] | None = None,
) -> list[SearchResult]:
    """Search memories using hybrid BM25 + vector search with RRF fusion.

    Falls back to vector-only search if hybrid_search is not available on the db.
    The scope filters (``user_id`` and ``filters``) go to the engine, which keeps
    only the matching nodes before it fuses the text and vector results (grafeo
    0.6), so a user in a shared database gets up to ``k`` of their own memories.
    They are checked again on the returned nodes, with the expiry check.

    Default fusion is RRF with k=1, tuned for small memory collections
    (10-100 memories) where score differentiation matters.
    """
    if query_embedding is None:
        query_embedding = embedder.embed([query])[0]

    if not hasattr(db, "hybrid_search"):
        return vector_search(
            db,
            embedder,
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            vector_property=vector_property,
            query_embedding=query_embedding,
        )

    # Scope filters: applied by the engine, and checked again on the returned nodes.
    all_filters: dict = {"user_id": user_id}
    if filters:
        all_filters.update(filters)

    engine_filters, fetch_k = _engine_filters(all_filters, k)
    try:
        results = db.hybrid_search(  # ty: ignore[call-non-callable]
            MEMORY_LABEL,
            text_property,
            vector_property,
            query,
            fetch_k,
            query_vector=query_embedding,
            fusion=fusion or "rrf",
            rrf_k=rrf_k,
            filters=engine_filters,
        )
    except Exception:
        logger.warning("hybrid_search failed, falling back to vector_search", exc_info=True)
        return vector_search(
            db,
            embedder,
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            vector_property=vector_property,
            query_embedding=query_embedding,
        )

    # Re-rank by cosine similarity: hybrid finds candidates, vector ranks them.
    # This eliminates BM25 noise from short memories with repeated entity names.
    from grafeo_memory.core.search.graph import _cosine_similarity

    search_results: list[SearchResult] = []
    for node_id, _fused_score in results:
        if len(search_results) >= k:
            break
        node = db.get_node(node_id)
        if node is None:
            continue
        props = _get_props(node)
        # Skip expired memories (soft-deleted)
        if props.get("expired_at") is not None:
            continue
        # Check the scope filters again (the engine applied them; a db passed in with db= may not)
        if not _matches_filters(props, all_filters):
            continue
        relations = _get_node_relations(db, node_id)
        # Use cosine similarity against the stored embedding for ranking
        stored_emb = props.get(vector_property)
        if stored_emb is not None:
            score = max(0.0, _cosine_similarity(query_embedding, stored_emb))
        else:
            score = max(0.0, 1.0 - _fused_score) if _fused_score < 1.0 else 0.5
        search_results.append(
            SearchResult(
                memory_id=str(node_id),
                text=props.get("text", ""),
                score=score,
                user_id=user_id,
                metadata=_parse_metadata(props.get("metadata")),
                relations=relations if relations else None,
                actor_id=props.get("actor_id"),
                role=props.get("role"),
                memory_type=props.get("memory_type", "semantic"),
                source="vector",
                created_at=props.get("created_at"),
                learned_at=props.get("learned_at"),
                session_id=props.get("session_id"),
                expired_at=props.get("expired_at"),
                valid_at=props.get("valid_at"),
                invalid_at=props.get("invalid_at"),
            )
        )

    search_results.sort(key=lambda r: r.score, reverse=True)
    return search_results


def diverse_search(
    db: GrafeoDBProtocol,
    embedder: EmbeddingClient,
    query: str,
    *,
    user_id: str,
    k: int = 10,
    filters: dict | None = None,
    vector_property: str = "embedding",
    query_embedding: list[float] | None = None,
    lambda_mult: float = 0.5,
) -> list[SearchResult]:
    """Search memories using MMR (Maximal Marginal Relevance) for diversity.

    Balances relevance with diversity: lambda_mult=1.0 is pure relevance,
    lambda_mult=0.0 is pure diversity. Default 0.5 is balanced.
    Falls back to vector_search if mmr_search is not available.
    """
    if query_embedding is None:
        query_embedding = embedder.embed([query])[0]

    search_filters = {"user_id": user_id}
    if filters:
        search_filters.update(filters)

    if not hasattr(db, "mmr_search"):
        return vector_search(
            db,
            embedder,
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            vector_property=vector_property,
            query_embedding=query_embedding,
        )

    engine_filters, fetch_k = _engine_filters(search_filters, k)
    try:
        results = db.mmr_search(  # ty: ignore[call-non-callable]
            MEMORY_LABEL,
            vector_property,
            query_embedding,
            fetch_k,
            fetch_k=fetch_k * 4,
            lambda_mult=lambda_mult,
            filters=engine_filters,
        )
    except Exception:
        logger.warning("mmr_search failed, falling back to vector_search", exc_info=True)
        return vector_search(
            db,
            embedder,
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            vector_property=vector_property,
            query_embedding=query_embedding,
        )

    search_results: list[SearchResult] = []
    for node_id, distance in results:
        if len(search_results) >= k:
            break
        node = db.get_node(node_id)
        if node is None:
            continue
        props = _get_props(node)
        if props.get("expired_at") is not None:
            continue
        # Post-hoc scope filtering
        all_filters: dict = {"user_id": user_id}
        if filters:
            all_filters.update(filters)
        if not _matches_filters(props, all_filters):
            continue
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
                actor_id=props.get("actor_id"),
                role=props.get("role"),
                memory_type=props.get("memory_type", "semantic"),
                source="vector",
                created_at=props.get("created_at"),
                learned_at=props.get("learned_at"),
                session_id=props.get("session_id"),
                expired_at=props.get("expired_at"),
                valid_at=props.get("valid_at"),
                invalid_at=props.get("invalid_at"),
            )
        )

    return search_results


def search_similar(
    db: GrafeoDBProtocol,
    embeddings: list[list[float]],
    *,
    user_id: str,
    threshold: float = 0.3,
    vector_property: str = "embedding",
    filters: dict | None = None,
) -> list[dict]:
    """Search for existing memories similar to any of the given embeddings.

    Returns a deduplicated list of dicts: {id, text, score}.
    Used during add() to find candidates for reconciliation.

    Args:
        threshold: Minimum similarity score (0.0-1.0). Candidates below
            this similarity are excluded from reconciliation.
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
            similarity = max(0.0, 1.0 - float(distance))
            if similarity < threshold:
                continue  # below minimum similarity
            seen.add(node_id)

            node = db.get_node(node_id)
            if node is None:
                continue
            props = _get_props(node)
            results.append(
                {
                    "id": str(node_id),
                    "text": props.get("text", ""),
                    "score": similarity,
                }
            )

    logger.debug("search_similar: returning %d total results", len(results))
    return results


def _engine_filters(filters: dict, k: int) -> tuple[dict, int]:
    """The filters the engine applies, and how many hits to ask it for.

    ``memory_type="semantic"`` stays a post-hoc check: a memory written before
    memory_type existed has no such property and counts as semantic
    (``_matches_filters``), while the engine's equality filter would drop it. The
    engine is then asked for twice ``k`` hits so the post-hoc check still fills ``k``.
    Any other filter set goes to the engine unchanged (the same dict).
    """
    if filters.get("memory_type") != "semantic":
        return filters, k
    return {key: v for key, v in filters.items() if key != "memory_type"}, k * 2


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


def _get_node_relations(db: GrafeoDBProtocol, node_id: int) -> list[dict]:
    """Get relations for a memory node by traversing HAS_ENTITY and RELATION edges."""
    from grafeo_memory.common.types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL, RELATION_EDGE

    relations: list[dict] = []
    try:
        query = (
            f"MATCH (m:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL})"
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
            return p()  # ty: ignore[invalid-return-type]
        return p  # ty: ignore[invalid-return-type]
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
