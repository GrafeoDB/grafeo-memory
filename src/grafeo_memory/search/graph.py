"""Entity-based graph traversal search."""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable
from typing import TYPE_CHECKING

from ..extraction import extract_entities
from ..types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL, RELATION_EDGE, Fact, ModelType, SearchResult
from .vector import _get_node_relations, _get_props

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

    from ..embedding import EmbeddingClient
    from ..protocol import GrafeoDBProtocol


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def graph_search(
    db: GrafeoDBProtocol,
    model: ModelType,
    query: str,
    *,
    user_id: str,
    embedder: EmbeddingClient | None = None,
    k: int = 20,
    vector_property: str = "embedding",
    _on_usage: Callable[[str, RunUsage], None] | None = None,
    _entities: list | None = None,
    query_embedding: list[float] | None = None,
    search_depth: int = 1,
) -> list[SearchResult]:
    """Search by extracting entities from the query and finding linked memories.

    Traverses Entity nodes reachable via HAS_ENTITY edges back to Memory nodes.
    When an embedder is provided, scores results by cosine similarity against
    the query embedding. Otherwise falls back to a low fixed score (0.3).
    Makes 1 LLM call (entity extraction from query) unless _entities is provided.

    _entities: Pre-extracted entities to use instead of calling extract_entities().
               Pass this when calling from an async context to avoid nested run_sync().
    query_embedding: Pre-computed query embedding to avoid redundant embed() calls.
    """
    if _entities is not None:
        entities = _entities
    else:
        try:
            entities, _ = extract_entities(model, [Fact(text=query)], user_id, _on_usage=_on_usage)
        except Exception:
            logger.warning("graph_search: entity extraction failed for query=%r", query, exc_info=True)
            return []

    if not entities:
        return []

    # Compute query embedding for scoring (skip if pre-computed)
    if query_embedding is None and embedder is not None:
        try:
            query_embedding = embedder.embed([query])[0]
        except Exception:
            logger.warning("graph_search: failed to embed query", exc_info=True)

    results: list[SearchResult] = []
    seen_memory_ids: set[str] = set()

    for entity in entities:
        # Find Entity nodes by name, scoped to the Entity label and user.
        entity_nids: list[int] = []
        try:
            rows = db.execute(
                f"MATCH (e:{ENTITY_LABEL}) WHERE e.name = $name AND e.user_id = $uid RETURN id(e)",
                {"name": entity.name, "uid": user_id},
            )
            entity_nids = [int(next(iter(r.values()))) for r in rows if isinstance(r, dict) and r]
        except Exception:
            logger.warning("graph_search: entity lookup failed for %r", entity.name, exc_info=True)

        # Case-insensitive fallback
        if not entity_nids:
            with contextlib.suppress(Exception):
                rows = db.execute(
                    f"MATCH (e:{ENTITY_LABEL}) WHERE toLower(e.name) = $name AND e.user_id = $uid RETURN id(e)",
                    {"name": entity.name.lower(), "uid": user_id},
                )
                entity_nids = [int(next(iter(r.values()))) for r in rows if isinstance(r, dict) and r]

        for entity_nid in entity_nids:
            # Traverse HAS_ENTITY edges back to Memory nodes
            try:
                query_str = (
                    f"MATCH (m:{MEMORY_LABEL})"
                    f"-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) "
                    f"WHERE id(e) = $eid AND m.user_id = $uid "
                    f"RETURN id(m), m.text"
                )
                result = db.execute(query_str, {"eid": entity_nid, "uid": user_id})
                for row in result:
                    if not isinstance(row, dict):
                        continue
                    vals = list(row.values())
                    if len(vals) < 2:
                        continue
                    mem_id = str(vals[0])
                    if mem_id in seen_memory_ids:
                        continue
                    seen_memory_ids.add(mem_id)
                    mem_text = str(vals[1]) if vals[1] else ""
                    relations = _get_node_relations(db, int(vals[0]))
                    mem_node = db.get_node(int(vals[0]))
                    mem_props = _get_props(mem_node) if mem_node else {}

                    # Skip expired memories (soft-deleted)
                    if mem_props.get("expired_at") is not None:
                        continue

                    # Compute score from embedding similarity if available
                    score = 0.3  # fallback: low score so vector results rank higher
                    if query_embedding is not None:
                        mem_embedding = mem_props.get(vector_property)
                        if mem_embedding is not None and isinstance(mem_embedding, (list, tuple)):
                            score = max(0.0, _cosine_similarity(query_embedding, mem_embedding))

                    results.append(
                        SearchResult(
                            memory_id=mem_id,
                            text=mem_text,
                            score=score,
                            user_id=user_id,
                            relations=relations if relations else None,
                            memory_type=mem_props.get("memory_type", "semantic"),
                            source="graph",
                            created_at=mem_props.get("created_at"),
                            learned_at=mem_props.get("learned_at"),
                            session_id=mem_props.get("session_id"),
                            expired_at=mem_props.get("expired_at"),
                        )
                    )
            except Exception:
                logger.warning("graph_search: traversal failed for entity_nid=%s", entity_nid, exc_info=True)
                continue

    # 2-hop traversal: Entity → RELATION → Entity → Memory (finds indirect connections)
    if search_depth >= 2 and entities:
        entity_names = [e.name for e in entities]
        try:
            two_hop_query = (
                f"MATCH (e1:{ENTITY_LABEL})-[:{RELATION_EDGE}]->(e2:{ENTITY_LABEL})"
                f"<-[:{HAS_ENTITY_EDGE}]-(m:{MEMORY_LABEL}) "
                f"WHERE e1.name IN $names AND e1.user_id = $uid AND m.user_id = $uid "
                f"RETURN DISTINCT id(m), m.text"
            )
            two_hop_result = db.execute(two_hop_query, {"names": entity_names, "uid": user_id})
            for row in two_hop_result:
                if not isinstance(row, dict):
                    continue
                vals = list(row.values())
                if len(vals) < 2:
                    continue
                mem_id = str(vals[0])
                if mem_id in seen_memory_ids:
                    continue
                seen_memory_ids.add(mem_id)
                mem_node = db.get_node(int(vals[0]))
                mem_props = _get_props(mem_node) if mem_node else {}
                if mem_props.get("expired_at") is not None:
                    continue
                mem_text = str(vals[1]) if vals[1] else ""
                # 2-hop results get a discount (0.7x) to prefer direct matches
                score = 0.2
                if query_embedding is not None:
                    mem_embedding = mem_props.get(vector_property)
                    if mem_embedding is not None and isinstance(mem_embedding, (list, tuple)):
                        score = max(0.0, _cosine_similarity(query_embedding, mem_embedding) * 0.7)
                relations = _get_node_relations(db, int(vals[0]))
                results.append(
                    SearchResult(
                        memory_id=mem_id,
                        text=mem_text,
                        score=score,
                        user_id=user_id,
                        relations=relations if relations else None,
                        memory_type=mem_props.get("memory_type", "semantic"),
                        source="graph",
                        created_at=mem_props.get("created_at"),
                        learned_at=mem_props.get("learned_at"),
                        session_id=mem_props.get("session_id"),
                        expired_at=mem_props.get("expired_at"),
                    )
                )
        except Exception:
            logger.warning("graph_search: 2-hop traversal failed", exc_info=True)

    # Sort by score descending and limit to k results
    results.sort(key=lambda r: r.score, reverse=True)
    return results[:k]
