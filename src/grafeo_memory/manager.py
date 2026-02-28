"""MemoryManager — the public API for grafeo-memory."""

from __future__ import annotations

import json
import logging
import time

from ._compat import run_sync
from .embedding import EmbeddingClient
from .extraction import extract_async
from .history import HistoryEntry, get_history, record_history
from .messages import Message, parse_messages
from .reconciliation import reconcile_async, reconcile_relations_async
from .search import graph_search, hybrid_search, search_similar
from .search.vector import _get_props, _parse_metadata
from .types import (
    DERIVED_FROM_EDGE,
    ENTITY_LABEL,
    HAS_ENTITY_EDGE,
    MEMORY_LABEL,
    RELATION_EDGE,
    AddResult,
    Entity,
    ExtractionResult,
    MemoryAction,
    MemoryConfig,
    MemoryEvent,
    MemoryType,
    ReconciliationDecision,
    SearchResponse,
    SearchResult,
)

logger = logging.getLogger(__name__)


class _MemoryCore:
    """Shared initialization and async internal methods.

    Both MemoryManager (sync) and AsyncMemoryManager (async) inherit
    from this base. All orchestration logic lives here as async methods.
    """

    def __init__(
        self,
        model: object,
        config: MemoryConfig | None = None,
        *,
        embedder: EmbeddingClient,
        reranker: object | None = None,
    ):
        import grafeo

        self._config = config or MemoryConfig()
        self._model = model
        self._embedder = embedder
        self._reranker = reranker

        if self._config.db_path:
            self._db = grafeo.GrafeoDB(self._config.db_path)
        else:
            self._db = grafeo.GrafeoDB()

        self._vector_index_ready = False
        self._ensure_indexes()

        # OpenTelemetry: instrument all pydantic-ai agents when enabled
        if self._config.instrument:
            from pydantic_ai import Agent

            Agent.instrument_all(self._config.instrument if self._config.instrument is not True else True)

    def _ensure_indexes(self) -> None:
        vp = self._config.vector_property
        tp = self._config.text_property
        dims = self._config.embedding_dimensions

        try:
            dims = self._embedder.dimensions
        except Exception:
            logger.debug("Could not read embedder.dimensions, using config default %d", dims)

        self._embedding_dims = dims

        try:
            self._db.create_vector_index(MEMORY_LABEL, vp, dimensions=dims, metric="cosine")
            self._vector_index_ready = True
        except Exception:
            # Expected on first run — no nodes with embedding property yet.
            # Will retry after first memory is stored.
            logger.debug("Vector index deferred (no embedding data yet)")

        try:
            self._db.create_text_index(MEMORY_LABEL, tp)
        except Exception:
            logger.debug("Text index creation deferred or already exists")

    def _ensure_vector_index(self) -> None:
        """Lazily create the vector index after the first embedding is stored."""
        if self._vector_index_ready:
            return
        vp = self._config.vector_property
        try:
            self._db.create_vector_index(MEMORY_LABEL, vp, dimensions=self._embedding_dims, metric="cosine")
            self._vector_index_ready = True
            logger.debug("Vector index created successfully (deferred)")
        except Exception:
            logger.warning("Failed to create vector index (deferred)", exc_info=True)

    def close(self) -> None:
        """Close the database connection.

        The async runner is intentionally NOT closed here — it is shared across
        sessions and cleaned up automatically at process exit via atexit. Closing
        it per-session corrupts httpx/anyio transport state on Windows.
        """
        if hasattr(self._db, "close"):
            self._db.close()

    # --- Scope filter building ---

    def _build_filters(self, user_id: str, extra: dict | None = None) -> dict:
        """Build query filters from config scoping (user_id, agent_id, run_id)."""
        filters: dict = {"user_id": user_id}
        if self._config.agent_id:
            filters["agent_id"] = self._config.agent_id
        if self._config.run_id:
            filters["run_id"] = self._config.run_id
        if extra:
            filters.update(extra)
        return filters

    def _make_usage_collector(self):
        """Create a usage collector that accumulates RunUsage and optionally fires a user callback."""
        from pydantic_ai.usage import RunUsage

        total = RunUsage()
        cb = self._config.usage_callback

        def collector(operation, usage):
            total.incr(usage)
            if cb is not None:
                try:
                    cb(operation, usage)
                except Exception:
                    logger.warning("Usage callback failed for '%s'", operation, exc_info=True)

        return collector, total

    # --- Async internal orchestration ---

    async def _add(
        self,
        messages: str | dict | list[dict],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        on_usage, total = self._make_usage_collector()
        uid = user_id or self._config.user_id
        sid = session_id or self._config.session_id
        now_ms = int(time.time() * 1000)
        mtype = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type

        text, parsed, images = parse_messages(messages)
        actor_id, role = _extract_actor(parsed)

        # Vision: convert images to text descriptions via LLM
        if images and self._config.enable_vision:
            from .vision import describe_images_async

            vision_model = self._config.vision_model or self._model
            descriptions = await describe_images_async(vision_model, images, _on_usage=on_usage)
            image_text = "\n".join(f"[Image: {d}]" for d in descriptions if d)
            text = f"{text}\n{image_text}" if text else image_text

        if not infer:
            return AddResult(
                self._raw_add(
                    text,
                    uid,
                    sid,
                    metadata,
                    now_ms,
                    actor_id,
                    role,
                    importance=importance,
                    memory_type=mtype,
                )
            )

        # Select prompt based on memory type
        custom_prompt = self._config.custom_fact_prompt
        if mtype == MemoryType.PROCEDURAL:
            custom_prompt = self._config.custom_procedural_prompt

        extraction = await extract_async(
            self._model,
            text,
            uid,
            custom_fact_prompt=custom_prompt,
            memory_type=mtype.value,
            _on_usage=on_usage,
        )
        if not extraction.facts:
            return AddResult(usage=total)

        fact_texts = [f.text for f in extraction.facts]
        embeddings = self._embedder.embed(fact_texts)

        # Scope reconciliation to same memory type (only for procedural — semantic skips for backward compat)
        similar_filters = {"memory_type": mtype.value} if mtype != MemoryType.SEMANTIC else None
        existing = search_similar(
            self._db,
            embeddings,
            user_id=uid,
            threshold=self._config.similarity_threshold,
            vector_property=self._config.vector_property,
            filters=similar_filters,
        )
        logger.debug("Found %d existing memories for reconciliation", len(existing))

        decisions = await reconcile_async(self._model, extraction.facts, existing, _on_usage=on_usage)

        events = await self._execute_decisions(
            decisions,
            embeddings,
            extraction,
            uid,
            sid,
            metadata,
            now_ms,
            actor_id,
            role,
            importance=importance,
            memory_type=mtype,
            _on_usage=on_usage,
        )
        return AddResult(events, usage=total)

    async def _add_batch(
        self,
        messages_list: list[str | dict | list[dict]],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        """Add multiple messages in batch.

        For infer=False: batch-embeds all texts in a single call for efficiency.
        For infer=True: processes each message through the full extraction/reconciliation
        pipeline sequentially (reconciliation depends on prior state).
        """
        uid = user_id or self._config.user_id
        sid = session_id or self._config.session_id
        now_ms = int(time.time() * 1000)
        mtype = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type

        if not infer:
            return AddResult(
                self._raw_add_batch(
                    messages_list,
                    uid,
                    sid,
                    metadata,
                    now_ms,
                    importance=importance,
                    memory_type=mtype,
                )
            )

        from pydantic_ai.usage import RunUsage

        combined_usage = RunUsage()
        all_events: list[MemoryEvent] = []
        for messages in messages_list:
            result = await self._add(
                messages,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                infer=True,
                importance=importance,
                memory_type=mtype,
            )
            all_events.extend(result)
            combined_usage.incr(result.usage)
        return AddResult(all_events, usage=combined_usage)

    def _raw_add_batch(
        self,
        messages_list: list[str | dict | list[dict]],
        user_id: str,
        session_id: str | None,
        metadata: dict | None,
        timestamp: int,
        *,
        importance: float = 1.0,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> list[MemoryEvent]:
        """Batch store texts directly without LLM. Embeds all texts in one call."""
        texts: list[str] = []
        actors: list[tuple[str | None, str | None]] = []
        for msg in messages_list:
            text, parsed, _images = parse_messages(msg)
            actor_id, role = _extract_actor(parsed)
            texts.append(text)
            actors.append((actor_id, role))

        embeddings = self._embedder.embed(texts)

        events: list[MemoryEvent] = []
        for text, embedding, (actor_id, role) in zip(texts, embeddings, actors, strict=True):
            memory_id = self._create_memory(
                text,
                embedding,
                user_id,
                session_id,
                metadata,
                timestamp,
                actor_id,
                role,
                importance=importance,
                memory_type=memory_type,
            )
            record_history(
                self._db,
                int(memory_id),
                HistoryEntry(event="ADD", new_text=text, timestamp=timestamp, actor_id=actor_id, role=role),
            )
            events.append(
                MemoryEvent(
                    action=MemoryAction.ADD,
                    memory_id=memory_id,
                    text=text,
                    actor_id=actor_id,
                    role=role,
                    memory_type=memory_type.value,
                )
            )
        return events

    def _raw_add(
        self,
        text: str,
        user_id: str,
        session_id: str | None,
        metadata: dict | None,
        timestamp: int,
        actor_id: str | None,
        role: str | None,
        *,
        importance: float = 1.0,
        memory_type: MemoryType = MemoryType.SEMANTIC,
    ) -> list[MemoryEvent]:
        """Store text directly as a memory without LLM extraction/reconciliation."""
        embedding = self._embedder.embed([text])[0]
        memory_id = self._create_memory(
            text,
            embedding,
            user_id,
            session_id,
            metadata,
            timestamp,
            actor_id,
            role,
            importance=importance,
            memory_type=memory_type,
        )
        record_history(
            self._db,
            int(memory_id),
            HistoryEntry(event="ADD", new_text=text, timestamp=timestamp, actor_id=actor_id, role=role),
        )
        return [
            MemoryEvent(
                action=MemoryAction.ADD,
                memory_id=memory_id,
                text=text,
                actor_id=actor_id,
                role=role,
                memory_type=memory_type.value,
            )
        ]

    async def _search(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        filters: dict | None = None,
        rerank: bool = True,
        memory_type: MemoryType | str | None = None,
    ) -> SearchResponse:
        on_usage, total = self._make_usage_collector()
        uid = user_id or self._config.user_id
        search_filters = self._build_filters(uid, filters)

        if memory_type is not None:
            mtype = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type
            search_filters["memory_type"] = mtype.value

        # Embed query once, share across both search paths
        query_embedding = self._embedder.embed([query])[0]

        vector_results = hybrid_search(
            self._db,
            self._embedder,
            query,
            user_id=uid,
            k=k,
            filters=search_filters,
            vector_property=self._config.vector_property,
            text_property=self._config.text_property,
            query_embedding=query_embedding,
        )

        # Extract entities asynchronously to avoid nested run_sync() deadlock
        # when graph_search is called from within this async context.
        from .extraction.entities import extract_entities_async
        from .types import Fact

        try:
            entities, _ = await extract_entities_async(self._model, [Fact(text=query)], uid, _on_usage=on_usage)
        except Exception:
            logger.warning("_search: entity extraction failed for query=%r", query, exc_info=True)
            entities = []

        graph_results = graph_search(
            self._db,
            self._model,
            query,
            user_id=uid,
            embedder=self._embedder,
            k=k,
            vector_property=self._config.vector_property,
            _on_usage=on_usage,
            _entities=entities,
            query_embedding=query_embedding,
        )

        # Post-hoc filter graph results by memory_type if specified
        if memory_type is not None:
            mtype_val = MemoryType(memory_type).value if isinstance(memory_type, str) else memory_type.value
            graph_results = [r for r in graph_results if (r.memory_type or "semantic") == mtype_val]

        # Merge and deduplicate (keep highest score per memory)
        all_results = vector_results + graph_results
        seen: dict[str, SearchResult] = {}
        for r in all_results:
            if r.memory_id not in seen or r.score > seen[r.memory_id].score:
                seen[r.memory_id] = r

        final = sorted(seen.values(), key=lambda r: r.score, reverse=True)

        # Topology boost: lightweight structural re-ranking (no LLM call)
        if self._config.enable_topology_boost:
            from .scoring import apply_topology_boost

            final = apply_topology_boost(final, self._db, self._config)

        if rerank and self._reranker is not None:
            if hasattr(self._reranker, "rerank_async"):
                final = await self._reranker.rerank_async(query, final, top_k=k, _on_usage=on_usage)
            else:
                final = self._reranker.rerank(query, final, top_k=k, _on_usage=on_usage)

        if self._config.enable_importance:
            from .scoring import apply_importance_scoring

            final = apply_importance_scoring(final, self._db, self._config)

        return SearchResponse(final[:k], usage=total)

    async def _update(self, memory_id: str, text: str) -> MemoryEvent:
        """Update a memory's text directly. Re-embeds and records history."""
        now_ms = int(time.time() * 1000)
        old_text = self._update_memory(memory_id, text, now_ms)
        record_history(
            self._db,
            int(memory_id),
            HistoryEntry(event="UPDATE", old_text=old_text, new_text=text, timestamp=now_ms),
        )
        return MemoryEvent(
            action=MemoryAction.UPDATE,
            memory_id=memory_id,
            text=text,
            old_text=old_text,
        )

    # --- Internal helpers (sync, called from async context) ---

    async def _execute_decisions(
        self,
        decisions: list[ReconciliationDecision],
        embeddings: list[list[float]],
        extraction: ExtractionResult,
        user_id: str,
        session_id: str | None,
        metadata: dict | None,
        timestamp: int,
        actor_id: str | None,
        role: str | None,
        *,
        importance: float = 1.0,
        memory_type: MemoryType = MemoryType.SEMANTIC,
        _on_usage=None,
    ) -> list[MemoryEvent]:
        events: list[MemoryEvent] = []

        for i, decision in enumerate(decisions):
            if decision.action == MemoryAction.ADD:
                emb = embeddings[i] if i < len(embeddings) else None
                memory_id = self._create_memory(
                    decision.text,
                    emb,
                    user_id,
                    session_id,
                    metadata,
                    timestamp,
                    actor_id,
                    role,
                    importance=importance,
                    memory_type=memory_type,
                )
                record_history(
                    self._db,
                    int(memory_id),
                    HistoryEntry(
                        event="ADD", new_text=decision.text, timestamp=timestamp, actor_id=actor_id, role=role
                    ),
                )
                events.append(
                    MemoryEvent(
                        action=MemoryAction.ADD,
                        memory_id=memory_id,
                        text=decision.text,
                        actor_id=actor_id,
                        role=role,
                        memory_type=memory_type.value,
                    )
                )

            elif decision.action == MemoryAction.UPDATE and not decision.target_memory_id:
                logger.warning("UPDATE without target_memory_id, falling back to ADD: %r", decision.text)
                emb = embeddings[i] if i < len(embeddings) else None
                memory_id = self._create_memory(
                    decision.text,
                    emb,
                    user_id,
                    session_id,
                    metadata,
                    timestamp,
                    actor_id,
                    role,
                    importance=importance,
                    memory_type=memory_type,
                )
                record_history(
                    self._db,
                    int(memory_id),
                    HistoryEntry(
                        event="ADD", new_text=decision.text, timestamp=timestamp, actor_id=actor_id, role=role
                    ),
                )
                events.append(
                    MemoryEvent(
                        action=MemoryAction.ADD,
                        memory_id=memory_id,
                        text=decision.text,
                        actor_id=actor_id,
                        role=role,
                        memory_type=memory_type.value,
                    )
                )

            elif decision.action == MemoryAction.UPDATE and decision.target_memory_id:
                old_text = self._update_memory(decision.target_memory_id, decision.text, timestamp)
                record_history(
                    self._db,
                    int(decision.target_memory_id),
                    HistoryEntry(
                        event="UPDATE",
                        old_text=old_text,
                        new_text=decision.text,
                        timestamp=timestamp,
                        actor_id=actor_id,
                        role=role,
                    ),
                )
                events.append(
                    MemoryEvent(
                        action=MemoryAction.UPDATE,
                        memory_id=decision.target_memory_id,
                        text=decision.text,
                        old_text=old_text,
                        actor_id=actor_id,
                        role=role,
                        memory_type=memory_type.value,
                    )
                )

            elif decision.action == MemoryAction.DELETE and not decision.target_memory_id:
                logger.warning("DELETE without target_memory_id, skipping: %r", decision.text)

            elif decision.action == MemoryAction.DELETE and decision.target_memory_id:
                old_text = self._delete_memory(decision.target_memory_id)
                record_history(
                    self._db,
                    int(decision.target_memory_id),
                    HistoryEntry(
                        event="DELETE",
                        old_text=old_text,
                        timestamp=timestamp,
                        actor_id=actor_id,
                        role=role,
                    ),
                )
                events.append(
                    MemoryEvent(
                        action=MemoryAction.DELETE,
                        memory_id=decision.target_memory_id,
                        old_text=old_text,
                        actor_id=actor_id,
                        role=role,
                        memory_type=memory_type.value,
                    )
                )

            elif decision.action == MemoryAction.NONE:
                continue

        memory_node_ids = [e.memory_id for e in events if e.memory_id is not None]
        await self._store_graph(extraction, user_id, memory_node_ids, _on_usage=_on_usage)

        return events

    def _create_memory(
        self,
        text: str,
        embedding: list[float] | None,
        user_id: str,
        session_id: str | None,
        metadata: dict | None,
        timestamp: int,
        actor_id: str | None = None,
        role: str | None = None,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> str:
        mtype_val = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        props: dict = {
            "text": text,
            "user_id": user_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "memory_type": mtype_val,
        }
        if session_id:
            props["session_id"] = session_id
        if metadata:
            props["metadata"] = json.dumps(metadata)
        if self._config.agent_id:
            props["agent_id"] = self._config.agent_id
        if self._config.run_id:
            props["run_id"] = self._config.run_id
        if actor_id:
            props["actor_id"] = actor_id
        if role:
            props["role"] = role
        if self._config.enable_importance:
            props["importance"] = importance
            props["access_count"] = 0
            props["last_accessed"] = timestamp

        node = self._db.create_node([MEMORY_LABEL], props)
        node_id = node.id if hasattr(node, "id") else node

        if embedding is not None:
            self._db.set_node_property(node_id, self._config.vector_property, embedding)
            self._ensure_vector_index()

        return str(node_id)

    def _update_memory(self, memory_id: str, new_text: str, timestamp: int) -> str | None:
        try:
            node_id = int(memory_id)
        except ValueError:
            return None

        node = self._db.get_node(node_id)
        if node is None:
            return None

        props = _get_props(node)
        old_text = props.get("text", "")

        self._db.set_node_property(node_id, "text", new_text)
        self._db.set_node_property(node_id, "updated_at", timestamp)

        new_embedding = self._embedder.embed([new_text])[0]
        self._db.set_node_property(node_id, self._config.vector_property, new_embedding)

        return old_text

    def _delete_memory(self, memory_id: str) -> str | None:
        try:
            node_id = int(memory_id)
        except ValueError:
            return None

        node = self._db.get_node(node_id)
        if node is None:
            return None

        props = _get_props(node)
        old_text = props.get("text", "")

        self._db.delete_node(node_id)
        return old_text

    async def _store_graph(
        self,
        extraction: ExtractionResult,
        user_id: str,
        memory_node_ids: list[str] | None = None,
        *,
        _on_usage=None,
    ) -> None:
        if not extraction.entities:
            return

        entity_ids: dict[str, int] = {}
        for entity in extraction.entities:
            node_id = self._find_or_create_entity(entity, user_id)
            entity_ids[entity.name] = node_id

        if memory_node_ids:
            for mid_str in memory_node_ids:
                try:
                    mid = int(mid_str)
                except ValueError:
                    continue
                for eid in entity_ids.values():
                    self._db.create_edge(mid, eid, HAS_ENTITY_EDGE)

        if extraction.relations:
            existing_rels = self._get_existing_relations(entity_ids)
            if existing_rels:
                to_delete = await reconcile_relations_async(
                    self._model, extraction.relations, existing_rels, _on_usage=_on_usage
                )
                self._delete_relations(to_delete, existing_rels)

        for relation in extraction.relations:
            source_id = entity_ids.get(relation.source)
            target_id = entity_ids.get(relation.target)
            if source_id is not None and target_id is not None:
                self._db.create_edge(source_id, target_id, RELATION_EDGE, {"relation_type": relation.relation_type})

    def _find_or_create_entity(self, entity: Entity, user_id: str) -> int:
        try:
            nodes = self._db.find_nodes_by_property("name", entity.name)
            for nid in nodes:
                node = self._db.get_node(nid)
                if node is None:
                    continue
                props = _get_props(node)
                labels = node.labels if hasattr(node, "labels") else []
                if ENTITY_LABEL in labels and props.get("user_id") == user_id:
                    return nid
        except Exception:
            logger.warning("_find_or_create_entity: lookup failed for %r", entity.name, exc_info=True)

        node = self._db.create_node(
            [ENTITY_LABEL],
            {"name": entity.name, "entity_type": entity.entity_type, "user_id": user_id},
        )
        return node.id if hasattr(node, "id") else node

    def _get_existing_relations(self, entity_ids: dict[str, int]) -> list[dict]:
        relations: list[dict] = []
        seen_edge_ids: set[int] = set()

        for nid in entity_ids.values():
            try:
                query = (
                    f"MATCH (s:{ENTITY_LABEL})-[r:{RELATION_EDGE}]->(t:{ENTITY_LABEL}) "
                    f"WHERE id(s) = $nid "
                    f"RETURN id(r), s.name, r.relation_type, t.name"
                )
                result = self._db.execute(query, {"nid": nid})
                for row in result:
                    if not isinstance(row, dict):
                        continue
                    vals = list(row.values())
                    if len(vals) < 4:
                        continue
                    edge_id = vals[0]
                    if edge_id in seen_edge_ids:
                        continue
                    seen_edge_ids.add(edge_id)
                    relations.append(
                        {
                            "edge_id": edge_id,
                            "source": vals[1],
                            "relation_type": vals[2],
                            "target": vals[3],
                        }
                    )
            except Exception:
                logger.warning("_get_existing_relations: query failed for nid=%s", nid, exc_info=True)
                continue

        return relations

    def _delete_relations(self, to_delete: list[dict], existing_rels: list[dict]) -> None:
        for item in to_delete:
            src = item.get("source", "")
            tgt = item.get("target", "")
            rel = item.get("relation_type", "")
            for existing in existing_rels:
                if (
                    existing.get("source") == src
                    and existing.get("target") == tgt
                    and existing.get("relation_type") == rel
                ):
                    edge_id = existing.get("edge_id")
                    if edge_id is not None:
                        self._db.delete_edge(edge_id)
                    break

    def _get_all_impl(
        self, user_id: str | None = None, memory_type: MemoryType | str | None = None
    ) -> list[SearchResult]:
        """Shared get_all implementation."""
        uid = user_id or self._config.user_id

        try:
            nodes = self._db.get_nodes_by_label(MEMORY_LABEL)
        except Exception:
            return []

        filters = self._build_filters(uid)
        memories: list[SearchResult] = []
        for node_id, props in nodes:
            # Apply all scope filters
            if not all(props.get(k) == v for k, v in filters.items()):
                continue
            # Filter by memory_type if specified (treat missing as "semantic")
            if memory_type is not None:
                mtype_val = MemoryType(memory_type).value if isinstance(memory_type, str) else memory_type.value
                node_type = props.get("memory_type", "semantic")
                if node_type != mtype_val:
                    continue
            memories.append(
                SearchResult(
                    memory_id=str(node_id),
                    text=props.get("text", ""),
                    score=1.0,
                    user_id=uid,
                    metadata=_parse_metadata(props.get("metadata")),
                    actor_id=props.get("actor_id"),
                    role=props.get("role"),
                    memory_type=props.get("memory_type", "semantic"),
                )
            )
        return memories

    def _get_memories_with_timestamps(self, user_id: str) -> list[tuple[str, str, int]]:
        """Get all memories as (memory_id, text, created_at) sorted oldest-first."""
        try:
            nodes = self._db.get_nodes_by_label(MEMORY_LABEL)
        except Exception:
            return []

        filters = self._build_filters(user_id)
        results: list[tuple[str, str, int]] = []
        for node_id, props in nodes:
            if not all(props.get(k) == v for k, v in filters.items()):
                continue
            results.append((str(node_id), props.get("text", ""), int(props.get("created_at", 0))))

        results.sort(key=lambda x: x[2])  # oldest first
        return results

    async def _summarize(
        self,
        user_id: str | None = None,
        *,
        preserve_recent: int = 5,
        batch_size: int = 20,
    ) -> AddResult:
        """Consolidate old memories into fewer, more concise entries.

        Groups related memories by topic via LLM, creates new consolidated Memory
        nodes, and deletes the originals. Recent memories are preserved.
        """
        from pydantic_ai import Agent

        from .prompts import SUMMARIZE_SYSTEM, SUMMARIZE_USER
        from .schemas import SummarizeOutput

        on_usage, total = self._make_usage_collector()
        uid = user_id or self._config.user_id
        now_ms = int(time.time() * 1000)

        memory_data = self._get_memories_with_timestamps(uid)
        if not memory_data or len(memory_data) <= preserve_recent:
            return AddResult(usage=total)

        to_consolidate = memory_data[:-preserve_recent] if preserve_recent > 0 else memory_data

        # Topology-aware consolidation: protect well-connected memories
        threshold = self._config.consolidation_protect_threshold
        if threshold > 0:
            from .scoring import _batch_topology_scores

            candidate_ids = [mid for mid, _, _ in to_consolidate]
            topo_cache = _batch_topology_scores(self._db, candidate_ids, self._config)
            to_consolidate = [
                (mid, text, ts) for mid, text, ts in to_consolidate if topo_cache.get(mid, (0.0, 0.0))[0] < threshold
            ]
            if not to_consolidate:
                return AddResult(usage=total)

        agent = Agent(self._model, system_prompt=SUMMARIZE_SYSTEM, output_type=SummarizeOutput)
        all_events: list[MemoryEvent] = []

        for i in range(0, len(to_consolidate), batch_size):
            batch = to_consolidate[i : i + batch_size]
            memories_text = "\n".join(f"{j + 1}. {text}" for j, (_, text, _) in enumerate(batch))

            try:
                result = await agent.run(SUMMARIZE_USER.format(count=len(batch), memories=memories_text))
            except Exception:
                logger.warning("Summarization failed for batch %d, skipping", i // batch_size, exc_info=True)
                continue

            on_usage("summarize", result.usage())

            # Create new consolidated Memory nodes
            consolidated_texts = [m for m in result.output.memories if m]
            embeddings = self._embedder.embed(consolidated_texts)
            new_memory_ids: list[int] = []
            for text, embedding in zip(consolidated_texts, embeddings, strict=True):
                memory_id = self._create_memory(text, embedding, uid, None, None, now_ms, None, None)
                self._db.set_node_property(int(memory_id), "source", "summarize")
                record_history(self._db, int(memory_id), HistoryEntry(event="ADD", new_text=text, timestamp=now_ms))
                all_events.append(MemoryEvent(action=MemoryAction.ADD, memory_id=memory_id, text=text))
                new_memory_ids.append(int(memory_id))

            # Create DERIVED_FROM edges: each summary derives from all originals in the batch
            for new_mid in new_memory_ids:
                for mid, _, _ in batch:
                    try:
                        self._db.create_edge(new_mid, int(mid), DERIVED_FROM_EDGE)
                    except Exception:
                        logger.debug("Failed to create DERIVED_FROM edge %d->%s", new_mid, mid, exc_info=True)

            # Delete originals (record history before deletion)
            for mid, old_text, _ in batch:
                record_history(self._db, int(mid), HistoryEntry(event="DELETE", old_text=old_text, timestamp=now_ms))
                self._db.delete_node(int(mid))
                all_events.append(MemoryEvent(action=MemoryAction.DELETE, memory_id=mid, old_text=old_text))

        return AddResult(all_events, usage=total)

    def _history_impl(self, memory_id: str) -> list[HistoryEntry]:
        """Shared history implementation using graph-native History nodes."""
        try:
            node_id = int(memory_id)
        except ValueError:
            return []

        return get_history(self._db, node_id)

    def _set_importance_impl(self, memory_id: str, importance: float) -> bool:
        """Set the base importance score for a memory."""
        if not 0.0 <= importance <= 1.0:
            raise ValueError("importance must be between 0.0 and 1.0")
        try:
            node_id = int(memory_id)
        except ValueError:
            return False
        node = self._db.get_node(node_id)
        if node is None:
            return False
        self._db.set_node_property(node_id, "importance", importance)
        return True


def _extract_actor(parsed: list[Message]) -> tuple[str | None, str | None]:
    """Extract actor_id and role from the last message with a name."""
    actor_id: str | None = None
    role: str | None = None
    for msg in reversed(parsed):
        if msg.get("name"):
            actor_id = msg["name"]
            role = msg.get("role")
            break
    if actor_id is None and parsed:
        role = parsed[-1].get("role")
    return actor_id, role


class MemoryManager(_MemoryCore):
    """Sync AI memory layer powered by GrafeoDB.

    Usage::

        from openai import OpenAI
        from grafeo_memory import MemoryManager, MemoryConfig, OpenAIEmbedder

        config = MemoryConfig(db_path="./memory.db")
        embedder = OpenAIEmbedder(OpenAI())

        with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
            memory.add("I work at Acme Corp as a data scientist")
            results = memory.search("Where does the user work?")
    """

    def __enter__(self) -> MemoryManager:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()

    def add(
        self,
        messages: str | dict | list[dict],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        """Extract facts from text and store them as memories.

        Args:
            messages: Input text, a single message dict, or a list of message dicts.
            user_id: Override the config user_id for this call.
            session_id: Override the config session_id for this call.
            metadata: Arbitrary metadata to store with the memory.
            infer: If False, store text directly without LLM extraction/reconciliation.
            importance: Base importance score (0.0-1.0) when enable_importance is True.
            memory_type: Type of memory — "semantic" (facts) or "procedural" (instructions/preferences).
        """
        return run_sync(
            self._add(
                messages,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                infer=infer,
                importance=importance,
                memory_type=memory_type,
            )
        )

    def add_batch(
        self,
        messages_list: list[str | dict | list[dict]],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        """Add multiple messages in batch.

        For infer=False: batch-embeds all texts in a single call.
        For infer=True: processes each message through the full pipeline sequentially.
        """
        return run_sync(
            self._add_batch(
                messages_list,
                user_id=user_id,
                session_id=session_id,
                metadata=metadata,
                infer=infer,
                importance=importance,
                memory_type=memory_type,
            )
        )

    def search(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        filters: dict | None = None,
        rerank: bool = True,
        memory_type: MemoryType | str | None = None,
    ) -> SearchResponse:
        """Search memories by semantic similarity and graph context."""
        return run_sync(
            self._search(query, user_id=user_id, k=k, filters=filters, rerank=rerank, memory_type=memory_type)
        )

    def update(self, memory_id: str, text: str) -> MemoryEvent:
        """Update a memory's text directly. Re-embeds and records history."""
        return run_sync(self._update(memory_id, text))

    def get_all(self, user_id: str | None = None, memory_type: MemoryType | str | None = None) -> list[SearchResult]:
        """Retrieve all memories for a user."""
        return self._get_all_impl(user_id, memory_type=memory_type)

    def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by its ID."""
        try:
            node_id = int(memory_id)
        except ValueError:
            return False
        return self._db.delete_node(node_id)

    def delete_all(self, user_id: str | None = None) -> int:
        """Delete all memories for a user. Returns the count of deleted memories."""
        memories = self.get_all(user_id)
        count = 0
        for m in memories:
            if self.delete(m.memory_id):
                count += 1
        return count

    def summarize(
        self,
        user_id: str | None = None,
        *,
        preserve_recent: int = 5,
        batch_size: int = 20,
    ) -> AddResult:
        """Consolidate old memories into fewer, more concise entries.

        Groups related memories by topic via LLM and replaces originals
        with consolidated versions. Recent memories are preserved.

        Args:
            user_id: Override the config user_id.
            preserve_recent: Number of most recent memories to leave untouched.
            batch_size: Max memories to consolidate per LLM call.
        """
        return run_sync(self._summarize(user_id, preserve_recent=preserve_recent, batch_size=batch_size))

    def set_importance(self, memory_id: str, importance: float) -> bool:
        """Set the base importance score (0.0-1.0) for a memory."""
        return self._set_importance_impl(memory_id, importance)

    def history(self, memory_id: str) -> list[HistoryEntry]:
        """Get the change history for a memory."""
        return self._history_impl(memory_id)


class AsyncMemoryManager(_MemoryCore):
    """Async AI memory layer powered by GrafeoDB.

    Usage::

        from openai import OpenAI
        from grafeo_memory import AsyncMemoryManager, MemoryConfig, OpenAIEmbedder

        config = MemoryConfig(db_path="./memory.db")
        embedder = OpenAIEmbedder(OpenAI())

        async with AsyncMemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
            await memory.add("I work at Acme Corp as a data scientist")
            results = await memory.search("Where does the user work?")
    """

    async def __aenter__(self) -> AsyncMemoryManager:
        return self

    async def __aexit__(self, *args: object) -> None:
        self.close()

    async def add(
        self,
        messages: str | dict | list[dict],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        """Extract facts from text and store them as memories."""
        return await self._add(
            messages,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            infer=infer,
            importance=importance,
            memory_type=memory_type,
        )

    async def add_batch(
        self,
        messages_list: list[str | dict | list[dict]],
        user_id: str | None = None,
        session_id: str | None = None,
        metadata: dict | None = None,
        *,
        infer: bool = True,
        importance: float = 1.0,
        memory_type: MemoryType | str = MemoryType.SEMANTIC,
    ) -> AddResult:
        """Add multiple messages in batch."""
        return await self._add_batch(
            messages_list,
            user_id=user_id,
            session_id=session_id,
            metadata=metadata,
            infer=infer,
            importance=importance,
            memory_type=memory_type,
        )

    async def search(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        filters: dict | None = None,
        rerank: bool = True,
        memory_type: MemoryType | str | None = None,
    ) -> SearchResponse:
        """Search memories by semantic similarity and graph context."""
        return await self._search(
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            rerank=rerank,
            memory_type=memory_type,
        )

    async def update(self, memory_id: str, text: str) -> MemoryEvent:
        """Update a memory's text directly. Re-embeds and records history."""
        return await self._update(memory_id, text)

    async def get_all(
        self, user_id: str | None = None, memory_type: MemoryType | str | None = None
    ) -> list[SearchResult]:
        """Retrieve all memories for a user."""
        return self._get_all_impl(user_id, memory_type=memory_type)

    async def delete(self, memory_id: str) -> bool:
        """Delete a specific memory by its ID."""
        try:
            node_id = int(memory_id)
        except ValueError:
            return False
        return self._db.delete_node(node_id)

    async def delete_all(self, user_id: str | None = None) -> int:
        """Delete all memories for a user. Returns the count of deleted memories."""
        memories = await self.get_all(user_id)
        count = 0
        for m in memories:
            if await self.delete(m.memory_id):
                count += 1
        return count

    async def summarize(
        self,
        user_id: str | None = None,
        *,
        preserve_recent: int = 5,
        batch_size: int = 20,
    ) -> AddResult:
        """Consolidate old memories into fewer, more concise entries."""
        return await self._summarize(user_id, preserve_recent=preserve_recent, batch_size=batch_size)

    def set_importance(self, memory_id: str, importance: float) -> bool:
        """Set the base importance score (0.0-1.0) for a memory."""
        return self._set_importance_impl(memory_id, importance)

    async def history(self, memory_id: str) -> list[HistoryEntry]:
        """Get the change history for a memory."""
        return self._history_impl(memory_id)
