"""MemoryManager — the public API for grafeo-memory."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import time

from ._compat import run_sync
from .embedding import EmbeddingClient
from .extraction import extract_async
from .history import HistoryEntry, get_history, record_history
from .messages import Message, parse_messages
from .protocol import GrafeoDBProtocol
from .reconciliation import reconcile_async, reconcile_relations_async
from .search import graph_search, hybrid_search, search_similar
from .search.vector import _get_props, _parse_metadata
from .types import (
    DERIVED_FROM_EDGE,
    ENTITY_LABEL,
    EPISODE_LABEL,
    HAS_ENTITY_EDGE,
    LEADS_TO_EDGE,
    MEMORY_LABEL,
    MENTIONS_EDGE,
    NEXT_EPISODE_EDGE,
    PRODUCED_EDGE,
    RELATION_EDGE,
    SUPERSEDES_EDGE,
    AddResult,
    Entity,
    EpisodeResult,
    ExplainResult,
    ExplainStep,
    ExtractionResult,
    MemoryAction,
    MemoryConfig,
    MemoryEvent,
    MemoryStats,
    MemoryType,
    ModelType,
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
        model: ModelType,
        config: MemoryConfig | None = None,
        *,
        embedder: EmbeddingClient,
        reranker: object | None = None,
        db: GrafeoDBProtocol | None = None,
    ):
        import grafeo

        self._config = config or MemoryConfig()
        self._model: ModelType = model
        self._embedder = embedder
        self._reranker = reranker

        self._db: GrafeoDBProtocol
        self._db_external = db is not None
        if db is not None:
            self._db = db
        elif self._config.db_path:
            self._db = grafeo.GrafeoDB(self._config.db_path)
        else:
            self._db = grafeo.GrafeoDB()

        self._vector_index_ready = False
        self._user_locks: dict[str, asyncio.Lock] = {}
        self._graph_dirty = False
        self._ensure_indexes()

        # OpenTelemetry: instrument all pydantic-ai agents when enabled
        if self._config.instrument:
            from pydantic_ai import Agent

            Agent.instrument_all(self._config.instrument if self._config.instrument is not True else True)

    def _ensure_indexes(self) -> None:
        """Create indexes for efficient memory queries.

        Vector and text indexes are label-scoped to Memory nodes. Property
        indexes (user_id, created_at, memory_type, name) are database-wide
        because GrafeoDB's ``create_property_index`` does not support
        label-scoped indexing. In a shared database these indexes will also
        cover non-memory nodes — harmless but slightly wasteful.
        """
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

        # Property indexes for fast filtered lookups (database-wide; see docstring).
        for prop in ("user_id", "created_at", "memory_type"):
            with contextlib.suppress(Exception):
                self._db.create_property_index(prop)
        with contextlib.suppress(Exception):
            self._db.create_property_index("name")

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

        If the database was externally provided via the ``db`` parameter,
        it is **not** closed here — the caller owns its lifecycle.

        The async runner is intentionally NOT closed here — it is shared across
        sessions and cleaned up automatically at process exit via atexit. Closing
        it per-session corrupts httpx/anyio transport state on Windows.
        """
        if self._db_external:
            return
        if hasattr(self._db, "close"):
            self._db.close()  # ty: ignore[call-non-callable]

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
                    cb_name = getattr(cb, "__name__", repr(cb))
                    logger.warning("Usage callback %s failed for '%s'", cb_name, operation, exc_info=True)

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
            events = self._raw_add(
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
            run = self._config.run_id or sid
            new_ids = [e.memory_id for e in events if e.action == MemoryAction.ADD and e.memory_id]
            if self._config.enable_episodes and events:
                ep_id = self._create_episode(text, uid, sid, now_ms, None, events)
                if run:
                    self._link_episode_chain(ep_id, uid, run)
            elif run and new_ids:
                self._link_session_chain(new_ids, uid, run)
            if events:
                self._graph_dirty = True
            return AddResult(events)

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

        # Bi-temporal: annotate facts with real-world validity timestamps
        if self._config.enable_bitemporal:
            from .extraction.temporal import annotate_temporal_async

            extraction.temporal_annotations = await annotate_temporal_async(
                self._model, extraction.facts, text, _on_usage=on_usage
            )

        fact_texts = [f.text for f in extraction.facts]
        embeddings = self._embedder.embed(fact_texts)

        # Scope reconciliation to same memory type (only for procedural — semantic skips for backward compat)
        similar_filters = {"memory_type": mtype.value} if mtype != MemoryType.SEMANTIC else None

        # Per-user lock prevents race conditions where concurrent adds reconcile
        # against the same existing memory. The LLM call (reconcile_async) happens
        # inside the lock but outside any DB transaction.
        lock = self._user_locks.setdefault(uid, asyncio.Lock())
        async with lock:
            existing = search_similar(
                self._db,
                embeddings,
                user_id=uid,
                threshold=self._config.reconciliation_threshold,
                vector_property=self._config.vector_property,
                filters=similar_filters,
            )
            logger.debug("Found %d existing memories for reconciliation", len(existing))

            decisions = await reconcile_async(self._model, extraction.facts, existing, _on_usage=on_usage)

            # Execute memory mutations inside a transaction for atomicity
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

            # Session ordering: episodes or LEADS_TO edges
            run = self._config.run_id or sid
            new_add_ids = [e.memory_id for e in events if e.action == MemoryAction.ADD and e.memory_id]
            if self._config.enable_episodes and events:
                ep_id = self._create_episode(text, uid, sid, now_ms, extraction, events)
                if run:
                    self._link_episode_chain(ep_id, uid, run)
            elif run and new_add_ids:
                self._link_session_chain(new_add_ids, uid, run)

            if events:
                self._graph_dirty = True

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

        # Build property dicts for all nodes
        mtype_val = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        metadata_json = json.dumps(metadata) if metadata else None
        properties_list: list[dict] = []
        for text, embedding, (actor_id, role) in zip(texts, embeddings, actors, strict=True):
            props: dict = {
                "text": text,
                "user_id": user_id,
                "created_at": timestamp,
                "updated_at": timestamp,
                "learned_at": timestamp,
                "memory_type": mtype_val,
            }
            if session_id:
                props["session_id"] = session_id
            if metadata_json:
                props["metadata"] = metadata_json
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
            if embedding is not None:
                if self._embedding_dims and len(embedding) != self._embedding_dims:
                    raise ValueError(
                        f"Embedding dimension mismatch: got {len(embedding)}, expected {self._embedding_dims}"
                    )
                props[self._config.vector_property] = embedding
            properties_list.append(props)

        # Try batch creation, fall back to individual nodes
        try:
            node_ids = self._db.batch_create_nodes_with_props(MEMORY_LABEL, properties_list)  # ty: ignore[unresolved-attribute]
        except AttributeError:
            # batch_create_nodes_with_props not available, fall back to per-node creation
            node_ids = []
            for props in properties_list:
                embedding = props.pop(self._config.vector_property, None)
                node = self._db.create_node([MEMORY_LABEL], props)
                nid = node.id if hasattr(node, "id") else node
                if embedding is not None:
                    self._db.set_node_property(nid, self._config.vector_property, embedding)
                node_ids.append(nid)

        self._ensure_vector_index()

        # Record history and build events
        events: list[MemoryEvent] = []
        for nid, text, (actor_id, role) in zip(node_ids, texts, actors, strict=True):
            memory_id = str(nid)
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
        min_score: float | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        point_in_time: int | None = None,
        _trace: list[ExplainStep] | None = None,
    ) -> SearchResponse:
        from .temporal import detect_temporal_hints

        self._recompute_graph_metrics()

        # Deferred community materialization (requires async context for LLM calls)
        if hasattr(self, "_pending_communities") and self._pending_communities:
            from .communities import materialize_communities_async

            uid_for_communities = user_id or self._config.user_id
            await materialize_communities_async(self._db, self._model, uid_for_communities, self._pending_communities)
            self._pending_communities = None

        on_usage, total = self._make_usage_collector()
        uid = user_id or self._config.user_id
        search_filters = self._build_filters(uid, filters)

        if memory_type is not None:
            mtype = MemoryType(memory_type) if isinstance(memory_type, str) else memory_type
            search_filters["memory_type"] = mtype.value

        # Detect temporal signals in the query and adjust search behavior
        temporal_hints = detect_temporal_hints(query)
        if temporal_hints.include_expired and not include_expired:
            include_expired = True
        effective_k = k * 2 if temporal_hints.expand_limit else k

        # Embed query once, share across both search paths
        query_embedding = self._embedder.embed([query])[0]

        if _trace is not None:
            detail: dict = {"dimensions": len(query_embedding)}
            if temporal_hints.is_temporal:
                detail["temporal_hints"] = temporal_hints.signals
            _trace.append(ExplainStep(name="embed_query", detail=detail))

        if diverse:
            from .search.vector import diverse_search

            vector_results = diverse_search(
                self._db,
                self._embedder,
                query,
                user_id=uid,
                k=effective_k,
                filters=search_filters,
                vector_property=self._config.vector_property,
                query_embedding=query_embedding,
                lambda_mult=self._config.mmr_lambda,
            )
        else:
            vector_results = hybrid_search(
                self._db,
                self._embedder,
                query,
                user_id=uid,
                k=effective_k,
                filters=search_filters,
                vector_property=self._config.vector_property,
                text_property=self._config.text_property,
                query_embedding=query_embedding,
            )

        if _trace is not None:
            _trace.append(
                ExplainStep(
                    name="hybrid_search",
                    detail={
                        "candidates": len(vector_results),
                        "top_scores": [round(r.score, 4) for r in vector_results[:5]],
                    },
                )
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

        if _trace is not None:
            _trace.append(
                ExplainStep(
                    name="entity_extraction",
                    detail={"entities": [e.name for e in entities]},
                )
            )

        graph_results = graph_search(
            self._db,
            self._model,
            query,
            user_id=uid,
            embedder=self._embedder,
            k=effective_k,
            vector_property=self._config.vector_property,
            _on_usage=on_usage,
            _entities=entities,
            query_embedding=query_embedding,
            search_depth=self._config.graph_search_depth,
        )

        # Post-hoc filter graph results by memory_type if specified
        if memory_type is not None:
            mtype_val = MemoryType(memory_type).value if isinstance(memory_type, str) else memory_type.value
            graph_results = [r for r in graph_results if (r.memory_type or "semantic") == mtype_val]

        if _trace is not None:
            _trace.append(
                ExplainStep(
                    name="graph_search",
                    detail={
                        "candidates": len(graph_results),
                        "top_scores": [round(r.score, 4) for r in graph_results[:5]],
                    },
                )
            )

        # Merge and deduplicate with agreement bonus
        vector_map = {r.memory_id: r for r in vector_results}
        graph_map = {r.memory_id: r for r in graph_results}
        all_ids = set(vector_map) | set(graph_map)
        bonus = self._config.agreement_bonus
        agreement_count = 0
        seen: dict[str, SearchResult] = {}
        for mid in all_ids:
            v = vector_map.get(mid)
            g = graph_map.get(mid)
            if v and g:
                agreement_count += 1
                best = v if v.score >= g.score else g
                score = best.score * (1.0 + bonus)
                seen[mid] = SearchResult(
                    memory_id=best.memory_id,
                    text=best.text,
                    score=score,
                    user_id=best.user_id,
                    metadata=best.metadata,
                    relations=best.relations,
                    actor_id=best.actor_id,
                    role=best.role,
                    importance=best.importance,
                    access_count=best.access_count,
                    memory_type=best.memory_type,
                    source="both",
                )
            elif v:
                seen[mid] = v
            else:
                seen[mid] = g  # ty: ignore[invalid-assignment]

        final = sorted(seen.values(), key=lambda r: r.score, reverse=True)

        if _trace is not None:
            _trace.append(
                ExplainStep(
                    name="merge",
                    detail={
                        "vector_count": len(vector_results),
                        "graph_count": len(graph_results),
                        "after_dedup": len(final),
                        "agreement_count": agreement_count,
                    },
                )
            )

        # Temporal filtering: exclude expired and apply time range
        if not include_expired:
            final = [r for r in final if r.expired_at is None]
        if time_after is not None:
            final = [r for r in final if r.created_at is not None and r.created_at >= time_after]
        if time_before is not None:
            final = [r for r in final if r.created_at is not None and r.created_at <= time_before]
        if (time_after is not None or time_before is not None or include_expired) and _trace is not None:
            _trace.append(
                ExplainStep(
                    name="temporal_filter",
                    detail={
                        "time_after": time_after,
                        "time_before": time_before,
                        "include_expired": include_expired,
                        "remaining": len(final),
                    },
                )
            )

        # Bi-temporal point-in-time filter: only include facts valid at the given moment
        if point_in_time is not None:
            before_pit = len(final)
            final = [r for r in final if _valid_at_point(r, point_in_time)]
            if _trace is not None:
                _trace.append(
                    ExplainStep(
                        name="point_in_time_filter",
                        detail={"point_in_time": point_in_time, "before": before_pit, "after": len(final)},
                    )
                )

        # Topology boost: lightweight structural re-ranking (no LLM call)
        if self._config.enable_topology_boost:
            from .scoring import apply_topology_boost

            final = apply_topology_boost(final, self._db, self._config)
            if _trace is not None:
                _trace.append(ExplainStep(name="topology_boost", detail={"applied": True}))

        # Cross-session boost: use cached graph algorithm scores
        if self._config.cross_session_factor > 0:
            from .scoring import apply_cross_session_boost

            final = apply_cross_session_boost(final, self._db, self._config)
            if _trace is not None:
                _trace.append(ExplainStep(name="cross_session_boost", detail={"applied": True}))

        if rerank and self._reranker is not None:
            if hasattr(self._reranker, "rerank_async"):
                final = await self._reranker.rerank_async(query, final, top_k=k, _on_usage=on_usage)  # ty: ignore[call-non-callable]
            else:
                final = self._reranker.rerank(query, final, top_k=k, _on_usage=on_usage)  # ty: ignore[unresolved-attribute]
            if _trace is not None:
                _trace.append(ExplainStep(name="rerank", detail={"applied": True}))

        if self._config.enable_importance:
            from .scoring import apply_importance_scoring

            final = apply_importance_scoring(final, self._db, self._config)
            if _trace is not None:
                _trace.append(ExplainStep(name="importance_scoring", detail={"applied": True}))

        # Min-score filtering: drop results below threshold
        effective_min = min_score if min_score is not None else self._config.search_min_score
        if effective_min > 0:
            before_filter = len(final)
            final = [r for r in final if r.score >= effective_min]
            if _trace is not None:
                _trace.append(
                    ExplainStep(
                        name="min_score_filter",
                        detail={"threshold": effective_min, "before": before_filter, "after": len(final)},
                    )
                )

        # Temporal sort: when temporal keywords detected, sort chronologically instead of by score
        if temporal_hints.sort_chronologically:
            final.sort(key=lambda r: r.created_at or 0)
            if _trace is not None:
                _trace.append(ExplainStep(name="temporal_sort", detail={"order": "chronological"}))

        return SearchResponse(final[:k], usage=total)

    async def _explain(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        memory_type: MemoryType | str | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        point_in_time: int | None = None,
    ) -> ExplainResult:
        """Run a search with full pipeline tracing."""
        trace: list[ExplainStep] = []
        response = await self._search(
            query,
            user_id=user_id,
            k=k,
            memory_type=memory_type,
            time_before=time_before,
            time_after=time_after,
            include_expired=include_expired,
            diverse=diverse,
            point_in_time=point_in_time,
            _trace=trace,
        )

        trace.append(
            ExplainStep(
                name="final",
                detail={
                    "count": len(response),
                    "top_results": [
                        {"id": r.memory_id, "score": round(r.score, 4), "text": r.text[:80]} for r in response[:5]
                    ],
                },
            )
        )

        return ExplainResult(query=query, steps=trace, results=list(response))

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
        temporal = extraction.temporal_annotations

        for i, decision in enumerate(decisions):
            fact_valid_at, _ = temporal.get(i, (None, None))

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
                    valid_at=fact_valid_at,
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
                        valid_at=fact_valid_at,
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
                    valid_at=fact_valid_at,
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
                        valid_at=fact_valid_at,
                    )
                )

            elif decision.action == MemoryAction.UPDATE and decision.target_memory_id:
                # Soft-expiry update: expire old memory, create new one, link with SUPERSEDES
                # In bi-temporal mode, set invalid_at on old memory to when the new fact became true
                old_text = self._expire_memory(
                    decision.target_memory_id, timestamp, invalid_at=fact_valid_at or timestamp
                )
                emb = self._embedder.embed([decision.text])[0]
                new_memory_id = self._create_memory(
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
                    valid_at=fact_valid_at,
                )
                try:
                    self._db.create_edge(int(new_memory_id), int(decision.target_memory_id), SUPERSEDES_EDGE)
                except Exception:
                    logger.warning(
                        "Failed to create SUPERSEDES edge %s->%s",
                        new_memory_id,
                        decision.target_memory_id,
                        exc_info=True,
                    )
                # Inherit entity edges from the expired memory so graph search
                # can still reach the new memory via the old entity connections.
                self._inherit_entity_edges(decision.target_memory_id, new_memory_id)
                record_history(
                    self._db,
                    int(new_memory_id),
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
                        memory_id=new_memory_id,
                        text=decision.text,
                        old_text=old_text,
                        actor_id=actor_id,
                        role=role,
                        memory_type=memory_type.value,
                        valid_at=fact_valid_at,
                    )
                )

            elif decision.action == MemoryAction.DELETE and not decision.target_memory_id:
                logger.warning("DELETE without target_memory_id, skipping: %r", decision.text)

            elif decision.action == MemoryAction.DELETE and decision.target_memory_id:
                old_text = self._expire_memory(decision.target_memory_id, timestamp)
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
        learned_at: int | None = None,
        valid_at: int | None = None,
    ) -> str:
        mtype_val = memory_type.value if isinstance(memory_type, MemoryType) else memory_type
        props: dict = {
            "text": text,
            "user_id": user_id,
            "created_at": timestamp,
            "updated_at": timestamp,
            "learned_at": learned_at or timestamp,
            "memory_type": mtype_val,
        }
        if valid_at is not None:
            props["valid_at"] = valid_at
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
            if self._embedding_dims and len(embedding) != self._embedding_dims:
                raise ValueError(f"Embedding dimension mismatch: got {len(embedding)}, expected {self._embedding_dims}")
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

    def _expire_memory(self, memory_id: str, timestamp: int, *, invalid_at: int | None = None) -> str | None:
        """Soft-expire a memory by setting expired_at instead of deleting it.

        When bi-temporal mode is enabled, also sets invalid_at on the node to
        record when the fact ceased being true in reality.
        """
        try:
            node_id = int(memory_id)
        except ValueError:
            return None

        node = self._db.get_node(node_id)
        if node is None:
            return None

        props = _get_props(node)
        old_text = props.get("text", "")

        self._db.set_node_property(node_id, "expired_at", timestamp)
        if self._config.enable_bitemporal and invalid_at is not None:
            self._db.set_node_property(node_id, "invalid_at", invalid_at)
        return old_text

    def _inherit_entity_edges(self, old_memory_id: str, new_memory_id: str) -> None:
        """Copy HAS_ENTITY edges from an expired memory to its replacement.

        When UPDATE expires a memory and creates a new one, the old memory's
        entity connections would be lost to graph search. This preserves them
        so the new memory is reachable via all the same entity paths.
        """
        try:
            old_id = int(old_memory_id)
            new_id = int(new_memory_id)
        except ValueError:
            return

        try:
            query = (
                f"MATCH (m:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) WHERE id(m) = $old_id RETURN id(e)"
            )
            rows = self._db.execute(query, {"old_id": old_id})
            for row in rows:
                if not isinstance(row, dict):
                    continue
                vals = list(row.values())
                if vals:
                    entity_id = int(vals[0])
                    with contextlib.suppress(Exception):
                        self._db.create_edge(new_id, entity_id, HAS_ENTITY_EDGE)
        except Exception:
            logger.debug("_inherit_entity_edges failed for %s->%s", old_memory_id, new_memory_id, exc_info=True)

    def _link_session_chain(self, new_memory_ids: list[str], user_id: str, run_id: str) -> None:
        """Create LEADS_TO edges linking new memories to the session's previous memory."""
        # Find the most recent existing memory in this session (by created_at, excluding the new ones)
        new_set = set(new_memory_ids)
        try:
            nodes = self._db.get_nodes_by_label(MEMORY_LABEL)
        except Exception:
            return

        prev_id: str | None = None
        prev_ts: int = 0
        for node_id, props in nodes:
            mid = str(node_id)
            if mid in new_set:
                continue
            if props.get("user_id") != user_id:
                continue
            node_run = props.get("run_id") or props.get("session_id")
            if node_run != run_id:
                continue
            if props.get("expired_at") is not None:
                continue
            ts = int(props.get("created_at", 0))
            if ts > prev_ts:
                prev_ts = ts
                prev_id = mid

        # Chain: prev → first new, then new[0] → new[1] → ...
        ordered = new_memory_ids
        if prev_id is not None:
            try:
                self._db.create_edge(int(prev_id), int(ordered[0]), LEADS_TO_EDGE, {"sequence": 0})
            except Exception:
                logger.debug("Failed to create LEADS_TO edge %s->%s", prev_id, ordered[0], exc_info=True)

        for i in range(len(ordered) - 1):
            try:
                self._db.create_edge(int(ordered[i]), int(ordered[i + 1]), LEADS_TO_EDGE, {"sequence": i + 1})
            except Exception:
                logger.debug("Failed to create LEADS_TO edge %s->%s", ordered[i], ordered[i + 1], exc_info=True)

    # --- Episode provenance ---

    def _create_episode(
        self,
        content: str,
        user_id: str,
        session_id: str | None,
        timestamp: int,
        extraction: ExtractionResult | None,
        events: list[MemoryEvent],
    ) -> str:
        """Create an Episode node and link it to produced memories and mentioned entities."""
        props: dict = {
            "content": content,
            "source": "message",
            "user_id": user_id,
            "created_at": timestamp,
        }
        if session_id:
            props["session_id"] = session_id
        if self._config.run_id:
            props["run_id"] = self._config.run_id

        node = self._db.create_node([EPISODE_LABEL], props)
        ep_id = node.id if hasattr(node, "id") else node

        # PRODUCED edges: Episode -> Memory
        for event in events:
            if event.memory_id is not None:
                with contextlib.suppress(Exception):
                    self._db.create_edge(ep_id, int(event.memory_id), PRODUCED_EDGE)

        # MENTIONS edges: Episode -> Entity
        if extraction and extraction.entities:
            for entity in extraction.entities:
                entity_node = self._find_entity_by_name(entity.name, user_id)
                if entity_node is not None:
                    with contextlib.suppress(Exception):
                        self._db.create_edge(ep_id, entity_node, MENTIONS_EDGE)

        return str(ep_id)

    def _find_entity_by_name(self, name: str, user_id: str) -> int | None:
        """Find an entity node by name and user_id."""
        try:
            query = f"MATCH (e:{ENTITY_LABEL}) WHERE e.name = $name AND e.user_id = $uid RETURN id(e)"
            rows = self._db.execute(query, {"name": name, "uid": user_id})
            for row in rows:
                if not isinstance(row, dict):
                    continue
                vals = list(row.values())
                if vals:
                    return int(vals[0])
        except Exception:
            logger.debug("_find_entity_by_name failed for %s", name, exc_info=True)
        return None

    def _link_episode_chain(self, episode_id: str, user_id: str, run_id: str) -> None:
        """Create NEXT_EPISODE edge from the previous episode to this one."""
        try:
            nodes = self._db.get_nodes_by_label(EPISODE_LABEL)
        except Exception:
            return

        prev_id: str | None = None
        prev_ts: int = 0
        for node_id, props in nodes:
            mid = str(node_id)
            if mid == episode_id:
                continue
            if props.get("user_id") != user_id:
                continue
            node_run = props.get("run_id") or props.get("session_id")
            if node_run != run_id:
                continue
            ts = int(props.get("created_at", 0))
            if ts > prev_ts:
                prev_ts = ts
                prev_id = mid

        if prev_id is not None:
            with contextlib.suppress(Exception):
                self._db.create_edge(int(prev_id), int(episode_id), NEXT_EPISODE_EDGE)

    def _get_episodes_impl(
        self,
        user_id: str | None = None,
        session_id: str | None = None,
        limit: int = 50,
    ) -> list[EpisodeResult]:
        """Retrieve episode nodes."""
        uid = user_id or self._config.user_id
        try:
            nodes = self._db.get_nodes_by_label(EPISODE_LABEL)
        except Exception:
            return []

        episodes: list[EpisodeResult] = []
        for node_id, props in nodes:
            if props.get("user_id") != uid:
                continue
            if session_id is not None and props.get("session_id") != session_id:
                continue

            # Collect produced memory IDs
            produced: list[str] = []
            mentioned: list[str] = []
            try:
                query = f"MATCH (e)-[:{PRODUCED_EDGE}]->(m:{MEMORY_LABEL}) WHERE id(e) = $eid RETURN id(m)"
                for row in self._db.execute(query, {"eid": node_id}):
                    if isinstance(row, dict):
                        vals = list(row.values())
                        if vals:
                            produced.append(str(vals[0]))
            except Exception:
                pass
            try:
                query = f"MATCH (e)-[:{MENTIONS_EDGE}]->(ent:{ENTITY_LABEL}) WHERE id(e) = $eid RETURN ent.name"
                for row in self._db.execute(query, {"eid": node_id}):
                    if isinstance(row, dict):
                        vals = list(row.values())
                        if vals and vals[0]:
                            mentioned.append(str(vals[0]))
            except Exception:
                pass

            episodes.append(
                EpisodeResult(
                    episode_id=str(node_id),
                    content=props.get("content", ""),
                    source=props.get("source", "message"),
                    user_id=uid,
                    session_id=props.get("session_id"),
                    run_id=props.get("run_id"),
                    created_at=props.get("created_at"),
                    valid_at=props.get("valid_at"),
                    produced_memories=produced,
                    mentioned_entities=mentioned,
                )
            )

        episodes.sort(key=lambda e: e.created_at or 0)
        return episodes[:limit]

    def _get_provenance_impl(self, memory_id: str) -> list[EpisodeResult]:
        """Get the episodes that produced a given memory."""
        try:
            mid = int(memory_id)
        except ValueError:
            return []

        try:
            query = f"MATCH (e:{EPISODE_LABEL})-[:{PRODUCED_EDGE}]->(m:{MEMORY_LABEL}) WHERE id(m) = $mid RETURN id(e)"
            ep_ids: list[int] = []
            for row in self._db.execute(query, {"mid": mid}):
                if isinstance(row, dict):
                    vals = list(row.values())
                    if vals:
                        ep_ids.append(int(vals[0]))
        except Exception:
            return []

        results: list[EpisodeResult] = []
        for ep_id in ep_ids:
            node = self._db.get_node(ep_id)
            if node is None:
                continue
            props = _get_props(node)
            results.append(
                EpisodeResult(
                    episode_id=str(ep_id),
                    content=props.get("content", ""),
                    source=props.get("source", "message"),
                    user_id=props.get("user_id", ""),
                    session_id=props.get("session_id"),
                    run_id=props.get("run_id"),
                    created_at=props.get("created_at"),
                    valid_at=props.get("valid_at"),
                )
            )
        return results

    def _episode_chain_impl(
        self,
        episode_id: str,
        direction: str = "forward",
        max_depth: int = 10,
    ) -> list[EpisodeResult]:
        """Follow NEXT_EPISODE edges for session replay."""
        try:
            start_id = int(episode_id)
        except ValueError:
            return []

        results: list[EpisodeResult] = []

        if direction in ("forward", "both"):
            self._traverse_episodes(start_id, "forward", max_depth, results)

        if direction in ("backward", "both"):
            backward: list[EpisodeResult] = []
            self._traverse_episodes(start_id, "backward", max_depth, backward)
            results = list(reversed(backward)) + results

        return results

    def _traverse_episodes(self, start_id: int, direction: str, max_depth: int, results: list[EpisodeResult]) -> None:
        current_id = start_id
        for _ in range(max_depth):
            if direction == "forward":
                query = (
                    f"MATCH (a:{EPISODE_LABEL})-[:{NEXT_EPISODE_EDGE}]->(b:{EPISODE_LABEL}) "
                    f"WHERE id(a) = $cid RETURN id(b)"
                )
            else:
                query = (
                    f"MATCH (b:{EPISODE_LABEL})-[:{NEXT_EPISODE_EDGE}]->(a:{EPISODE_LABEL}) "
                    f"WHERE id(a) = $cid RETURN id(b)"
                )
            try:
                rows = self._db.execute(query, {"cid": current_id})
            except Exception:
                break
            next_id = None
            for row in rows:
                if isinstance(row, dict):
                    vals = list(row.values())
                    if vals:
                        next_id = int(vals[0])
                        break
            if next_id is None:
                break
            node = self._db.get_node(next_id)
            if node is None:
                break
            props = _get_props(node)
            results.append(
                EpisodeResult(
                    episode_id=str(next_id),
                    content=props.get("content", ""),
                    source=props.get("source", "message"),
                    user_id=props.get("user_id", ""),
                    session_id=props.get("session_id"),
                    run_id=props.get("run_id"),
                    created_at=props.get("created_at"),
                    valid_at=props.get("valid_at"),
                )
            )
            current_id = next_id

    def temporal_chain(
        self,
        memory_id: str,
        user_id: str | None = None,
        direction: str = "forward",
        max_depth: int = 5,
    ) -> list:
        """Follow LEADS_TO edges to build a temporal chain of memories.

        Args:
            memory_id: Starting memory node ID.
            user_id: If provided, only include memories belonging to this user.
            direction: "forward", "backward", or "both".
            max_depth: Maximum number of hops to traverse.

        Returns:
            List of dicts with memory_id, text, created_at, session_id.
        """
        mid = int(memory_id)
        user_filter = " AND rest.user_id = $uid" if user_id else ""
        params: dict = {"mid": mid}
        if user_id:
            params["uid"] = user_id

        results: list[dict] = []

        if direction in ("forward", "both"):
            query = (
                f"MATCH (m:{MEMORY_LABEL})-[:{LEADS_TO_EDGE}*1..{max_depth}]->(rest:{MEMORY_LABEL}) "
                f"WHERE id(m) = $mid{user_filter} "
                f"RETURN id(rest), rest.text, rest.created_at, rest.session_id "
                f"ORDER BY rest.created_at"
            )
            try:
                rows = self._db.execute(query, params)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    vals = list(row.values())
                    if len(vals) < 4:
                        continue
                    results.append(
                        {
                            "memory_id": str(vals[0]),
                            "text": vals[1],
                            "created_at": vals[2],
                            "session_id": vals[3],
                        }
                    )
            except Exception:
                logger.debug(
                    "temporal_chain forward query failed for %s",
                    memory_id,
                    exc_info=True,
                )

        if direction in ("backward", "both"):
            query = (
                f"MATCH (rest:{MEMORY_LABEL})-[:{LEADS_TO_EDGE}*1..{max_depth}]->(m:{MEMORY_LABEL}) "
                f"WHERE id(m) = $mid{user_filter} "
                f"RETURN id(rest), rest.text, rest.created_at, rest.session_id "
                f"ORDER BY rest.created_at"
            )
            try:
                rows = self._db.execute(query, params)
                for row in rows:
                    if not isinstance(row, dict):
                        continue
                    vals = list(row.values())
                    if len(vals) < 4:
                        continue
                    entry = {
                        "memory_id": str(vals[0]),
                        "text": vals[1],
                        "created_at": vals[2],
                        "session_id": vals[3],
                    }
                    # Avoid duplicates when direction is "both"
                    if not any(r["memory_id"] == entry["memory_id"] for r in results):
                        results.append(entry)
            except Exception:
                logger.debug(
                    "temporal_chain backward query failed for %s",
                    memory_id,
                    exc_info=True,
                )

        # Sort merged results by created_at
        if direction == "both":
            results.sort(key=lambda r: r.get("created_at") or 0)

        return results

    def _group_results_by_session(
        self,
        results: list,
    ) -> dict[str, list]:
        """Group search results by session_id, chronological within each group."""
        from collections import defaultdict

        groups: dict[str, list] = defaultdict(list)
        for r in results:
            groups[getattr(r, "session_id", None) or "default"].append(r)
        for sid in groups:
            groups[sid].sort(key=lambda r: getattr(r, "created_at", 0) or 0)
        return dict(groups)

    def _recompute_graph_metrics(self) -> None:
        """Recompute graph algorithm scores and cache as node properties.

        Runs pagerank, betweenness centrality, and louvain community detection.
        Results are stored as node properties for fast access during search scoring.
        Only runs when enable_graph_algorithms is True and the graph has changed.

        Only Memory and Entity nodes are updated. Algorithm computation may span
        the full graph (GrafeoDB API limitation), so scores can be influenced by
        non-memory topology when using a shared database.
        """
        if not self._config.enable_graph_algorithms or not self._graph_dirty:
            return
        self._graph_dirty = False

        if not hasattr(self._db, "algorithms"):
            logger.debug("Graph algorithms not available on this db instance")
            return

        # Collect IDs of nodes we own so we only write to Memory/Entity nodes.
        owned_ids: set[int] = set()
        for node_id, _ in self._db.get_nodes_by_label(MEMORY_LABEL):
            owned_ids.add(node_id)
        for node_id, _ in self._db.get_nodes_by_label(ENTITY_LABEL):
            owned_ids.add(node_id)
        if not owned_ids:
            return

        algos = self._db.algorithms

        try:
            ranks = algos.pagerank(0.85, 100, 1e-6)  # ty: ignore[unresolved-attribute]
            for node_id, score in ranks.items():
                if node_id in owned_ids:
                    with contextlib.suppress(Exception):
                        self._db.set_node_property(node_id, "_pagerank", score)
        except Exception:
            logger.debug("pagerank computation failed", exc_info=True)

        try:
            centrality = algos.betweenness_centrality(True)  # ty: ignore[unresolved-attribute]
            for node_id, score in centrality.items():
                if node_id in owned_ids:
                    with contextlib.suppress(Exception):
                        self._db.set_node_property(node_id, "_betweenness", score)
        except Exception:
            logger.debug("betweenness_centrality computation failed", exc_info=True)

        try:
            result = algos.louvain(1.0)  # ty: ignore[unresolved-attribute]
            communities = result.get("communities", {}) if isinstance(result, dict) else {}
            for node_id, community_id in communities.items():
                if node_id in owned_ids:
                    with contextlib.suppress(Exception):
                        self._db.set_node_property(node_id, "_community", community_id)
            # Store louvain result for deferred community materialization
            if self._config.enable_community_summaries and communities:
                self._pending_communities = result
        except Exception:
            logger.debug("louvain community detection failed", exc_info=True)

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
            rows = self._db.execute(
                f"MATCH (e:{ENTITY_LABEL}) WHERE e.name = $name AND e.user_id = $uid RETURN id(e)",
                {"name": entity.name, "uid": user_id},
            )
            for row in rows:
                if isinstance(row, dict):
                    vals = list(row.values())
                    if vals:
                        return int(vals[0])
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
                logger.warning(
                    "_get_existing_relations: query failed for entity nid=%s (%d entities total)",
                    nid,
                    len(entity_ids),
                    exc_info=True,
                )
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
        self,
        user_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        include_expired: bool = False,
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
            # Exclude expired unless explicitly requested
            if not include_expired and props.get("expired_at") is not None:
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
                    created_at=props.get("created_at"),
                    expired_at=props.get("expired_at"),
                    valid_at=props.get("valid_at"),
                    invalid_at=props.get("invalid_at"),
                )
            )
        return memories

    def _get_memories_with_timestamps(self, user_id: str) -> list[tuple[str, str, int]]:
        """Get all active (non-expired) memories as (memory_id, text, created_at) sorted oldest-first."""
        try:
            nodes = self._db.get_nodes_by_label(MEMORY_LABEL)
        except Exception:
            return []

        filters = self._build_filters(user_id)
        results: list[tuple[str, str, int]] = []
        for node_id, props in nodes:
            if not all(props.get(k) == v for k, v in filters.items()):
                continue
            if props.get("expired_at") is not None:
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
            consolidated_texts = [m for m in result.output.memories if m]  # ty: ignore[unresolved-attribute]
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
                        logger.warning("Failed to create DERIVED_FROM edge %d->%s", new_mid, mid, exc_info=True)

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

    def _stats_impl(self) -> MemoryStats:
        """Collect database introspection statistics."""
        try:
            memory_nodes = self._db.get_nodes_by_label(MEMORY_LABEL)
        except Exception:
            memory_nodes = []

        semantic = 0
        procedural = 0
        episodic = 0
        for item in memory_nodes:
            # get_nodes_by_label returns list of (id, props_dict) tuples
            props = item[1] if isinstance(item, tuple) else _get_props(item)
            mtype = props.get("memory_type", "semantic")
            if mtype == "procedural":
                procedural += 1
            elif mtype == "episodic":
                episodic += 1
            else:
                semantic += 1

        try:
            entity_nodes = self._db.get_nodes_by_label(ENTITY_LABEL)
            entity_count = len(entity_nodes)
        except Exception:
            entity_count = 0

        relation_count = 0
        try:
            rows = self._db.execute(
                f"MATCH (:{ENTITY_LABEL})-[r:{RELATION_EDGE}]->(:{ENTITY_LABEL}) RETURN count(r)", {}
            )
            for row in rows:
                vals = list(row.values()) if isinstance(row, dict) else [row]
                relation_count = int(vals[0]) if vals else 0
        except Exception:
            pass

        ep_count = 0
        try:
            ep_nodes = self._db.get_nodes_by_label(EPISODE_LABEL)
            ep_count = len(ep_nodes)
        except Exception:
            pass

        comm_count = 0
        try:
            from .types import COMMUNITY_LABEL as _CL

            comm_nodes = self._db.get_nodes_by_label(_CL)
            comm_count = len(comm_nodes)
        except Exception:
            pass

        total = semantic + procedural + episodic
        return MemoryStats(
            total_memories=total,
            semantic_count=semantic,
            procedural_count=procedural,
            episodic_count=episodic,
            entity_count=entity_count,
            relation_count=relation_count,
            episode_count=ep_count,
            community_count=comm_count,
            db_info={
                "memory_node_count": total,
                "entity_node_count": entity_count,
                "relation_edge_count": relation_count,
                "episode_node_count": ep_count,
                "community_node_count": comm_count,
            },
        )

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


def _valid_at_point(result: SearchResult, pit: int) -> bool:
    """Check whether a memory was valid at a given point in time.

    Permissive: memories with no valid_at are always included (backward compat).
    """
    if result.valid_at is None:
        return True
    if result.valid_at > pit:
        return False
    return result.invalid_at is None or result.invalid_at > pit


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
        min_score: float | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        grouped: bool = False,
        point_in_time: int | None = None,
    ) -> SearchResponse | dict[str, list]:
        """Search memories by semantic similarity and graph context."""
        response = run_sync(
            self._search(
                query,
                user_id=user_id,
                k=k,
                filters=filters,
                rerank=rerank,
                memory_type=memory_type,
                min_score=min_score,
                time_before=time_before,
                time_after=time_after,
                include_expired=include_expired,
                diverse=diverse,
                point_in_time=point_in_time,
            )
        )
        if grouped:
            return self._group_results_by_session(response)
        return response

    def update(self, memory_id: str, text: str) -> MemoryEvent:
        """Update a memory's text directly. Re-embeds and records history."""
        return run_sync(self._update(memory_id, text))

    def get_all(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        include_expired: bool = False,
    ) -> list[SearchResult]:
        """Retrieve all memories for a user."""
        return self._get_all_impl(user_id, memory_type=memory_type, include_expired=include_expired)

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

    def temporal_chain(
        self,
        memory_id: str,
        user_id: str | None = None,
        direction: str = "forward",
        max_depth: int = 5,
    ) -> list:
        """Follow LEADS_TO edges to build a temporal chain of memories."""
        return super().temporal_chain(
            memory_id,
            user_id=user_id,
            direction=direction,
            max_depth=max_depth,
        )

    def get_episodes(
        self, user_id: str | None = None, session_id: str | None = None, limit: int = 50
    ) -> list[EpisodeResult]:
        """Get episodes, optionally filtered by session."""
        return self._get_episodes_impl(user_id, session_id=session_id, limit=limit)

    def get_provenance(self, memory_id: str) -> list[EpisodeResult]:
        """Get the episodes that produced a given memory."""
        return self._get_provenance_impl(memory_id)

    def episode_chain(self, episode_id: str, direction: str = "forward", max_depth: int = 10) -> list[EpisodeResult]:
        """Follow NEXT_EPISODE edges for session replay."""
        return self._episode_chain_impl(episode_id, direction=direction, max_depth=max_depth)

    def get_communities(self, user_id: str | None = None) -> list:
        """Get all detected communities and their summaries."""
        from .communities import get_communities as _get_communities

        uid = user_id or self._config.user_id
        return _get_communities(self._db, uid)

    def stats(self) -> MemoryStats:
        """Return database introspection statistics (no LLM calls)."""
        return self._stats_impl()

    def explain(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        memory_type: MemoryType | str | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        point_in_time: int | None = None,
    ) -> ExplainResult:
        """Run a search and return a step-by-step pipeline trace."""
        return run_sync(
            self._explain(
                query,
                user_id=user_id,
                k=k,
                memory_type=memory_type,
                time_before=time_before,
                time_after=time_after,
                include_expired=include_expired,
                diverse=diverse,
                point_in_time=point_in_time,
            )
        )


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
        min_score: float | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        grouped: bool = False,
        point_in_time: int | None = None,
    ) -> SearchResponse | dict[str, list]:
        """Search memories by semantic similarity and graph context."""
        response = await self._search(
            query,
            user_id=user_id,
            k=k,
            filters=filters,
            rerank=rerank,
            memory_type=memory_type,
            min_score=min_score,
            time_before=time_before,
            time_after=time_after,
            include_expired=include_expired,
            diverse=diverse,
            point_in_time=point_in_time,
        )
        if grouped:
            return self._group_results_by_session(response)
        return response

    async def update(self, memory_id: str, text: str) -> MemoryEvent:
        """Update a memory's text directly. Re-embeds and records history."""
        return await self._update(memory_id, text)

    async def get_all(
        self,
        user_id: str | None = None,
        memory_type: MemoryType | str | None = None,
        include_expired: bool = False,
    ) -> list[SearchResult]:
        """Retrieve all memories for a user."""
        return self._get_all_impl(user_id, memory_type=memory_type, include_expired=include_expired)

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

    def get_episodes(
        self, user_id: str | None = None, session_id: str | None = None, limit: int = 50
    ) -> list[EpisodeResult]:
        """Get episodes, optionally filtered by session."""
        return self._get_episodes_impl(user_id, session_id=session_id, limit=limit)

    def get_provenance(self, memory_id: str) -> list[EpisodeResult]:
        """Get the episodes that produced a given memory."""
        return self._get_provenance_impl(memory_id)

    def episode_chain(self, episode_id: str, direction: str = "forward", max_depth: int = 10) -> list[EpisodeResult]:
        """Follow NEXT_EPISODE edges for session replay."""
        return self._episode_chain_impl(episode_id, direction=direction, max_depth=max_depth)

    def get_communities(self, user_id: str | None = None) -> list:
        """Get all detected communities and their summaries."""
        from .communities import get_communities as _get_communities

        uid = user_id or self._config.user_id
        return _get_communities(self._db, uid)

    def stats(self) -> MemoryStats:
        """Return database introspection statistics (no LLM calls)."""
        return self._stats_impl()

    async def explain(
        self,
        query: str,
        user_id: str | None = None,
        k: int = 10,
        *,
        memory_type: MemoryType | str | None = None,
        time_before: int | None = None,
        time_after: int | None = None,
        include_expired: bool = False,
        diverse: bool = False,
        point_in_time: int | None = None,
    ) -> ExplainResult:
        """Run a search and return a step-by-step pipeline trace."""
        return await self._explain(
            query,
            user_id=user_id,
            k=k,
            memory_type=memory_type,
            time_before=time_before,
            time_after=time_after,
            include_expired=include_expired,
            diverse=diverse,
            point_in_time=point_in_time,
        )
