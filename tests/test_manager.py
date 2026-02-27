"""Integration tests for MemoryManager with mock model and real GrafeoDB."""

import asyncio

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryAction, MemoryConfig, MemoryManager, MemoryType


def _make_manager(outputs, dims=16, **config_kwargs):
    """Create a MemoryManager with mock model and in-memory GrafeoDB."""
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)
    return MemoryManager(model, config, embedder=embedder)


def _make_async_manager(outputs, dims=16, **config_kwargs):
    """Create an AsyncMemoryManager with mock model and in-memory GrafeoDB."""
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)
    return AsyncMemoryManager(model, config, embedder=embedder)


class TestMemoryManagerAdd:
    def test_add_new_facts(self):
        manager = _make_manager(
            [
                # combined extraction
                {
                    "facts": ["alice works at acme corp"],
                    "entities": [
                        {"name": "alice", "entity_type": "person"},
                        {"name": "acme_corp", "entity_type": "organization"},
                    ],
                    "relations": [
                        {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
                    ],
                },
                # reconcile (no existing memories, so won't be called — falls through to ADD)
            ]
        )

        events = manager.add("Alice works at Acme Corp")
        assert len(events) >= 1
        assert events[0].action == MemoryAction.ADD
        assert events[0].memory_id is not None
        assert "acme" in events[0].text.lower()

        manager.close()

    def test_add_returns_empty_for_no_facts(self):
        manager = _make_manager(
            [
                {"facts": []},
            ]
        )
        events = manager.add("Hello there!")
        assert events == []
        manager.close()

    def test_add_with_user_id(self):
        manager = _make_manager(
            [
                {"facts": ["bob likes hiking"], "entities": [{"name": "bob", "entity_type": "person"}], "relations": []},
            ]
        )
        events = manager.add("Bob likes hiking", user_id="bob")
        assert len(events) >= 1
        manager.close()

    def test_add_with_message_dict(self):
        """add() should accept a single message dict."""
        manager = _make_manager(
            [
                {"facts": ["alice likes tea"], "entities": [], "relations": []},
            ]
        )
        events = manager.add({"role": "user", "content": "Alice likes tea"})
        assert len(events) == 1
        assert events[0].action == MemoryAction.ADD
        manager.close()

    def test_add_with_message_list(self):
        """add() should accept a list of message dicts."""
        manager = _make_manager(
            [
                {"facts": ["alice works at acme"], "entities": [], "relations": []},
            ]
        )
        events = manager.add(
            [
                {"role": "user", "content": "I work at Acme Corp"},
                {"role": "assistant", "content": "That's great!"},
            ]
        )
        assert len(events) == 1
        manager.close()

    def test_add_with_named_messages(self):
        """add() should extract actor from named messages."""
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"], "entities": [], "relations": []},
            ]
        )
        events = manager.add({"role": "user", "content": "I like hiking", "name": "alice"})
        assert len(events) == 1
        assert events[0].actor_id == "alice"
        assert events[0].role == "user"
        manager.close()


class TestCustomPrompts:
    def test_custom_fact_prompt(self):
        """Custom fact prompt should be passed to the extraction agent."""
        manager = _make_manager(
            [
                {"facts": ["custom extracted fact"], "entities": [], "relations": []},
            ],
            custom_fact_prompt="You are a custom extractor. Extract only food preferences.",
        )
        events = manager.add("Alice likes pizza and works at Acme")
        assert len(events) == 1
        manager.close()


class TestBatchAdd:
    def test_batch_add_raw_mode(self):
        """add_batch(infer=False) should store multiple texts with batched embedding."""
        manager = _make_manager([])
        events = manager.add_batch(
            ["fact one", "fact two", "fact three"],
            infer=False,
        )
        assert len(events) == 3
        assert all(e.action == MemoryAction.ADD for e in events)
        assert events[0].text == "fact one"
        assert events[1].text == "fact two"
        assert events[2].text == "fact three"

        memories = manager.get_all()
        assert len(memories) == 3
        manager.close()

    def test_batch_add_with_infer(self):
        """add_batch(infer=True) should process each message through extraction."""
        manager = _make_manager(
            [
                # First message combined extraction
                {"facts": ["alice likes hiking"], "entities": [], "relations": []},
                # Second message combined extraction
                {"facts": ["bob likes swimming"], "entities": [], "relations": []},
            ]
        )
        events = manager.add_batch(["Alice likes hiking", "Bob likes swimming"])
        assert len(events) == 2
        assert all(e.action == MemoryAction.ADD for e in events)
        manager.close()

    def test_batch_add_records_history(self):
        """add_batch() should record history for each memory."""
        manager = _make_manager([])
        events = manager.add_batch(["fact one", "fact two"], infer=False)
        assert len(events) == 2

        for event in events:
            history = manager.history(event.memory_id)
            assert len(history) == 1
            assert history[0]["event"] == "ADD"
        manager.close()

    def test_batch_add_empty_list(self):
        """add_batch() with empty list should return empty events."""
        manager = _make_manager([])
        events = manager.add_batch([], infer=False)
        assert events == []
        manager.close()

    def test_async_batch_add(self):
        """AsyncMemoryManager.add_batch() should work."""
        manager = _make_async_manager([])

        async def _run():
            events = await manager.add_batch(["async fact 1", "async fact 2"], infer=False)
            assert len(events) == 2
            all_mem = await manager.get_all()
            assert len(all_mem) == 2
            manager.close()

        asyncio.run(_run())


class TestMemoryManagerRawMode:
    def test_add_infer_false(self):
        """infer=False should store text directly without LLM extraction."""
        manager = _make_manager([])  # no LLM outputs needed
        events = manager.add("raw fact: alice is 30 years old", infer=False)
        assert len(events) == 1
        assert events[0].action == MemoryAction.ADD
        assert events[0].text == "raw fact: alice is 30 years old"

        # Should appear in get_all
        memories = manager.get_all()
        assert len(memories) == 1
        assert "alice is 30" in memories[0].text
        manager.close()


class TestMemoryManagerUpdate:
    def test_update_changes_text(self):
        """update() should replace memory text and re-embed."""
        manager = _make_manager(
            [
                {"facts": ["alice works at acme"], "entities": [], "relations": []},
            ]
        )
        events = manager.add("Alice works at Acme Corp")
        memory_id = events[0].memory_id

        event = manager.update(memory_id, "alice works at globex corp")
        assert event.action == MemoryAction.UPDATE
        assert event.text == "alice works at globex corp"
        assert event.old_text == "alice works at acme"

        # Verify updated text in DB
        memories = manager.get_all()
        assert any("globex" in m.text for m in memories)
        manager.close()

    def test_update_records_history(self):
        """update() should record a history entry."""
        manager = _make_manager(
            [
                {"facts": ["alice works at acme"], "entities": [], "relations": []},
            ]
        )
        events = manager.add("Alice works at Acme Corp")
        memory_id = events[0].memory_id

        manager.update(memory_id, "alice works at globex")

        history = manager.history(memory_id)
        assert len(history) == 2  # ADD + UPDATE
        assert history[0]["event"] == "ADD"
        assert history[1]["event"] == "UPDATE"
        assert history[1]["old_text"] == "alice works at acme"
        assert history[1]["new_text"] == "alice works at globex"
        manager.close()


class TestMemoryManagerSearch:
    def test_search_returns_results(self):
        manager = _make_manager(
            [
                # add call: combined extraction
                {"facts": ["test_user prefers python"], "entities": [], "relations": []},
                # search will use vector_search internally + graph search (entity extraction)
                {"entities": [], "relations": []},
            ]
        )
        manager.add("I prefer Python for data science")

        results = manager.search("programming language preference")
        # Results depend on vector similarity of mock embeddings
        # At minimum, the search should not error
        assert isinstance(results, list)
        manager.close()

    def test_search_empty_db(self):
        manager = _make_manager(
            [
                # graph_search entity extraction (returns no entities)
                {"entities": [], "relations": []},
            ]
        )
        results = manager.search("anything")
        assert results == []
        manager.close()


class TestMemoryManagerDelete:
    def test_delete_memory(self):
        manager = _make_manager(
            [
                {"facts": ["carol likes tea"], "entities": [], "relations": []},
            ]
        )
        events = manager.add("Carol likes tea")
        assert len(events) == 1

        memory_id = events[0].memory_id
        assert memory_id is not None

        deleted = manager.delete(memory_id)
        assert deleted is True

        # Verify node is gone
        deleted_again = manager.delete(memory_id)
        assert deleted_again is False

        manager.close()

    def test_delete_invalid_id(self):
        manager = _make_manager([])
        assert manager.delete("not_a_number") is False
        manager.close()


class TestMemoryManagerGetAll:
    def test_get_all_empty(self):
        manager = _make_manager([])
        results = manager.get_all()
        assert results == []
        manager.close()


class TestMemoryManagerHistory:
    def test_history_after_add(self):
        """add() should create a history entry."""
        manager = _make_manager(
            [
                {"facts": ["test fact"], "entities": [], "relations": []},
            ]
        )
        events = manager.add("test fact")
        memory_id = events[0].memory_id

        history = manager.history(memory_id)
        assert len(history) == 1
        assert history[0]["event"] == "ADD"
        assert history[0]["new_text"] == "test fact"
        manager.close()

    def test_history_empty_for_nonexistent(self):
        manager = _make_manager([])
        history = manager.history("99999")
        assert history == []
        manager.close()

    def test_history_invalid_id(self):
        manager = _make_manager([])
        history = manager.history("not_a_number")
        assert history == []
        manager.close()


class TestMemoryManagerContextManager:
    def test_context_manager(self):
        model = make_test_model(
            [
                {"facts": ["context manager works"], "entities": [], "relations": []},
            ]
        )
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, embedding_dimensions=16)
        with MemoryManager(model, config, embedder=embedder) as memory:
            events = memory.add("context manager works")
            assert len(events) >= 1


class TestHasEntityEdges:
    def test_has_entity_edges_created(self):
        """After add(), Memory nodes should be linked to Entity nodes via HAS_ENTITY."""
        manager = _make_manager(
            [
                # combined extraction
                {
                    "facts": ["alice works at acme corp"],
                    "entities": [
                        {"name": "alice", "entity_type": "person"},
                        {"name": "acme_corp", "entity_type": "organization"},
                    ],
                    "relations": [
                        {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
                    ],
                },
            ]
        )
        events = manager.add("Alice works at Acme Corp")
        assert len(events) == 1
        memory_id = int(events[0].memory_id)

        # Query for HAS_ENTITY edges from the memory node
        result = manager._db.execute(
            "MATCH (m:Memory)-[:HAS_ENTITY]->(e:Entity) WHERE id(m) = $mid RETURN e.name",
            {"mid": memory_id},
        )
        entity_names = [next(iter(row.values())) for row in result if isinstance(row, dict)]
        assert len(entity_names) == 2
        assert "alice" in entity_names
        assert "acme_corp" in entity_names

        manager.close()

    def test_entity_dedup(self):
        """Adding the same entity name twice should reuse the existing node."""
        manager = _make_manager(
            [
                # First add: combined extraction
                {"facts": ["alice likes hiking"], "entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
                # Second add: combined extraction
                {"facts": ["alice likes cooking"], "entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
            ]
        )
        manager.add("Alice likes hiking")
        manager.add("Alice likes cooking")

        # Should be only one Entity node named "alice"
        nodes = manager._db.find_nodes_by_property("name", "alice")
        entity_nodes = []
        for nid in nodes:
            node = manager._db.get_node(nid)
            if node and "Entity" in (node.labels if hasattr(node, "labels") else []):
                entity_nodes.append(nid)
        assert len(entity_nodes) == 1

        manager.close()


class TestGraphSearch:
    def test_graph_search_finds_memories(self):
        """Graph search should find memories linked to queried entities."""
        manager = _make_manager(
            [
                # add() call: combined extraction
                {
                    "facts": ["alice works at acme corp"],
                    "entities": [
                        {"name": "alice", "entity_type": "person"},
                        {"name": "acme_corp", "entity_type": "organization"},
                    ],
                    "relations": [
                        {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
                    ],
                },
                # search() -> _graph_search() -> extract_entities (entity extraction from query)
                {"entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
            ]
        )

        manager.add("Alice works at Acme Corp")
        results = manager.search("Tell me about alice")

        # Graph search should find the memory via HAS_ENTITY -> alice
        assert len(results) >= 1
        texts = [r.text for r in results]
        assert any("acme" in t.lower() for t in texts)

        manager.close()

    def test_score_is_similarity(self):
        """Search scores should be similarity (higher = more relevant)."""
        manager = _make_manager(
            [
                {"facts": ["test_user prefers python"]},
                {"entities": [], "relations": []},
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("I prefer Python for data science")

        results = manager.search("python preference")
        for r in results:
            # Similarity should be between 0 and 1 (converted from distance)
            assert 0.0 <= r.score <= 1.0

        manager.close()


class TestHybridSearch:
    def test_search_uses_hybrid_when_available(self):
        """search() should use hybrid search (BM25 + vector) by default."""
        manager = _make_manager(
            [
                {"facts": ["alice works at acme"]},
                {"entities": [], "relations": []},
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("Alice works at Acme Corp")

        # hybrid_search falls back to vector_search internally when db supports it
        # Just verify search still works end-to-end
        results = manager.search("acme corp work")
        assert isinstance(results, list)
        manager.close()

    def test_search_fallback_without_hybrid(self):
        """hybrid_search() should fall back to vector-only if db lacks hybrid_search."""
        from grafeo_memory.search.vector import hybrid_search as hs_fn

        manager = _make_manager(
            [
                {"facts": ["bob likes hiking"]},
                {"entities": [], "relations": []},
            ]
        )
        manager.add("Bob likes hiking")

        # Use a wrapper that hides the hybrid_search method
        class _NoHybridDb:
            def __init__(self, db):
                self._db = db

            def __getattr__(self, name):
                if name == "hybrid_search":
                    raise AttributeError
                return getattr(self._db, name)

            def __hasattr__(self, name):
                if name == "hybrid_search":
                    return False
                return hasattr(self._db, name)

        wrapped_db = _NoHybridDb(manager._db)
        assert not hasattr(wrapped_db, "hybrid_search")

        results = hs_fn(wrapped_db, MockEmbedder(16), "hiking", user_id="test_user")
        assert isinstance(results, list)
        manager.close()

    def test_hybrid_search_respects_user_isolation(self):
        """Hybrid search should respect user_id filtering."""
        manager = _make_manager(
            [
                {"facts": ["alice likes tea"]},
                {"entities": [], "relations": []},
                {"facts": ["bob likes coffee"]},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )
        manager.add("Alice likes tea", user_id="user1")
        manager.add("Bob likes coffee", user_id="user2")

        results = manager.search("drinks", user_id="user1")
        for r in results:
            assert r.user_id == "user1"
        manager.close()


class TestAdvancedFilters:
    def test_filter_with_gt_operator(self):
        """search() should support $gt filter on created_at."""
        manager = _make_manager(
            [
                {"facts": ["old fact"]},
                {"entities": [], "relations": []},
                {"facts": ["new fact"]},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )
        manager.add("Old fact")
        import time

        time.sleep(0.01)
        cutoff = int(time.time() * 1000)
        time.sleep(0.01)
        manager.add("New fact")

        results = manager.search("fact", filters={"created_at": {"$gt": cutoff}})
        # Only the new fact should match
        for r in results:
            assert "new" in r.text.lower()
        manager.close()

    def test_filter_with_in_operator(self):
        """search() should support $in filter."""
        manager = _make_manager([])
        manager.add({"role": "user", "content": "alpha fact", "name": "alice"}, infer=False)
        manager.add({"role": "assistant", "content": "beta fact", "name": "bob"}, infer=False)
        manager.add({"role": "user", "content": "gamma fact", "name": "charlie"}, infer=False)

        # Filter for specific actors — this goes through vector_search which supports $in natively
        results = manager.search("fact", filters={"actor_id": {"$in": ["alice", "charlie"]}})
        for r in results:
            assert r.actor_id in ("alice", "charlie")
        manager.close()

    def test_matches_filters_unit(self):
        """_matches_filters should handle all operator types."""
        from grafeo_memory.search.vector import _matches_filters

        props = {"age": 30, "name": "alice", "score": 0.9}

        assert _matches_filters(props, {"age": 30})
        assert not _matches_filters(props, {"age": 25})
        assert _matches_filters(props, {"age": {"$gt": 25}})
        assert not _matches_filters(props, {"age": {"$gt": 35}})
        assert _matches_filters(props, {"age": {"$gte": 30}})
        assert _matches_filters(props, {"age": {"$lt": 35}})
        assert not _matches_filters(props, {"age": {"$lt": 25}})
        assert _matches_filters(props, {"age": {"$lte": 30}})
        assert _matches_filters(props, {"age": {"$ne": 25}})
        assert not _matches_filters(props, {"age": {"$ne": 30}})
        assert _matches_filters(props, {"name": {"$in": ["alice", "bob"]}})
        assert not _matches_filters(props, {"name": {"$in": ["bob", "carol"]}})
        assert _matches_filters(props, {"name": {"$nin": ["bob", "carol"]}})
        assert not _matches_filters(props, {"name": {"$nin": ["alice", "bob"]}})
        assert _matches_filters(props, {"name": {"$contains": "lic"}})
        assert not _matches_filters(props, {"name": {"$contains": "xyz"}})


class TestMultiUserIsolation:
    def test_user_isolation(self):
        """Memories from different users should not interfere."""
        manager = _make_manager(
            [
                # user1 add
                {"facts": ["alice likes hiking"]},
                {"entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
                # user2 add
                {"facts": ["bob likes swimming"]},
                {"entities": [{"name": "bob", "entity_type": "person"}], "relations": []},
            ]
        )

        manager.add("Alice likes hiking", user_id="user1")
        manager.add("Bob likes swimming", user_id="user2")

        user1_memories = manager.get_all(user_id="user1")
        user2_memories = manager.get_all(user_id="user2")

        assert len(user1_memories) == 1
        assert "hiking" in user1_memories[0].text
        assert len(user2_memories) == 1
        assert "swimming" in user2_memories[0].text

        manager.close()

    def test_delete_all_scoped(self):
        """delete_all should only remove the specified user's memories."""
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"]},
                {"entities": [], "relations": []},
                {"facts": ["bob likes swimming"]},
                {"entities": [], "relations": []},
            ]
        )

        manager.add("Alice likes hiking", user_id="user1")
        manager.add("Bob likes swimming", user_id="user2")

        deleted = manager.delete_all(user_id="user1")
        assert deleted == 1

        assert len(manager.get_all(user_id="user1")) == 0
        assert len(manager.get_all(user_id="user2")) == 1

        manager.close()


class TestScoping:
    def test_agent_id_scoping(self):
        """Memories should be scoped by agent_id when set."""
        manager = _make_manager(
            [
                {"facts": ["fact from agent1"]},
                {"entities": [], "relations": []},
            ],
            agent_id="agent1",
        )
        events = manager.add("fact from agent1")
        assert len(events) == 1

        # get_all should return memories with matching agent_id
        memories = manager.get_all()
        assert len(memories) == 1

        manager.close()

    def test_run_id_scoping(self):
        """Memories should be scoped by run_id when set."""
        manager = _make_manager(
            [
                {"facts": ["fact from run1"]},
                {"entities": [], "relations": []},
            ],
            run_id="run_001",
        )
        events = manager.add("fact from run1")
        assert len(events) == 1

        memories = manager.get_all()
        assert len(memories) == 1

        manager.close()


class TestActorTracking:
    def test_actor_in_search_results(self):
        """Search results should include actor_id and role."""
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"]},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )
        manager.add({"role": "user", "content": "I like hiking", "name": "alice"})

        all_memories = manager.get_all()
        assert len(all_memories) == 1
        assert all_memories[0].actor_id == "alice"
        assert all_memories[0].role == "user"
        manager.close()


class TestSummarize:
    def test_summarize_consolidates_old_memories(self):
        """summarize() should consolidate old memories and keep recent ones."""
        manager = _make_manager(
            [
                # summarize LLM call
                {"memories": ["alice works at acme corp as a data scientist who prefers python"]},
            ]
        )
        for text in [
            "alice works at acme corp",
            "alice is a data scientist",
            "alice prefers python",
            "alice likes hiking",
            "alice has a dog named max",
            "alice moved to sf",
        ]:
            manager.add(text, infer=False)
            import time

            time.sleep(0.002)

        assert len(manager.get_all()) == 6

        events = manager.summarize(preserve_recent=2, batch_size=20)

        deletes = [e for e in events if e.action == MemoryAction.DELETE]
        adds = [e for e in events if e.action == MemoryAction.ADD]
        assert len(deletes) == 4  # 6 - 2 preserved
        assert len(adds) >= 1

        remaining = manager.get_all()
        assert len(remaining) == 2 + len(adds)
        manager.close()

    def test_summarize_nothing_to_consolidate(self):
        """summarize() should return empty when all memories are recent."""
        manager = _make_manager([])
        manager.add("fact one", infer=False)
        manager.add("fact two", infer=False)

        events = manager.summarize(preserve_recent=5)
        assert events == []
        assert len(manager.get_all()) == 2
        manager.close()

    def test_summarize_empty_db(self):
        """summarize() on empty database should return empty list."""
        manager = _make_manager([])
        events = manager.summarize()
        assert events == []
        manager.close()

    def test_summarize_preserves_recent(self):
        """summarize() should not touch the most recent memories."""
        manager = _make_manager(
            [
                {"memories": ["consolidated old facts"]},
            ]
        )
        import time

        for text in ["old1", "old2", "old3", "recent1", "recent2"]:
            manager.add(text, infer=False)
            time.sleep(0.002)

        manager.summarize(preserve_recent=2, batch_size=20)

        remaining = manager.get_all()
        remaining_texts = [m.text for m in remaining]
        assert "recent1" in remaining_texts
        assert "recent2" in remaining_texts
        # old1, old2, old3 should be gone
        assert "old1" not in remaining_texts
        assert "old2" not in remaining_texts
        assert "old3" not in remaining_texts
        manager.close()

    def test_summarize_records_history(self):
        """summarize() should record history for new summary memories."""
        manager = _make_manager(
            [
                {"memories": ["consolidated memory"]},
            ]
        )
        manager.add("old fact 1", infer=False)
        manager.add("old fact 2", infer=False)
        manager.add("recent fact", infer=False)

        events = manager.summarize(preserve_recent=1)

        adds = [e for e in events if e.action == MemoryAction.ADD]
        assert len(adds) >= 1
        history = manager.history(adds[0].memory_id)
        assert len(history) >= 1
        assert history[0]["event"] == "ADD"
        manager.close()

    def test_summarize_batching(self):
        """summarize() should process memories in batches."""
        manager = _make_manager(
            [
                {"memories": ["batch1 summary"]},
                {"memories": ["batch2 summary"]},
            ]
        )
        import time

        for i in range(6):
            manager.add(f"fact {i}", infer=False)
            time.sleep(0.002)

        events = manager.summarize(preserve_recent=2, batch_size=2)

        adds = [e for e in events if e.action == MemoryAction.ADD]
        assert len(adds) == 2  # one summary per batch
        manager.close()

    def test_summarize_llm_failure_skips_batch(self):
        """If LLM fails for a batch, that batch is skipped and originals preserved."""
        from mock_llm import MockEmbedder, make_error_model

        model = make_error_model()
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=embedder)

        for i in range(4):
            # Raw mode — no LLM needed for add
            run_sync = __import__("grafeo_memory._compat", fromlist=["run_sync"]).run_sync
            run_sync(manager._add(f"fact {i}", infer=False))

        events = manager.summarize(preserve_recent=1)
        assert events == []
        assert len(manager.get_all()) == 4
        manager.close()

    def test_async_summarize(self):
        """AsyncMemoryManager.summarize() should work."""
        manager = _make_async_manager(
            [
                {"memories": ["async summary"]},
            ]
        )

        async def _run():
            await manager.add("old 1", infer=False)
            await manager.add("old 2", infer=False)
            await manager.add("recent", infer=False)

            events = await manager.summarize(preserve_recent=1)
            adds = [e for e in events if e.action == MemoryAction.ADD]
            assert len(adds) >= 1
            manager.close()

        asyncio.run(_run())


class TestImportanceScoring:
    def test_disabled_by_default(self):
        """With enable_importance=False, search results have no importance/access_count."""
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"]},
                {"entities": [], "relations": []},
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("Alice likes hiking")
        results = manager.search("hiking")
        for r in results:
            assert r.importance is None
            assert r.access_count is None
        manager.close()

    def test_enabled_populates_fields(self):
        """With enable_importance=True, results have importance and access_count."""
        manager = _make_manager(
            [
                # graph search entity extraction
                {"entities": [], "relations": []},
            ],
            enable_importance=True,
        )
        manager.add("alice likes hiking", infer=False)
        manager._ensure_indexes()  # re-index after adding embeddings

        results = manager.search("alice likes hiking")
        assert len(results) >= 1
        assert results[0].importance is not None
        assert results[0].access_count is not None
        manager.close()

    def test_importance_stored_on_creation(self):
        """add(importance=0.5) should store importance on the node."""
        manager = _make_manager([], enable_importance=True)
        events = manager.add("critical fact", infer=False, importance=0.5)
        assert len(events) == 1

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert float(props["importance"]) == 0.5
        assert int(props["access_count"]) == 0
        manager.close()

    def test_importance_default_is_one(self):
        """add() without importance should store importance=1.0."""
        manager = _make_manager([], enable_importance=True)
        events = manager.add("normal fact", infer=False)

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert float(props["importance"]) == 1.0
        manager.close()

    def test_access_count_incremented_on_search(self):
        """search() should increment access_count on returned memories."""
        manager = _make_manager(
            [
                # graph search entity extraction (2 search calls)
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ],
            enable_importance=True,
        )
        events = manager.add("alice likes hiking", infer=False)
        memory_id = events[0].memory_id
        manager._ensure_indexes()  # re-index after adding embeddings

        manager.search("alice likes hiking")
        manager.search("alice likes hiking")

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(memory_id))
        props = _get_props(node)
        assert int(props["access_count"]) == 2
        manager.close()

    def test_set_importance(self):
        """set_importance() should update the importance property."""
        manager = _make_manager([], enable_importance=True)
        events = manager.add("some fact", infer=False)
        memory_id = events[0].memory_id

        result = manager.set_importance(memory_id, 0.3)
        assert result is True

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(memory_id))
        props = _get_props(node)
        assert float(props["importance"]) == 0.3
        manager.close()

    def test_set_importance_invalid_range(self):
        """set_importance() should raise ValueError for out-of-range values."""
        manager = _make_manager([], enable_importance=True)
        events = manager.add("fact", infer=False)
        memory_id = events[0].memory_id

        import pytest

        with pytest.raises(ValueError):
            manager.set_importance(memory_id, 1.5)
        with pytest.raises(ValueError):
            manager.set_importance(memory_id, -0.1)
        manager.close()

    def test_set_importance_nonexistent_id(self):
        """set_importance() should return False for non-existent ID."""
        manager = _make_manager([])
        assert manager.set_importance("99999", 0.5) is False
        manager.close()

    def test_set_importance_invalid_id(self):
        """set_importance() should return False for non-numeric ID."""
        manager = _make_manager([])
        assert manager.set_importance("abc", 0.5) is False
        manager.close()

    def test_recency_affects_ranking(self):
        """Newer memories should rank higher when similarity is equal."""
        import time

        manager = _make_manager(
            [{"entities": [], "relations": []}],
            enable_importance=True,
        )
        # Add two memories with the same text (same similarity)
        manager.add("fact about python", infer=False)
        manager.add("fact about python", infer=False)
        manager._ensure_indexes()  # re-index after adding embeddings

        # Make the first memory appear 30 days old
        all_mems = manager.get_all()
        assert len(all_mems) == 2
        old_id = int(all_mems[0].memory_id)
        now_ms = int(time.time() * 1000)
        thirty_days_ago = now_ms - (30 * 24 * 60 * 60 * 1000)
        manager._db.set_node_property(old_id, "created_at", thirty_days_ago)

        results = manager.search("fact about python")
        # Both have same similarity, but the newer one should rank higher
        assert len(results) == 2
        new_id = int(all_mems[1].memory_id)
        assert int(results[0].memory_id) == new_id
        manager.close()

    def test_raw_mode_with_importance(self):
        """add(infer=False, importance=0.8) should store importance."""
        manager = _make_manager([], enable_importance=True)
        events = manager.add("raw fact", infer=False, importance=0.8)

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert float(props["importance"]) == 0.8
        manager.close()

    def test_backward_compat_old_memories(self):
        """Memories without importance props should get default values."""
        # Create manager without importance, add a memory
        manager = _make_manager(
            [{"entities": [], "relations": []}],
        )
        manager.add("old memory", infer=False)
        manager._ensure_indexes()  # re-index after adding embeddings

        # Now enable importance and search — should use defaults (1.0, 0)
        manager._config.enable_importance = True
        results = manager.search("old memory")
        if results:
            assert results[0].importance == 1.0
            assert results[0].access_count == 0
        manager.close()

    def test_async_set_importance(self):
        """AsyncMemoryManager.set_importance() should work."""
        manager = _make_async_manager([], enable_importance=True)

        async def _run():
            events = await manager.add("async fact", infer=False)
            memory_id = events[0].memory_id
            result = manager.set_importance(memory_id, 0.6)
            assert result is True

            from grafeo_memory.search.vector import _get_props

            node = manager._db.get_node(int(memory_id))
            props = _get_props(node)
            assert float(props["importance"]) == 0.6
            manager.close()

        asyncio.run(_run())


class TestAsyncManager:
    def test_async_add_and_search(self):
        """AsyncMemoryManager should work with asyncio.run."""
        manager = _make_async_manager(
            [
                {"facts": ["alice likes hiking"]},
                {"entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )

        async def _run():
            events = await manager.add("Alice likes hiking")
            assert len(events) == 1
            assert events[0].action == MemoryAction.ADD

            results = await manager.search("hiking")
            assert isinstance(results, list)

            all_mem = await manager.get_all()
            assert len(all_mem) == 1

            manager.close()

        asyncio.run(_run())

    def test_async_context_manager(self):
        """AsyncMemoryManager should work as async context manager."""
        model = make_test_model(
            [
                {"facts": ["async context test"]},
                {"entities": [], "relations": []},
            ]
        )
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, embedding_dimensions=16)

        async def _run():
            async with AsyncMemoryManager(model, config, embedder=embedder) as memory:
                events = await memory.add("async context test")
                assert len(events) >= 1

        asyncio.run(_run())

    def test_async_update(self):
        """AsyncMemoryManager.update() should work."""
        manager = _make_async_manager(
            [
                {"facts": ["alice works at acme"]},
                {"entities": [], "relations": []},
            ]
        )

        async def _run():
            events = await manager.add("alice works at acme")
            memory_id = events[0].memory_id

            event = await manager.update(memory_id, "alice works at globex")
            assert event.action == MemoryAction.UPDATE
            assert event.text == "alice works at globex"

            history = await manager.history(memory_id)
            assert len(history) == 2
            manager.close()

        asyncio.run(_run())

    def test_async_raw_mode(self):
        """AsyncMemoryManager should support infer=False."""
        manager = _make_async_manager([])

        async def _run():
            events = await manager.add("raw async fact", infer=False)
            assert len(events) == 1
            assert events[0].text == "raw async fact"
            manager.close()

        asyncio.run(_run())


class TestProceduralMemory:
    def test_add_procedural_memory(self):
        """add(memory_type='procedural') should extract and store with correct type."""
        manager = _make_manager(
            [
                # extract_facts (procedural prompt used internally)
                {"facts": ["always use formal tone in responses"]},
                # extract_entities
                {"entities": [], "relations": []},
            ]
        )
        events = manager.add(
            "Always use formal tone in responses",
            memory_type="procedural",
        )
        assert len(events) == 1
        assert events[0].action == MemoryAction.ADD
        assert events[0].memory_type == "procedural"

        # Verify node has memory_type property stored
        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert props["memory_type"] == "procedural"
        manager.close()

    def test_add_procedural_raw_mode(self):
        """add(infer=False, memory_type='procedural') should store type on node."""
        manager = _make_manager([])
        events = manager.add("use pytest for testing", infer=False, memory_type="procedural")
        assert len(events) == 1
        assert events[0].memory_type == "procedural"

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert props["memory_type"] == "procedural"
        manager.close()

    def test_add_default_is_semantic(self):
        """add() without memory_type should store 'semantic'."""
        manager = _make_manager([])
        events = manager.add("alice works at acme", infer=False)
        assert len(events) == 1
        assert events[0].memory_type == "semantic"

        from grafeo_memory.search.vector import _get_props

        node = manager._db.get_node(int(events[0].memory_id))
        props = _get_props(node)
        assert props["memory_type"] == "semantic"
        manager.close()

    def test_procedural_scoped_reconciliation(self):
        """Procedural add should only reconcile against procedural memories.

        Semantic and procedural memories with similar text should not interfere.
        """
        manager = _make_manager(
            [
                # First add (semantic): extraction
                {"facts": ["use python for data science"]},
                {"entities": [], "relations": []},
                # Second add (procedural): extraction
                # Since reconciliation is scoped, the semantic memory won't be found
                # as a candidate, so the procedural fact will be ADD'd
                {"facts": ["always use python for data science"]},
                {"entities": [], "relations": []},
            ]
        )
        # Add semantic memory
        sem_events = manager.add("Use python for data science")
        assert len(sem_events) == 1

        # Add procedural memory with similar text
        proc_events = manager.add(
            "Always use python for data science",
            memory_type="procedural",
        )
        assert len(proc_events) == 1
        assert proc_events[0].action == MemoryAction.ADD

        # Both should exist
        all_memories = manager.get_all()
        assert len(all_memories) == 2
        manager.close()

    def test_search_all_types_by_default(self):
        """search() with no memory_type should return both types."""
        manager = _make_manager(
            [
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("alice works at acme", infer=False)
        manager.add("always use type hints", infer=False, memory_type="procedural")
        manager._ensure_indexes()

        results = manager.search("anything")
        assert len(results) == 2
        types = {r.memory_type for r in results}
        assert types == {"semantic", "procedural"}
        manager.close()

    def test_search_filter_procedural(self):
        """search(memory_type='procedural') should return only procedural memories."""
        manager = _make_manager(
            [
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("alice works at acme", infer=False)
        manager.add("always use type hints", infer=False, memory_type="procedural")
        manager.add("prefer pytest over unittest", infer=False, memory_type="procedural")
        manager._ensure_indexes()

        results = manager.search("coding", memory_type="procedural")
        assert len(results) == 2
        assert all(r.memory_type == "procedural" for r in results)
        manager.close()

    def test_search_filter_semantic(self):
        """search(memory_type='semantic') should return only semantic memories."""
        manager = _make_manager(
            [
                # graph search entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("alice works at acme", infer=False)
        manager.add("always use type hints", infer=False, memory_type="procedural")
        manager._ensure_indexes()

        results = manager.search("anything", memory_type="semantic")
        assert len(results) == 1
        assert results[0].memory_type == "semantic"
        manager.close()

    def test_get_all_filter_by_type(self):
        """get_all(memory_type='procedural') should filter correctly."""
        manager = _make_manager([])
        manager.add("alice works at acme", infer=False)
        manager.add("always use type hints", infer=False, memory_type="procedural")
        manager.add("prefer pytest", infer=False, memory_type="procedural")

        proc = manager.get_all(memory_type="procedural")
        assert len(proc) == 2
        assert all(m.memory_type == "procedural" for m in proc)

        sem = manager.get_all(memory_type="semantic")
        assert len(sem) == 1
        assert sem[0].memory_type == "semantic"

        all_mem = manager.get_all()
        assert len(all_mem) == 3
        manager.close()

    def test_memory_event_includes_type(self):
        """MemoryEvent.memory_type should be set for both semantic and procedural."""
        manager = _make_manager(
            [
                {"facts": ["alice works at acme"]},
                {"entities": [], "relations": []},
                {"facts": ["always use formal tone"]},
                {"entities": [], "relations": []},
            ]
        )
        sem_events = manager.add("Alice works at Acme")
        proc_events = manager.add("Always use formal tone", memory_type="procedural")

        assert sem_events[0].memory_type == "semantic"
        assert proc_events[0].memory_type == "procedural"
        manager.close()

    def test_search_result_includes_type(self):
        """SearchResult.memory_type should be populated from node property."""
        manager = _make_manager(
            [
                {"entities": [], "relations": []},
            ]
        )
        manager.add("prefer dark mode", infer=False, memory_type="procedural")
        manager._ensure_indexes()

        results = manager.search("dark mode")
        assert len(results) >= 1
        assert results[0].memory_type == "procedural"
        manager.close()

    def test_backward_compat_old_memories(self):
        """Old nodes without memory_type property should be treated as semantic."""
        manager = _make_manager(
            [
                {"entities": [], "relations": []},
            ]
        )
        # Add a memory — it has memory_type="semantic" by default
        manager.add("old memory without type", infer=False)

        # Verify get_all with memory_type="semantic" includes it
        # get_all with memory_type="semantic" should include it
        sem = manager.get_all(memory_type="semantic")
        assert len(sem) == 1

        # search should also work
        manager._ensure_indexes()
        results = manager.search("old memory", memory_type="semantic")
        assert len(results) >= 1
        manager.close()

    def test_batch_add_procedural(self):
        """add_batch(memory_type='procedural') should store all as procedural."""
        manager = _make_manager([])
        events = manager.add_batch(
            ["always use type hints", "prefer pytest over unittest"],
            infer=False,
            memory_type="procedural",
        )
        assert len(events) == 2
        assert all(e.memory_type == "procedural" for e in events)

        proc = manager.get_all(memory_type="procedural")
        assert len(proc) == 2
        manager.close()

    def test_memory_type_string_accepted(self):
        """String 'procedural' should work the same as MemoryType.PROCEDURAL enum."""
        manager = _make_manager([])
        str_events = manager.add("rule one", infer=False, memory_type="procedural")
        enum_events = manager.add("rule two", infer=False, memory_type=MemoryType.PROCEDURAL)

        assert str_events[0].memory_type == "procedural"
        assert enum_events[0].memory_type == "procedural"
        manager.close()

    def test_async_procedural(self):
        """AsyncMemoryManager should support memory_type parameter."""
        manager = _make_async_manager([])

        async def _run():
            events = await manager.add("always use type hints", infer=False, memory_type="procedural")
            assert len(events) == 1
            assert events[0].memory_type == "procedural"

            all_proc = await manager.get_all(memory_type="procedural")
            assert len(all_proc) == 1
            assert all_proc[0].memory_type == "procedural"

            all_sem = await manager.get_all(memory_type="semantic")
            assert len(all_sem) == 0
            manager.close()

        asyncio.run(_run())

    def test_custom_procedural_prompt(self):
        """custom_procedural_prompt in config should override the default procedural prompt."""
        manager = _make_manager(
            [
                {"facts": ["custom procedural extraction"]},
                {"entities": [], "relations": []},
            ],
            custom_procedural_prompt="You are a custom procedural extractor. Extract only deployment rules.",
        )
        events = manager.add(
            "When deploying, always use us-east-1 region",
            memory_type="procedural",
        )
        assert len(events) == 1
        assert events[0].memory_type == "procedural"
        manager.close()
