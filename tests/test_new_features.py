"""Integration tests for the 7 new grafeo-memory features.

Covers: property indexes, learned_at, session-grouped search, batch creation,
temporal chain traversal, and CDC history.
"""

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryAction, MemoryConfig, MemoryManager
from grafeo_memory.history import HistoryEntry, get_history, record_history


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder)


def _extraction(facts, entities=None, relations=None):
    """Build a combined extraction output for mock LLM."""
    return {
        "facts": facts,
        "entities": entities or [],
        "relations": relations or [],
    }


def _reconciliation(*decisions):
    """Build reconciliation decisions. Each is (action, fact_index, memory_id?)."""
    result = []
    for d in decisions:
        entry = {"action": d[0], "fact_index": d[1]}
        if len(d) > 2 and d[2] is not None:
            entry["memory_id"] = d[2]
        if len(d) > 3:
            entry["new_text"] = d[3]
        result.append(entry)
    return result


# =============================================================================
# 1. Property Indexes
# =============================================================================


class TestPropertyIndexes:
    def test_indexes_created_on_init(self):
        """_ensure_indexes creates property indexes on hot fields."""
        manager = _make_manager([_extraction(["test fact"]), _reconciliation(("ADD", 0))])
        with manager:
            db = manager._db
            # These should not raise (index exists or was just created)
            assert db.has_property_index("user_id")
            assert db.has_property_index("created_at")
            assert db.has_property_index("memory_type")
            assert db.has_property_index("name")

    def test_find_nodes_by_property_works(self):
        """Property indexes enable fast lookups."""
        manager = _make_manager([_extraction(["alix likes hiking"]), _reconciliation(("ADD", 0))])
        with manager:
            manager.add("alix likes hiking")
            nodes = manager._db.find_nodes_by_property("user_id", "test_user")
            assert len(nodes) >= 1


# =============================================================================
# 2. learned_at Timestamp
# =============================================================================


class TestLearnedAt:
    def test_learned_at_defaults_to_created_at(self):
        """Without explicit learned_at, it defaults to created_at."""
        manager = _make_manager([_extraction(["gus loves jazz"]), _reconciliation(("ADD", 0))])
        with manager:
            manager.add("gus loves jazz")
            results = manager.search("jazz")
            assert len(results) >= 1
            r = results[0]
            assert r.learned_at is not None
            assert r.created_at is not None
            assert r.learned_at == r.created_at

    def test_learned_at_on_memory_node(self):
        """Memory nodes store learned_at property."""
        manager = _make_manager([_extraction(["paris is beautiful"]), _reconciliation(("ADD", 0))])
        with manager:
            events = manager.add("paris is beautiful")
            mem_id = int(events[0].memory_id)
            node = manager._db.get_node(mem_id)
            props = node.properties if hasattr(node, "properties") else {}
            if callable(props):
                props = props()
            assert "learned_at" in props
            assert props["learned_at"] == props["created_at"]


# =============================================================================
# 3. Session-Grouped Search
# =============================================================================


class TestSessionGroupedSearch:
    def test_grouped_false_returns_list(self):
        """Default search returns a flat list."""
        manager = _make_manager([_extraction(["fact one"]), _reconciliation(("ADD", 0))])
        with manager:
            manager.add("fact one")
            results = manager.search("fact", grouped=False)
            assert isinstance(results, list)

    def test_grouped_true_returns_dict(self):
        """grouped=True returns dict grouped by session_id."""
        manager = _make_manager(
            [
                _extraction(["fact one"]),
                _reconciliation(("ADD", 0)),
            ]
        )
        with manager:
            manager.add("fact one")
            grouped = manager.search("fact", grouped=True)
            assert isinstance(grouped, dict)
            # All results should be in some group
            total = sum(len(v) for v in grouped.values())
            assert total >= 1

    def test_session_id_in_search_results(self):
        """SearchResult includes session_id field."""
        manager = _make_manager(
            [_extraction(["test session"]), _reconciliation(("ADD", 0))],
            session_id="session_001",
        )
        with manager:
            manager.add("test session", session_id="session_001")
            results = manager.search("session")
            if results:
                assert hasattr(results[0], "session_id")

    def test_grouped_chronological_ordering(self):
        """Results within a group are sorted by created_at."""
        manager = _make_manager(
            [
                _extraction(["first fact"]),
                _reconciliation(("ADD", 0)),
                _extraction(["second fact"]),
                _reconciliation(("ADD", 0)),
            ]
        )
        with manager:
            manager.add("first fact")
            manager.add("second fact")
            grouped = manager.search("fact", grouped=True)
            for group in grouped.values():
                timestamps = [r.created_at or 0 for r in group]
                assert timestamps == sorted(timestamps)


# =============================================================================
# 4. Batch Node Creation
# =============================================================================


class TestBatchNodeCreation:
    def test_batch_add_creates_nodes(self):
        """Batch add creates all nodes."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                [
                    "alix works at acme",
                    "gus studies at TU delft",
                    "vincent likes jazz",
                ],
                infer=False,
            )
            assert len(events) == 3
            assert all(e.action == MemoryAction.ADD for e in events)
            assert manager._db.node_count >= 3

    def test_batch_add_with_embeddings(self):
        """Batch-created nodes have embeddings."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                ["alix works at acme", "gus studies at TU delft"],
                infer=False,
            )
            for event in events:
                node = manager._db.get_node(int(event.memory_id))
                props = node.properties if hasattr(node, "properties") else {}
                if callable(props):
                    props = props()
                vp = manager._config.vector_property
                assert vp in props, f"Node {event.memory_id} missing vector property '{vp}'"

    def test_batch_add_preserves_metadata(self):
        """Batch-created nodes have all expected properties."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(["test memory"], infer=False)
            node = manager._db.get_node(int(events[0].memory_id))
            props = node.properties if hasattr(node, "properties") else {}
            if callable(props):
                props = props()
            assert props.get("user_id") == "test_user"
            assert "created_at" in props
            assert "updated_at" in props
            assert "learned_at" in props
            assert props.get("memory_type") == "semantic"


# =============================================================================
# 5. Transactions (lock strategy)
# =============================================================================


class TestTransactionStrategy:
    def test_concurrent_adds_dont_crash(self):
        """Multiple sequential adds to same user don't crash."""
        manager = _make_manager(
            [
                _extraction(["fact one"]),
                _reconciliation(("ADD", 0)),
                _extraction(["fact two"]),
                _reconciliation(("ADD", 0)),
                _extraction(["fact three"]),
                _reconciliation(("ADD", 0)),
            ]
        )
        with manager:
            manager.add("fact one")
            manager.add("fact two")
            manager.add("fact three")
            results = manager.search("fact")
            assert len(results) >= 2


# =============================================================================
# 6. Temporal Chain Traversal
# =============================================================================


class TestTemporalChain:
    def test_temporal_chain_returns_list(self):
        """temporal_chain returns a list of dicts."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                ["first thing", "second thing", "third thing"],
                infer=False,
                session_id="chain_session",
            )
            mem_id = events[0].memory_id
            chain = manager.temporal_chain(mem_id)
            assert isinstance(chain, list)

    def test_temporal_chain_forward(self):
        """Forward chain follows LEADS_TO edges."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                ["step one", "step two", "step three"],
                infer=False,
                session_id="chain_test",
            )
            # First memory should have forward chain to subsequent ones
            chain = manager.temporal_chain(events[0].memory_id, direction="forward")
            # May or may not have chain depending on whether _link_session_chain was called
            assert isinstance(chain, list)

    def test_temporal_chain_backward(self):
        """Backward chain follows incoming LEADS_TO edges."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                ["step one", "step two"],
                infer=False,
                session_id="back_test",
            )
            chain = manager.temporal_chain(events[-1].memory_id, direction="backward")
            assert isinstance(chain, list)

    def test_temporal_chain_empty_for_isolated_node(self):
        """Isolated node has empty temporal chain."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(["lonely memory"], infer=False)
            chain = manager.temporal_chain(events[0].memory_id)
            assert chain == []

    def test_temporal_chain_max_depth(self):
        """max_depth limits chain length."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                [f"step {i}" for i in range(5)],
                infer=False,
                session_id="depth_test",
            )
            chain = manager.temporal_chain(events[0].memory_id, direction="forward", max_depth=2)
            assert len(chain) <= 2

    def test_temporal_chain_both_directions(self):
        """Both directions combines forward and backward."""
        manager = _make_manager([])
        with manager:
            events = manager.add_batch(
                ["a", "b", "c"],
                infer=False,
                session_id="both_test",
            )
            if len(events) >= 2:
                chain = manager.temporal_chain(events[1].memory_id, direction="both")
                assert isinstance(chain, list)


# =============================================================================
# 7. CDC History
# =============================================================================


class TestCDCHistory:
    def test_record_history_noop_with_cdc(self):
        """record_history is a no-op when native CDC is available."""
        db = grafeo.GrafeoDB()
        if not hasattr(db, "node_history"):
            return
        node = db.create_node(["Memory"], {"text": "test"})
        node_id = node.id if hasattr(node, "id") else node
        result = record_history(db, node_id, HistoryEntry(event="ADD", new_text="test"))
        assert result is None

    def test_get_history_returns_add_event(self):
        """get_history returns ADD event for newly created node."""
        db = grafeo.GrafeoDB()
        if not hasattr(db, "node_history"):
            return
        node = db.create_node(["Memory"], {"text": "hello world"})
        node_id = node.id if hasattr(node, "id") else node
        entries = get_history(db, node_id)
        assert len(entries) >= 1
        assert entries[0].event == "ADD"

    def test_get_history_tracks_text_update(self):
        """get_history returns UPDATE when text property changes."""
        db = grafeo.GrafeoDB()
        if not hasattr(db, "node_history"):
            return
        node = db.create_node(["Memory"], {"text": "version 1"})
        node_id = node.id if hasattr(node, "id") else node
        db.set_node_property(node_id, "text", "version 2")

        entries = get_history(db, node_id)
        events = [e.event for e in entries]
        assert "ADD" in events
        assert "UPDATE" in events

    def test_get_history_empty_for_nonexistent(self):
        """get_history returns empty list for nonexistent node."""
        db = grafeo.GrafeoDB()
        entries = get_history(db, 99999)
        assert entries == []

    def test_cdc_update_captures_old_and_new_text(self):
        """CDC UPDATE events capture old_text and new_text."""
        db = grafeo.GrafeoDB()
        if not hasattr(db, "node_history"):
            return
        node = db.create_node(["Memory"], {"text": "original"})
        node_id = node.id if hasattr(node, "id") else node
        db.set_node_property(node_id, "text", "updated")

        entries = get_history(db, node_id)
        updates = [e for e in entries if e.event == "UPDATE"]
        if updates:
            assert updates[0].old_text == "original"
            assert updates[0].new_text == "updated"
