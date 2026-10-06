"""Tests for durable, graph-native history tracking (history.py).

History used to come from engine CDC (``db.node_history()``). Since grafeo
0.5.32 CDC is opt-in and in-memory only, so that design returned no history at
all on a default database and lost it on reopen. These tests pin the design
that replaced it: grafeo-memory writes its own ``:History`` nodes.
"""

import grafeo

from grafeo_memory.history import (
    HAS_HISTORY_EDGE,
    HISTORY_LABEL,
    MEMORY_ID_PROPERTY,
    HistoryEntry,
    adopt_legacy_history,
    get_history,
    record_history,
    record_history_batch,
)


def _make_db():
    return grafeo.GrafeoDB()


def _props(db, node_id):
    props = db.get_node(node_id).properties
    return props() if callable(props) else props


def _history_edges(db, node_id):
    rows = db.execute(f"MATCH (h:{HISTORY_LABEL})-[r]-() WHERE id(h) = $hid RETURN id(r)", {"hid": node_id})
    return list(rows)


class TestRecordHistory:
    def test_creates_history_node(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "alice likes hiking"})

        hist_id = record_history(db, mem.id, HistoryEntry(event="ADD", new_text="alice likes hiking", timestamp=1000))

        props = _props(db, hist_id)
        assert props["event"] == "ADD"
        assert props["new_text"] == "alice likes hiking"
        assert props["timestamp"] == 1000
        assert props[MEMORY_ID_PROPERTY] == mem.id
        assert HISTORY_LABEL in db.get_node(hist_id).labels

    def test_history_node_has_no_edges(self):
        """History nodes are keyed by memory_id, so they do not change the graph topology."""
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "test"})
        hist_id = record_history(db, mem.id, HistoryEntry(event="ADD", new_text="test", timestamp=1000))
        assert _history_edges(db, hist_id) == []
        # The memory can still be deleted with the direct API: no HAS_HISTORY edge holds it.
        assert db.delete_node(mem.id) is True

    def test_works_without_cdc(self):
        """CDC is off by default since grafeo 0.5.32: recording must not depend on it."""
        db = _make_db()
        assert db.cdc_enabled is False
        mem = db.create_node(["Memory"], {"text": "test"})
        record_history(db, mem.id, HistoryEntry(event="ADD", new_text="test", timestamp=1000))
        assert [e.event for e in get_history(db, mem.id)] == ["ADD"]

    def test_actor_and_scope_fields_stored(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "test"})
        entry = HistoryEntry(
            event="UPDATE", old_text="old", new_text="new", timestamp=2000, actor_id="alix", role="user"
        )

        hist_id = record_history(db, mem.id, entry, user_id="u1", graph_name="g1")

        props = _props(db, hist_id)
        assert props["actor_id"] == "alix"
        assert props["role"] == "user"
        assert props["user_id"] == "u1"
        assert props["graph_name"] == "g1"

    def test_optional_fields_left_out(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "test"})
        hist_id = record_history(db, mem.id, HistoryEntry(event="ADD", timestamp=1))
        props = _props(db, hist_id)
        for key in ("old_text", "new_text", "actor_id", "role", "user_id", "graph_name"):
            assert key not in props


class TestRecordHistoryBatch:
    def test_batch_entries(self):
        db = _make_db()
        a = db.create_node(["Memory"], {"text": "a"})
        b = db.create_node(["Memory"], {"text": "b"})
        ids = record_history_batch(
            db,
            [
                (a.id, HistoryEntry(event="ADD", new_text="a", timestamp=1)),
                (b.id, HistoryEntry(event="ADD", timestamp=1)),
            ],
            user_id="u1",
        )
        assert len(ids) == 2
        assert _props(db, ids[0])["user_id"] == "u1"
        assert [e.new_text for e in get_history(db, a.id)] == ["a"]
        assert [e.event for e in get_history(db, b.id)] == ["ADD"]
        assert record_history_batch(db, []) == []

    def test_without_batch_api(self):
        """Databases that only implement the protocol's create_node get one node per entry."""

        class NoBatch:
            def __init__(self, inner):
                self._inner = inner

            def create_node(self, labels, props):
                return self._inner.create_node(labels, props)

        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "m"})
        ids = record_history_batch(NoBatch(db), [(mem.id, HistoryEntry(event="ADD", timestamp=1))])  # type: ignore[arg-type]
        assert len(ids) == 1
        assert [e.event for e in get_history(db, mem.id)] == ["ADD"]


class TestGetHistory:
    def test_retrieves_entries_in_order(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "v1"})
        record_history(db, mem.id, HistoryEntry(event="UPDATE", old_text="v1", new_text="v2", timestamp=2000))
        record_history(db, mem.id, HistoryEntry(event="ADD", new_text="v1", timestamp=1000))

        entries = get_history(db, mem.id)
        assert [e.event for e in entries] == ["ADD", "UPDATE"]
        assert entries[0].timestamp < entries[1].timestamp
        assert entries[1].old_text == "v1"
        assert entries[1].new_text == "v2"

    def test_same_timestamp_keeps_recording_order(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "v1"})
        for event in ("ADD", "UPDATE", "DELETE"):
            record_history(db, mem.id, HistoryEntry(event=event, timestamp=5000))
        assert [e.event for e in get_history(db, mem.id)] == ["ADD", "UPDATE", "DELETE"]

    def test_only_entries_of_that_memory(self):
        db = _make_db()
        a = db.create_node(["Memory"], {"text": "a"})
        b = db.create_node(["Memory"], {"text": "b"})
        record_history(db, a.id, HistoryEntry(event="ADD", new_text="a", timestamp=1))
        record_history(db, b.id, HistoryEntry(event="ADD", new_text="b", timestamp=1))
        assert [e.new_text for e in get_history(db, a.id)] == ["a"]
        assert [e.new_text for e in get_history(db, b.id)] == ["b"]

    def test_survives_deletion_of_the_memory(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "gone soon"})
        record_history(db, mem.id, HistoryEntry(event="ADD", new_text="gone soon", timestamp=1))
        record_history(db, mem.id, HistoryEntry(event="DELETE", old_text="gone soon", timestamp=2))
        assert db.delete_node(mem.id) is True
        assert [e.event for e in get_history(db, mem.id)] == ["ADD", "DELETE"]

    def test_graph_name_scope(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "t"})
        record_history(db, mem.id, HistoryEntry(event="ADD", timestamp=1), graph_name="g1")
        record_history(db, mem.id, HistoryEntry(event="UPDATE", timestamp=2), graph_name="g2")
        assert [e.event for e in get_history(db, mem.id, graph_name="g1")] == ["ADD"]
        assert [e.event for e in get_history(db, mem.id, graph_name="g2")] == ["UPDATE"]
        assert [e.event for e in get_history(db, mem.id)] == ["ADD", "UPDATE"]

    def test_raw_engine_writes_are_not_history(self):
        """Only changes made through grafeo-memory are history; CDC events are not read."""
        db = _make_db()
        db.enable_cdc()
        mem = db.create_node(["Memory"], {"text": "v1"})
        db.set_node_property(mem.id, "text", "v2")
        assert db.node_history(mem.id)  # the engine has CDC events...
        assert get_history(db, mem.id) == []  # ...but they are not memory history

    def test_empty_for_nonexistent_node(self):
        db = _make_db()
        entries = get_history(db, 99999)
        assert entries == []


class TestLegacyHistory:
    """grafeo-memory 0.1.x linked History nodes with HAS_HISTORY edges and no memory_id."""

    def _legacy_entry(self, db, mem_id, event, timestamp, **extra):
        hist = db.create_node([HISTORY_LABEL], {"event": event, "timestamp": timestamp, **extra})
        db.create_edge(mem_id, hist.id, HAS_HISTORY_EDGE)
        return hist.id

    def test_legacy_entries_are_read(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "v2"})
        self._legacy_entry(db, mem.id, "ADD", 1000, new_text="v1", actor_id="alix", role="user")
        record_history(db, mem.id, HistoryEntry(event="UPDATE", old_text="v1", new_text="v2", timestamp=2000))

        entries = get_history(db, mem.id)
        assert [e.event for e in entries] == ["ADD", "UPDATE"]
        assert entries[0].new_text == "v1"
        assert entries[0].actor_id == "alix"
        assert entries[0].role == "user"

    def test_adopted_entries_survive_detach_delete_without_duplicates(self):
        db = _make_db()
        mem = db.create_node(["Memory"], {"text": "v1"})
        self._legacy_entry(db, mem.id, "ADD", 1000, new_text="v1")

        adopt_legacy_history(db, mem.id)
        # Before the delete, the entry is reachable both ways and must be returned once.
        assert [e.event for e in get_history(db, mem.id)] == ["ADD"]

        db.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": mem.id})
        assert db.get_node(mem.id) is None
        assert [e.event for e in get_history(db, mem.id)] == ["ADD"]
