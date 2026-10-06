"""End-to-end history through the manager: every write path, persistence, shared databases.

On grafeo 0.5.32 to 0.5.44 ``history()`` returned nothing at all, because it
read engine CDC, which is opt-in and in-memory only. These tests make sure
history is recorded on every path that changes a memory and that it outlives
the process that wrote it.
"""

from __future__ import annotations

import asyncio
import time

import grafeo
import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryAction, MemoryConfig, MemoryManager

ALICE = {"name": "alice", "entity_type": "person"}


def _config(**kwargs) -> MemoryConfig:
    defaults = {"user_id": "test_user", "embedding_dimensions": 16}
    defaults.update(kwargs)
    return MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]


def _make_manager(outputs=(), *, db=None, **kwargs) -> MemoryManager:
    return MemoryManager(make_test_model(list(outputs)), _config(**kwargs), embedder=MockEmbedder(16), db=db)


def _events(entries) -> list[str]:
    return [e.event for e in entries]


@pytest.fixture(params=["memory.grafeo", "memory.db"], ids=["single-file", "wal-directory"])
def db_path(request, tmp_path) -> str:
    return str(tmp_path / request.param)


class TestHistorySurvivesReopen:
    def test_full_lifecycle_across_sessions(self, db_path):
        with _make_manager(db_path=db_path) as memory:
            kept = memory.add("alice likes tea", infer=False, metadata={"src": "chat"})[0].memory_id
            memory.update(kept, "alice likes green tea")
            gone = memory.add("temporary note", infer=False)[0].memory_id

        with _make_manager(db_path=db_path) as memory:
            assert _events(memory.history(kept)) == ["ADD", "UPDATE"]
            assert memory.delete(gone) is True

        with _make_manager(db_path=db_path) as memory:
            entries = memory.history(kept)
            assert entries[1].old_text == "alice likes tea"
            assert entries[1].new_text == "alice likes green tea"
            assert all(e.timestamp > 1_600_000_000_000 for e in entries), "timestamps are epoch milliseconds"
            assert _events(memory.history(gone)) == ["ADD", "DELETE"]
            assert memory.history(gone)[1].old_text == "temporary note"
            assert [m.text for m in memory.get_all()] == ["alice likes green tea"]

    def test_history_of_batch_and_summarize_across_sessions(self, db_path):
        with _make_manager([{"memories": ["alice: a, b"]}], db_path=db_path) as memory:
            events = memory.add_batch(["a", "b", "c"], infer=False)
            ids = [e.memory_id for e in events]
            time.sleep(0.002)
            memory.add("recent", infer=False)
            result = memory.summarize(preserve_recent=1)
            summary_id = next(e.memory_id for e in result if e.action == MemoryAction.ADD)

        with _make_manager(db_path=db_path) as memory:
            for mid in ids:
                assert _events(memory.history(mid)) == ["ADD", "DELETE"]
            assert _events(memory.history(summary_id)) == ["ADD"]


class TestHistoryOnEveryWritePath:
    def test_reconciliation_update_and_delete(self):
        db = grafeo.GrafeoDB()
        first = _make_manager([{"facts": ["alice works at acme"], "entities": [ALICE], "relations": []}], db=db)
        old_id = first.add("Alice works at Acme")[0].memory_id
        second = _make_manager(
            [
                {"facts": ["alice works at globex"], "entities": [ALICE], "relations": []},
                {"decisions": [{"action": "update", "text": "alice works at globex", "target_memory_id": old_id}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        new_id = second.add({"role": "user", "content": "I moved to Globex", "name": "alice"})[0].memory_id

        update = second.history(new_id)
        assert _events(update) == ["UPDATE"]
        assert update[0].old_text == "alice works at acme"
        assert update[0].new_text == "alice works at globex"
        assert update[0].actor_id == "alice"
        assert update[0].role == "user"

        third = _make_manager(
            [
                {"facts": ["alice is unemployed"], "entities": [ALICE], "relations": []},
                {"decisions": [{"action": "delete", "text": "", "target_memory_id": new_id}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        third.add("Alice quit")
        assert _events(third.history(new_id)) == ["UPDATE", "DELETE"]
        assert third.history(new_id)[1].old_text == "alice works at globex"
        assert _events(third.history(old_id)) == ["ADD"]

    def test_history_entries_carry_scope(self):
        manager = _make_manager(graph_name="team-a")
        mid = manager.add("scoped", infer=False, user_id="carol")[0].memory_id
        query = "MATCH (h:History) RETURN h.user_id AS u, h.graph_name AS g, h.memory_id AS m"
        rows = list(manager._db.execute(query, {}))
        assert rows == [{"u": "carol", "g": "team-a", "m": int(mid)}]
        manager.close()

    def test_history_nodes_are_not_memories(self):
        manager = _make_manager()
        mid = manager.add("one", infer=False)[0].memory_id
        manager.update(mid, "two")
        stats = manager.stats()
        assert stats.total_memories == 1
        assert len(manager.get_all()) == 1
        manager.close()

    def test_history_nodes_do_not_feed_graph_metrics(self):
        manager = _make_manager(
            [
                {"facts": ["alice likes tea"], "entities": [ALICE], "relations": []},
                {"entities": [], "relations": []},
            ],
            enable_graph_algorithms=True,
        )
        mid = manager.add("Alice likes tea")[0].memory_id
        for i in range(5):
            manager.update(mid, f"alice likes tea v{i}")
        manager.search("tea")
        props = manager._db.get_node(int(mid)).properties()
        # History nodes have no edges, so the memory is not a bridge to them.
        assert props["_betweenness"] == 0.0
        manager.close()


class TestHistoryPurge:
    def test_documented_purge_of_a_users_history(self):
        """The README shows how to remove a user's history after delete_all()."""
        manager = _make_manager()
        mid = manager.add("private", infer=False, user_id="alice")[0].memory_id
        other = manager.add("bob note", infer=False, user_id="bob")[0].memory_id
        assert manager.delete_all(user_id="alice") == 1
        assert _events(manager.history(mid)) == ["ADD", "DELETE"]

        manager._db.execute("MATCH (h:History) WHERE h.user_id = $uid DELETE h", {"uid": "alice"})

        assert manager.history(mid) == []
        assert _events(manager.history(other)) == ["ADD"]
        manager.close()


class TestHistoryInSharedDatabases:
    def test_user_supplied_db_keeps_its_cdc_setting(self):
        for cdc in (False, True):
            db = grafeo.GrafeoDB()
            if cdc:
                db.enable_cdc()
            manager = _make_manager(db=db)
            mid = manager.add("x", infer=False)[0].memory_id
            manager.update(mid, "y")
            assert _events(manager.history(mid)) == ["ADD", "UPDATE"]
            assert db.cdc_enabled is cdc
            manager.close()

    def test_graph_name_tenants_see_only_their_history(self):
        db = grafeo.GrafeoDB()
        a = _make_manager(db=db, graph_name="a")
        b = _make_manager(db=db, graph_name="b")
        unscoped = _make_manager(db=db)
        a_id = a.add("from a", infer=False)[0].memory_id
        b_id = b.add("from b", infer=False)[0].memory_id

        assert _events(a.history(a_id)) == ["ADD"]
        assert a.history(b_id) == []
        assert b.history(a_id) == []
        assert _events(unscoped.history(b_id)) == ["ADD"]

    def test_external_db_history_survives_manager_close(self):
        db = grafeo.GrafeoDB()
        manager = _make_manager(db=db)
        mid = manager.add("x", infer=False)[0].memory_id
        manager.close()
        assert _events(_make_manager(db=db).history(mid)) == ["ADD"]


class TestAsyncHistory:
    def test_async_manager_history_and_persistence(self, tmp_path):
        path = str(tmp_path / "async.grafeo")

        async def first_session() -> str:
            async with AsyncMemoryManager(make_test_model([]), _config(db_path=path), embedder=MockEmbedder(16)) as m:
                mid = (await m.add("async fact", infer=False))[0].memory_id
                event = await m.update(mid, "async fact v2")
                assert event.old_text == "async fact"
                assert _events(await m.history(mid)) == ["ADD", "UPDATE"]
                return mid

        async def second_session(mid: str) -> None:
            async with AsyncMemoryManager(make_test_model([]), _config(db_path=path), embedder=MockEmbedder(16)) as m:
                assert _events(await m.history(mid)) == ["ADD", "UPDATE"]
                assert await m.delete(mid) is True
                assert _events(await m.history(mid)) == ["ADD", "UPDATE", "DELETE"]
                with pytest.raises(ValueError):
                    await m.update(mid, "gone")

        mid = asyncio.run(first_session())
        asyncio.run(second_session(mid))
