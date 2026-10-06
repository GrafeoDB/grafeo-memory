"""Hard deletes on grafeo 0.5.44+, where ``delete_node`` refuses a node that still has edges.

Every node grafeo-memory deletes (memories, dissolved communities) can have
edges: HAS_ENTITY, LEADS_TO, SUPERSEDES, DERIVED_FROM, PRODUCED, HAS_MEMBER and
the HAS_HISTORY edges of 0.1.x databases. These tests delete through each of them.
"""

from __future__ import annotations

import asyncio

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryAction, MemoryConfig, MemoryManager
from grafeo_memory.communities import get_communities, materialize_communities_async
from grafeo_memory.history import HAS_HISTORY_EDGE, HISTORY_LABEL

ALICE = {"name": "alice", "entity_type": "person"}
ACME = {"name": "acme_corp", "entity_type": "organization"}


def _config(**kwargs) -> MemoryConfig:
    defaults = {"user_id": "test_user", "embedding_dimensions": 16}
    defaults.update(kwargs)
    return MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]


def _make_manager(outputs, *, db=None, **kwargs) -> MemoryManager:
    return MemoryManager(make_test_model(outputs), _config(**kwargs), embedder=MockEmbedder(16), db=db)


def _edge_count(db, edge_type: str) -> int:
    return len(list(db.execute(f"MATCH ()-[r:{edge_type}]->() RETURN id(r)")))


class TestDeleteThroughEachEdgeType:
    def test_entities_and_relations_survive(self):
        manager = _make_manager(
            [
                {
                    "facts": ["alice works at acme corp"],
                    "entities": [ALICE, ACME],
                    "relations": [{"source": "alice", "target": "acme_corp", "relation_type": "works_at"}],
                }
            ]
        )
        mid = manager.add("Alice works at Acme Corp")[0].memory_id
        db = manager._db
        assert _edge_count(db, "HAS_ENTITY") == 2

        assert manager.delete(mid) is True

        assert db.get_node(int(mid)) is None
        assert _edge_count(db, "HAS_ENTITY") == 0
        assert len(db.get_nodes_by_label("Entity")) == 2, "entities are shared knowledge and are kept"
        assert _edge_count(db, "RELATION") == 1
        manager.close()

    def test_middle_of_session_chain(self):
        manager = _make_manager([], run_id="run-1")
        ids = [manager.add(f"step {i}", infer=False)[0].memory_id for i in range(3)]
        assert _edge_count(manager._db, "LEADS_TO") == 2

        assert manager.delete(ids[1]) is True

        assert _edge_count(manager._db, "LEADS_TO") == 0
        assert manager.temporal_chain(ids[0]) == []
        assert {m.memory_id for m in manager.get_all()} == {ids[0], ids[2]}
        manager.close()

    def test_superseded_memory_and_its_successor(self):
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
        new_id = second.add("Alice works at Globex")[0].memory_id
        assert _edge_count(db, "SUPERSEDES") == 1

        assert second.delete(old_id) is True, "an expired memory can still be hard-deleted"
        assert _edge_count(db, "SUPERSEDES") == 0
        assert second.delete(new_id) is True
        assert db.get_nodes_by_label("Memory") == []
        first.close()
        second.close()

    def test_memory_produced_by_an_episode(self):
        manager = _make_manager(
            [{"facts": ["alice likes tea"], "entities": [ALICE], "relations": []}],
            enable_episodes=True,
            run_id="run-1",
        )
        mid = manager.add("Alice likes tea")[0].memory_id
        (episode,) = manager.get_episodes()
        assert episode.produced_memories == [mid]

        assert manager.delete(mid) is True

        (episode,) = manager.get_episodes()
        assert episode.produced_memories == []
        assert episode.mentioned_entities == ["alice"]
        assert manager.get_provenance(mid) == []
        manager.close()

    def test_summarize_deletes_originals_with_derived_from_edges(self):
        manager = _make_manager([{"memories": ["alice: acme, python"]}], run_id="run-1")
        for text in ("alice works at acme", "alice likes python", "recent"):
            manager.add(text, infer=False)
            time_step()

        events = manager.summarize(preserve_recent=1)

        deletes = [e for e in events if e.action == MemoryAction.DELETE]
        assert len(deletes) == 2
        for e in deletes:
            assert manager._db.get_node(int(e.memory_id)) is None
            assert [h.event for h in manager.history(e.memory_id)] == ["ADD", "DELETE"]
        assert sorted(m.text for m in manager.get_all()) == ["alice: acme, python", "recent"]
        manager.close()

    def test_memory_with_legacy_history_edges(self):
        """Databases written by grafeo-memory 0.1.x link History nodes with HAS_HISTORY edges."""
        manager = _make_manager([])
        db = manager._db
        mid = manager.add("legacy memory", infer=False)[0].memory_id
        legacy = db.create_node([HISTORY_LABEL], {"event": "UPDATE", "old_text": "a", "new_text": "b", "timestamp": 0})
        db.create_edge(int(mid), legacy.id, HAS_HISTORY_EDGE)

        assert manager.delete(mid) is True

        assert db.get_node(legacy.id) is not None
        assert [h.event for h in manager.history(mid)] == ["UPDATE", "ADD", "DELETE"]
        manager.close()


class TestRelationReconciliation:
    def test_contradicted_relation_edge_is_deleted(self):
        globex = {"name": "globex", "entity_type": "organization"}
        manager = _make_manager(
            [
                {
                    "facts": ["alice works at acme corp"],
                    "entities": [ALICE, ACME],
                    "relations": [{"source": "alice", "target": "acme_corp", "relation_type": "works_at"}],
                },
                {
                    "facts": ["alice works at globex"],
                    "entities": [ALICE, globex],
                    "relations": [{"source": "alice", "target": "globex", "relation_type": "works_at"}],
                },
                {"decisions": [{"action": "add", "text": "alice works at globex"}]},
                {"delete": [{"source": "alice", "target": "acme_corp", "relation_type": "works_at"}]},
            ],
            reconciliation_threshold=0.0,
        )
        manager.add("Alice works at Acme Corp")
        manager.add("Alice moved to Globex")

        rows = manager._db.execute(
            "MATCH (s:Entity)-[r:RELATION]->(t:Entity) RETURN s.name AS s, r.relation_type AS r, t.name AS t", {}
        )
        assert [(row["s"], row["r"], row["t"]) for row in rows] == [("alice", "works_at", "globex")]
        manager.close()


class TestDeleteAll:
    def test_deletes_connected_memories_and_keeps_other_users(self):
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"], "entities": [ALICE], "relations": []},
                {"facts": ["alice likes cooking"], "entities": [ALICE], "relations": []},
            ],
            run_id="run-1",
        )
        manager.add("Alice likes hiking")
        manager.add("Alice likes cooking")
        other = manager.add("bob's note", user_id="bob", infer=False)[0].memory_id

        assert manager.delete_all() == 2

        assert manager.get_all() == []
        assert [m.memory_id for m in manager.get_all(user_id="bob")] == [other]
        manager.close()

    def test_async_delete_all(self):
        async def run():
            config = _config(run_id="run-1")
            async with AsyncMemoryManager(make_test_model([]), config, embedder=MockEmbedder(16)) as memory:
                for i in range(3):
                    await memory.add(f"fact {i}", infer=False)
                assert await memory.delete_all() == 3
                assert await memory.get_all() == []
                assert await memory.delete("0") is False

        asyncio.run(run())


class TestDissolvedCommunities:
    def test_community_with_member_edges_is_deleted(self):
        db = grafeo.GrafeoDB()
        ids = [db.create_node(["Entity"], {"name": n, "user_id": "u"}).id for n in ("a", "b", "c", "d")]
        model = make_test_model([{"name": "AB", "summary": "a and b"}, {"name": "CD", "summary": "c and d"}])
        partition = {"communities": {ids[0]: 0, ids[1]: 0, ids[2]: 1, ids[3]: 1}}
        asyncio.run(materialize_communities_async(db, model, "u", partition))
        assert _edge_count(db, "HAS_MEMBER") == 4

        asyncio.run(
            materialize_communities_async(db, make_test_model([]), "u", {"communities": {ids[0]: 0, ids[1]: 0}})
        )

        assert [c.name for c in get_communities(db, "u")] == ["AB"]
        assert _edge_count(db, "HAS_MEMBER") == 2
        assert all(db.get_node(i) is not None for i in ids), "member entities are kept"


def time_step() -> None:
    """Make consecutive created_at values distinct (summarize orders by created_at)."""
    import time

    time.sleep(0.002)
