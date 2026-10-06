"""Regression tests for bugs found while preparing for grafeo 0.5.44 and 0.6.0.

Each class names the bug it guards against. All tests use a mock LLM and a
real in-memory GrafeoDB.
"""

from __future__ import annotations

import hashlib
import time

import grafeo
import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import LLMReranker, MemoryAction, MemoryConfig, MemoryManager
from grafeo_memory.search.vector import _get_props

ALICE = {"name": "alice", "entity_type": "person"}


def _make_manager(outputs, *, db=None, embedder=None, dims=16, **config_kwargs):
    defaults = {"user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(make_test_model(outputs), config, embedder=embedder or MockEmbedder(dims), db=db)


def _props(db, node_id):
    return _get_props(db.get_node(int(node_id)))


class _IntEmbedder:
    """An embedder that returns ints, like a quantized model or numpy integer output."""

    dimensions = 8

    def embed(self, texts: list[str]) -> list[list[int]]:
        return [[(b % 5) + 1 for b in hashlib.sha256(t.encode()).digest()[: self.dimensions]] for t in texts]


# ---------------------------------------------------------------------------
# Search pipeline stages dropped SearchResult fields
# ---------------------------------------------------------------------------


class TestSearchKeepsResultFields:
    """The vector/graph merge, topology boost, importance scoring and the reranker
    rebuilt SearchResult by hand and dropped created_at, session_id, valid_at and more.
    Results found by both paths then failed time_after/time_before, ignored
    point_in_time, and grouped=True put them all under "default".
    """

    def _add_alice(self, manager, session_id="s1"):
        events = manager.add("Alice likes hiking", session_id=session_id)
        assert events[0].action == MemoryAction.ADD
        return events[0].memory_id

    def test_result_found_by_both_paths_passes_time_filter(self):
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"], "entities": [ALICE], "relations": []},
                {"entities": [ALICE], "relations": []},  # query entity extraction
            ]
        )
        start = int(time.time() * 1000) - 1
        mid = self._add_alice(manager)

        results = manager.search("alice likes hiking", time_after=start)

        assert [r.memory_id for r in results] == [mid]
        assert results[0].source == "both"
        assert results[0].created_at is not None
        assert results[0].session_id == "s1"
        manager.close()

    def test_point_in_time_applies_to_results_found_by_both_paths(self):
        db = grafeo.GrafeoDB()
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"], "entities": [ALICE], "relations": []},
                {"entities": [ALICE], "relations": []},
            ],
            db=db,
        )
        mid = self._add_alice(manager)
        db.set_node_property(int(mid), "valid_at", 2_000)

        results = manager.search("alice likes hiking", point_in_time=1_000)

        assert results == [], "a fact that only became valid at 2000 must not be returned for 1000"
        manager.close()

    def test_grouped_search_keeps_sessions_with_importance_and_topology(self):
        manager = _make_manager(
            [
                {"facts": ["alice likes hiking"], "entities": [ALICE], "relations": []},
                {"entities": [ALICE], "relations": []},
            ],
            enable_importance=True,
            enable_topology_boost=True,
        )
        mid = self._add_alice(manager, session_id="trip")

        groups = manager.search("alice likes hiking", grouped=True)

        assert isinstance(groups, dict)
        assert list(groups) == ["trip"]
        (result,) = groups["trip"]
        assert result.memory_id == mid
        assert result.created_at is not None
        assert result.importance is not None
        manager.close()

    def test_reranker_keeps_fields(self):
        reranker = LLMReranker(make_test_model([{"score": 0.9, "reasoning": "relevant"}]))
        config = MemoryConfig(user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(
            make_test_model([{"entities": [], "relations": []}]),
            config,
            embedder=MockEmbedder(16),
            reranker=reranker,
        )
        manager.add("alice likes hiking", infer=False, session_id="s9", memory_type="procedural")

        (result,) = manager.search("alice likes hiking")

        assert result.score == 0.9
        assert result.memory_type == "procedural"
        assert result.session_id == "s9"
        assert result.created_at is not None
        manager.close()


# ---------------------------------------------------------------------------
# Embeddings that are not float lists were silently unsearchable
# ---------------------------------------------------------------------------


class TestEmbeddingsStoredAsFloats:
    """The engine indexes a property as a vector only when every element is a float.
    An embedder returning ints stored memories that vector search never found.
    """

    def test_int_embedder_memories_are_searchable(self):
        manager = _make_manager([{"entities": [], "relations": []}], embedder=_IntEmbedder(), dims=8)
        events = manager.add("alice likes hiking", infer=False)
        stored = _props(manager._db, events[0].memory_id)["embedding"]
        assert all(isinstance(x, float) for x in stored)

        results = manager.search("alice likes hiking")

        assert [r.memory_id for r in results] == [events[0].memory_id]
        manager.close()

    def test_int_embedder_batch_add(self):
        manager = _make_manager([], embedder=_IntEmbedder(), dims=8)
        events = manager.add_batch(["one", "two"], infer=False)
        for event in events:
            assert all(isinstance(x, float) for x in _props(manager._db, event.memory_id)["embedding"])
        manager.close()


# ---------------------------------------------------------------------------
# Reconciliation acted on any node id the LLM returned
# ---------------------------------------------------------------------------


class TestReconciliationTargetsAreCandidates:
    """UPDATE and DELETE decisions acted on whatever target_memory_id the LLM returned:
    a hallucinated or injected id could expire another user's memory or a foreign
    node in a shared database, and on grafeo 0.5.44 a missing id made the
    SUPERSEDES edge fail.
    """

    def _setup(self):
        db = grafeo.GrafeoDB()
        foreign = db.create_node(["Domain"], {"name": "keep me"})
        other = _make_manager([], db=db, user_id="other_user")
        other_id = other.add("bob likes chess", infer=False)[0].memory_id
        own = _make_manager([], db=db)
        own_id = own.add("alice likes hiking", infer=False)[0].memory_id
        return db, foreign.id, other_id, own_id

    def test_update_of_non_candidate_becomes_add(self):
        db, foreign_id, other_id, own_id = self._setup()
        manager = _make_manager(
            [
                {"facts": ["alice likes climbing"], "entities": [], "relations": []},
                {
                    "decisions": [
                        {"action": "update", "text": "alice likes climbing", "target_memory_id": other_id},
                    ]
                },
            ],
            db=db,
            reconciliation_threshold=0.0,
        )

        events = manager.add("Alice likes climbing")

        assert [e.action for e in events] == [MemoryAction.ADD]
        assert "expired_at" not in _props(db, other_id), "another user's memory was expired"
        assert "expired_at" not in _props(db, own_id)
        assert _props(db, foreign_id) == {"name": "keep me"}
        manager.close()

    def test_delete_of_foreign_node_is_skipped(self):
        db, foreign_id, _, own_id = self._setup()
        manager = _make_manager(
            [
                {"facts": ["alice stopped hiking"], "entities": [], "relations": []},
                {"decisions": [{"action": "delete", "text": "", "target_memory_id": str(foreign_id)}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )

        events = manager.add("Alice stopped hiking")

        assert events == []
        assert _props(db, foreign_id) == {"name": "keep me"}
        assert "expired_at" not in _props(db, own_id)
        manager.close()

    def test_candidate_targets_still_work(self):
        db, _, _, own_id = self._setup()
        manager = _make_manager(
            [
                {"facts": ["alice likes climbing"], "entities": [], "relations": []},
                {"decisions": [{"action": "update", "text": "alice likes climbing", "target_memory_id": own_id}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )

        events = manager.add("Alice likes climbing")

        assert [e.action for e in events] == [MemoryAction.UPDATE]
        assert _props(db, own_id)["expired_at"] is not None
        manager.close()


# ---------------------------------------------------------------------------
# The public API modified nodes it does not own
# ---------------------------------------------------------------------------


class TestPublicApiOnlyTouchesOwnedMemories:
    """delete(), update() and set_importance() accepted any node id, so in a shared
    database they could delete or change foreign nodes (and memories of another
    graph_name). update() of a missing id also reported success.
    """

    def test_delete_refuses_foreign_node(self):
        db = grafeo.GrafeoDB()
        a = db.create_node(["Domain"], {"name": "a"})
        b = db.create_node(["Domain"], {"name": "b"})
        db.create_edge(a.id, b.id, "CONNECTED_TO")
        lone = db.create_node(["Domain"], {"name": "lone"})
        manager = _make_manager([], db=db)

        assert manager.delete(str(a.id)) is False
        assert manager.delete(str(lone.id)) is False
        assert db.get_node(a.id) is not None
        assert db.get_node(lone.id) is not None
        assert len(list(db.execute("MATCH (:Domain)-[r:CONNECTED_TO]->(:Domain) RETURN id(r)"))) == 1
        manager.close()

    def test_update_refuses_foreign_and_missing(self):
        db = grafeo.GrafeoDB()
        foreign = db.create_node(["Domain"], {"text": "original"})
        manager = _make_manager([], db=db)

        with pytest.raises(ValueError, match="not found"):
            manager.update(str(foreign.id), "changed")
        with pytest.raises(ValueError, match="not found"):
            manager.update("999999", "changed")
        with pytest.raises(ValueError, match="not found"):
            manager.update("not-a-number", "changed")

        assert _props(db, foreign.id) == {"text": "original"}
        assert list(db.execute("MATCH (h:History) RETURN id(h)")) == [], "history was written for a non-memory"
        manager.close()

    def test_set_importance_refuses_foreign_node(self):
        db = grafeo.GrafeoDB()
        foreign = db.create_node(["Domain"], {"name": "x"})
        manager = _make_manager([], db=db)
        assert manager.set_importance(str(foreign.id), 0.5) is False
        assert "importance" not in _props(db, foreign.id)
        manager.close()

    def test_graph_name_scopes_mutations(self):
        db = grafeo.GrafeoDB()
        team_a = _make_manager([], db=db, graph_name="a")
        team_b = _make_manager([], db=db, graph_name="b")
        mid = team_a.add("a's memory", infer=False)[0].memory_id

        assert team_b.delete(mid) is False
        with pytest.raises(ValueError):
            team_b.update(mid, "b was here")
        assert team_b.set_importance(mid, 0.1) is False
        assert team_b.history(mid) == []

        assert _props(db, mid)["text"] == "a's memory"
        assert [e.event for e in team_a.history(mid)] == ["ADD"]
        assert team_a.delete(mid) is True
        team_a.close()
        team_b.close()


# ---------------------------------------------------------------------------
# delete_all() left expired memories behind
# ---------------------------------------------------------------------------


class TestDeleteAllIncludesExpired:
    """delete_all() only deleted active memories: superseded (expired) ones, text
    included, stayed in the database.
    """

    def test_superseded_memories_are_deleted(self):
        db = grafeo.GrafeoDB()
        first = _make_manager([{"facts": ["alice works at acme"], "entities": [], "relations": []}], db=db)
        old_id = first.add("Alice works at Acme")[0].memory_id
        second = _make_manager(
            [
                {"facts": ["alice works at globex"], "entities": [], "relations": []},
                {"decisions": [{"action": "update", "text": "alice works at globex", "target_memory_id": old_id}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        second.add("Alice works at Globex")
        assert len(second.get_all(include_expired=True)) == 2

        assert second.delete_all() == 2

        assert second.get_all(include_expired=True) == []
        assert db.get_nodes_by_label("Memory") == []
        first.close()
        second.close()


# ---------------------------------------------------------------------------
# Session chains branched when adds shared a millisecond
# ---------------------------------------------------------------------------


class TestChainsWithinOneMillisecond:
    """LEADS_TO and NEXT_EPISODE picked the previous node by created_at alone, so
    adds within one millisecond all linked to the first one (a fan, not a chain).
    """

    def _edges(self, db, edge_type):
        rows = db.execute(f"MATCH (a)-[:{edge_type}]->(b) RETURN id(a) AS a, id(b) AS b ORDER BY a")
        return [(row["a"], row["b"]) for row in rows]

    def test_session_chain(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
        manager = _make_manager([], run_id="run-1")
        ids = [int(manager.add(f"step {i}", infer=False)[0].memory_id) for i in range(3)]
        assert self._edges(manager._db, "LEADS_TO") == [(ids[0], ids[1]), (ids[1], ids[2])]
        manager.close()

    def test_episode_chain(self, monkeypatch):
        monkeypatch.setattr(time, "time", lambda: 1_700_000_000.0)
        manager = _make_manager([], run_id="run-1", enable_episodes=True)
        for i in range(3):
            manager.add(f"step {i}", infer=False)
        episodes = [int(e.episode_id) for e in manager.get_episodes()]
        assert len(episodes) == 3
        assert self._edges(manager._db, "NEXT_EPISODE") == [(episodes[0], episodes[1]), (episodes[1], episodes[2])]
        manager.close()
