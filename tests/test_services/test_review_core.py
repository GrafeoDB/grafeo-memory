"""0.3.0 review regressions in the manager: decision-to-fact association, manager
scopes on writes by id, update dimension check, scoped legacy history."""

from __future__ import annotations

import grafeo
import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.common.types import MemoryAction, ReconciliationDecision
from grafeo_memory.core.history import HAS_HISTORY_EDGE, HISTORY_LABEL
from grafeo_memory.services.memory import _fact_slots


def _make_manager(outputs=(), *, db=None, **kwargs) -> MemoryManager:
    defaults = {"user_id": "test_user", "embedding_dimensions": 16}
    defaults.update(kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(make_test_model(list(outputs)), config, embedder=MockEmbedder(16), db=db)


def _add(text: str) -> ReconciliationDecision:
    return ReconciliationDecision(action=MemoryAction.ADD, text=text)


class TestFactSlots:
    def test_one_per_fact_keeps_position_for_reworded_text(self):
        assert _fact_slots([_add("A, reworded"), _add("b")], ["a", "b"]) == [0, 1]

    def test_reordered_answers_follow_their_text(self):
        assert _fact_slots([_add("b"), _add("a")], ["a", "b"]) == [1, 0]

    def test_count_mismatch_matches_by_text_only(self):
        assert _fact_slots([_add("b"), _add("extra"), _add("a")], ["a", "b"]) == [1, None, 0]


class TestDecisionsGetTheirFactsEmbedding:
    def _stored_vector(self, manager: MemoryManager, memory_id: str) -> list[float]:
        return [float(x) for x in manager._db.get_node(int(memory_id)).properties()["embedding"]]

    def test_reordered_and_extra_decisions(self):
        manager = _make_manager(
            [
                {"facts": ["alice likes tea", "alice likes chess"], "entities": [], "relations": []},
                {
                    "decisions": [
                        {"action": "add", "text": "alice likes chess"},
                        {"action": "add", "text": "alice plays chess on sundays"},
                        {"action": "add", "text": "alice likes tea"},
                    ]
                },
            ],
            reconciliation_threshold=0.0,
        )
        manager.add("seed memory", infer=False)
        events = [e for e in manager.add("Alice likes tea and chess") if e.action == MemoryAction.ADD]
        assert len(events) == 3
        embedder = MockEmbedder(16)
        for e in events:
            assert e.memory_id is not None and e.text is not None
            assert self._stored_vector(manager, e.memory_id) == pytest.approx(embedder.embed([e.text])[0])
        manager.close()


class TestManagerScopesOnWritesById:
    def test_other_agent_cannot_modify(self):
        db = grafeo.GrafeoDB()
        mid = _make_manager(db=db, agent_id="agent-a").add("agent a note", infer=False)[0].memory_id
        other = _make_manager(db=db, agent_id="agent-b")
        assert other.delete(mid) is False
        assert other.set_importance(mid, 0.5) is False
        with pytest.raises(ValueError):
            other.update(mid, "rewritten")
        assert db.get_node(int(mid)).properties()["text"] == "agent a note"
        assert _make_manager(db=db, agent_id="agent-a").delete(mid) is True

    def test_update_checks_dimensions_before_writing(self):
        manager = _make_manager()
        mid = manager.add("original", infer=False)[0].memory_id
        manager._embedder = MockEmbedder(8)
        with pytest.raises(ValueError, match="dimension"):
            manager.update(mid, "changed")
        assert manager._db.get_node(int(mid)).properties()["text"] == "original"
        manager.close()


class TestScopedLegacyHistory:
    def test_graph_scoped_manager_reads_legacy_entries_before_and_after_delete(self):
        manager = _make_manager(graph_name="team-a")
        db = manager._db
        mid = manager.add("legacy memory", infer=False)[0].memory_id
        legacy = db.create_node([HISTORY_LABEL], {"event": "UPDATE", "old_text": "a", "new_text": "b", "timestamp": 0})
        db.create_edge(int(mid), legacy.id, HAS_HISTORY_EDGE)
        assert [h.event for h in manager.history(mid)] == ["UPDATE", "ADD"]

        assert manager.delete(mid) is True
        assert [h.event for h in manager.history(mid)] == ["UPDATE", "ADD", "DELETE"]
        assert _make_manager(db=db, graph_name="team-b").history(mid) == []
        manager.close()


class TestLegacyMemoryTypeInSearch:
    def test_semantic_filter_keeps_memories_without_memory_type(self):
        from grafeo_memory.core.search.vector import hybrid_search, vector_search

        manager = _make_manager()
        manager.add("alice likes green tea", infer=False)
        manager.add("brew tea at 80 degrees", infer=False, memory_type="procedural")
        legacy = manager.add("bob likes black tea", infer=False)[0].memory_id
        assert legacy is not None
        manager._db.remove_node_property(int(legacy), "memory_type")
        embedder = MockEmbedder(16)
        for search in (vector_search, hybrid_search):
            hits = search(manager._db, embedder, "tea", user_id="test_user", k=5, filters={"memory_type": "semantic"})
            texts = {h.text for h in hits}
            assert texts == {"alice likes green tea", "bob likes black tea"}, search.__name__
        manager.close()


def _facts(*facts: str) -> dict:
    return {"facts": list(facts), "entities": [], "relations": []}


class TestSupersededMemoriesStayOut:
    """Dogfooding 0.3.0: an UPDATE expired the old memory, and the next add() offered that expired
    memory as a candidate again, so the reconciler "updated" it a second time and the live memory
    stayed next to the newer fact."""

    def test_expired_memory_is_no_reconciliation_candidate(self):
        from grafeo_memory.core.search.vector import search_similar

        manager = _make_manager(
            [_facts("alice works at acme"), {"decisions": [{"action": "add", "text": "alice works at acme"}]}]
        )
        mid = manager.add("Alice works at Acme")[0].memory_id
        emb = manager._embedder.embed(["alice works at acme"])
        assert [c["id"] for c in search_similar(manager._db, emb, user_id="test_user", threshold=0.0)] == [mid]
        manager._expire_memory(mid, 1)
        assert search_similar(manager._db, emb, user_id="test_user", threshold=0.0) == []
        manager.close()

    def test_update_of_a_superseded_memory_becomes_add(self):
        db = grafeo.GrafeoDB()
        first = _make_manager(
            [_facts("alice works at acme"), {"decisions": [{"action": "add", "text": "alice works at acme"}]}], db=db
        )
        old = first.add("Alice works at Acme")[0].memory_id
        second = _make_manager(
            [
                _facts("alice is a senior engineer at acme"),
                {
                    "decisions": [
                        # old is the only candidate, shown to the model as "m1"
                        {"action": "update", "text": "alice is a senior engineer at acme", "target_memory_id": "m1"}
                    ]
                },
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        promoted = second.add("Alice was promoted")
        assert [e.action for e in promoted] == [MemoryAction.UPDATE], "the setup supersedes old"
        live = promoted[0].memory_id
        third = _make_manager(
            [
                _facts("alice works at beta"),
                # old is expired, so it has no alias; its real id is not a valid target
                {"decisions": [{"action": "update", "text": "alice works at beta", "target_memory_id": old}]},
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        events = third.add("Alice joined Beta")
        assert [e.action for e in events] == [MemoryAction.ADD], "the expired memory is not a valid target"
        assert {m.memory_id for m in third.get_all()} >= {live}
        first.close()

    def test_two_updates_of_one_memory_in_one_add(self):
        db = grafeo.GrafeoDB()
        first = _make_manager(
            [_facts("alice works at acme"), {"decisions": [{"action": "add", "text": "alice works at acme"}]}], db=db
        )
        old = first.add("Alice works at Acme")[0].memory_id
        second = _make_manager(
            [
                _facts("alice left acme", "alice joined beta"),
                {
                    "decisions": [
                        # old is the only candidate, shown to the model as "m1"
                        {"action": "update", "text": "alice left acme", "target_memory_id": "m1"},
                        {"action": "update", "text": "alice joined beta", "target_memory_id": "m1"},
                    ]
                },
            ],
            db=db,
            reconciliation_threshold=0.0,
        )
        events = second.add("I left Acme and joined Beta")
        assert [e.action for e in events] == [MemoryAction.UPDATE, MemoryAction.ADD]
        assert old not in {m.memory_id for m in second.get_all()}  # superseded once, by the first update
        assert sorted(m.text for m in second.get_all()) == ["alice joined beta", "alice left acme"]
        first.close()
