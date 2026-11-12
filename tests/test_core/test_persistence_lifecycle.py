"""Tests for persistence round-trip and multi-session lifecycle.

Covers T4 (write/close/reopen) and T5 (multi-session sequential open/close).
"""

from __future__ import annotations

from conftest import make_manager
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager


class TestPersistenceRoundTrip:
    """T4: write/close/reopen should preserve all memories."""

    def test_memories_persist_across_close_reopen(self, tmp_path):
        db_file = str(tmp_path / "test.db")

        # Session 1: add memories
        manager = make_manager([], db_path=db_file)
        manager.add("fact one", infer=False)
        manager.add("fact two", infer=False)
        manager.add("fact three", infer=False)
        manager.close()

        # Session 2: reopen and verify
        manager2 = make_manager([], db_path=db_file)
        memories = manager2.get_all()
        texts = sorted(m.text for m in memories)
        assert texts == ["fact one", "fact three", "fact two"]
        manager2.close()

    def test_search_works_after_reopen(self, tmp_path):
        db_file = str(tmp_path / "test.db")

        # Session 1: add memories
        manager = make_manager(
            [{"entities": [], "relations": []}],
            db_path=db_file,
        )
        manager.add("alice likes hiking", infer=False)
        manager.close()

        # Session 2: search
        manager2 = make_manager(
            [{"entities": [], "relations": []}],
            db_path=db_file,
        )
        results = manager2.search("hiking")
        assert len(results) >= 1
        assert any("hiking" in r.text for r in results)
        manager2.close()


class TestMultiSessionSequential:
    """T5: open/close 3 managers sequentially in the same process."""

    def test_three_sessions_all_data_accessible(self, tmp_path):
        db_file = str(tmp_path / "multi.db")

        # Session 1
        m1 = make_manager([], db_path=db_file)
        m1.add("session one fact", infer=False)
        m1.close()

        # Session 2
        m2 = make_manager([], db_path=db_file)
        m2.add("session two fact", infer=False)
        m2.close()

        # Session 3
        m3 = make_manager([], db_path=db_file)
        m3.add("session three fact", infer=False)

        # All memories from all sessions should be accessible
        memories = m3.get_all()
        texts = sorted(m.text for m in memories)
        assert texts == ["session one fact", "session three fact", "session two fact"]
        m3.close()

    def test_context_manager_sequential_sessions(self, tmp_path):
        db_file = str(tmp_path / "ctx.db")
        embedder = MockEmbedder(16)

        for i in range(3):
            model = make_test_model([])
            config = MemoryConfig(db_path=db_file, user_id="test_user", embedding_dimensions=16)
            with MemoryManager(model, config, embedder=embedder) as mem:
                mem.add(f"fact from session {i}", infer=False)

        # Final session: verify all facts present
        model = make_test_model([])
        config = MemoryConfig(db_path=db_file, user_id="test_user", embedding_dimensions=16)
        with MemoryManager(model, config, embedder=embedder) as mem:
            memories = mem.get_all()
            assert len(memories) == 3


class TestSingleFileDatabase:
    """A ``.grafeo`` path is a single-file database, the default format from grafeo 0.6."""

    def test_round_trip_with_graph_search_and_indexes(self, tmp_path):
        path = tmp_path / "memory.grafeo"
        alice = {"name": "alice", "entity_type": "person"}

        manager = make_manager(
            [{"facts": ["alice likes hiking"], "entities": [alice], "relations": []}], db_path=str(path)
        )
        mid = manager.add("Alice likes hiking")[0].memory_id
        manager.add("unrelated note", infer=False)
        manager.close()
        assert path.is_file()

        manager = make_manager([{"entities": [alice], "relations": []}], db_path=str(path))
        try:
            results = manager.search("alice likes hiking")
            assert results[0].memory_id == mid
            assert results[0].source == "both", "vector and graph search both work after reopen"
            assert manager.stats().entity_count == 1
            for prop in ("user_id", "memory_id"):
                assert manager._db.has_property_index(prop)
        finally:
            manager.close()

    def test_close_releases_the_file(self, tmp_path):
        path = str(tmp_path / "memory.grafeo")
        for i in range(3):
            with make_manager([], db_path=path) as manager:
                manager.add(f"fact {i}", infer=False)
        with make_manager([], db_path=path) as manager:
            assert sorted(m.text for m in manager.get_all()) == ["fact 0", "fact 1", "fact 2"]
