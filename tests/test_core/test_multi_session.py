"""Regression test for B3: event loop corruption with multiple MemoryManager sessions.

On Windows, opening and closing multiple MemoryManager instances in the same process
could corrupt the asyncio event loop (ProactorEventLoop teardown). This test verifies
that sequential sessions work correctly without event loop errors.
"""

from __future__ import annotations

from conftest import make_manager
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager


class TestMultiSessionEventLoop:
    """B3 regression: multiple sessions must not corrupt the event loop."""

    def test_open_add_close_three_times_in_memory(self):
        """Open, add, close three managers sequentially with in-memory DB."""
        for i in range(3):
            manager = make_manager([])
            events = manager.add(f"fact {i}", infer=False)
            assert len(events) == 1
            assert events[0].text == f"fact {i}"
            manager.close()

    def test_open_add_close_three_times_persistent(self, tmp_path):
        """Open, add, close three managers sequentially with persistent DB.

        After the third session, search should find all memories.
        """
        db_file = str(tmp_path / "b3_regression.db")

        for i in range(3):
            manager = make_manager([], db_path=db_file)
            manager.add(f"fact {i}", infer=False)
            manager.close()

        # Reopen and verify all memories are present
        manager = make_manager(
            [{"entities": [], "relations": []}],
            db_path=db_file,
        )
        memories = manager.get_all()
        assert len(memories) == 3
        texts = sorted(m.text for m in memories)
        assert texts == ["fact 0", "fact 1", "fact 2"]

        # Search should also work (exercises the event loop)
        results = manager.search("fact")
        assert len(results) >= 1
        manager.close()

    def test_context_manager_reuse_pattern(self):
        """Context manager pattern: sequential with blocks."""
        embedder = MockEmbedder(16)
        results_per_session = []

        for i in range(3):
            model = make_test_model([])
            config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
            with MemoryManager(model, config, embedder=embedder) as mem:
                events = mem.add(f"session {i} fact", infer=False)
                results_per_session.append(events)

        # All three sessions should have succeeded
        assert len(results_per_session) == 3
        for events in results_per_session:
            assert len(events) == 1

    def test_add_with_infer_across_sessions(self):
        """Ensure LLM-based extraction works across multiple sessions."""
        for i in range(3):
            model = make_test_model([{"facts": [f"fact {i}"], "entities": [], "relations": []}])
            embedder = MockEmbedder(16)
            config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
            with MemoryManager(model, config, embedder=embedder) as mem:
                events = mem.add(f"Some text for session {i}")
                assert len(events) >= 1
