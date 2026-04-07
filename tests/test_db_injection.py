"""Tests for injecting an external GrafeoDB instance into MemoryManager."""

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryAction, MemoryConfig, MemoryManager


def _extraction_output():
    """Standard extraction output for a single fact."""
    return {
        "facts": ["alice works at acme corp"],
        "entities": [
            {"name": "alice", "entity_type": "person"},
            {"name": "acme_corp", "entity_type": "organization"},
        ],
        "relations": [
            {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
        ],
    }


class TestDBInjection:
    """Test that an external GrafeoDB can be injected into MemoryManager."""

    def test_inject_db_into_memory_manager(self):
        """MemoryManager should use the provided db instead of creating its own."""
        db = grafeo.GrafeoDB()
        model = make_test_model([_extraction_output()])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        manager = MemoryManager(model, config, embedder=embedder, db=db)
        assert manager._db is db
        assert manager._db_external is True

        events = manager.add("Alice works at Acme Corp")
        assert len(events) >= 1
        assert events[0].action == MemoryAction.ADD

        # Verify data was written to the shared db
        memories = db.get_nodes_by_label("Memory")
        assert len(memories) >= 1

        manager.close()

    def test_inject_db_into_async_memory_manager(self):
        """AsyncMemoryManager should use the provided db instead of creating its own."""
        import asyncio

        db = grafeo.GrafeoDB()
        model = make_test_model([_extraction_output()])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        manager = AsyncMemoryManager(model, config, embedder=embedder, db=db)
        assert manager._db is db
        assert manager._db_external is True

        events = asyncio.run(manager.add("Alice works at Acme Corp"))
        assert len(events) >= 1

        manager.close()

    def test_default_creates_internal_db(self):
        """Omitting db should preserve existing behavior — internal DB is created."""
        model = make_test_model([{"facts": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        manager = MemoryManager(model, config, embedder=embedder)
        assert manager._db_external is False
        assert manager._db is not None

        manager.close()


class TestCloseLifecycle:
    """Test that close() respects external DB ownership."""

    def test_close_does_not_close_external_db(self):
        """When db is externally provided, close() should leave it open."""
        db = grafeo.GrafeoDB()
        model = make_test_model([{"facts": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        manager = MemoryManager(model, config, embedder=embedder, db=db)
        manager.close()

        # DB should still be usable after manager.close()
        node = db.create_node(["Test"], {"name": "after_close"})
        assert node.id is not None

    def test_close_closes_internal_db(self):
        """When db is internally created, close() should close it."""
        model = make_test_model([{"facts": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        manager = MemoryManager(model, config, embedder=embedder)
        manager.close()
        # Internal DB should be closed — we just verify close() ran without error

    def test_context_manager_does_not_close_external_db(self):
        """Using `with` statement should not close an external db."""
        db = grafeo.GrafeoDB()
        model = make_test_model([{"facts": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)

        with MemoryManager(model, config, embedder=embedder, db=db):
            pass

        # DB should still be usable
        node = db.create_node(["Test"], {"name": "after_context"})
        assert node.id is not None
