"""Tests that grafeo-memory operations are scoped to Memory/Entity nodes.

These tests create a shared in-memory GrafeoDB containing foreign "Domain"
nodes alongside Memory/Entity nodes, then verify grafeo-memory never reads,
modifies, or deletes the foreign nodes.
"""

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.search.vector import _get_props
from grafeo_memory.types import ENTITY_LABEL, MEMORY_LABEL


def _extraction_output(facts=None, entities=None, relations=None):
    return {
        "facts": facts or ["alice works at acme corp"],
        "entities": entities
        or [
            {"name": "alice", "entity_type": "person"},
            {"name": "acme_corp", "entity_type": "organization"},
        ],
        "relations": relations or [{"source": "alice", "target": "acme_corp", "relation_type": "works_at"}],
    }


def _make_shared_manager(db, outputs, dims=16, **config_kwargs):
    """Create a MemoryManager backed by a shared database."""
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder, db=db)


def _add_foreign_nodes(db):
    """Add non-memory nodes to the database that grafeo-memory must not touch."""
    n1 = db.create_node(["Domain", "CADPart"], {"name": "cylinder", "radius": 5.0})
    n2 = db.create_node(["Domain", "CADPart"], {"name": "base_plate", "thickness": 2.0})
    n3 = db.create_node(["Timeline"], {"step": 1, "operation": "extrude"})
    db.create_edge(n1.id, n2.id, "CONNECTED_TO", {"joint": "weld"})
    db.create_edge(n3.id, n1.id, "PRODUCES")
    return [n1, n2, n3]


class TestSharedDBMetrics:
    """Graph algorithm scores must only be written to Memory/Entity nodes."""

    def test_metrics_skip_foreign_nodes(self):
        db = grafeo.GrafeoDB()
        foreign = _add_foreign_nodes(db)

        manager = _make_shared_manager(
            db,
            [_extraction_output(), _extraction_output(["bob likes hiking"])],
            enable_graph_algorithms=True,
        )
        manager.add("Alice works at Acme Corp")
        assert manager._graph_dirty is True

        manager._recompute_graph_metrics()

        # Foreign nodes must NOT have algorithm properties
        for fnode in foreign:
            node = db.get_node(fnode.id)
            assert node is not None
            props = _get_props(node)
            assert "_pagerank" not in props, f"Foreign node {fnode.id} got _pagerank"
            assert "_betweenness" not in props, f"Foreign node {fnode.id} got _betweenness"
            assert "_community" not in props, f"Foreign node {fnode.id} got _community"

        # Memory/Entity nodes SHOULD have algorithm properties
        memory_nodes = db.get_nodes_by_label(MEMORY_LABEL)
        assert len(memory_nodes) > 0
        for mid, props in memory_nodes:
            assert "_pagerank" in props, f"Memory node {mid} missing _pagerank"

        manager.close()

    def test_foreign_node_data_unchanged(self):
        """Foreign node properties must be completely untouched after memory operations."""
        db = grafeo.GrafeoDB()
        foreign = _add_foreign_nodes(db)

        manager = _make_shared_manager(db, [_extraction_output()])
        manager.add("Alice works at Acme Corp")
        manager.close()

        # Verify foreign node properties are exactly as created
        cyl = db.get_node(foreign[0].id)
        assert cyl is not None
        props = _get_props(cyl)
        assert props["name"] == "cylinder"
        assert props["radius"] == 5.0


class TestSharedDBEntityLookup:
    """Entity lookup must not match foreign nodes with the same name."""

    def test_foreign_node_with_same_name_not_reused(self):
        db = grafeo.GrafeoDB()
        # Create a foreign node named "alice" — NOT an Entity
        db.create_node(["Person"], {"name": "alice", "user_id": "test_user"})

        manager = _make_shared_manager(db, [_extraction_output()])
        manager.add("Alice works at Acme Corp")

        # A new Entity node should have been created, not reusing the Person node
        entity_nodes = db.get_nodes_by_label(ENTITY_LABEL)
        entity_names = [props.get("name") for _, props in entity_nodes]
        assert "alice" in entity_names

        # The Person node should still exist untouched
        person_nodes = db.get_nodes_by_label("Person")
        assert len(person_nodes) == 1

        manager.close()


class TestSharedDBTemporalChain:
    """temporal_chain must only traverse Memory nodes."""

    def test_temporal_chain_ignores_foreign_nodes(self):
        db = grafeo.GrafeoDB()

        # Create a foreign node and a Memory node, connect them with LEADS_TO
        foreign = db.create_node(["Domain"], {"name": "step_1", "text": "foreign data"})
        memory = db.create_node(
            [MEMORY_LABEL],
            {"text": "a real memory", "user_id": "test_user", "memory_type": "semantic"},
        )
        db.create_edge(foreign.id, memory.id, "LEADS_TO")

        manager = _make_shared_manager(db, [{"facts": []}])

        # Query temporal chain starting from the foreign node — should get nothing
        # because the query now requires (m:Memory)
        result = manager.temporal_chain(str(foreign.id), direction="forward")
        assert result == []

        manager.close()


class TestSharedDBStats:
    """Stats must only report memory-scoped counts."""

    def test_stats_exclude_foreign_nodes(self):
        db = grafeo.GrafeoDB()
        _add_foreign_nodes(db)

        manager = _make_shared_manager(db, [_extraction_output()])
        manager.add("Alice works at Acme Corp")

        stats = manager.stats()
        # Stats should reflect only Memory/Entity nodes, not the 3 foreign nodes
        assert stats.total_memories >= 1
        assert stats.entity_count >= 1

        # db_info should contain scoped counts, not raw db.info()
        assert "memory_node_count" in stats.db_info
        assert stats.db_info["memory_node_count"] == stats.total_memories
        assert stats.db_info["entity_node_count"] == stats.entity_count

        manager.close()

    def test_stats_db_info_has_no_total_node_count(self):
        """db_info must not leak total database node counts."""
        db = grafeo.GrafeoDB()
        _add_foreign_nodes(db)

        manager = _make_shared_manager(db, [{"facts": []}])
        stats = manager.stats()

        # Should not contain keys from raw db.info() like "node_count" or "total_nodes"
        for key in stats.db_info:
            assert key in ("memory_node_count", "entity_node_count", "relation_edge_count"), (
                f"Unexpected key in db_info: {key}"
            )

        manager.close()


class TestSharedDBDeleteAll:
    """delete_all must only remove Memory nodes, not foreign nodes."""

    def test_delete_all_preserves_foreign_nodes(self):
        db = grafeo.GrafeoDB()
        foreign = _add_foreign_nodes(db)

        manager = _make_shared_manager(db, [_extraction_output()])
        manager.add("Alice works at Acme Corp")
        assert manager.stats().total_memories >= 1

        manager.delete_all()
        assert manager.stats().total_memories == 0

        # All foreign nodes must still exist
        for fnode in foreign:
            assert db.get_node(fnode.id) is not None

        manager.close()
