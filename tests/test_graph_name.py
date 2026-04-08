"""Tests for graph_name scoping in MemoryConfig.

Two MemoryManagers sharing a single GrafeoDB instance, each with a
different graph_name, should be fully isolated: memories, entities,
search results, stats, and temporal chains should not bleed across.
"""

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryAction, MemoryConfig, MemoryManager
from grafeo_memory.types import ENTITY_LABEL, MEMORY_LABEL


def _extraction(facts=None, entities=None, relations=None):
    return {
        "facts": facts or ["alice works at acme corp"],
        "entities": entities
        or [
            {"name": "alice", "entity_type": "person"},
            {"name": "acme_corp", "entity_type": "organization"},
        ],
        "relations": relations or [{"source": "alice", "target": "acme_corp", "relation_type": "works_at"}],
    }


def _make_manager(db, graph_name, outputs, dims=16, **kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {
        "db_path": None,
        "user_id": "test_user",
        "embedding_dimensions": dims,
        "graph_name": graph_name,
    }
    defaults.update(kwargs)
    config = MemoryConfig(**defaults)
    return MemoryManager(model, config, embedder=embedder, db=db)


class TestGraphNameWriteScoping:
    """Nodes created under a graph_name carry the property."""

    def test_memory_node_stamped_with_graph_name(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, "design_a", [_extraction()])
        mgr.add("Alice works at Acme Corp")

        for _nid, props in db.get_nodes_by_label(MEMORY_LABEL):
            assert props.get("graph_name") == "design_a"

        mgr.close()

    def test_entity_node_stamped_with_graph_name(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, "design_a", [_extraction()])
        mgr.add("Alice works at Acme Corp")

        for _nid, props in db.get_nodes_by_label(ENTITY_LABEL):
            assert props.get("graph_name") == "design_a"

        mgr.close()

    def test_no_graph_name_means_no_property(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, None, [_extraction()])
        mgr.add("Alice works at Acme Corp")

        for _nid, props in db.get_nodes_by_label(MEMORY_LABEL):
            assert "graph_name" not in props

        for _nid, props in db.get_nodes_by_label(ENTITY_LABEL):
            assert "graph_name" not in props

        mgr.close()

    def test_batch_add_stamps_graph_name(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, "batch_graph", [_extraction()])
        mgr.add("Alice works at Acme Corp", infer=False)

        for _nid, props in db.get_nodes_by_label(MEMORY_LABEL):
            assert props.get("graph_name") == "batch_graph"

        mgr.close()


class TestGraphNameReadIsolation:
    """Queries from one graph_name must not see nodes from another."""

    def test_search_isolated_between_graphs(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [_extraction()])
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")

        results_a = mgr_a.search("Alice", user_id="test_user")
        results_b = mgr_b.search("Bob", user_id="test_user")

        texts_a = [r.text for r in results_a]
        texts_b = [r.text for r in results_b]

        # graph_a should only see alice's memory
        assert any("alice" in t.lower() or "acme" in t.lower() for t in texts_a)
        assert not any("bob" in t.lower() or "hiking" in t.lower() for t in texts_a)

        # graph_b should only see bob's memory
        assert any("bob" in t.lower() or "hiking" in t.lower() for t in texts_b)
        assert not any("alice" in t.lower() or "acme" in t.lower() for t in texts_b)

        mgr_a.close()
        mgr_b.close()

    def test_get_all_isolated_between_graphs(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [_extraction()])
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")

        all_a = mgr_a.get_all(user_id="test_user")
        all_b = mgr_b.get_all(user_id="test_user")

        assert len(all_a) == 1
        assert len(all_b) == 1
        assert "alice" in all_a[0].text.lower() or "acme" in all_a[0].text.lower()
        assert "bob" in all_b[0].text.lower() or "hiking" in all_b[0].text.lower()

        mgr_a.close()
        mgr_b.close()

    def test_no_graph_name_sees_everything(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [_extraction()])
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ])
        mgr_all = _make_manager(db, None, [])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")

        all_memories = mgr_all.get_all(user_id="test_user")
        assert len(all_memories) == 2

        mgr_a.close()
        mgr_b.close()
        mgr_all.close()


class TestGraphNameEntityIsolation:
    """Same entity name in different graphs should be separate nodes."""

    def test_same_entity_different_graphs(self):
        db = grafeo.GrafeoDB()
        extraction_with_alice = _extraction()

        mgr_a = _make_manager(db, "graph_a", [extraction_with_alice])
        mgr_b = _make_manager(db, "graph_b", [extraction_with_alice])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Alice works at Acme Corp")

        # Should have two separate "alice" entity nodes
        entity_nodes = db.get_nodes_by_label(ENTITY_LABEL)
        alice_nodes = [
            (nid, props) for nid, props in entity_nodes
            if props.get("name") == "alice"
        ]
        assert len(alice_nodes) == 2

        graphs = {props.get("graph_name") for _, props in alice_nodes}
        assert graphs == {"graph_a", "graph_b"}

        mgr_a.close()
        mgr_b.close()


class TestGraphNameStats:
    """stats() should return scoped counts when graph_name is set."""

    def test_stats_scoped_to_graph(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [_extraction()])
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
            _extraction(facts=["bob plays guitar"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")
        mgr_b.add("Bob plays guitar")

        stats_a = mgr_a.stats()
        stats_b = mgr_b.stats()

        assert stats_a.total_memories == 1
        assert stats_b.total_memories == 2

        # Entity counts should be scoped too
        assert stats_a.entity_count == 2  # alice + acme_corp
        assert stats_b.entity_count == 1  # bob

        mgr_a.close()
        mgr_b.close()

    def test_stats_no_graph_name_counts_all(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [_extraction()])
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ])
        mgr_all = _make_manager(db, None, [])

        mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")

        stats_all = mgr_all.stats()
        assert stats_all.total_memories == 2

        mgr_a.close()
        mgr_b.close()
        mgr_all.close()


class TestGraphNameTemporalChain:
    """temporal_chain should not cross graph boundaries."""

    def test_temporal_chain_scoped(self):
        db = grafeo.GrafeoDB()
        mgr_a = _make_manager(db, "graph_a", [
            _extraction(),
            _extraction(facts=["alice got promoted"]),
        ], run_id="session1")
        mgr_b = _make_manager(db, "graph_b", [
            _extraction(facts=["bob likes hiking"], entities=[{"name": "bob", "entity_type": "person"}], relations=[]),
        ], run_id="session1")

        events_a = mgr_a.add("Alice works at Acme Corp")
        mgr_b.add("Bob likes hiking")
        mgr_a.add("Alice got promoted")

        # Get chain from graph_a's first memory
        first_id = events_a[0].memory_id
        chain = mgr_a.temporal_chain(first_id, user_id="test_user")

        # Chain should only contain graph_a memories, not graph_b
        chain_texts = [c["text"] for c in chain]
        for text in chain_texts:
            assert "bob" not in text.lower()

        mgr_a.close()
        mgr_b.close()


class TestGraphNameBuildFilters:
    """_build_filters should include graph_name when configured."""

    def test_build_filters_with_graph_name(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, "my_graph", [])
        filters = mgr._build_filters("test_user")
        assert filters["graph_name"] == "my_graph"
        mgr.close()

    def test_build_filters_without_graph_name(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, None, [])
        filters = mgr._build_filters("test_user")
        assert "graph_name" not in filters
        mgr.close()


class TestGraphNameBackwardsCompat:
    """Existing behavior unchanged when graph_name is not set."""

    def test_default_config_has_no_graph_name(self):
        config = MemoryConfig()
        assert config.graph_name is None

    def test_manager_without_graph_name_works(self):
        db = grafeo.GrafeoDB()
        mgr = _make_manager(db, None, [_extraction()])
        events = mgr.add("Alice works at Acme Corp")
        assert len(events) >= 1
        assert events[0].action == MemoryAction.ADD

        results = mgr.search("Alice", user_id="test_user")
        assert len(results) >= 1
        mgr.close()
