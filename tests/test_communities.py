"""Tests for community summaries: Community nodes, HAS_MEMBER edges, LLM summaries."""

from __future__ import annotations

import grafeo
import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.communities import get_communities, get_community_context, materialize_communities_async
from grafeo_memory.types import ENTITY_LABEL


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    db = config_kwargs.pop("db", None)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return (
        MemoryManager(model, config, embedder=embedder, db=db)
        if db
        else MemoryManager(model, config, embedder=embedder)
    )


def _setup_entity_graph(db):
    """Create a small entity graph with two clusters for testing community detection."""
    # Cluster 1: alice, bob at acme
    e1 = db.create_node([ENTITY_LABEL], {"name": "alice", "entity_type": "person", "user_id": "test_user"})
    e2 = db.create_node([ENTITY_LABEL], {"name": "bob", "entity_type": "person", "user_id": "test_user"})
    e3 = db.create_node([ENTITY_LABEL], {"name": "acme", "entity_type": "org", "user_id": "test_user"})

    # Cluster 2: charlie, diana at globex
    e4 = db.create_node([ENTITY_LABEL], {"name": "charlie", "entity_type": "person", "user_id": "test_user"})
    e5 = db.create_node([ENTITY_LABEL], {"name": "diana", "entity_type": "person", "user_id": "test_user"})
    e6 = db.create_node([ENTITY_LABEL], {"name": "globex", "entity_type": "org", "user_id": "test_user"})

    # Get node IDs
    ids = {}
    for node in [e1, e2, e3, e4, e5, e6]:
        nid = node.id if hasattr(node, "id") else node
        ids[nid] = nid

    e1_id = e1.id if hasattr(e1, "id") else e1
    e2_id = e2.id if hasattr(e2, "id") else e2
    e3_id = e3.id if hasattr(e3, "id") else e3
    e4_id = e4.id if hasattr(e4, "id") else e4
    e5_id = e5.id if hasattr(e5, "id") else e5
    e6_id = e6.id if hasattr(e6, "id") else e6

    # Relations within clusters
    db.create_edge(e1_id, e3_id, "RELATION", {"relation_type": "works_at"})
    db.create_edge(e2_id, e3_id, "RELATION", {"relation_type": "works_at"})
    db.create_edge(e4_id, e6_id, "RELATION", {"relation_type": "works_at"})
    db.create_edge(e5_id, e6_id, "RELATION", {"relation_type": "works_at"})

    return {
        "cluster1": [e1_id, e2_id, e3_id],
        "cluster2": [e4_id, e5_id, e6_id],
    }


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestCommunityConfig:
    def test_requires_graph_algorithms(self):
        """enable_community_summaries=True requires enable_graph_algorithms=True."""
        with pytest.raises(ValueError, match="enable_community_summaries requires"):
            MemoryConfig(enable_community_summaries=True, enable_graph_algorithms=False)

    def test_valid_config(self):
        """Both enabled together is valid."""
        config = MemoryConfig(enable_community_summaries=True, enable_graph_algorithms=True)
        assert config.enable_community_summaries is True


# ---------------------------------------------------------------------------
# Community materialization
# ---------------------------------------------------------------------------


class TestCommunityMaterialization:
    def test_communities_created_from_louvain(self):
        """materialize_communities_async creates Community nodes from Louvain result."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        # Simulate louvain result: cluster1 entities in community 0, cluster2 in community 1
        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
                **{nid: 1 for nid in clusters["cluster2"]},
            }
        }

        model = make_test_model(
            [
                # Community 0 summary
                {"name": "Acme Team", "summary": "Alice and Bob work at Acme."},
                # Community 1 summary
                {"name": "Globex Team", "summary": "Charlie and Diana work at Globex."},
            ]
        )

        result = asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        assert len(result) == 2
        names = {c.name for c in result}
        assert "Acme Team" in names
        assert "Globex Team" in names

    def test_single_entity_not_community(self):
        """A community with only 1 member is not materialized."""
        import asyncio

        db = grafeo.GrafeoDB()
        e1 = db.create_node([ENTITY_LABEL], {"name": "lonely", "entity_type": "person", "user_id": "test_user"})
        e1_id = e1.id if hasattr(e1, "id") else e1

        louvain_result = {"communities": {e1_id: 0}}
        model = make_test_model([])

        result = asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))
        assert len(result) == 0

    def test_community_not_created_when_disabled(self):
        """When enable_community_summaries=False, no Community nodes are created."""
        db = grafeo.GrafeoDB()
        _setup_entity_graph(db)

        # Manager without community summaries
        manager = _make_manager([], db=db)
        communities = manager.get_communities()
        assert len(communities) == 0
        manager.close()


# ---------------------------------------------------------------------------
# Get communities
# ---------------------------------------------------------------------------


class TestGetCommunities:
    def test_get_communities_returns_all(self):
        """get_communities returns all Community nodes for user."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
                **{nid: 1 for nid in clusters["cluster2"]},
            }
        }

        model = make_test_model(
            [
                {"name": "Team A", "summary": "First team."},
                {"name": "Team B", "summary": "Second team."},
            ]
        )

        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        communities = get_communities(db, "test_user")
        assert len(communities) == 2

    def test_get_communities_empty_without_data(self):
        """No entities -> no communities."""
        db = grafeo.GrafeoDB()
        communities = get_communities(db, "test_user")
        assert len(communities) == 0


# ---------------------------------------------------------------------------
# Community context search
# ---------------------------------------------------------------------------


class TestCommunityContext:
    def test_community_context_for_entity(self):
        """get_community_context returns communities containing the queried entity."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
                **{nid: 1 for nid in clusters["cluster2"]},
            }
        }

        model = make_test_model(
            [
                {"name": "Acme Team", "summary": "Alice and Bob at Acme."},
                {"name": "Globex Team", "summary": "Charlie and Diana at Globex."},
            ]
        )

        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        # Query for "alice" -> should find the Acme Team community
        context = get_community_context(db, ["alice"], "test_user")
        assert len(context) == 1
        assert context[0].name == "Acme Team"

    def test_community_context_no_match(self):
        """get_community_context returns empty when no entity matches."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
            }
        }

        model = make_test_model(
            [
                {"name": "Acme Team", "summary": "Alice and Bob at Acme."},
            ]
        )

        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        context = get_community_context(db, ["unknown_person"], "test_user")
        assert len(context) == 0


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestCommunityStats:
    def test_stats_includes_community_count(self):
        """stats() returns correct community_count."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
                **{nid: 1 for nid in clusters["cluster2"]},
            }
        }

        model = make_test_model(
            [
                {"name": "Team A", "summary": "First."},
                {"name": "Team B", "summary": "Second."},
            ]
        )

        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        manager = _make_manager([], db=db)
        stats = manager.stats()
        assert stats.community_count == 2
        manager.close()


# ---------------------------------------------------------------------------
# Edge cases and error paths
# ---------------------------------------------------------------------------


class TestCommunityEdgeCases:
    def test_empty_communities_map(self):
        """Empty communities dict returns empty list."""
        import asyncio

        db = grafeo.GrafeoDB()
        model = make_test_model([])
        result = asyncio.run(materialize_communities_async(db, model, "test_user", {"communities": {}}))
        assert result == []

    def test_empty_louvain_result(self):
        """Missing 'communities' key returns empty list."""
        import asyncio

        db = grafeo.GrafeoDB()
        model = make_test_model([])
        result = asyncio.run(materialize_communities_async(db, model, "test_user", {}))
        assert result == []

    def test_unchanged_membership_reuses_existing(self):
        """Re-running with same membership skips LLM re-summarization."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {
            "communities": {**{nid: 0 for nid in clusters["cluster1"]}},
        }

        model1 = make_test_model([{"name": "Acme Team", "summary": "Alice and Bob at Acme."}])
        result1 = asyncio.run(materialize_communities_async(db, model1, "test_user", louvain_result))
        assert len(result1) == 1

        # Re-run with same membership: no LLM call needed (model has no outputs left)
        model2 = make_test_model([])
        result2 = asyncio.run(materialize_communities_async(db, model2, "test_user", louvain_result))
        assert len(result2) == 1
        assert result2[0].name == "Acme Team"

    def test_changed_membership_re_summarizes(self):
        """When member count changes, community is re-summarized."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        # First run: 3 members
        louvain_result = {"communities": {**{nid: 0 for nid in clusters["cluster1"]}}}
        model1 = make_test_model([{"name": "Acme Team", "summary": "Original summary."}])
        result1 = asyncio.run(materialize_communities_async(db, model1, "test_user", louvain_result))
        assert len(result1) == 1

        # Second run: 2 members (different count triggers re-summary)
        louvain_result2 = {"communities": {nid: 0 for nid in clusters["cluster1"][:2]}}
        model2 = make_test_model([{"name": "Acme Duo", "summary": "Updated summary."}])
        result2 = asyncio.run(materialize_communities_async(db, model2, "test_user", louvain_result2))
        assert len(result2) == 1
        assert result2[0].name == "Acme Duo"

    def test_dissolved_communities_deleted(self):
        """Communities no longer in the partition are removed."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        # Create two communities
        louvain_result = {
            "communities": {
                **{nid: 0 for nid in clusters["cluster1"]},
                **{nid: 1 for nid in clusters["cluster2"]},
            }
        }
        model1 = make_test_model(
            [
                {"name": "Acme Team", "summary": "First."},
                {"name": "Globex Team", "summary": "Second."},
            ]
        )
        asyncio.run(materialize_communities_async(db, model1, "test_user", louvain_result))
        assert len(get_communities(db, "test_user")) == 2

        # Re-run with only cluster1: cluster2 community should be dissolved
        louvain_result2 = {"communities": {**{nid: 0 for nid in clusters["cluster1"]}}}
        model2 = make_test_model([])  # Unchanged membership, no LLM call
        asyncio.run(materialize_communities_async(db, model2, "test_user", louvain_result2))
        remaining = get_communities(db, "test_user")
        assert len(remaining) == 1
        assert remaining[0].name == "Acme Team"

    def test_get_communities_filters_by_user(self):
        """get_communities only returns communities for the specified user."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {"communities": {**{nid: 0 for nid in clusters["cluster1"]}}}
        model = make_test_model([{"name": "Acme Team", "summary": "Team A."}])
        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        # Different user should see nothing
        assert len(get_communities(db, "other_user")) == 0

    def test_get_community_context_empty_entity_list(self):
        """get_community_context returns empty for empty entity_names."""
        db = grafeo.GrafeoDB()
        assert get_community_context(db, [], "test_user") == []

    def test_get_community_context_filters_by_user(self):
        """get_community_context only matches communities for the specified user."""
        import asyncio

        db = grafeo.GrafeoDB()
        clusters = _setup_entity_graph(db)

        louvain_result = {"communities": {**{nid: 0 for nid in clusters["cluster1"]}}}
        model = make_test_model([{"name": "Acme Team", "summary": "Team A."}])
        asyncio.run(materialize_communities_async(db, model, "test_user", louvain_result))

        # Different user should find nothing even with matching entity name
        assert len(get_community_context(db, ["alice"], "other_user")) == 0
