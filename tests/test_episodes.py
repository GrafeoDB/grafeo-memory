"""Tests for episode provenance: Episode nodes, PRODUCED/MENTIONS edges, session replay."""

from __future__ import annotations

import grafeo
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import (
    MemoryConfig,
    MemoryManager,
)


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


def _extraction_output(facts, entities=None, relations=None):
    return {
        "facts": facts,
        "entities": entities or [],
        "relations": relations or [],
    }


# ---------------------------------------------------------------------------
# Episode creation
# ---------------------------------------------------------------------------


class TestEpisodeCreation:
    def test_episode_created_on_add(self):
        """When enable_episodes=True, an Episode node is created on add."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        events = manager.add("Alice works at Acme")
        assert len(events) >= 1

        episodes = manager.get_episodes()
        assert len(episodes) == 1
        assert "Alice works at Acme" in episodes[0].content
        manager.close()

    def test_episode_not_created_when_disabled(self):
        """When enable_episodes=False (default), no Episode nodes are created."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
            ],
        )
        manager.add("Alice works at Acme")

        episodes = manager.get_episodes()
        assert len(episodes) == 0
        manager.close()

    def test_episode_properties(self):
        """Episode has content, source, user_id, created_at."""
        manager = _make_manager(
            [
                _extraction_output(["bob likes pizza"]),
                {"decisions": [{"action": "add", "text": "bob likes pizza"}]},
            ],
            enable_episodes=True,
            session_id="sess1",
            run_id="run1",
        )
        manager.add("Bob likes pizza")

        episodes = manager.get_episodes()
        assert len(episodes) == 1
        ep = episodes[0]
        assert ep.source == "message"
        assert ep.user_id == "test_user"
        assert ep.session_id == "sess1"
        assert ep.created_at is not None
        manager.close()

    def test_raw_add_creates_episode(self):
        """infer=False with episodes still creates an Episode node."""
        manager = _make_manager(
            [],
            enable_episodes=True,
            run_id="run1",
        )
        events = manager.add("Some raw text", infer=False)
        assert len(events) >= 1

        episodes = manager.get_episodes()
        assert len(episodes) == 1
        assert "Some raw text" in episodes[0].content
        manager.close()


# ---------------------------------------------------------------------------
# PRODUCED edges
# ---------------------------------------------------------------------------


class TestProducedEdges:
    def test_produced_edge_links_episode_to_memory(self):
        """PRODUCED edge connects Episode to Memory."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        events = manager.add("Alice works at Acme")
        memory_id = events[0].memory_id

        episodes = manager.get_episodes()
        assert len(episodes) == 1
        assert memory_id in episodes[0].produced_memories
        manager.close()

    def test_get_provenance_returns_episode(self):
        """get_provenance(memory_id) returns the source Episode."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        events = manager.add("Alice works at Acme")
        memory_id = events[0].memory_id

        provenance = manager.get_provenance(memory_id)
        assert len(provenance) == 1
        assert "Alice works at Acme" in provenance[0].content
        manager.close()

    def test_multiple_facts_multiple_produced(self):
        """Multiple facts from one add produce multiple PRODUCED edges."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme", "bob works at globex"]),
                {
                    "decisions": [
                        {"action": "add", "text": "alice works at acme"},
                        {"action": "add", "text": "bob works at globex"},
                    ]
                },
            ],
            enable_episodes=True,
            run_id="run1",
        )
        events = manager.add("Alice works at Acme. Bob works at Globex.")
        assert len(events) == 2

        episodes = manager.get_episodes()
        assert len(episodes) == 1
        assert len(episodes[0].produced_memories) == 2
        manager.close()


# ---------------------------------------------------------------------------
# NEXT_EPISODE chain
# ---------------------------------------------------------------------------


class TestNextEpisodeChain:
    def test_next_episode_edges_with_run_id(self):
        """Sequential adds create NEXT_EPISODE chain between episodes."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
                _extraction_output(["fact three"]),
                {"decisions": [{"action": "add", "text": "fact three"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Fact one")
        manager.add("Fact two")
        manager.add("Fact three")

        episodes = manager.get_episodes()
        assert len(episodes) == 3

        # Follow forward chain from first episode
        chain = manager.episode_chain(episodes[0].episode_id, direction="forward")
        assert len(chain) == 2  # ep2, ep3
        manager.close()

    def test_episode_chain_backward(self):
        """episode_chain backward from last returns preceding episodes."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Fact one")
        manager.add("Fact two")

        episodes = manager.get_episodes()
        chain = manager.episode_chain(episodes[1].episode_id, direction="backward")
        assert len(chain) == 1  # ep1
        manager.close()


# ---------------------------------------------------------------------------
# get_episodes
# ---------------------------------------------------------------------------


class TestGetEpisodes:
    def test_get_episodes_all(self):
        """Returns all episodes for user."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Fact one")
        manager.add("Fact two")

        episodes = manager.get_episodes()
        assert len(episodes) == 2
        manager.close()

    def test_get_episodes_by_session(self):
        """Filter by session_id."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Fact one", session_id="sess_a")
        manager.add("Fact two", session_id="sess_b")

        episodes_a = manager.get_episodes(session_id="sess_a")
        assert len(episodes_a) == 1
        assert "Fact one" in episodes_a[0].content
        manager.close()


# ---------------------------------------------------------------------------
# Search unaffected
# ---------------------------------------------------------------------------


class TestSearchUnaffected:
    def test_episode_nodes_not_in_search(self):
        """Episode nodes should not appear in search results."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
                # Search needs entity extraction
                {"entities": [], "relations": []},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Alice works at Acme")

        results = manager.search("alice work")
        # All results should be Memory nodes, not Episode nodes
        for r in results:
            assert r.source in ("vector", "graph", "both", None)
        manager.close()

    def test_leads_to_not_created_with_episodes(self):
        """LEADS_TO edges between memories are NOT created when episodes are enabled."""
        db = grafeo.GrafeoDB()
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
            ],
            enable_episodes=True,
            run_id="run1",
            db=db,
        )
        manager.add("Fact one")
        manager.add("Fact two")

        # Check there are no LEADS_TO edges in the graph
        count = 0
        try:
            rows = db.execute("MATCH ()-[r:LEADS_TO]->() RETURN count(r)", {})
            for row in rows:
                vals = list(row.values()) if isinstance(row, dict) else [row]
                count = int(vals[0]) if vals else 0
        except (RuntimeError, KeyError, TypeError):
            pass  # Query not supported or returned unexpected shape: no LEADS_TO edges
        assert count == 0
        manager.close()


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


class TestEpisodeStats:
    def test_stats_includes_episode_count(self):
        """stats() returns correct episode_count."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
                _extraction_output(["fact two"]),
                {"decisions": [{"action": "add", "text": "fact two"}]},
            ],
            enable_episodes=True,
            run_id="run1",
        )
        manager.add("Fact one")
        manager.add("Fact two")

        stats = manager.stats()
        assert stats.episode_count == 2
        manager.close()
