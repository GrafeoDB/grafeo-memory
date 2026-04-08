"""Integration tests: cross-feature interactions between bi-temporal, episodes, and communities."""

from __future__ import annotations

from datetime import UTC, datetime

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager


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


def _temporal_annotation_output(annotations):
    return {"annotations": annotations}


class TestBiTemporalWithEpisodes:
    def test_bitemporal_and_episodes_together(self):
        """Both enabled: Episode created, Memory gets valid_at, PRODUCED links."""
        jan_15 = int(datetime(2024, 1, 15, tzinfo=UTC).timestamp() * 1000)

        manager = _make_manager(
            [
                _extraction_output(
                    ["alice started at acme in january 2024"],
                    [{"name": "alice", "entity_type": "person"}],
                ),
                _temporal_annotation_output([{"fact_index": 0, "valid_at": "2024-01-15", "invalid_at": None}]),
                {"decisions": [{"action": "add", "text": "alice started at acme in january 2024"}]},
            ],
            enable_bitemporal=True,
            enable_episodes=True,
            run_id="run1",
        )

        events = manager.add("Alice started at Acme in January 2024")
        assert len(events) >= 1
        assert events[0].valid_at == jan_15

        # Episode should exist with PRODUCED edge
        episodes = manager.get_episodes()
        assert len(episodes) == 1
        assert events[0].memory_id in episodes[0].produced_memories

        # Memory should have valid_at
        memories = manager.get_all()
        assert len(memories) >= 1
        assert memories[0].valid_at == jan_15
        manager.close()


class TestBackwardCompatDefaults:
    def test_all_features_default_off(self):
        """All new features default to off, existing behavior unchanged."""
        config = MemoryConfig()
        assert config.enable_bitemporal is False
        assert config.enable_episodes is False
        assert config.enable_community_summaries is False

    def test_add_without_features(self):
        """Basic add/search still works with all features off."""
        manager = _make_manager(
            [
                _extraction_output(["alice works at acme"]),
                {"decisions": [{"action": "add", "text": "alice works at acme"}]},
                {"entities": [], "relations": []},
            ],
        )
        events = manager.add("Alice works at Acme")
        assert len(events) >= 1

        results = manager.search("alice work")
        assert len(results) >= 1
        # No valid_at/invalid_at set
        assert results[0].valid_at is None
        assert results[0].invalid_at is None
        manager.close()

    def test_episodes_not_created_by_default(self):
        """No episodes created when feature is off."""
        manager = _make_manager(
            [
                _extraction_output(["fact one"]),
                {"decisions": [{"action": "add", "text": "fact one"}]},
            ],
        )
        manager.add("Fact one")
        episodes = manager.get_episodes()
        assert len(episodes) == 0
        manager.close()


class TestYoloConfig:
    def test_yolo_still_works(self):
        """MemoryConfig.yolo() creates a valid config with all traditional features."""
        config = MemoryConfig.yolo()
        assert config.enable_importance is True
        assert config.enable_vision is True
        # New features remain off by default in yolo
        assert config.enable_bitemporal is False
        assert config.enable_episodes is False
        assert config.enable_community_summaries is False
