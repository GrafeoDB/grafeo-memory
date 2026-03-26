"""Tests for MemoryManager.stats() and MemoryStats."""

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager, MemoryType
from grafeo_memory.types import MemoryStats


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return MemoryManager(model, config, embedder=embedder)


class TestStats:
    def test_empty_db_stats(self):
        manager = _make_manager([])
        s = manager.stats()
        assert isinstance(s, MemoryStats)
        assert s.total_memories == 0
        assert s.semantic_count == 0
        assert s.procedural_count == 0
        assert s.episodic_count == 0
        assert s.entity_count == 0
        assert s.relation_count == 0
        manager.close()

    def test_stats_after_add(self):
        manager = _make_manager(
            [
                {
                    "facts": ["alice works at acme"],
                    "entities": [
                        {"name": "alice", "entity_type": "person"},
                        {"name": "acme", "entity_type": "organization"},
                    ],
                    "relations": [
                        {"source": "alice", "target": "acme", "relation_type": "works_at"},
                    ],
                },
            ]
        )
        manager.add("Alice works at Acme Corp")
        s = manager.stats()
        assert s.total_memories == 1
        assert s.semantic_count == 1
        assert s.entity_count == 2
        assert s.relation_count >= 1
        manager.close()

    def test_stats_type_breakdown(self):
        manager = _make_manager(
            [
                # semantic add
                {
                    "facts": ["fact one"],
                    "entities": [],
                    "relations": [],
                },
                # procedural add
                {
                    "facts": ["always use type hints"],
                    "entities": [],
                    "relations": [],
                },
            ]
        )
        manager.add("Fact one", memory_type=MemoryType.SEMANTIC)
        manager.add("Always use type hints", memory_type=MemoryType.PROCEDURAL)
        s = manager.stats()
        assert s.total_memories == 2
        assert s.semantic_count == 1
        assert s.procedural_count == 1
        manager.close()

    def test_stats_raw_add(self):
        """Stats count raw (infer=False) memories correctly."""
        manager = _make_manager([])
        manager.add("raw text one", infer=False)
        manager.add("raw text two", infer=False, memory_type="procedural")
        s = manager.stats()
        assert s.total_memories == 2
        assert s.semantic_count == 1
        assert s.procedural_count == 1
        manager.close()
