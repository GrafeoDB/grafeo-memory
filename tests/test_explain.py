"""Tests for MemoryManager.explain() and ExplainResult."""

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.types import ExplainResult, ExplainStep


def _make_manager(outputs, dims=16, **config_kwargs):
    model = make_test_model(outputs)
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)
    return MemoryManager(model, config, embedder=embedder)


class TestExplain:
    def test_explain_returns_explain_result(self):
        manager = _make_manager(
            [
                # add: combined extraction
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "alice", "entity_type": "person"}],
                    "relations": [],
                },
                # explain/search: entity extraction for the query
                {"entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
            ]
        )
        manager.add("Alice works at Acme Corp")
        result = manager.explain("Where does Alice work?")

        assert isinstance(result, ExplainResult)
        assert result.query == "Where does Alice work?"
        manager.close()

    def test_explain_has_core_steps(self):
        manager = _make_manager(
            [
                # add
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "alice", "entity_type": "person"}],
                    "relations": [],
                },
                # explain: entity extraction
                {"entities": [{"name": "alice", "entity_type": "person"}], "relations": []},
            ]
        )
        manager.add("Alice works at Acme Corp")
        result = manager.explain("Alice")

        step_names = [s.name for s in result.steps]
        assert "embed_query" in step_names
        assert "hybrid_search" in step_names
        assert "entity_extraction" in step_names
        assert "graph_search" in step_names
        assert "merge" in step_names
        assert "final" in step_names
        manager.close()

    def test_explain_embed_step_has_dimensions(self):
        manager = _make_manager(
            [
                {"facts": ["fact"], "entities": [], "relations": []},
                {"entities": [], "relations": []},
            ]
        )
        manager.add("some fact", infer=False)
        result = manager.explain("test")

        embed_step = next(s for s in result.steps if s.name == "embed_query")
        assert isinstance(embed_step, ExplainStep)
        assert embed_step.detail["dimensions"] == 16
        manager.close()

    def test_explain_results_match_search(self):
        manager = _make_manager(
            [
                # add (raw)
                # search: entity extraction
                {"entities": [], "relations": []},
                # explain: entity extraction
                {"entities": [], "relations": []},
            ]
        )
        manager.add("some raw fact", infer=False)

        search_results = manager.search("some raw fact")
        explain_result = manager.explain("some raw fact")

        # Same number of results
        assert len(explain_result.results) == len(search_results)
        manager.close()

    def test_explain_empty_db(self):
        manager = _make_manager(
            [
                # explain: entity extraction (empty db, no add needed)
                {"entities": [], "relations": []},
            ]
        )
        result = manager.explain("test query")

        assert result.query == "test query"
        assert len(result.results) == 0
        step_names = [s.name for s in result.steps]
        assert "embed_query" in step_names
        assert "final" in step_names
        manager.close()

    def test_explain_no_topology_or_importance_steps_by_default(self):
        manager = _make_manager(
            [
                {"entities": [], "relations": []},
            ]
        )
        result = manager.explain("test")

        step_names = [s.name for s in result.steps]
        assert "topology_boost" not in step_names
        assert "importance_scoring" not in step_names
        assert "rerank" not in step_names
        manager.close()
