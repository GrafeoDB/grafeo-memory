"""Tests for the Reranker protocol and LLMReranker."""

from grafeo_memory.reranker import LLMReranker, Reranker
from grafeo_memory.types import SearchResult


class TestRerankerProtocol:
    def test_llm_reranker_satisfies_protocol(self):
        """LLMReranker should satisfy the Reranker protocol."""
        assert isinstance(LLMReranker.__new__(LLMReranker), Reranker)

    def test_custom_reranker_satisfies_protocol(self):
        """A custom class with rerank() should satisfy the protocol."""

        class MyReranker:
            def rerank(self, query, results, *, top_k=None):
                return results[:top_k] if top_k else results

        assert isinstance(MyReranker(), Reranker)


class TestLLMReranker:
    def test_rerank_empty(self):
        """Reranking empty results returns empty."""
        from mock_llm import make_test_model

        model = make_test_model([])
        reranker = LLMReranker(model)
        result = reranker.rerank("query", [])
        assert result == []

    def test_rerank_scores_and_sorts(self):
        """Reranker should score results and sort by score descending."""
        from mock_llm import make_test_model

        results = [
            SearchResult(memory_id="1", text="alice likes hiking", score=0.5, user_id="u1"),
            SearchResult(memory_id="2", text="bob likes cooking", score=0.8, user_id="u1"),
        ]

        model = make_test_model(
            [
                {"score": 0.9, "reasoning": "very relevant"},
                {"score": 0.2, "reasoning": "not relevant"},
            ]
        )

        reranker = LLMReranker(model)
        reranked = reranker.rerank("hiking", results)

        assert len(reranked) == 2
        assert reranked[0].memory_id == "1"  # higher score
        assert reranked[0].score == 0.9
        assert reranked[1].memory_id == "2"
        assert reranked[1].score == 0.2

    def test_rerank_top_k(self):
        """top_k should limit results."""
        from mock_llm import make_test_model

        results = [
            SearchResult(memory_id="1", text="a", score=0.5, user_id="u1"),
            SearchResult(memory_id="2", text="b", score=0.5, user_id="u1"),
            SearchResult(memory_id="3", text="c", score=0.5, user_id="u1"),
        ]

        model = make_test_model(
            [
                {"score": 0.9, "reasoning": "high"},
                {"score": 0.7, "reasoning": "mid"},
                {"score": 0.3, "reasoning": "low"},
            ]
        )

        reranker = LLMReranker(model)
        reranked = reranker.rerank("query", results, top_k=2)
        assert len(reranked) == 2
