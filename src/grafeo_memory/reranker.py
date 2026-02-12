"""Reranker protocol and LLM-based implementation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from pydantic import BaseModel
from pydantic_ai import Agent

from ._compat import run_sync
from .types import SearchResult

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

logger = logging.getLogger(__name__)

DEFAULT_RERANK_PROMPT = """\
You are a relevance scoring assistant. Given a query and a memory, rate how \
relevant the memory is to the query on a scale from 0.0 to 1.0.

- 1.0 = perfectly relevant, directly answers the query.
- 0.5 = somewhat relevant, tangentially related.
- 0.0 = completely irrelevant.

Be strict: only give high scores to memories that directly address the query."""

RERANK_USER_PROMPT = """\
Query: {query}

Memory: {memory}

Rate the relevance of this memory to the query."""


class RelevanceScore(BaseModel):
    score: float
    reasoning: str


@runtime_checkable
class Reranker(Protocol):
    """Protocol for reranking search results by relevance."""

    def rerank(self, query: str, results: list[SearchResult], *, top_k: int | None = None) -> list[SearchResult]: ...


class LLMReranker:
    """Score each result's relevance to the query via LLM.

    Usage::

        from grafeo_memory import LLMReranker

        reranker = LLMReranker("openai:gpt-4o-mini")
        reranked = reranker.rerank("hobbies", results, top_k=5)
    """

    def __init__(self, model: object, *, prompt: str | None = None):
        self._model = model
        self._prompt = prompt or DEFAULT_RERANK_PROMPT

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
        _on_usage: Callable[[str, RunUsage], None] | None = None,
    ) -> list[SearchResult]:
        """Rerank results by LLM-scored relevance (sync)."""
        return run_sync(self.rerank_async(query, results, top_k=top_k, _on_usage=_on_usage))

    async def rerank_async(
        self,
        query: str,
        results: list[SearchResult],
        *,
        top_k: int | None = None,
        _on_usage: Callable[[str, RunUsage], None] | None = None,
    ) -> list[SearchResult]:
        """Rerank results by LLM-scored relevance (async)."""
        if not results:
            return []

        agent = Agent(self._model, system_prompt=self._prompt, output_type=RelevanceScore)

        scored: list[tuple[float, SearchResult]] = []
        for r in results:
            try:
                result = await agent.run(RERANK_USER_PROMPT.format(query=query, memory=r.text))
                score = max(0.0, min(1.0, result.output.score))
                if _on_usage is not None:
                    _on_usage("rerank", result.usage())
            except Exception:
                logger.warning("Reranker scoring failed for memory %s", r.memory_id, exc_info=True)
                score = r.score  # fall back to original score
            scored.append((score, r))

        scored.sort(key=lambda x: x[0], reverse=True)

        reranked = []
        for score, r in scored:
            reranked.append(
                SearchResult(
                    memory_id=r.memory_id,
                    text=r.text,
                    score=score,
                    user_id=r.user_id,
                    metadata=r.metadata,
                    relations=r.relations,
                    actor_id=r.actor_id,
                    role=r.role,
                )
            )

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked
