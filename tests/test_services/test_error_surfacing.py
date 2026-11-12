"""Tests for error surfacing: verify failures propagate instead of being silently swallowed.

Covers T1 (broken embedder on add), T2 (broken embedder on search), T3 (index creation failure).
"""

from __future__ import annotations

import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import MemoryConfig, MemoryManager


class BrokenEmbedder:
    """Embedder that always raises on embed()."""

    def __init__(self, dims: int = 16):
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embed failed")

    @property
    def dimensions(self) -> int:
        return self._dims


class _ZeroDimEmbedder:
    """Embedder that reports 0 dimensions."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]

    @property
    def dimensions(self) -> int:
        return 0


class TestBrokenEmbedderOnAdd:
    """T1: broken embedder on add should surface the error, not silently succeed."""

    def test_add_with_broken_embedder_raises(self):
        model = make_test_model(
            [
                {"facts": ["alice works at acme"], "entities": [], "relations": []},
            ]
        )
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=BrokenEmbedder(16))

        try:
            # add() with infer=True calls embedder after extraction.
            # The error should propagate, not be swallowed.
            with pytest.raises(RuntimeError, match="embed failed"):
                manager.add("Alice works at Acme Corp")
        finally:
            manager.close()

    def test_add_raw_with_broken_embedder_raises(self):
        model = make_test_model([])
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=BrokenEmbedder(16))

        try:
            # infer=False still embeds the text for storage.
            with pytest.raises(RuntimeError, match="embed failed"):
                manager.add("raw fact", infer=False)
        finally:
            manager.close()


class TestBrokenEmbedderOnSearch:
    """T2: broken embedder on search should raise, not return empty list."""

    def test_search_with_broken_embedder(self):
        # Use a working embedder for add, then swap to broken for search.
        working_embedder = MockEmbedder(16)
        model = make_test_model(
            [
                {"facts": ["alice works at acme"], "entities": [], "relations": []},
            ]
        )
        config = MemoryConfig(db_path=None, user_id="test_user", embedding_dimensions=16)
        manager = MemoryManager(model, config, embedder=working_embedder)
        try:
            manager.add("Alice works at Acme Corp")

            # Swap embedder to broken one
            manager._embedder = BrokenEmbedder(16)

            # Search embeds the query via _embedder.embed() with no try/except,
            # so a broken embedder must propagate the error.
            with pytest.raises(RuntimeError, match="embed failed"):
                manager.search("alice")
        finally:
            manager.close()


class TestIndexCreationFailure:
    """T3: creating indexes with zero dimensions should surface an error."""

    def test_zero_dimension_config_rejected(self):
        """MemoryConfig should reject embedding_dimensions=0."""
        with pytest.raises(ValueError, match="embedding_dimensions must be positive"):
            MemoryConfig(embedding_dimensions=0)
