"""Embedding client protocols and implementations."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class EmbeddingClient(Protocol):
    """Protocol for text embedding generation."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        ...

    @property
    def dimensions(self) -> int:
        """Return the embedding vector dimensionality."""
        ...


class OpenAIEmbedder:
    """Embedding client backed by the OpenAI Python SDK.

    Usage::

        from openai import OpenAI
        from grafeo_memory import OpenAIEmbedder

        embedder = OpenAIEmbedder(OpenAI())
    """

    def __init__(
        self,
        client: Any,
        model: str = "text-embedding-3-small",
    ):
        self._client = client
        self._model = model
        self._dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, input=texts)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            result = self.embed(["dimension probe"])
            self._dimensions = len(result[0])
        return self._dimensions


class MistralEmbedder:
    """Embedding client backed by the Mistral Python SDK.

    Usage::

        from mistralai import Mistral
        from grafeo_memory import MistralEmbedder

        embedder = MistralEmbedder(Mistral())
    """

    def __init__(
        self,
        client: Any,
        model: str = "mistral-embed",
    ):
        self._client = client
        self._model = model
        self._dimensions: int | None = None

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(model=self._model, inputs=texts)
        return [item.embedding for item in response.data]

    @property
    def dimensions(self) -> int:
        if self._dimensions is None:
            result = self.embed(["dimension probe"])
            self._dimensions = len(result[0])
        return self._dimensions
