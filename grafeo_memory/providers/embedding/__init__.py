"""Embedding client protocols and implementations."""

from __future__ import annotations

from typing import Any

from grafeo_memory.common.types import EmbeddingClient as EmbeddingClient  # re-exported protocol


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


def mistral_client_class() -> Any:
    """Return the ``Mistral`` client class of the installed ``mistralai`` SDK.

    mistralai 2 and later export it from ``mistralai.client``; mistralai 1 from
    ``mistralai``. Raises ImportError when the SDK is not installed.
    """
    import importlib

    for name in ("mistralai.client", "mistralai"):  # mistralai 2 and later, then 1.x
        try:
            module = importlib.import_module(name)
        except ImportError:
            continue
        if hasattr(module, "Mistral"):
            return module.Mistral
    raise ImportError("the mistralai package is not installed")


class MistralEmbedder:
    """Embedding client backed by the Mistral Python SDK.

    Usage::

        from mistralai.client import Mistral  # mistralai 1.x: from mistralai import Mistral
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
