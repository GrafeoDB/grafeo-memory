"""Mock embedding client and test model utilities for pydantic-ai based testing."""

from __future__ import annotations

import hashlib
import struct

from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel


class MockEmbedder:
    """Deterministic mock embedding client.

    Generates stable pseudo-random vectors from text content using hashing.
    """

    def __init__(self, dims: int = 16):
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_embed(t) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims

    def _hash_embed(self, text: str) -> list[float]:
        """Generate a deterministic unit vector from text."""
        h = hashlib.sha256(text.encode()).digest()
        # Expand hash to fill dimensions
        while len(h) < self._dims * 4:
            h += hashlib.sha256(h).digest()
        # Convert to floats in [-1, 1]
        floats = []
        for i in range(self._dims):
            (val,) = struct.unpack_from("<f", h, i * 4)
            floats.append(val / (abs(val) + 1.0))  # Squash to (-1, 1)
        # Normalize to unit vector
        norm = sum(f * f for f in floats) ** 0.5
        if norm > 0:
            floats = [f / norm for f in floats]
        return floats


def make_test_model(outputs: list[dict]) -> FunctionModel:
    """Create a FunctionModel that returns pre-defined structured outputs sequentially.

    Each output should be a dict matching the expected Pydantic model fields, e.g.:
        {"facts": ["alice works at acme"]}
        {"entities": [...], "relations": [...]}
        {"decisions": [...]}
        {"delete": [...]}
    """
    index = [0]  # mutable container to allow closure mutation

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        output = outputs[index[0]]
        index[0] += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=output)])

    return FunctionModel(function=handler)


def make_error_model() -> FunctionModel:
    """Create a FunctionModel that always raises an error (for testing fallback behavior)."""

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        raise ValueError("Simulated LLM error")

    return FunctionModel(function=handler)


def make_error_then_succeed_model(outputs: list[dict]) -> FunctionModel:
    """Create a FunctionModel that errors on the first call, then returns outputs sequentially.

    Useful for testing fallback paths where combined extraction fails but separate calls succeed.
    """
    index = [0]

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        if index[0] == 0:
            index[0] += 1
            raise ValueError("Simulated combined extraction failure")
        output = outputs[index[0] - 1]
        index[0] += 1
        tool_name = info.output_tools[0].name
        return ModelResponse(parts=[ToolCallPart(tool_name=tool_name, args=output)])

    return FunctionModel(function=handler)
