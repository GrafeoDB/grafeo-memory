"""Error hierarchy for grafeo-memory. One base, so callers can catch broadly."""

from __future__ import annotations


class GrafeoMemoryError(Exception):
    """Base error for grafeo-memory."""


class ProviderError(GrafeoMemoryError):
    """An external provider (LLM, decision model, embedder) failed."""


class SecretNotFoundError(GrafeoMemoryError):
    """A required API key/secret is not configured."""


class ConfigurationError(GrafeoMemoryError):
    """Invalid configuration (MemoryConfig, settings, pipeline stages)."""
