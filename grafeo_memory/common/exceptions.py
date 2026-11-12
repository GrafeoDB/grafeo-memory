"""Error hierarchy for grafeo-memory. One base, so callers can catch broadly.

Subclasses are added when something raises them: configuration errors stay
``ValueError`` (what ``MemoryConfig`` callers already catch).
"""

from __future__ import annotations


class GrafeoMemoryError(Exception):
    """Base error for grafeo-memory."""


class SecretNotFoundError(GrafeoMemoryError):
    """A required API key/secret is not configured."""
