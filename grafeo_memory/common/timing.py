"""Retry with exponential backoff for transient provider failures (429/5xx/timeouts)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

TRANSIENT_MARKERS = ("429", "Timeout", "Connect", "rate_limit", "500", "503", "529")


def is_transient(exc: BaseException) -> bool:
    return any(m in repr(exc) for m in TRANSIENT_MARKERS)


def retry_sync[T](fn: Callable[[], T], *, retries: int = 8, max_sleep: float = 30.0) -> T:
    """Call fn() until it stops raising transient errors."""
    for attempt in range(retries):
        try:
            return fn()
        except Exception as e:
            if is_transient(e) and attempt < retries - 1:
                time.sleep(min(2**attempt, max_sleep))
                continue
            raise
    raise RuntimeError("retry_sync: retries exhausted")


async def retry_async[T](fn: Callable[[], Awaitable[T]], *, retries: int = 8, max_sleep: float = 30.0) -> T:
    """Await fn() until it stops raising transient errors."""
    for attempt in range(retries):
        try:
            return await fn()
        except Exception as e:
            if is_transient(e) and attempt < retries - 1:
                await asyncio.sleep(min(2**attempt, max_sleep))
                continue
            raise
    raise RuntimeError("retry_async: retries exhausted")
