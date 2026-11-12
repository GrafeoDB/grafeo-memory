"""Retry with exponential backoff for transient provider failures (429/5xx/timeouts)."""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

TRANSIENT_STATUS = frozenset({429, 500, 502, 503, 504, 529})
# Matched against exception class names (and their bases), never the message, so a
# number or word echoed from the prompt cannot make a permanent error look transient.
TRANSIENT_MARKERS = ("Timeout", "Connect", "RateLimit", "Overloaded", "Unavailable", "ServerError")


def _status_code(exc: BaseException) -> int | None:
    for obj in (exc, getattr(exc, "response", None)):  # SDK errors, then httpx.HTTPStatusError
        status = getattr(obj, "status_code", None)
        if isinstance(status, int):
            return status
    return None


def is_transient(exc: BaseException) -> bool:
    """True for HTTP 429/5xx, and for timeout, connection and rate-limit error types.

    Walks the ``__cause__``/``__context__`` chain, so a wrapped SDK error still counts."""
    seen: set[int] = set()
    e: BaseException | None = exc
    while e is not None and id(e) not in seen:
        seen.add(id(e))
        status = _status_code(e)
        if status is not None:
            return status in TRANSIENT_STATUS
        if any(m in cls.__name__ for cls in type(e).__mro__ for m in TRANSIENT_MARKERS):
            return True
        e = e.__cause__ or e.__context__
    return False


def retry_sync[T](fn: Callable[[], T], *, retries: int = 8, max_sleep: float = 30.0) -> T:
    """Call fn() until it stops raising transient errors (at least one attempt)."""
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return fn()
        except Exception as e:
            if is_transient(e) and attempt < attempts - 1:
                time.sleep(min(2**attempt, max_sleep))
                continue
            raise
    raise RuntimeError("retry_sync: retries exhausted")


async def retry_async[T](fn: Callable[[], Awaitable[T]], *, retries: int = 8, max_sleep: float = 30.0) -> T:
    """Await fn() until it stops raising transient errors (at least one attempt)."""
    attempts = max(1, retries)
    for attempt in range(attempts):
        try:
            return await fn()
        except Exception as e:
            if is_transient(e) and attempt < attempts - 1:
                await asyncio.sleep(min(2**attempt, max_sleep))
                continue
            raise
    raise RuntimeError("retry_async: retries exhausted")
