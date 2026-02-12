"""Async/sync compatibility utilities."""

from __future__ import annotations

import asyncio
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import TypeVar

T = TypeVar("T")

# --- Windows ProactorEventLoop safety net ---
# Python 3.13+ incremental GC can trigger httpx transport __del__ after the
# event loop is closed, raising RuntimeError on Windows. This monkey-patch
# silences the spurious error in transport teardown.
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _original_del = _ProactorBasePipeTransport.__del__

        def _safe_del(self, _warn=None):  # type: ignore[no-untyped-def]
            try:
                _original_del(self, _warn)
            except RuntimeError:
                pass

        _ProactorBasePipeTransport.__del__ = _safe_del  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass

# --- Persistent asyncio.Runner (Python 3.12+) ---
# Keeps the event loop alive across multiple run_sync() calls, avoiding
# the create/destroy overhead and the httpx transport teardown crash.
_runner: asyncio.Runner | None = None


def _get_runner() -> asyncio.Runner:
    global _runner
    if _runner is None:
        _runner = asyncio.Runner()
    return _runner


def run_sync(coro: object) -> object:
    """Run an async coroutine synchronously.

    If no event loop is running, uses a persistent asyncio.Runner to keep the
    loop alive across calls (avoids Windows ProactorEventLoop teardown issues).
    If called from within an existing event loop, runs the coroutine
    in a separate thread to avoid deadlock.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return _get_runner().run(coro)

    # Already in an event loop — run in a thread to avoid blocking
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(lambda: asyncio.Runner().run(coro)).result()


def shutdown() -> None:
    """Close the persistent event loop runner.

    Call on application exit or from MemoryManager.close() to cleanly
    shut down the event loop and allow transports to finalize.
    """
    global _runner
    if _runner is not None:
        _runner.close()
        _runner = None
