"""Async/sync compatibility utilities."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import sys
from collections.abc import Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any

# --- Windows ProactorEventLoop safety net ---
# Python 3.13+ incremental GC can trigger httpx transport __del__ after the
# event loop is closed, raising RuntimeError on Windows. This monkey-patch
# silences the spurious error in transport teardown.
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        _original_del = _ProactorBasePipeTransport.__del__

        def _safe_del(self, _warn: object = None) -> None:
            with contextlib.suppress(RuntimeError):
                _original_del(self, _warn)  # type: ignore[too-many-positional-arguments]

        _ProactorBasePipeTransport.__del__ = _safe_del  # type: ignore[assignment]
    except (ImportError, AttributeError):
        pass

# --- Persistent asyncio.Runner (Python 3.12+) ---
# Keeps the event loop alive across multiple run_sync() calls and across
# multiple MemoryManager sessions. Closing the runner between sessions
# corrupts httpx/anyio global state on Windows, so we keep it alive for the
# process lifetime and clean up via atexit.
_runner: asyncio.Runner | None = None


def _get_runner() -> asyncio.Runner:
    global _runner
    if _runner is None:
        _runner = asyncio.Runner()
    return _runner


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
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

    Called automatically at process exit via atexit. Should NOT be called
    when closing individual MemoryManager sessions — the runner is shared
    across sessions to avoid corrupting httpx/anyio transport state.
    """
    global _runner
    if _runner is not None:
        with contextlib.suppress(Exception):
            _runner.close()
        _runner = None


atexit.register(shutdown)
