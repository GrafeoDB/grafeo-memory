"""Async/sync compatibility utilities."""

from __future__ import annotations

import asyncio
import atexit
import contextlib
import os
import sys
import threading
import warnings
from collections.abc import Callable, Coroutine
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

# --- Windows ProactorEventLoop safety net ---
# Python 3.13+ incremental GC can trigger httpx transport __del__ after the
# event loop is closed, raising RuntimeError on Windows. This monkey-patch
# silences the spurious error in transport teardown.
#
# CPython's _ProactorBasePipeTransport.__del__ has signature
# `def __del__(self, _warn=warnings.warn)`, `_warn` is bound at module-load
# time so __del__ keeps a working warner even after interpreter teardown.
# Our wrapper must preserve that contract: defaulting `_warn=None` causes
# `TypeError: 'NoneType' object is not callable` deep inside __del__.
if sys.platform == "win32":
    try:
        from asyncio.proactor_events import _ProactorBasePipeTransport

        # typed loosely: CPython passes the bound warner positionally (see above)
        _original_del = cast("Callable[..., None]", _ProactorBasePipeTransport.__del__)

        def _safe_del(self, _warn=warnings.warn) -> None:
            with contextlib.suppress(Exception):
                _original_del(self, _warn)

        _ProactorBasePipeTransport.__del__ = _safe_del
    except (ImportError, AttributeError):
        pass

# --- One persistent event loop for every sync call ---
# Model clients built on httpx (Mistral, OpenAI, Anthropic, Ollama) pool their
# connections, and pydantic-ai shares one client per process. A pooled
# connection belongs to the event loop that opened it, so every sync call must
# run on the same loop: a second loop fails with "Event loop is closed" or
# "bound to a different event loop". The loop runs in a daemon thread, which
# serves a plain script and a caller whose own loop is running (Jupyter, an
# async web app calling the sync API) alike. It lives for the process and is
# stopped via atexit.
_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_lock = threading.Lock()


def _get_loop() -> asyncio.AbstractEventLoop:
    global _loop, _thread
    with _lock:
        if _loop is None or _thread is None or not _thread.is_alive():
            loop = asyncio.new_event_loop()
            thread = threading.Thread(target=loop.run_forever, name="grafeo-memory-loop", daemon=True)
            thread.start()
            _loop, _thread = loop, thread
        return _loop


def run_sync[T](coro: Coroutine[Any, Any, T]) -> T:
    """Run an async coroutine synchronously, on grafeo-memory's persistent event loop.

    Works with or without an event loop running in the calling thread. A call
    made from a coroutine already on that loop runs in a fresh thread instead,
    since waiting on the loop from inside it would deadlock.
    """
    if threading.current_thread() is _thread:
        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(lambda: asyncio.run(coro)).result()

    future = asyncio.run_coroutine_threadsafe(coro, _get_loop())
    try:
        return future.result()
    except BaseException:  # KeyboardInterrupt included: stop the coroutine too
        future.cancel()
        raise


def shutdown() -> None:
    """Stop the persistent event loop.

    Called automatically at process exit via atexit. Should NOT be called
    when closing individual MemoryManager sessions, the loop is shared
    across sessions to keep pooled httpx connections valid.
    """
    global _loop, _thread
    with _lock:
        loop, thread = _loop, _thread
        _loop = _thread = None
    if loop is None:
        return
    with contextlib.suppress(Exception):
        loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=5)
        if not loop.is_running():
            loop.close()


def _reset_after_fork() -> None:
    """A forked child has no loop thread; drop the parent's loop so the next call starts one."""
    global _loop, _thread, _lock
    _loop = _thread = None
    _lock = threading.Lock()


atexit.register(shutdown)
if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_after_fork)
