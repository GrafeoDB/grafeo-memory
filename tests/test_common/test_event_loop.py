"""run_sync() and model clients whose pooled connections belong to one event loop.

Model clients built on httpx (Mistral, OpenAI, Anthropic, Ollama) keep pooled connections, and
pydantic-ai shares one client per process. Every sync call must therefore run on the same event
loop. A sync call made while a loop is running (a Jupyter cell, an async app calling the sync API)
used a new loop per call, so the second call failed with "bound to a different event loop" (with a
real client: "Event loop is closed"). These tests use a pooled ``httpx.AsyncClient`` against a
local server, as a real client does.
"""

from __future__ import annotations

import asyncio
import gc
import http.server
import threading
from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from grafeo_memory.common.compat import run_sync


class _Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"  # keep-alive, so the client pools the connection

    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Length", "2")
        self.end_headers()
        self.wfile.write(b"ok")

    def log_message(self, format: str, *args: Any) -> None:
        pass


@pytest.fixture(scope="module")
def url() -> Iterator[str]:
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/"
    server.shutdown()


def _sync_get(client: httpx.AsyncClient, url: str) -> int:
    async def get() -> int:
        return (await client.get(url)).status_code

    return run_sync(get())


def test_sync_calls_inside_a_running_loop_share_pooled_connections(url: str) -> None:
    client = httpx.AsyncClient()

    async def notebook_cell() -> list[int]:
        statuses = []
        for _ in range(3):
            statuses.append(_sync_get(client, url))
            gc.collect()
        return statuses

    assert asyncio.run(notebook_cell()) == [200, 200, 200]
    # The same client keeps working from plain sync code in the same process.
    assert [_sync_get(client, url) for _ in range(3)] == [200, 200, 200]


def test_run_sync_from_the_loop_itself_does_not_deadlock() -> None:
    async def inner() -> int:
        return 7

    async def outer() -> int:
        return run_sync(inner())  # a sync helper called from a coroutine on grafeo-memory's loop

    assert run_sync(outer()) == 7


def test_run_sync_propagates_exceptions() -> None:
    async def boom() -> None:
        raise ValueError("no")

    with pytest.raises(ValueError, match="no"):
        run_sync(boom())
