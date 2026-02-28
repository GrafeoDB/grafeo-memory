"""Tests for MCP tools wrapping the grafeo-memory API."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

from grafeo_memory.history import HistoryEntry
from grafeo_memory.types import MemoryAction, MemoryEvent, SearchResult

# ---------------------------------------------------------------------------
# Mock MCP context: ctx.request_context.lifespan_context.manager
# ---------------------------------------------------------------------------


class FakeAsyncManager:
    """Fake AsyncMemoryManager that records calls and returns canned responses."""

    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    async def add(self, text, *, user_id=None, memory_type="semantic", infer=True, **kw):
        self.calls.append(("add", {"text": text, "user_id": user_id, "memory_type": memory_type, "infer": infer}))
        return [MemoryEvent(action=MemoryAction.ADD, memory_id="1", text=text, memory_type=memory_type)]

    async def add_batch(self, texts, *, user_id=None, memory_type="semantic", infer=True, **kw):
        self.calls.append(("add_batch", {"texts": texts, "user_id": user_id}))
        return [MemoryEvent(action=MemoryAction.ADD, memory_id=str(i), text=t) for i, t in enumerate(texts)]

    async def search(self, query, *, user_id=None, k=10, memory_type=None, **kw):
        self.calls.append(("search", {"query": query, "user_id": user_id, "k": k, "memory_type": memory_type}))
        return [SearchResult(memory_id="1", text="alice works at acme", score=0.95, user_id=user_id or "default")]

    async def update(self, memory_id, text, **kw):
        self.calls.append(("update", {"memory_id": memory_id, "text": text}))
        return MemoryEvent(action=MemoryAction.UPDATE, memory_id=memory_id, text=text, old_text="old text")

    async def delete(self, memory_id, **kw):
        self.calls.append(("delete", {"memory_id": memory_id}))
        return True

    async def delete_all(self, *, user_id=None, **kw):
        self.calls.append(("delete_all", {"user_id": user_id}))
        return 5

    async def get_all(self, *, user_id=None, memory_type=None, **kw):
        self.calls.append(("get_all", {"user_id": user_id, "memory_type": memory_type}))
        return [SearchResult(memory_id="1", text="a fact", score=1.0, user_id=user_id or "default")]

    async def summarize(self, *, user_id=None, preserve_recent=5, batch_size=20, **kw):
        self.calls.append(("summarize", {"user_id": user_id, "preserve_recent": preserve_recent}))
        return [MemoryEvent(action=MemoryAction.ADD, memory_id="10", text="consolidated")]

    async def history(self, memory_id, **kw):
        self.calls.append(("history", {"memory_id": memory_id}))
        return [HistoryEntry(event="ADD", new_text="original text", timestamp=1000000)]


@dataclass
class _LifespanContext:
    manager: Any


@dataclass
class _RequestContext:
    lifespan_context: _LifespanContext


@dataclass
class MockContext:
    request_context: _RequestContext


def _make_ctx(manager=None):
    mgr = manager or FakeAsyncManager()
    return MockContext(request_context=_RequestContext(lifespan_context=_LifespanContext(manager=mgr))), mgr


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMemoryAdd:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_add

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_add("alice likes pizza", ctx=ctx)))
        assert "events" in result
        assert len(result["events"]) == 1
        assert result["events"][0]["action"] == "add"
        assert result["events"][0]["text"] == "alice likes pizza"
        call_name, call_args = mgr.calls[0]
        assert call_name == "add"
        assert call_args == {"text": "alice likes pizza", "user_id": None, "memory_type": "semantic", "infer": True}

    def test_with_user_and_type(self):
        from grafeo_memory.mcp.tools import memory_add

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_add("use pytest", user_id="alice", memory_type="procedural", ctx=ctx)))
        assert result["events"][0]["action"] == "add"
        assert mgr.calls[0][1]["user_id"] == "alice"
        assert mgr.calls[0][1]["memory_type"] == "procedural"

    def test_no_infer(self):
        from grafeo_memory.mcp.tools import memory_add

        ctx, mgr = _make_ctx()
        asyncio.run(memory_add("raw text", infer=False, ctx=ctx))
        assert mgr.calls[0][1]["infer"] is False


class TestMemoryAddBatch:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_add_batch

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_add_batch(["fact 1", "fact 2"], ctx=ctx)))
        assert len(result["events"]) == 2
        assert mgr.calls[0][0] == "add_batch"


class TestMemorySearch:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_search

        ctx, _ = _make_ctx()
        result = json.loads(asyncio.run(memory_search("alice work", ctx=ctx)))
        assert "results" in result
        assert len(result["results"]) == 1
        assert result["results"][0]["text"] == "alice works at acme"
        assert result["results"][0]["score"] == 0.95

    def test_with_filters(self):
        from grafeo_memory.mcp.tools import memory_search

        ctx, mgr = _make_ctx()
        asyncio.run(memory_search("query", user_id="bob", k=5, memory_type="procedural", ctx=ctx))
        assert mgr.calls[0][1]["user_id"] == "bob"
        assert mgr.calls[0][1]["k"] == 5
        assert mgr.calls[0][1]["memory_type"] == "procedural"


class TestMemoryUpdate:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_update

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_update("42", "new text", ctx=ctx)))
        assert result["event"]["action"] == "update"
        assert result["event"]["text"] == "new text"
        assert result["event"]["old_text"] == "old text"
        assert mgr.calls[0] == ("update", {"memory_id": "42", "text": "new text"})


class TestMemoryDelete:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_delete

        ctx, _ = _make_ctx()
        result = json.loads(asyncio.run(memory_delete("42", ctx=ctx)))
        assert result["deleted"] is True
        assert result["memory_id"] == "42"

    def test_delete_all(self):
        from grafeo_memory.mcp.tools import memory_delete_all

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_delete_all(user_id="alice", ctx=ctx)))
        assert result["deleted_count"] == 5
        assert mgr.calls[0][1]["user_id"] == "alice"


class TestMemoryList:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_list

        ctx, _ = _make_ctx()
        result = json.loads(asyncio.run(memory_list(ctx=ctx)))
        assert "memories" in result
        assert len(result["memories"]) == 1
        assert result["memories"][0]["text"] == "a fact"

    def test_with_type_filter(self):
        from grafeo_memory.mcp.tools import memory_list

        ctx, mgr = _make_ctx()
        asyncio.run(memory_list(memory_type="procedural", ctx=ctx))
        assert mgr.calls[0][1]["memory_type"] == "procedural"


class TestMemorySummarize:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_summarize

        ctx, mgr = _make_ctx()
        result = json.loads(asyncio.run(memory_summarize(ctx=ctx)))
        assert "events" in result
        assert len(result["events"]) == 1
        assert result["events"][0]["text"] == "consolidated"
        assert mgr.calls[0][1]["preserve_recent"] == 5

    def test_custom_params(self):
        from grafeo_memory.mcp.tools import memory_summarize

        ctx, mgr = _make_ctx()
        asyncio.run(memory_summarize(preserve_recent=10, batch_size=50, ctx=ctx))
        assert mgr.calls[0][1]["preserve_recent"] == 10


class TestMemoryHistory:
    def test_basic(self):
        from grafeo_memory.mcp.tools import memory_history

        ctx, _ = _make_ctx()
        result = json.loads(asyncio.run(memory_history("42", ctx=ctx)))
        assert "history" in result
        assert len(result["history"]) == 1
        assert result["history"][0]["event"] == "ADD"
        assert result["history"][0]["new_text"] == "original text"


class TestErrorHandling:
    def test_tool_returns_error_json(self):
        """When the manager raises, the tool should return a JSON error, not crash."""
        from grafeo_memory.mcp.tools import memory_search

        class BrokenManager(FakeAsyncManager):
            async def search(self, *a, **kw):
                raise RuntimeError("DB connection failed")

        ctx, _ = _make_ctx(BrokenManager())
        result = json.loads(asyncio.run(memory_search("test", ctx=ctx)))
        assert "error" in result
        assert "DB connection failed" in result["error"]
