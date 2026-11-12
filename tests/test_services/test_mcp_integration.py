"""MCP server tools, resources and lifespan against a real AsyncMemoryManager and GrafeoDB.

test_mcp_tools.py checks the tool wiring with a fake manager; these tests run
the same tools end to end (mock LLM, real database), including the error paths
that only a real manager produces.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Any

import pytest

pytest.importorskip("mcp", reason="mcp package not installed")

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryConfig
from grafeo_memory.mcp import resources, server, tools


@dataclass
class _Lifespan:
    manager: Any


@dataclass
class _Request:
    lifespan_context: _Lifespan


@dataclass
class _Ctx:
    request_context: _Request


def _manager(outputs=(), db_path=None) -> AsyncMemoryManager:
    config = MemoryConfig(db_path=db_path, user_id="mcp_user", embedding_dimensions=16)
    return AsyncMemoryManager(make_test_model(list(outputs)), config, embedder=MockEmbedder(16))


def _ctx(manager) -> Any:
    return _Ctx(_Request(_Lifespan(manager)))


def _call(coro) -> dict:
    return json.loads(asyncio.run(coro))


class TestToolsEndToEnd:
    def test_add_update_history_delete(self):
        manager = _manager()
        ctx = _ctx(manager)

        added = _call(tools.memory_add("alice likes tea", infer=False, ctx=ctx))
        mid = added["events"][0]["memory_id"]
        assert added["events"][0]["action"] == "add"

        updated = _call(tools.memory_update(mid, "alice likes green tea", ctx=ctx))
        assert updated["event"]["old_text"] == "alice likes tea"

        history = _call(tools.memory_history(mid, ctx=ctx))["history"]
        assert [h["event"] for h in history] == ["ADD", "UPDATE"]

        assert _call(tools.memory_delete(mid, ctx=ctx)) == {"deleted": True, "memory_id": mid}
        assert _call(tools.memory_delete(mid, ctx=ctx)) == {"deleted": False, "memory_id": mid}
        history = _call(tools.memory_history(mid, ctx=ctx))["history"]
        assert [h["event"] for h in history] == ["ADD", "UPDATE", "DELETE"]
        manager.close()

    def test_update_of_missing_memory_reports_error(self):
        manager = _manager()
        result = _call(tools.memory_update("12345", "x", ctx=_ctx(manager)))
        assert "not found" in result["error"]
        manager.close()

    def test_add_with_entities_then_delete(self):
        """Deleting a memory linked to entities failed on grafeo 0.5.44 (edges)."""
        manager = _manager(
            [
                {
                    "facts": ["alice works at acme"],
                    "entities": [{"name": "alice", "entity_type": "person"}],
                    "relations": [],
                }
            ]
        )
        ctx = _ctx(manager)
        mid = _call(tools.memory_add("Alice works at Acme", ctx=ctx))["events"][0]["memory_id"]
        assert _call(tools.memory_delete(mid, ctx=ctx))["deleted"] is True
        assert _call(tools.memory_list(ctx=ctx)) == {"memories": []}
        manager.close()

    def test_batch_list_search_stats_explain(self):
        manager = _manager([{"entities": [], "relations": []}, {"entities": [], "relations": []}])
        ctx = _ctx(manager)
        batch = _call(tools.memory_add_batch(["alice likes tea", "bob likes chess"], infer=False, ctx=ctx))
        assert len(batch["events"]) == 2
        assert len(_call(tools.memory_list(ctx=ctx))["memories"]) == 2

        results = _call(tools.memory_search("alice likes tea", ctx=ctx))["results"]
        assert results[0]["text"] == "alice likes tea"

        stats = _call(tools.memory_stats(ctx=ctx))
        assert stats["total_memories"] == 2

        explained = _call(tools.memory_explain_search("alice likes tea", ctx=ctx))
        assert explained["results"][0]["text"] == "alice likes tea"
        assert any(step["name"] == "final" for step in explained["steps"])
        manager.close()

    def test_summarize_and_delete_all(self):
        manager = _manager([{"memories": ["consolidated"]}])
        ctx = _ctx(manager)
        for text in ("one", "two", "three"):
            _call(tools.memory_add(text, infer=False, ctx=ctx))
        summary = _call(tools.memory_summarize(preserve_recent=1, ctx=ctx))
        assert sorted(e["action"] for e in summary["events"]) == ["add", "delete", "delete"]
        assert _call(tools.memory_delete_all(ctx=ctx)) == {"deleted_count": 2}
        manager.close()


class TestResources:
    def test_config_and_stats(self, tmp_path):
        path = str(tmp_path / "mcp.grafeo")
        manager = _manager(db_path=path)
        ctx = _ctx(manager)
        asyncio.run(manager.add("x", infer=False))
        config = json.loads(asyncio.run(resources.memory_config(ctx)))
        assert config["db_path"] == path
        assert config["default_user_id"] == "mcp_user"
        stats = json.loads(asyncio.run(resources.memory_stats(ctx)))
        assert stats["total_memories"] == 1
        manager.close()


class TestServerLifespan:
    def test_manager_from_environment_persists_between_server_runs(self, tmp_path, monkeypatch):
        path = str(tmp_path / "server.grafeo")
        monkeypatch.setenv("GRAFEO_MEMORY_MODEL", "test")
        monkeypatch.setenv("GRAFEO_MEMORY_DB", path)
        monkeypatch.setenv("GRAFEO_MEMORY_USER", "env_user")
        monkeypatch.delenv("GRAFEO_MEMORY_YOLO", raising=False)
        monkeypatch.setattr(server, "_create_embedder", lambda model: MockEmbedder(16))

        async def run_server_once(text: str | None) -> list[str]:
            async with server.app_lifespan(server.mcp) as app:
                manager = app.manager
                assert manager._config.db_path == path
                assert manager._config.user_id == "env_user"
                if text is not None:
                    await manager.add(text, infer=False)
                return [m.text for m in await manager.get_all()]

        assert asyncio.run(run_server_once("first run")) == ["first run"]
        # The lifespan closed the database, so a second server can open the same file.
        assert asyncio.run(run_server_once(None)) == ["first run"]

    def test_yolo_environment(self, monkeypatch):
        from pydantic_ai import Agent

        monkeypatch.setenv("GRAFEO_MEMORY_MODEL", "test")
        monkeypatch.delenv("GRAFEO_MEMORY_DB", raising=False)
        monkeypatch.setenv("GRAFEO_MEMORY_YOLO", "1")
        monkeypatch.setattr(server, "_create_embedder", lambda model: MockEmbedder(16))
        # yolo turns on pydantic-ai instrumentation for every agent: restore it afterwards.
        monkeypatch.setattr(Agent, "_instrument_default", Agent._instrument_default)
        manager = server._create_manager()
        try:
            assert manager._config.enable_importance is True
            assert manager._config.db_path is None
        finally:
            manager.close()
