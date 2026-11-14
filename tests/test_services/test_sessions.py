"""start_session(): anonymous sessions stay in memory; binding joins them to a user, one way."""

from __future__ import annotations

import asyncio
import json

import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryConfig, MemoryManager
from grafeo_memory.common.types import TiersConfig, WorkingConfig

EXTRACTED = {"facts": ["the user likes tea"], "entities": [], "relations": []}


def _manager(outputs=(), **working) -> MemoryManager:
    tiers = TiersConfig(working=WorkingConfig(enabled=True, **working))
    config = MemoryConfig(user_id="admin", embedding_dimensions=16, tiers=tiers)
    return MemoryManager(make_test_model(list(outputs)), config, embedder=MockEmbedder(16))


def _memories(manager: MemoryManager) -> int:
    return len(manager._db.get_nodes_by_label("Memory"))


def test_an_anonymous_session_never_writes_long_term_memory():
    with _manager() as memory:  # no model output queued: any model call would fail
        sid = memory.start_session(anonymous=True)
        memory.add("I like tea", session_id=sid)
        memory.add("raw note", session_id=sid, infer=False)
        block = memory.assemble("tea", sid)
        assert [i.text for i in block.sections["working"]] == ["user: I like tea", "user: raw note"]
        assert _memories(memory) == 0
        assert list(memory.end_session(sid)) == []
        assert memory.assemble("tea", sid).sections.get("working") is None
        assert _memories(memory) == 0


def test_binding_stores_the_turns_for_that_user_once():
    with _manager([EXTRACTED]) as memory:
        sid = memory.start_session(anonymous=True)
        memory.add("I like tea", session_id=sid)
        alice = memory.for_user("alice")
        assert alice.start_session(sid) == sid
        assert [m.text for m in alice.get_all()] == ["the user likes tea"]
        assert [i.text for i in alice.assemble("tea", sid).sections["working"]] == ["user: I like tea"]
        with pytest.raises(ValueError, match="another user"):
            memory.for_user("bob").start_session(sid)
        with pytest.raises(ValueError, match="cannot become anonymous"):
            memory.start_session(sid, anonymous=True)


def test_with_defer_extraction_binding_waits_for_end_session():
    with _manager([EXTRACTED], defer_extraction=True, promote_on_session_end=True) as memory:
        sid = memory.start_session(anonymous=True)
        memory.add("I like tea", session_id=sid)
        alice = memory.for_user("alice")
        alice.start_session(sid)
        assert alice.get_all() == []
        alice.end_session(sid)
        assert [m.text for m in alice.get_all()] == ["the user likes tea"]


def test_anonymous_sessions_expire_after_the_ttl():
    with _manager(session_ttl=0.0) as memory:
        sid = memory.start_session(anonymous=True)
        memory.add("I like tea", session_id=sid)
        memory.start_session(anonymous=True)  # any session call purges expired ones
        assert memory.assemble("tea", sid).sections.get("working") is None
        # the id stays anonymous: a caller still holding it never writes long-term memory
        memory.add("I like coffee", session_id=sid)
        assert _memories(memory) == 0


def test_an_anonymous_window_keeps_its_last_turns():
    with _manager(window_messages=2) as memory:
        sid = memory.start_session(anonymous=True)
        for turn in ("one", "two", "three"):
            memory.add(turn, session_id=sid)
        assert [i.text for i in memory.assemble("x", sid).sections["working"]] == ["user: two", "user: three"]


def test_a_failed_bind_keeps_the_session_anonymous_with_its_turns(monkeypatch):
    with _manager([EXTRACTED]) as memory:
        sid = memory.start_session(anonymous=True)
        memory.add("I like tea", session_id=sid)
        alice = memory.for_user("alice")

        async def broken_add(*args, **kwargs):
            raise RuntimeError("database write failed")

        monkeypatch.setattr(alice, "_add", broken_add)
        with pytest.raises(RuntimeError):
            alice.start_session(sid)
        assert memory._sessions[sid][0] is None
        assert [i.text for i in memory.assemble("tea", sid).sections["working"]] == ["user: I like tea"]

        monkeypatch.undo()
        memory.for_user("alice").start_session(sid)  # binding again works
        assert [m.text for m in memory.for_user("alice").get_all()] == ["the user likes tea"]


def test_a_new_session_gets_an_id():
    with _manager() as memory:
        a, b = memory.start_session(), memory.start_session()
        assert a != b and len(a) == 32


def test_mcp_session_tools():
    from grafeo_memory.mcp import tools

    tiers = TiersConfig(working=WorkingConfig(enabled=True))
    config = MemoryConfig(user_id="mcp_user", embedding_dimensions=16, tiers=tiers)
    manager = AsyncMemoryManager(make_test_model([]), config, embedder=MockEmbedder(16))
    ctx = type("Ctx", (), {})()
    ctx.request_context = type("Req", (), {"lifespan_context": type("L", (), {"manager": manager})()})()

    def call(coro) -> dict:
        return json.loads(asyncio.run(coro))

    sid = call(tools.memory_session_start(anonymous=True, ctx=ctx))["session_id"]
    assert call(tools.memory_add("visitor likes tea", session_id=sid, ctx=ctx))["events"] == []
    assert call(tools.memory_session_end(sid, ctx=ctx))["events"] == []
    assert len(manager._db.get_nodes_by_label("Memory")) == 0
    manager.close()
