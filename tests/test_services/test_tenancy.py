"""for_user(): one user's view of a shared manager, which no call by id can leave."""

from __future__ import annotations

import asyncio
import json

import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryConfig, MemoryManager


def _manager() -> MemoryManager:
    config = MemoryConfig(user_id="admin", embedding_dimensions=16)
    return MemoryManager(make_test_model([]), config, embedder=MockEmbedder(16))


@pytest.fixture
def shared():
    manager = _manager()
    alice, bob = manager.for_user("alice"), manager.for_user("bob")
    a_id = alice.add("alice likes tea", infer=False)[0].memory_id
    b_id = bob.add("bob likes coffee", infer=False)[0].memory_id
    yield manager, alice, bob, a_id, b_id
    manager.close()


def test_a_view_acts_for_its_user(shared):
    manager, alice, _bob, _a, _b = shared
    assert [m.text for m in alice.get_all()] == ["alice likes tea"]
    assert [m.text for m in manager.get_all(user_id="bob")] == ["bob likes coffee"]
    assert alice.get_all(user_id="alice")  # naming its own user is fine


def test_a_view_refuses_another_user(shared):
    _m, alice, _bob, _a, _b = shared
    with pytest.raises(ValueError, match="bound to user 'alice'"):
        alice.search("coffee", user_id="bob")
    with pytest.raises(ValueError, match="bound to user 'alice'"):
        alice.for_user("bob")


def test_calls_by_id_never_reach_another_users_memory(shared):
    _m, alice, bob, a_id, b_id = shared
    with pytest.raises(ValueError, match="not found"):
        alice.update(b_id, "alice rewrote it")
    assert alice.delete(b_id) is False
    assert alice.set_importance(b_id, 0.1) is False
    assert alice.certify(b_id) is False
    assert alice.history(b_id) == []
    assert [e.event for e in bob.history(b_id)] == ["ADD"]
    assert [m.text for m in bob.get_all()] == ["bob likes coffee"]
    assert alice.delete(a_id) is True


def test_the_unbound_manager_is_unchanged(shared):
    manager, _alice, _bob, _a, b_id = shared
    assert manager.set_importance(b_id, 0.2) is True
    assert [e.event for e in manager.history(b_id)] == ["ADD"]


def test_views_share_state_and_closing_one_keeps_the_database(shared):
    manager, alice, bob, _a, _b = shared
    before = manager._write_version
    alice.add("alice likes green tea", infer=False)
    assert manager._write_version == bob._write_version == alice._write_version
    assert manager._write_version >= before
    alice.close()
    assert len(bob.get_all()) == 1 and len(manager.get_all(user_id="alice")) == 2


def test_roots_are_edited_on_the_manager_only(shared):
    _m, alice, _bob, _a, _b = shared
    with pytest.raises(ValueError, match="belong to the graph"):
        alice.set_root_scheme(add={"cooking": "recipes, food"})


def test_open_decisions_of_another_user_are_out_of_reach(shared):
    from grafeo_memory.services import bucket

    manager, alice, bob, _a, b_id = shared
    od_id = bucket.create(
        manager,
        "bob",
        kind="duplicate",
        question="same fact?",
        options=["yes", "no"],
        evidence=[b_id],
        proposed="yes",
        confidence=0.4,
        rung="decision",
    )
    assert alice.resolve_open_decision(od_id, "no") is False
    assert bob.list_open_decisions() and not alice.list_open_decisions()


def test_mcp_tools_reach_the_named_users_memories_only():
    from grafeo_memory.mcp import tools

    config = MemoryConfig(user_id="mcp_user", embedding_dimensions=16)
    manager = AsyncMemoryManager(make_test_model([]), config, embedder=MockEmbedder(16))
    ctx = type("Ctx", (), {})()
    ctx.request_context = type("Req", (), {"lifespan_context": type("L", (), {"manager": manager})()})()

    def call(coro) -> dict:
        return json.loads(asyncio.run(coro))

    bob_id = call(tools.memory_add("bob likes coffee", user_id="bob", infer=False, ctx=ctx))["events"][0]["memory_id"]
    assert call(tools.memory_delete(bob_id, ctx=ctx))["deleted"] is False  # server default user
    assert "error" in call(tools.memory_update(bob_id, "changed", ctx=ctx))
    assert call(tools.memory_delete(bob_id, user_id="bob", ctx=ctx))["deleted"] is True
    manager.close()
