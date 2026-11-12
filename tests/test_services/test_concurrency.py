"""Tests for concurrent add operations.

Covers T7 (concurrent add for same user) and T8 (concurrent add for different users).

Uses AsyncMemoryManager with asyncio.gather to test true concurrency,
since the sync MemoryManager uses a shared asyncio.Runner that cannot
handle concurrent calls from multiple threads.
"""

from __future__ import annotations

import asyncio

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import AsyncMemoryManager, MemoryConfig


def _make_async_manager(dims=16, **config_kwargs):
    """Create an AsyncMemoryManager with mock model and in-memory GrafeoDB."""
    model = make_test_model([])
    embedder = MockEmbedder(dims)
    defaults = {"db_path": None, "user_id": "test_user", "embedding_dimensions": dims}
    defaults.update(config_kwargs)
    config = MemoryConfig(**defaults)  # type: ignore[invalid-argument-type]
    return AsyncMemoryManager(model, config, embedder=embedder)


class TestConcurrentAddSameUser:
    """T7: concurrent add() calls for the same user_id should not corrupt data."""

    def test_concurrent_adds_no_data_loss(self):
        async def _run():
            async with _make_async_manager() as manager:
                user = "shared_user"
                tasks = []
                for thread_id in (1, 2):
                    for i in range(5):
                        tasks.append(manager.add(f"thread {thread_id} fact {i}", user_id=user, infer=False))
                await asyncio.gather(*tasks)

                memories = await manager.get_all(user_id=user)
                # Both "threads" added 5 facts each, so we expect 10 total
                assert len(memories) == 10
                texts = sorted(m.text for m in memories)
                # Verify no duplicates or missing entries
                assert len(set(texts)) == 10

        asyncio.run(_run())

    def test_concurrent_adds_all_texts_present(self):
        async def _run():
            async with _make_async_manager() as manager:
                user = "shared_user"
                texts_to_add = [f"concurrent fact {i}" for i in range(10)]
                tasks = [manager.add(t, user_id=user, infer=False) for t in texts_to_add]
                await asyncio.gather(*tasks)

                memories = await manager.get_all(user_id=user)
                stored_texts = {m.text for m in memories}
                for t in texts_to_add:
                    assert t in stored_texts, f"Missing: {t}"

        asyncio.run(_run())


class TestConcurrentAddDifferentUsers:
    """T8: concurrent adds for different users should not contaminate each other."""

    def test_user_isolation_under_concurrency(self):
        async def _run():
            async with _make_async_manager() as manager:

                async def add_for_user(user_id: str):
                    for i in range(5):
                        await manager.add(f"{user_id} fact {i}", user_id=user_id, infer=False)

                await asyncio.gather(add_for_user("alice"), add_for_user("bob"))

                alice_mems = await manager.get_all(user_id="alice")
                bob_mems = await manager.get_all(user_id="bob")

                assert len(alice_mems) == 5
                assert len(bob_mems) == 5

                # No cross-contamination
                alice_texts = {m.text for m in alice_mems}
                bob_texts = {m.text for m in bob_mems}
                assert alice_texts.isdisjoint(bob_texts)

                # Verify all alice facts mention alice
                for t in alice_texts:
                    assert t.startswith("alice")
                for t in bob_texts:
                    assert t.startswith("bob")

        asyncio.run(_run())

    def test_three_users_concurrent(self):
        async def _run():
            async with _make_async_manager() as manager:

                async def add_for_user(user_id: str):
                    for i in range(3):
                        await manager.add(f"{user_id} item {i}", user_id=user_id, infer=False)

                await asyncio.gather(
                    add_for_user("alice"),
                    add_for_user("bob"),
                    add_for_user("carol"),
                )

                for user_id in ("alice", "bob", "carol"):
                    mems = await manager.get_all(user_id=user_id)
                    assert len(mems) == 3
                    for m in mems:
                        assert m.text.startswith(user_id)

        asyncio.run(_run())
