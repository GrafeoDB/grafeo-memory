"""Async API, use grafeo-memory in async applications.

AsyncMemoryManager provides the same API as MemoryManager
but with async/await support for frameworks like FastAPI, aiohttp, etc.

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

import asyncio

from mistralai.client import Mistral

from grafeo_memory import AsyncMemoryManager, MemoryConfig, MistralEmbedder


async def main():
    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(user_id="jian-yang")

    async with AsyncMemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Async add ---
        print("Adding memories:")
        events = await memory.add(
            "I built Not Hotdog, an app that tells you if food is a hotdog or not", speaker="Jian-Yang"
        )
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        events = await memory.add("My fridge has a screen and it talks", speaker="Jian-Yang")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Async search ---
        print("\nSearch: 'What did Jian-Yang build?'")
        for r in await memory.search("What did Jian-Yang build?"):
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Async get all ---
        print("\nAll memories:")
        for m in await memory.get_all():
            print(f"  [{m.memory_id}] {m.text}")


if __name__ == "__main__":
    asyncio.run(main())
