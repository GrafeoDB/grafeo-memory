"""Async API — use grafeo-memory in async applications.

AsyncMemoryManager provides the same API as MemoryManager
but with async/await support for frameworks like FastAPI, aiohttp, etc.

Requires:
    pip install grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

import asyncio

from openai import OpenAI

from grafeo_memory import AsyncMemoryManager, MemoryConfig, OpenAIEmbedder


async def main():
    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(user_id="alice")

    async with AsyncMemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Async add ---
        print("Adding memories:")
        events = await memory.add("Alice is learning Rust and enjoys systems programming")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        events = await memory.add("Alice prefers Neovim over VS Code")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Async search ---
        print("\nSearch: 'What is Alice learning?'")
        results = await memory.search("What is Alice learning?")
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Async get all ---
        print("\nAll memories:")
        all_memories = await memory.get_all()
        for m in all_memories:
            print(f"  [{m.memory_id}] {m.text}")


if __name__ == "__main__":
    asyncio.run(main())
