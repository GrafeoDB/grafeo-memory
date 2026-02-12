"""Procedural memory — storing instructions, preferences, and rules.

grafeo-memory supports two memory types:
- semantic (default): facts, biographical details, events
- procedural: instructions, preferences, behavioral rules

Each type uses its own extraction prompt and reconciliation scope,
so procedural and semantic memories never interfere with each other.

Requires:
    pip install grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(user_id="developer")

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Add procedural memories (instructions / preferences) ---
        print("Adding procedural memories...")
        memory.add("Always use type hints in Python code", memory_type="procedural")
        memory.add("Prefer pytest over unittest for testing", memory_type="procedural")
        memory.add("Use Google-style docstrings", memory_type="procedural")

        # --- Add semantic memories (facts) ---
        print("Adding semantic memories...")
        memory.add("The project uses Python 3.12 and ruff for linting")
        memory.add("The main database is PostgreSQL 16")

        # --- Search procedural only ---
        print("\nSearching procedural memories for 'testing':")
        results = memory.search("testing", memory_type="procedural")
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Search semantic only ---
        print("\nSearching semantic memories for 'database':")
        results = memory.search("database", memory_type="semantic")
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")

        # --- List all by type ---
        print("\nAll procedural memories:")
        for m in memory.get_all(memory_type="procedural"):
            print(f"  [{m.memory_id}] {m.text}")

        print("\nAll semantic memories:")
        for m in memory.get_all(memory_type="semantic"):
            print(f"  [{m.memory_id}] {m.text}")


if __name__ == "__main__":
    main()
