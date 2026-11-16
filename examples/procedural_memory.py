"""Procedural memory, storing instructions, preferences, and rules.

grafeo-memory supports two memory types:
- semantic (default): facts, biographical details, events
- procedural: instructions, preferences, behavioral rules

Each type uses its own extraction prompt and reconciliation scope,
so procedural and semantic memories never interfere with each other.

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(user_id="richard")

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add procedural memories (instructions / preferences) ---
        print("Adding procedural memories...")
        memory.add("Always indent with tabs, never spaces", memory_type="procedural")
        memory.add("Never let Son of Anton push to production without a review", memory_type="procedural")
        memory.add("Write tests for every change to the compression code", memory_type="procedural")

        # --- Add semantic memories (facts) ---
        print("Adding semantic memories...")
        memory.add("Pied Piper's compression engine uses middle-out and scores 5.2 on the Weissman scale")
        memory.add("Pied Piper's servers run on Anton, which Gilfoyle built")

        # --- Search procedural only ---
        print("\nSearching procedural memories for 'code style':")
        for r in memory.search("code style", memory_type="procedural"):
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Search semantic only ---
        print("\nSearching semantic memories for 'servers':")
        for r in memory.search("servers", memory_type="semantic"):
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
