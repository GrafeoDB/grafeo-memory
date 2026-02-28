"""Quick-start example for grafeo-memory.

Demonstrates the core loop: add memories, watch reconciliation
(ADD / UPDATE / DELETE decisions), search, and list.

Requires:
    uv add grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(user_id="alice")

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Add memories from conversation ---
        print("Adding: 'I just started a new job at Acme Corp as a data scientist'")
        events = memory.add("I just started a new job at Acme Corp as a data scientist")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        print()
        print("Adding: 'I've been promoted to senior data scientist'")
        events = memory.add("I've been promoted to senior data scientist at Acme")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")
            if e.old_text:
                print(f"    (was: {e.old_text})")

        print()
        print("Adding: 'I left Acme and joined Beta Inc'")
        events = memory.add("I left Acme and joined Beta Inc")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")
            if e.old_text:
                print(f"    (was: {e.old_text})")

        # --- Search ---
        print()
        print("Searching: 'Where does Alice work?'")
        results = memory.search("Where does Alice work?")
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Get all memories ---
        print()
        print("All memories:")
        all_memories = memory.get_all()
        for m in all_memories:
            print(f"  [{m.memory_id}] {m.text}")


if __name__ == "__main__":
    main()
