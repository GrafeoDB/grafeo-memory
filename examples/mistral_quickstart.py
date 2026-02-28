"""Mistral quick-start — using grafeo-memory with Mistral models.

Demonstrates the full workflow with Mistral as both the LLM and
embedding provider: add, reconcile, search, and history.

Requires:
    uv add grafeo-memory[mistral]
    MISTRAL_API_KEY environment variable
"""

from mistralai import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    client = Mistral()
    embedder = MistralEmbedder(client)
    config = MemoryConfig(user_id="alice")

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add memories ---
        print("Adding: 'My name is Alice and I work at Acme Corp as a software engineer'")
        events = memory.add("My name is Alice and I work at Acme Corp as a software engineer")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        print()
        print("Adding: 'I love sushi and I'm learning Rust'")
        events = memory.add("I love sushi and I'm learning Rust")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Reconciliation: update a contradicting fact ---
        print()
        print("Adding: 'I just switched jobs — I now work at Google as a staff engineer'")
        events = memory.add("I just switched jobs — I now work at Google as a staff engineer")
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

        # --- History ---
        print()
        all_memories = memory.get_all()
        print(f"All memories ({len(all_memories)}):")
        for m in all_memories:
            print(f"  [{m.memory_id}] {m.text}")
            hist = memory.history(m.memory_id)
            for h in hist:
                print(f"    {h.event}: {h.new_text}")


if __name__ == "__main__":
    main()
