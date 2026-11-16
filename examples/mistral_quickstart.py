"""Mistral walkthrough: add, reconcile, search, and history.

Jared leaves Hooli for Pied Piper. Mistral is both the LLM and the
embedding provider.

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def show(events):
    for e in events:
        print(f"  [{e.action.value.upper()}] {e.text}")
        if e.old_text:
            print(f"    (was: {e.old_text})")


def main():
    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(user_id="jared")

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add memories ---
        text = "I work at Hooli in business development, on Gavin Belson's team"
        print(f"Adding: {text!r}")
        show(memory.add(text, speaker="Jared"))

        text = "My birth name is Donald Dunn"
        print(f"\nAdding: {text!r}")
        show(memory.add(text, speaker="Jared"))

        # --- Reconciliation: update a contradicting fact ---
        text = "I quit Hooli and joined Pied Piper as head of business development"
        print(f"\nAdding: {text!r}")
        show(memory.add(text, speaker="Jared"))

        # --- Search ---
        print("\nSearching: 'Where does Jared work?'")
        for r in memory.search("Where does Jared work?", k=3):
            print(f"  [{r.score:.2f}] {r.text}")

        # --- History ---
        all_memories = memory.get_all()
        print(f"\nAll memories ({len(all_memories)}):")
        for m in all_memories:
            print(f"  [{m.memory_id}] {m.text}")
            for h in memory.history(m.memory_id):
                print(f"    {h.event}: {h.new_text}")


if __name__ == "__main__":
    main()
