"""Quick-start example for grafeo-memory.

Demonstrates the core loop: add memories, watch reconciliation
(ADD / UPDATE / DELETE decisions), search, and list. Richard has just
left Hooli to start Pied Piper.

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
    config = MemoryConfig(user_id="richard")

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add memories from conversation ---
        text = (
            "I just left Hooli to start Pied Piper, a compression company built on middle-out. "
            "We work out of the Hacker Hostel in Palo Alto"
        )
        print(f"Adding: {text!r}")
        show(memory.add(text, speaker="Richard"))

        # --- Reconciliation: the move updates where Pied Piper is based ---
        text = "We moved Pied Piper out of the Hacker Hostel into our own office in Mountain View"
        print(f"\nAdding: {text!r}")
        show(memory.add(text, speaker="Richard"))

        # --- Search ---
        print("\nSearching: 'Where is Pied Piper based?'")
        for r in memory.search("Where is Pied Piper based?", k=3):
            print(f"  [{r.score:.2f}] {r.text}")

        # --- Get all memories ---
        print("\nAll memories:")
        for m in memory.get_all():
            print(f"  [{m.memory_id}] {m.text}")


if __name__ == "__main__":
    main()
