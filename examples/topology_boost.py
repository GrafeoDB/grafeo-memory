"""Topology boost, promote well-connected memories in search results.

When enabled, memories that share entities with many other memories
get a score boost during search. This is a lightweight, structural
re-ranking step, no extra LLM call required.

For example, "Gilfoyle built Anton, the server that runs Pied Piper"
connects to the "Gilfoyle", "Anton" and "Pied Piper" entities, which
link to other memories. This makes it rank higher than an isolated
memory like "Gilfoyle listens to death metal".

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    embedder = MistralEmbedder(Mistral())

    # Enable topology boost in config
    config = MemoryConfig(
        user_id="gilfoyle",
        enable_topology_boost=True,
        topology_boost_factor=0.2,  # 0.0 = no boost, higher = stronger
    )

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # Add interconnected work memories (share "Gilfoyle", "Pied Piper", "Anton" entities)
        memory.add("Gilfoyle is the systems architect at Pied Piper")
        memory.add("Gilfoyle built Anton, the server that runs Pied Piper")
        memory.add("Gilfoyle trained Son of Anton on Anton's code")
        memory.add("Gilfoyle built PiperCoin for Pied Piper")

        # Add isolated personal memories (fewer shared entities)
        memory.add("Gilfoyle listens to death metal")
        memory.add("Gilfoyle's girlfriend is Tara")

        # Search, well-connected work memories should rank higher
        print("Search: 'What does Gilfoyle do?'")
        print("(topology boost promotes memories with more entity connections)\n")
        for r in memory.search("What does Gilfoyle do?"):
            print(f"  [{r.score:.3f}] {r.text}")


if __name__ == "__main__":
    main()
