"""Topology boost — promote well-connected memories in search results.

When enabled, memories that share entities with many other memories
get a score boost during search. This is a lightweight, structural
re-ranking step — no extra LLM call required.

For example, "Alice works at Google" connects to both "Alice" and
"Google" entities, which may link to other memories. This makes it
rank higher than an isolated memory like "Alice has a dog named Max".

Requires:
    uv add grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    embedder = OpenAIEmbedder(OpenAI())

    # Enable topology boost in config
    config = MemoryConfig(
        user_id="alice",
        enable_topology_boost=True,
        topology_boost_factor=0.2,  # 0.0 = no boost, higher = stronger
    )

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # Add interconnected work memories (share "Alice", "Google" entities)
        memory.add("Alice is a software engineer at Google")
        memory.add("Alice works on the search ranking team at Google")
        memory.add("Alice uses Python and Go daily at work")
        memory.add("Alice mentors junior engineers on her team")

        # Add isolated personal memories (fewer shared entities)
        memory.add("Alice has a dog named Max")
        memory.add("Alice enjoys hiking on weekends")

        # Search — well-connected work memories should rank higher
        print("Search: 'What does Alice do?'")
        print("(topology boost promotes memories with more entity connections)\n")
        results = memory.search("What does Alice do?")
        for r in results:
            print(f"  [{r.score:.3f}] {r.text}")


if __name__ == "__main__":
    main()
