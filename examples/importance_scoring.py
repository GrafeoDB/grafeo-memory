"""Importance scoring — multi-factor search ranking.

When enabled, search results are ranked by a composite score:
  w_sim * similarity + w_rec * recency + w_freq * frequency + w_imp * importance

- similarity: vector cosine distance (always present)
- recency: exponential decay based on age
- frequency: how often this memory has been accessed
- importance: user-set priority (0.0 - 1.0)

Requires:
    pip install grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(
        user_id="alice",
        enable_importance=True,
        weight_similarity=0.4,
        weight_recency=0.3,
        weight_frequency=0.15,
        weight_importance=0.15,
    )

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Add memories (raw mode for predictable demo) ---
        r1 = memory.add("Alice's phone number is 555-0100", infer=False)
        r2 = memory.add("Alice prefers window seats on flights", infer=False)
        r3 = memory.add("Alice's passport expires in 2027", infer=False)

        # --- Set importance levels ---
        memory.set_importance(r1[0].memory_id, 0.9)  # high
        memory.set_importance(r2[0].memory_id, 0.3)  # low
        memory.set_importance(r3[0].memory_id, 0.7)  # medium

        print("Memories with importance scores:")
        for m in memory.get_all():
            print(f"  [{m.memory_id}] (imp={m.importance}) {m.text}")

        # --- Search — importance affects ranking ---
        print("\nSearch results (importance-weighted):")
        results = memory.search("Alice travel information")
        for r in results:
            imp = r.importance if r.importance is not None else "n/a"
            print(f"  [{r.score:.3f}] (imp={imp}) {r.text}")


if __name__ == "__main__":
    main()
