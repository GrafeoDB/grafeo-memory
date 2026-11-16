"""Importance scoring, multi-factor search ranking.

When enabled, search results are ranked by a composite score:
  w_sim * similarity + w_rec * recency + w_freq * frequency + w_imp * importance

- similarity: vector cosine distance (always present)
- recency: exponential decay based on age
- frequency: how often this memory has been accessed
- importance: user-set priority (0.0 - 1.0)

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(
        user_id="erlich",
        enable_importance=True,
        weight_similarity=0.4,
        weight_recency=0.3,
        weight_frequency=0.15,
        weight_importance=0.15,
    )

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add memories (raw mode for predictable demo) ---
        r1 = memory.add("Erlich owns ten percent of Pied Piper through his incubator", infer=False)
        r2 = memory.add("Erlich's catchphrase is Aviato", infer=False)
        r3 = memory.add("Erlich is flying to Tibet next week", infer=False)

        # --- Set importance levels ---
        memory.set_importance(r1[0].memory_id, 0.9)  # high
        memory.set_importance(r2[0].memory_id, 0.3)  # low
        memory.set_importance(r3[0].memory_id, 0.7)  # medium

        print("Memories with importance scores:")
        for m in memory.get_all():
            print(f"  [{m.memory_id}] (imp={m.importance}) {m.text}")

        # --- Search, importance affects ranking ---
        print("\nSearch results (importance-weighted):")
        for r in memory.search("Erlich travel plans"):
            imp = r.importance if r.importance is not None else "n/a"
            print(f"  [{r.score:.3f}] (imp={imp}) {r.text}")


if __name__ == "__main__":
    main()
