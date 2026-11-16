"""LLM usage tracking, monitor token consumption.

grafeo-memory tracks LLM usage across all operations:
1. Per-operation callbacks via MemoryConfig.usage_callback
2. Aggregated usage on results via the .usage attribute

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    def on_usage(operation, usage):
        tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        print(f"  [{operation}] {tokens} tokens ({usage.input_tokens} in / {usage.output_tokens} out)")

    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(user_id="russ", usage_callback=on_usage)

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add (fires the extraction and reconciliation callbacks) ---
        print("Adding memory (watching LLM calls):")
        result = memory.add("I led Pied Piper's Series A, and I have three commas: I'm a billionaire", speaker="Russ")
        print(
            f"  Total: {result.usage.input_tokens} in, {result.usage.output_tokens} out, "
            f"{result.usage.requests} requests\n"
        )

        # --- Search (no model call by default; RetrievalConfig.query_entities="llm" adds one) ---
        print("Searching (watching LLM calls):")
        search = memory.search("How rich is Russ?")
        print(
            f"  Total: {search.usage.input_tokens} in, {search.usage.output_tokens} out, "
            f"{search.usage.requests} requests"
        )

        print(f"\nFound {len(search)} results:")
        for r in search:
            print(f"  [{r.score:.2f}] {r.text}")


if __name__ == "__main__":
    main()
