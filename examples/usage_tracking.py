"""LLM usage tracking — monitor token consumption.

grafeo-memory tracks LLM usage across all operations:
1. Per-operation callbacks via MemoryConfig.usage_callback
2. Aggregated usage on results via the .usage attribute

Requires:
    pip install grafeo-memory[openai]
    OPENAI_API_KEY environment variable
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    def on_usage(operation, usage):
        tokens = (usage.input_tokens or 0) + (usage.output_tokens or 0)
        print(f"  [{operation}] {tokens} tokens ({usage.input_tokens} in / {usage.output_tokens} out)")

    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(user_id="alice", usage_callback=on_usage)

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Add (fires extract_facts + extract_entities callbacks) ---
        print("Adding memory (watching LLM calls):")
        result = memory.add("Alice is a data scientist at Acme Corp who loves hiking")
        print(
            f"  Total: {result.usage.input_tokens} in, {result.usage.output_tokens} out, "
            f"{result.usage.requests} requests\n"
        )

        # --- Search (fires extract_entities for graph search) ---
        print("Searching (watching LLM calls):")
        search = memory.search("What does Alice do?")
        print(
            f"  Total: {search.usage.input_tokens} in, {search.usage.output_tokens} out, "
            f"{search.usage.requests} requests"
        )

        print(f"\nFound {len(search)} results:")
        for r in search:
            print(f"  [{r.score:.2f}] {r.text}")


if __name__ == "__main__":
    main()
