"""Vision / multimodal, extract memories from images.

grafeo-memory can process images via a vision-capable LLM:
1. Images are described as text by the LLM
2. Text descriptions flow through normal extraction
3. No images are stored, only the extracted facts

Messages use the OpenAI multimodal content format, which grafeo-memory
accepts for every provider.

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
    A vision-capable model (mistral-small-latest, or another provider's)
"""

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder


def main():
    embedder = MistralEmbedder(Mistral())
    config = MemoryConfig(
        user_id="richard",
        enable_vision=True,
        # Optionally use a different model for image description:
        # vision_model="mistral:mistral-medium-latest",
    )

    with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
        # --- Add a multimodal message ---
        print("Adding a multimodal message with text + image:")
        events = memory.add(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a screenshot from our demo at TechCrunch Disrupt"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://raw.githubusercontent.com/GrafeoDB/grafeo/main/docs/assets/playground.png"
                        },
                    },
                ],
            },
            speaker="Richard",
        )
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Plain text still works normally ---
        print("\nAdding plain text:")
        events = memory.add("We won TechCrunch Disrupt with the middle-out demo", speaker="Richard")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Search across both ---
        print("\nSearch: 'What happened at TechCrunch Disrupt?'")
        for r in memory.search("What happened at TechCrunch Disrupt?"):
            print(f"  [{r.score:.2f}] {r.text}")


if __name__ == "__main__":
    main()
