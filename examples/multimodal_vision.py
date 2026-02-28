"""Vision / multimodal — extract memories from images.

grafeo-memory can process images via a vision-capable LLM:
1. Images are described as text by the LLM
2. Text descriptions flow through normal extraction
3. No images are stored — only the extracted facts

Messages use the OpenAI multimodal content format.

Requires:
    uv add grafeo-memory[openai]
    OPENAI_API_KEY environment variable
    A vision-capable model (gpt-4o, gpt-4o-mini)
"""

from openai import OpenAI

from grafeo_memory import MemoryConfig, MemoryManager, OpenAIEmbedder


def main():
    embedder = OpenAIEmbedder(OpenAI())
    config = MemoryConfig(
        user_id="alice",
        enable_vision=True,
        # Optionally use a different model for image description:
        # vision_model="openai:gpt-4o",
    )

    with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
        # --- Add a multimodal message ---
        print("Adding a multimodal message with text + image:")
        events = memory.add(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Here's a photo from the team offsite"},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png"
                        },
                    },
                ],
            }
        )
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Plain text still works normally ---
        print("\nAdding plain text:")
        events = memory.add("The offsite was in Lake Tahoe, we discussed Q3 roadmap")
        for e in events:
            print(f"  [{e.action.value.upper()}] {e.text}")

        # --- Search across both ---
        print("\nSearch: 'What happened at the offsite?'")
        results = memory.search("What happened at the offsite?")
        for r in results:
            print(f"  [{r.score:.2f}] {r.text}")


if __name__ == "__main__":
    main()
