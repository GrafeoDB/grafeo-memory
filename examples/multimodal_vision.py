"""Vision / multimodal, extract memories from images.

grafeo-memory can process images via a vision-capable LLM:
1. Images are described as text by the LLM
2. Text descriptions flow through normal extraction
3. No images are stored, only the extracted facts

Messages use the OpenAI multimodal content format, which grafeo-memory
accepts for every provider. An image is a URL the model can fetch, or a
data URL, as here: a photo of Pied Piper's whiteboard.

Requires:
    uv add "grafeo-memory[mistral]"
    MISTRAL_API_KEY environment variable
    A vision-capable model (mistral-small-latest, or another provider's)
"""

import base64
from pathlib import Path

from mistralai.client import Mistral

from grafeo_memory import MemoryConfig, MemoryManager, MistralEmbedder

WHITEBOARD = Path(__file__).with_name("pied_piper_whiteboard.png")


def main():
    whiteboard_url = "data:image/png;base64," + base64.b64encode(WHITEBOARD.read_bytes()).decode()
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
                    {"type": "text", "text": "Here's a photo of our whiteboard at Pied Piper"},
                    {"type": "image_url", "image_url": {"url": whiteboard_url}},
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

        # --- Search: the answer comes from the image ---
        print("\nSearch: 'What is Pied Piper's Weissman score?' (only the whiteboard says)")
        for r in memory.search("What is Pied Piper's Weissman score?", k=3):
            print(f"  [{r.score:.2f}] {r.text}")


if __name__ == "__main__":
    main()
