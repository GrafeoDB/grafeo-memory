"""FastMCP server for grafeo-memory."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

from grafeo_memory.manager import AsyncMemoryManager
from grafeo_memory.types import MemoryConfig


@dataclass
class AppContext:
    manager: AsyncMemoryManager


def _create_embedder(model: str):
    """Auto-detect and create the appropriate embedder based on the model string."""
    provider = model.split(":")[0] if ":" in model else "openai"

    if provider == "mistral":
        try:
            from mistralai import Mistral
        except ImportError:
            print("Error: mistralai package not installed.", file=sys.stderr)
            print("Install it with: pip install grafeo-memory[mistral]", file=sys.stderr)
            sys.exit(1)
        from grafeo_memory.embedding import MistralEmbedder

        api_key = os.environ.get("MISTRAL_API_KEY")
        return MistralEmbedder(Mistral(api_key=api_key))

    # Default to OpenAI for openai, anthropic, groq, etc.
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed.", file=sys.stderr)
        print("Install it with: pip install grafeo-memory[openai]", file=sys.stderr)
        sys.exit(1)
    from grafeo_memory.embedding import OpenAIEmbedder

    return OpenAIEmbedder(OpenAI())


def _create_manager() -> AsyncMemoryManager:
    """Build an AsyncMemoryManager from environment variables."""
    model = os.environ.get("GRAFEO_MEMORY_MODEL", "openai:gpt-4o-mini")
    db_path = os.environ.get("GRAFEO_MEMORY_DB")
    user_id = os.environ.get("GRAFEO_MEMORY_USER", "default")
    yolo = os.environ.get("GRAFEO_MEMORY_YOLO", "").strip() in ("1", "true", "yes")

    embedder = _create_embedder(model)
    if yolo:
        config = MemoryConfig.yolo(db_path=db_path, user_id=user_id)
    else:
        config = MemoryConfig(db_path=db_path, user_id=user_id)
    return AsyncMemoryManager(model, config, embedder=embedder)


@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    manager = _create_manager()
    try:
        yield AppContext(manager=manager)
    finally:
        manager.close()


mcp = FastMCP("grafeo-memory", lifespan=app_lifespan)


# Import modules to register tools/resources/prompts on the mcp instance.
import grafeo_memory.mcp.prompts  # noqa: E402
import grafeo_memory.mcp.resources  # noqa: E402
import grafeo_memory.mcp.tools  # noqa: E402, F401


def main():
    arg = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if arg == "sse":
        mcp.run(transport="sse")
    elif arg == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
