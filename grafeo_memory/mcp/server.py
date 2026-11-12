"""FastMCP server for grafeo-memory."""

from __future__ import annotations

import os
import sys
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from mcp.server.mcpserver import MCPServer

from grafeo_memory.common.types import MemoryConfig
from grafeo_memory.services.memory import AsyncMemoryManager


@dataclass
class AppContext:
    manager: AsyncMemoryManager


def _create_embedder(model: str):
    """Auto-detect and create the appropriate embedder based on the model string."""
    from grafeo_memory.common.settings import GrafeoMemorySettings

    provider = model.split(":")[0] if ":" in model else "openai"

    if provider == "mistral":
        from grafeo_memory.providers.embedding import mistral_client_class

        try:
            Mistral = mistral_client_class()
        except ImportError:
            print("Error: mistralai package not installed.", file=sys.stderr)
            print("Install it with: uv add grafeo-memory[mistral]", file=sys.stderr)
            sys.exit(1)
        from grafeo_memory.providers.embedding import MistralEmbedder

        try:
            settings = GrafeoMemorySettings()
            api_key = settings.mistral_api_key.get_secret_value() if settings.mistral_api_key else None
        except Exception:
            api_key = None
        api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        return MistralEmbedder(Mistral(api_key=api_key))

    # Default to OpenAI for openai, anthropic, groq, etc.
    try:
        from openai import OpenAI
    except ImportError:
        print("Error: openai package not installed.", file=sys.stderr)
        print("Install it with: uv add grafeo-memory[openai]", file=sys.stderr)
        sys.exit(1)
    from grafeo_memory.providers.embedding import OpenAIEmbedder

    return OpenAIEmbedder(OpenAI())


def _create_manager() -> AsyncMemoryManager:
    """Build an AsyncMemoryManager from settings (env/.env)."""
    from grafeo_memory.common.settings import GrafeoMemorySettings

    try:
        from dotenv import load_dotenv

        load_dotenv()
    except ImportError:
        pass
    settings = GrafeoMemorySettings()

    model = settings.model
    db_path = settings.db
    user_id = settings.user

    embedder = _create_embedder(model)
    if settings.yolo:
        config = MemoryConfig.yolo(db_path=db_path, user_id=user_id)
    else:
        config = MemoryConfig(db_path=db_path, user_id=user_id)
    settings.apply_to(config)
    return AsyncMemoryManager(model, config, embedder=embedder)


_current_app: AppContext | None = None


class AppContextView:
    """Context-shaped view of the AppContext: static resources cannot take a
    Context parameter in mcp 2.x, so they read this from the lifespan."""

    def __init__(self, app: AppContext) -> None:
        from types import SimpleNamespace

        self.request_context = SimpleNamespace(lifespan_context=app)


def current_app() -> AppContextView:
    """The app context of the running server, shaped like a Context."""
    assert _current_app is not None, "the server lifespan is not running"
    return AppContextView(_current_app)


@asynccontextmanager
async def app_lifespan(server: MCPServer) -> AsyncGenerator[AppContext]:
    global _current_app
    manager = _create_manager()
    _current_app = AppContext(manager=manager)
    try:
        yield _current_app
    finally:
        manager.close()
        _current_app = None


mcp = MCPServer("grafeo-memory", lifespan=app_lifespan)


def main():
    from grafeo_memory.common.logging import configure_entry_point_logging

    configure_entry_point_logging()  # stderr only: stdout carries the stdio transport
    arg = sys.argv[1] if len(sys.argv) > 1 else "stdio"
    if arg == "sse":
        mcp.run(transport="sse")
    elif arg == "streamable-http":
        mcp.run(transport="streamable-http")
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    # `python -m` runs a second copy of this module; use the package one, which has the handlers registered.
    import importlib

    importlib.import_module("grafeo_memory.mcp.server").main()
