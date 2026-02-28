"""MCP resources for grafeo-memory server."""

from __future__ import annotations

import json

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from grafeo_memory.mcp.server import AppContext, mcp


@mcp.resource("memory://config")
async def memory_config(ctx: Context[ServerSession, AppContext]) -> str:
    """Current grafeo-memory configuration: model, database path, default user, enabled features."""
    manager = ctx.request_context.lifespan_context.manager
    config = manager._config
    return json.dumps(
        {
            "model": str(manager._model),
            "db_path": config.db_path,
            "default_user_id": config.user_id,
            "session_id": config.session_id,
            "enable_importance": config.enable_importance,
            "enable_vision": config.enable_vision,
            "enable_topology_boost": config.enable_topology_boost,
            "instrument": bool(config.instrument),
        },
        default=str,
    )


@mcp.resource("memory://stats")
async def memory_stats(ctx: Context[ServerSession, AppContext]) -> str:
    """Memory system statistics: node/edge counts, database info."""
    manager = ctx.request_context.lifespan_context.manager
    db = manager._db
    try:
        info = db.info()
    except Exception:
        info = {}
    try:
        stats = db.detailed_stats()
    except Exception:
        stats = {}
    return json.dumps({"db_info": info, "db_stats": stats}, default=str)
