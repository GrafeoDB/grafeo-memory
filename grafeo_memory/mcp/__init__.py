"""grafeo-memory MCP server, expose the memory API as MCP tools."""

from grafeo_memory.mcp.server import mcp

# Importing these modules registers their tools, resources and prompts on `mcp`;
# the console entry point (grafeo_memory.mcp.server:main) loads this package first.
from grafeo_memory.mcp import prompts, resources, tools  # isort: skip

__all__ = ["mcp", "prompts", "resources", "tools"]
