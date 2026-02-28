"""MCP tools wrapping the grafeo-memory API."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from mcp.server.fastmcp import Context
from mcp.server.session import ServerSession

from grafeo_memory.mcp.server import AppContext, mcp


def _serialize(obj: Any) -> dict:
    """Convert a dataclass to a JSON-friendly dict, resolving Enums to their values."""
    d = asdict(obj)
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in d.items()}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def memory_add(
    text: str,
    user_id: str | None = None,
    memory_type: str = "semantic",
    infer: bool = True,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Add a memory by extracting facts from text via LLM.

    Use this tool when: you learn something new about the user or topic that
    should be remembered for future conversations.
    Do NOT use this for: searching existing memories (use memory_search),
    updating a specific memory (use memory_update), or storing instructions
    (use memory_type="procedural").

    Args:
        text: The text to extract memories from (e.g. "I work at Acme Corp").
        user_id: User to store the memory for. Uses server default if omitted.
        memory_type: "semantic" (facts), "procedural" (instructions), or "episodic" (interactions).
        infer: True (default) to use LLM extraction, False to store text as-is.

    Returns:
        JSON with events list (each with action, memory_id, text).

    Examples:
        memory_add("Alice works at Acme Corp as a data scientist")
        memory_add("Always use type hints in Python", memory_type="procedural")
        memory_add("Raw note to store", infer=False)
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        events = await manager.add(text, user_id=user_id, memory_type=memory_type, infer=infer)
        return json.dumps({"events": [_serialize(e) for e in events]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_add_batch(
    texts: list[str],
    user_id: str | None = None,
    memory_type: str = "semantic",
    infer: bool = True,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Add multiple memories in a single batch operation.

    Use this tool when: you have several pieces of information to store at once.
    Do NOT use this for: adding a single memory (use memory_add).

    Args:
        texts: List of text strings to extract memories from.
        user_id: User to store the memories for. Uses server default if omitted.
        memory_type: "semantic" (facts), "procedural" (instructions), or "episodic" (interactions).
        infer: True (default) to use LLM extraction, False to store as-is.

    Returns:
        JSON with events list for all memories added.

    Examples:
        memory_add_batch(["Alice likes pizza", "Bob works at Google"])
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        events = await manager.add_batch(texts, user_id=user_id, memory_type=memory_type, infer=infer)
        return json.dumps({"events": [_serialize(e) for e in events]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_search(
    query: str,
    user_id: str | None = None,
    k: int = 10,
    memory_type: str | None = None,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Search memories using hybrid vector similarity and graph context.

    Use this tool when: you need to recall stored information about a user
    or topic, or before adding new memories to check what already exists.
    Do NOT use this for: listing all memories (use memory_list) or adding
    memories (use memory_add).

    Args:
        query: Natural language search query.
        user_id: Search memories for this user. Uses server default if omitted.
        k: Maximum number of results to return (default 10).
        memory_type: Filter by type: "semantic", "procedural", "episodic", or null for all.

    Returns:
        JSON with results list (each with memory_id, text, score, metadata).

    Examples:
        memory_search("What does Alice do for work?")
        memory_search("python preferences", memory_type="procedural")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        results = await manager.search(query, user_id=user_id, k=k, memory_type=memory_type)
        return json.dumps({"results": [_serialize(r) for r in results]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_update(
    memory_id: str,
    text: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Update an existing memory's text. Re-embeds and records history.

    Use this tool when: you need to correct or modify a specific memory.
    Do NOT use this for: adding new memories (use memory_add).

    Args:
        memory_id: The ID of the memory to update.
        text: The new text for the memory.

    Returns:
        JSON with the update event (action, memory_id, text, old_text).

    Examples:
        memory_update("42", "Alice now works at Google as a senior engineer")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        event = await manager.update(memory_id, text)
        return json.dumps({"event": _serialize(event)}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_delete(
    memory_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Delete a single memory by ID.

    Use this tool when: a memory is incorrect, outdated, or should be removed.
    Do NOT use this for: deleting all memories (use memory_delete_all).

    Args:
        memory_id: The ID of the memory to delete.

    Returns:
        JSON with success status.

    Examples:
        memory_delete("42")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        ok = await manager.delete(memory_id)
        return json.dumps({"deleted": ok, "memory_id": memory_id})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_delete_all(
    user_id: str | None = None,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Delete all memories for a user. Use with caution.

    Use this tool when: you need to clear all stored memories for a fresh start.
    Do NOT use this for: deleting a single memory (use memory_delete).

    Args:
        user_id: The user whose memories to delete. Uses server default if omitted.

    Returns:
        JSON with the count of deleted memories.

    Examples:
        memory_delete_all()
        memory_delete_all(user_id="alice")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        count = await manager.delete_all(user_id=user_id)
        return json.dumps({"deleted_count": count})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_list(
    user_id: str | None = None,
    memory_type: str | None = None,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """List all stored memories for a user.

    Use this tool when: you want to see everything stored, or to audit
    what the system remembers.
    Do NOT use this for: searching by relevance (use memory_search).

    Args:
        user_id: The user whose memories to list. Uses server default if omitted.
        memory_type: Filter by type: "semantic", "procedural", "episodic", or null for all.

    Returns:
        JSON with memories list (each with memory_id, text, score, metadata).

    Examples:
        memory_list()
        memory_list(memory_type="procedural")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        memories = await manager.get_all(user_id=user_id, memory_type=memory_type)
        return json.dumps({"memories": [_serialize(m) for m in memories]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_summarize(
    user_id: str | None = None,
    preserve_recent: int = 5,
    batch_size: int = 20,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Consolidate old memories into fewer, topic-grouped entries via LLM.

    Use this tool when: the memory store has grown large and you want to
    compress older memories while keeping recent ones intact.
    Do NOT use this for: adding or searching memories.

    Args:
        user_id: The user whose memories to consolidate. Uses server default if omitted.
        preserve_recent: Number of most recent memories to keep untouched (default 5).
        batch_size: Memories per LLM consolidation batch (default 20).

    Returns:
        JSON with events list (ADD for new summaries, DELETE for removed originals).

    Examples:
        memory_summarize()
        memory_summarize(preserve_recent=10)
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        events = await manager.summarize(user_id=user_id, preserve_recent=preserve_recent, batch_size=batch_size)
        return json.dumps({"events": [_serialize(e) for e in events]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_history(
    memory_id: str,
    ctx: Context[ServerSession, AppContext] | None = None,
) -> str:
    """Show the change history for a specific memory.

    Use this tool when: you want to see how a memory was created or modified
    over time.
    Do NOT use this for: searching or listing memories.

    Args:
        memory_id: The ID of the memory to get history for.

    Returns:
        JSON with history entries (each with event, old_text, new_text, timestamp).

    Examples:
        memory_history("42")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        entries = await manager.history(memory_id)
        return json.dumps({"history": [_serialize(e) for e in entries]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
