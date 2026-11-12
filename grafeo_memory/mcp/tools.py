"""MCP tools wrapping the grafeo-memory API."""

from __future__ import annotations

import json
from dataclasses import asdict
from enum import Enum
from typing import Any

from mcp.server.mcpserver import Context
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
        events = await manager.add_batch(texts, user_id=user_id, memory_type=memory_type, infer=infer)  # ty: ignore[invalid-argument-type]
        return json.dumps({"events": [_serialize(e) for e in events]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_search(
    query: str,
    user_id: str | None = None,
    k: int = 10,
    memory_type: str | None = None,
    min_score: float | None = None,
    point_in_time: int | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
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
        min_score: Minimum score threshold (0.0-1.0). Results below this are excluded.
        point_in_time: Epoch milliseconds. Only return facts valid at this moment (bi-temporal).

    Returns:
        JSON with results list (each with memory_id, text, score, metadata).

    Examples:
        memory_search("What does Alice do for work?")
        memory_search("python preferences", memory_type="procedural")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        results = await manager.search(
            query, user_id=user_id, k=k, memory_type=memory_type, min_score=min_score, point_in_time=point_in_time
        )
        return json.dumps({"results": [_serialize(r) for r in results]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_update(
    memory_id: str,
    text: str,
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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
    ctx: Context[AppContext, ServerSession] | None = None,
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


@mcp.tool()
async def memory_stats(
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """Show memory system statistics: counts, type breakdown, database info.

    Use this tool when: you want to check the health or size of the memory store.

    Returns:
        JSON with total_memories, semantic_count, procedural_count, episodic_count,
        entity_count, relation_count, and db_info.

    Examples:
        memory_stats()
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        s = manager.stats()
        return json.dumps(asdict(s), default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_explain_search(
    query: str,
    user_id: str | None = None,
    k: int = 10,
    memory_type: str | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """Explain a search query step-by-step, showing the full pipeline trace.

    Use this tool when: you want to understand why certain memories ranked
    higher or lower, or to debug search quality.
    Do NOT use this for: regular searching (use memory_search).

    Args:
        query: Natural language search query to explain.
        user_id: Search memories for this user. Uses server default if omitted.
        k: Maximum number of results (default 10).
        memory_type: Filter by type: "semantic", "procedural", "episodic", or null for all.

    Returns:
        JSON with query, steps (pipeline trace), and results.

    Examples:
        memory_explain_search("What does Alice do for work?")
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        result = await manager.explain(query, user_id=user_id, k=k, memory_type=memory_type)
        return json.dumps(
            {
                "query": result.query,
                "steps": [asdict(step) for step in result.steps],
                "results": [_serialize(r) for r in result.results],
            },
            default=str,
        )
    except Exception as exc:
        return json.dumps({"error": str(exc)})


# ---------------------------------------------------------------------------
# Layers A, B and G: context assembly, instructions, roots, open decisions
# ---------------------------------------------------------------------------


@mcp.tool()
async def memory_open_decisions(
    status: str = "open",
    kind: str | None = None,
    ask: bool = False,
    user_id: str | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """List questions the memory parked because it was not sure (the open-decisions bucket).

    Use this tool when: you can relay a question to the person, for example "is
    Alice still at Acme, or did she move to Beta?". With ask=True only questions
    worth an interruption are returned, and they are marked as asked.

    Args:
        status: "open" (default, includes asked), "resolved", "expired", or null for all.
        kind: Filter by kind: contradiction, duplicate, entity_identity, entity_merge, root_migration.
        ask: Only questions above the importance floor; marks them asked.
        user_id: Uses the server default if omitted.

    Returns:
        JSON with decisions (id, kind, question, options, evidence, proposed, confidence).
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        ods = manager.list_open_decisions(status, kind, ask=ask, user_id=user_id)
        return json.dumps({"decisions": [asdict(o) for o in ods]}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_resolve_decision(
    decision_id: str,
    answer: str,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """Answer a parked question with one of its options; the answer is stored as ground truth.

    Args:
        decision_id: The id from memory_open_decisions.
        answer: One of the decision's options (for a contradiction: replace,
            invalidate, keep_both or duplicate).

    Returns:
        JSON with resolved (true when the answer was applied).
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        return json.dumps({"resolved": bool(manager.resolve_open_decision(decision_id, answer))})
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_manage_roots(
    add: dict[str, str] | None = None,
    rename: dict[str, str] | None = None,
    retire: list[str] | None = None,
    hard: dict[str, bool] | None = None,
    user_id: str | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """View or edit the subject roots that scope memories (career, health, ...).

    With no arguments, returns the current tree. Retiring a root never deletes
    memories: they stay, and a question asks where they belong.

    Args:
        add: New roots, id -> label (e.g. {"pets": "Pets"}).
        rename: id -> new label (the old label keeps working).
        retire: Root ids to retire.
        hard: id -> true to wall a root off from searches that do not route to it.
        user_id: Uses the server default if omitted.

    Returns:
        JSON with the scope tree and the changes made.
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        changes = None
        if add or rename or retire or hard:
            changes = manager.set_root_scheme(add=add, rename=rename, retire=retire, hard=hard, user_id=user_id)
        return json.dumps({"changes": changes, "scopes": manager.scopes(user_id=user_id)}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_assemble(
    query: str,
    session_id: str | None = None,
    budget: int | None = None,
    user_id: str | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """Assemble a prompt-ready context block for a query: session window, instructions,
    facts and past events within a token budget, each line with its provenance.

    Args:
        query: What the context is for.
        session_id: The session whose working window goes first.
        budget: Token budget (defaults to the configured working-tier budget).
        user_id: Uses the server default if omitted.

    Returns:
        JSON with text (rendered block), tokens and sections.
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        block = await manager.assemble(query, session_id, budget, user_id=user_id)
        return json.dumps({"text": block.render(), "tokens": block.tokens, "block": block.to_dict()}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.tool()
async def memory_add_instruction(
    text: str,
    user_id: str | None = None,
    ctx: Context[AppContext, ServerSession] | None = None,
) -> str:
    """Store a standing instruction from the user ("always answer in Dutch"), certified.

    Args:
        text: The instruction as the user stated it.
        user_id: Uses the server default if omitted.

    Returns:
        JSON with the stored event.
    """
    assert ctx is not None
    manager = ctx.request_context.lifespan_context.manager
    try:
        event = await manager.add_instruction(text, user_id=user_id)
        return json.dumps({"event": _serialize(event)}, default=str)
    except Exception as exc:
        return json.dumps({"error": str(exc)})
