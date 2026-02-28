"""MCP prompt templates for grafeo-memory server."""

from __future__ import annotations

from grafeo_memory.mcp.server import mcp


@mcp.prompt()
def manage_memories(topic: str = "") -> str:
    """Guide an AI agent through managing memories for a user conversation."""
    base = """\
You are managing a user's long-term memory system. Follow these steps:

1. SEARCH FIRST: Before adding anything, search for existing memories on the topic
   to avoid duplicates. Use memory_search with a relevant query.

2. ADD NEW MEMORIES: If the information is genuinely new, use memory_add to store it.
   Choose the right memory_type:
   - "semantic" for facts, knowledge, biographical details
   - "procedural" for instructions, preferences, behavioral rules
   - "episodic" for interaction events, questions asked and answers found

3. UPDATE EXISTING: If a memory exists but needs correction, use memory_update
   with the memory_id and new text.

4. REVIEW: Use memory_list to verify the current state of stored memories.

5. CONSOLIDATE: If there are many similar memories, use memory_summarize to
   group them into concise topic-based summaries."""

    if topic:
        return f"{base}\n\nFocus on the topic: {topic}"
    return base


@mcp.prompt()
def knowledge_capture(text: str = "") -> str:
    """Guide an AI agent through extracting and storing knowledge from a document or text."""
    base = """\
You are capturing knowledge from text into the memory system. Follow these steps:

1. READ the text carefully and identify the key facts, entities, and relationships.

2. SEARCH existing memories to see what is already stored. Use memory_search
   with key terms from the text.

3. ADD memories using memory_add. The system will automatically extract facts
   and reconcile with existing memories. For long texts, use memory_add_batch
   with multiple chunks.

4. VERIFY by searching for the newly added information to confirm it was stored
   correctly.

5. ORGANIZE: If the memory store is getting large, use memory_summarize to
   consolidate older entries."""

    if text:
        return f"{base}\n\nText to capture:\n{text}"
    return base
