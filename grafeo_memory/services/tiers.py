"""Layer A: the working tier, context assembly and procedural instructions.

- Working window: with ``tiers.working.enabled`` every ``add()`` also keeps its
  messages in a per-session window (``window_messages``); turns that fall out of
  the window are folded into a rolling summary (one LLM call per overflow).
  ``defer_extraction`` turns ``add()`` into buffering only, and ``end_session()``
  runs extraction once over the whole window.
- ``assemble(query, session_id, budget)``: a ContextBlock with the session window
  first, then instructions (procedural), facts (semantic) and events (episodic),
  each section capped by its budget share and ranked certified first. Every item
  carries its provenance.
- ``end_session()``: promotes the window (extraction when deferred) and stores the
  session summary as an episodic memory.
- ``add_instruction()``: a user-stated procedural memory, certified by definition.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

from grafeo_memory.common.types import (
    AddResult,
    ContextBlock,
    ContextItem,
    MemoryAction,
    MemoryEvent,
    MemoryType,
)
from grafeo_memory.core.tiers import detect_instructions, estimate_tokens, fill, rank_key, section_budgets

if TYPE_CHECKING:
    from grafeo_memory.services.memory import _MemoryCore

logger = logging.getLogger(__name__)


def _key(uid: str, sid: str | None) -> str:
    return f"{uid}\x00{sid or ''}"


async def buffer(m: _MemoryCore, uid: str, sid: str | None, parsed: Sequence[Mapping], on_usage=None) -> None:
    cfg = m._config.tiers.working
    key = _key(uid, sid)
    window = m._windows.setdefault(key, [])
    window.extend({"role": p.get("role", "user"), "content": str(p.get("content", ""))} for p in parsed)
    if cfg.defer_extraction:
        return  # the whole session is extracted at end_session(); no overflow folding
    overflow = window[: max(0, len(window) - cfg.window_messages)]
    if overflow:
        del window[: len(overflow)]
        if cfg.summarize_on_overflow:
            await _fold(m, key, overflow, on_usage)


async def _fold(m: _MemoryCore, key: str, turns: list[dict], on_usage=None) -> None:
    from pydantic_ai import Agent

    prev = m._summaries.get(key, "")
    text = "\n".join(f"{t['role']}: {t['content']}" for t in turns)
    prompt = (
        "Update the running summary of a conversation with the turns below. Keep names, dates, "
        "decisions and open items; at most 150 words.\n\n"
        f"Running summary:\n{prev or '(none)'}\n\nNew turns:\n{text}"
    )
    try:
        result = await Agent(m._model, output_type=str).run(prompt)
    except Exception:
        logger.warning("working tier: summary update failed, keeping the previous summary", exc_info=True)
        return
    if on_usage is not None:
        on_usage("working_summary", result.usage)
    m._summaries[key] = str(result.output).strip()


async def auto_instructions(m: _MemoryCore, uid: str, parsed: Sequence[Mapping]) -> list[MemoryEvent]:
    """Procedural write triggers: user turns that state standing instructions."""
    events: list[MemoryEvent] = []
    for text in detect_instructions(parsed):
        events.append(await add_instruction(m, text, uid, source="trigger"))
    return events


async def add_instruction(m: _MemoryCore, text: str, uid: str, *, source: str = "user") -> MemoryEvent:
    """Store a procedural memory. Explicit instructions are user-stated, so certified;
    triggered ones stay inferred (entrenchment: user-stated outranks inferred)."""
    res = await m._add(text, user_id=uid, infer=False, memory_type=MemoryType.PROCEDURAL)
    event = res[0] if res else MemoryEvent(action=MemoryAction.NONE, text=text)
    if event.memory_id:
        nid = int(event.memory_id)
        m._db.set_node_property(nid, "source", f"instruction:{source}")
        if source == "user":
            m._db.set_node_property(nid, "status", "certified")
            m._db.set_node_property(nid, "certified_by", "user")
    return event


async def end_session(m: _MemoryCore, uid: str, sid: str | None, on_usage=None) -> AddResult:
    """Promote the session: extraction over the deferred window, then the summary as an episodic memory."""
    cfg = m._config.tiers.working
    key = _key(uid, sid)
    window = m._windows.pop(key, [])
    summary = m._summaries.pop(key, None)
    from pydantic_ai.usage import RunUsage

    total = RunUsage()
    events: list[MemoryEvent] = []
    if not cfg.promote_on_session_end:
        return AddResult(events, usage=total)
    if cfg.defer_extraction and window:
        res = await m._add(window, user_id=uid, session_id=sid, infer=True, _from_window=True)
        events.extend(res)
        total.incr(res.usage)
        if cfg.summarize_on_overflow and len(window) > cfg.window_messages:
            await _fold(m, key, window, on_usage)
            summary = m._summaries.pop(key, summary)
    if summary:
        res = await m._add(
            f"Session summary: {summary}", user_id=uid, session_id=sid, infer=False, memory_type=MemoryType.EPISODIC
        )
        events.extend(res)
    return AddResult(events, usage=total)


def _item(r, section: str, props: dict) -> ContextItem:
    status = props.get("status") or ("provisional" if props.get("provisional") else "inferred")
    if props.get("provisional"):
        status = "provisional"
    return ContextItem(
        text=r.text,
        section=section,
        memory_id=r.memory_id,
        score=round(r.score, 4),
        confidence=props.get("confidence"),
        engine=props.get("engine"),
        status=status,
        session_id=r.session_id,
        created_at=r.created_at,
    )


async def assemble(
    m: _MemoryCore, query: str, uid: str, sid: str | None, budget: int | None, k: int = 30
) -> ContextBlock:
    from grafeo_memory.services.retrieval import fetch_props

    cfg = m._config.tiers
    total = budget or cfg.working.token_budget
    block = ContextBlock(query=query, budget=total)
    key = _key(uid, sid)
    window = list(m._windows.get(key, []))
    block.summary = m._summaries.get(key)

    results = list(await m._search(query, user_id=uid, k=k))
    procedural = [r for r in results if r.memory_type == "procedural"]
    if cfg.procedural.enabled:
        seen = {r.memory_id for r in procedural}
        extra = await m._search(query, user_id=uid, k=k, memory_type=MemoryType.PROCEDURAL)
        procedural += [r for r in extra if r.memory_id not in seen]
    episodic = [r for r in results if r.memory_type == "episodic"]
    semantic = [r for r in results if r.memory_type not in ("procedural", "episodic")]
    props = fetch_props(m, [r.memory_id for r in (*procedural, *episodic, *semantic)])

    candidates: dict[str, list[ContextItem]] = {
        "working": [ContextItem(text=f"{t['role']}: {t['content']}", section="working") for t in reversed(window)],
        "procedural": sorted((_item(r, "procedural", props.get(r.memory_id, {})) for r in procedural), key=rank_key),
        "semantic": sorted((_item(r, "semantic", props.get(r.memory_id, {})) for r in semantic), key=rank_key),
        "episodic": sorted((_item(r, "episodic", props.get(r.memory_id, {})) for r in episodic), key=rank_key),
    }
    present = [s for s, items in candidates.items() if items]
    summary_cost = estimate_tokens(block.summary) if block.summary else 0
    budgets = section_budgets(max(0, total - summary_cost), cfg.section_shares, present)
    used = summary_cost
    for name in present:
        taken, cost = fill(candidates[name], budgets[name])
        if name == "working":
            taken.reverse()  # oldest first reads naturally
        block.sections[name] = taken
        used += cost
    block.tokens = used
    if m._config.bucket_on:
        from grafeo_memory.services import bucket

        shown = {i.memory_id for i in block.items() if i.memory_id}
        block.open_questions = [od for od in bucket.list_decisions(m, uid) if shown & set(od.evidence)][:5]
    return block
