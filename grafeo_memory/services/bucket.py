"""Layer G: the open-decisions bucket, storage and resolution.

A judgment whose certainty stays below ``BucketConfig.floor`` is parked as an
``OpenDecision`` node instead of being forced. Its fact is kept, flagged
``provisional``, and the contested memory is left untouched. Three exits:

- the user answers (``resolve_open_decision``): ground truth, the surviving memory
  becomes certified;
- self-resolution: a later write that touches the evidence, or ``compile()``,
  re-judges the question one rung up (the LLM by default, or the decision model);
- expiry: past ``ttl_days`` (or beyond ``max_open``) the entry becomes an explicit
  ``unknown``, never a silent resolution.
"""

from __future__ import annotations

import json
import logging
import re
import time
from typing import TYPE_CHECKING

from grafeo_memory.common.types import (
    OPEN_DECISION_LABEL,
    SUPERSEDES_EDGE,
    OpenDecision,
)
from grafeo_memory.core.bucket import (
    DISTINCT,
    DUPLICATE,
    INVALIDATE,
    KEEP_BOTH,
    MERGE,
    NEW_ENTITY,
    REPLACE,
    rejudge_question,
)
from grafeo_memory.core.history import HistoryEntry
from grafeo_memory.core.search.vector import _get_props

if TYPE_CHECKING:
    from grafeo_memory.services.memory import _MemoryCore

logger = logging.getLogger(__name__)

_LIST_PROPS = ("options", "evidence")


def _now() -> int:
    return int(time.time() * 1000)


def _from_props(nid: int, p: dict) -> OpenDecision:
    payload = p.get("payload")
    try:
        payload = json.loads(payload) if isinstance(payload, str) else (payload or {})
    except ValueError:
        payload = {}
    return OpenDecision(
        id=str(nid),
        kind=str(p.get("kind", "")),
        question=str(p.get("question", "")),
        options=[str(x) for x in (p.get("options") or [])],
        evidence=[str(x) for x in (p.get("evidence") or [])],
        status=str(p.get("status", "open")),
        proposed=p.get("proposed"),
        confidence=p.get("confidence"),
        abstained_at_rung=p.get("abstained_at_rung"),
        created_at=p.get("created_at"),
        updated_at=p.get("updated_at"),
        resolution=p.get("resolution"),
        resolved_by=p.get("resolved_by"),
        resolved_at=p.get("resolved_at"),
        importance=float(p.get("importance", 0.5) or 0.5),
        payload=payload,
    )


def _all(m: _MemoryCore, user_id: str) -> list[OpenDecision]:
    gn = m._config.graph_name
    out: list[OpenDecision] = []
    try:
        nodes = m._db.get_nodes_by_label(OPEN_DECISION_LABEL)
    except Exception:
        return out
    for item in nodes:
        nid, props = (item[0], item[1]) if isinstance(item, tuple) else (getattr(item, "id", None), _get_props(item))
        if nid is None or props.get("user_id") != user_id:
            continue
        if gn and props.get("graph_name") != gn:
            continue
        out.append(_from_props(int(nid), props))
    out.sort(key=lambda d: (d.created_at or 0, int(d.id)))
    return out


def create(
    m: _MemoryCore,
    user_id: str,
    *,
    kind: str,
    question: str,
    options: list[str],
    evidence: list[str],
    proposed: str | None = None,
    confidence: float | None = None,
    rung: str | None = None,
    importance: float = 0.5,
    payload: dict | None = None,
) -> str:
    """Park one judgment. The same open question on the same evidence re-abstains
    into the existing entry (evidence and timestamp bumped) instead of duplicating."""
    now = _now()
    for od in _all(m, user_id):
        if od.status in ("open", "asked") and od.kind == kind and od.question == question:
            merged = sorted(set(od.evidence) | set(evidence))
            m._db.set_node_property(int(od.id), "evidence", merged)
            m._db.set_node_property(int(od.id), "updated_at", now)
            node = m._db.get_node(int(od.id))
            n = int(_get_props(node).get("reabstained", 0) or 0) if node is not None else 0
            m._db.set_node_property(int(od.id), "reabstained", n + 1)
            m._layer_counts["reabstained"] = m._layer_counts.get("reabstained", 0) + 1
            return od.id
    props: dict = {
        "kind": kind,
        "question": question,
        "options": list(options),
        "evidence": [str(e) for e in evidence],
        "status": "open",
        "created_at": now,
        "updated_at": now,
        "importance": float(importance),
        "user_id": user_id,
        "payload": json.dumps(payload or {}),
    }
    if proposed is not None:
        props["proposed"] = proposed
    if confidence is not None:
        props["confidence"] = float(confidence)
    if rung is not None:
        props["abstained_at_rung"] = rung
    if m._config.graph_name:
        props["graph_name"] = m._config.graph_name
    from grafeo_memory.services.schema import mark

    mark(m._db)
    node = m._db.create_node([OPEN_DECISION_LABEL], props)
    nid = node.id if hasattr(node, "id") else node
    m._layer_counts["parked"] = m._layer_counts.get("parked", 0) + 1
    return str(nid)


def get(m: _MemoryCore, decision_id: str) -> OpenDecision | None:
    try:
        node = m._db.get_node(int(decision_id))
    except (TypeError, ValueError):
        return None
    if node is None:
        return None
    props = _get_props(node)
    if props.get("kind") is None or "question" not in props:
        return None
    if m._config.graph_name and props.get("graph_name") != m._config.graph_name:
        return None
    return _from_props(int(decision_id), props)


def _close(m: _MemoryCore, od: OpenDecision, resolution: str, by: str, status: str = "resolved") -> None:
    nid = int(od.id)
    now = _now()
    m._db.set_node_property(nid, "status", status)
    m._db.set_node_property(nid, "resolution", resolution)
    m._db.set_node_property(nid, "resolved_by", by)
    m._db.set_node_property(nid, "resolved_at", now)
    m._db.set_node_property(nid, "updated_at", now)
    key = f"resolved_{by}"
    m._layer_counts[key] = m._layer_counts.get(key, 0) + 1


def expire(m: _MemoryCore, user_id: str) -> int:
    """TTL and max_open: stale or overflowing entries become an explicit unknown."""
    cfg = m._config.bucket
    now = _now()
    open_ = [d for d in _all(m, user_id) if d.status in ("open", "asked")]
    n = 0
    ttl_ms = cfg.ttl_days * 86_400_000 if cfg.ttl_days > 0 else None
    keep: list[OpenDecision] = []
    for od in open_:
        if ttl_ms is not None and od.created_at is not None and now - od.created_at > ttl_ms:
            _close(m, od, "unknown", "expiry", status="expired")
            n += 1
        else:
            keep.append(od)
    overflow = len(keep) - cfg.max_open
    for od in keep[: max(0, overflow)]:  # oldest first
        _close(m, od, "unknown", "expiry", status="expired")
        n += 1
    return n


def list_decisions(
    m: _MemoryCore, user_id: str, status: str | None = "open", kind: str | None = None, ask: bool = False
) -> list[OpenDecision]:
    expire(m, user_id)
    out = _all(m, user_id)
    if status is not None:
        wanted = {"open", "asked"} if status == "open" else {status}
        out = [d for d in out if d.status in wanted]
    if kind is not None:
        out = [d for d in out if d.kind == kind]
    if ask:
        out = [d for d in out if d.importance >= m._config.bucket.ask_importance_floor]
        for d in out:
            if d.status == "open":
                m._db.set_node_property(int(d.id), "status", "asked")
                d.status = "asked"
    return out


# --- parking from the write path ---------------------------------------------


def park_memory(
    m: _MemoryCore,
    user_id: str,
    *,
    new_memory_id: str,
    new_text: str,
    proposed: str,
    target_id: str | None,
    target_text: str | None,
    candidate_ids: list[str],
    confidence: float | None,
    rung: str,
    importance: float,
) -> str:
    """Record a contradiction (or a possible duplicate) and flag the new memory provisional."""
    from grafeo_memory.core.bucket import MEMORY_OPTIONS, contradiction_question

    kind = "contradiction" if target_id else "duplicate"
    evidence = [new_memory_id, *([target_id] if target_id else candidate_ids[:3])]
    od_id = create(
        m,
        user_id,
        kind=kind,
        question=contradiction_question(new_text, target_text),
        options=MEMORY_OPTIONS,
        evidence=evidence,
        proposed=proposed,
        confidence=confidence,
        rung=rung,
        importance=importance,
        payload={"new": new_memory_id, "target": target_id, "candidates": candidate_ids[:5]},
    )
    m._db.set_node_property(int(new_memory_id), "provisional", True)
    m._db.set_node_property(int(new_memory_id), "open_decision", od_id)
    return od_id


# --- resolution ----------------------------------------------------------------


def _clear_provisional(m: _MemoryCore, memory_id: str | None) -> None:
    if memory_id is None:
        return
    try:
        m._db.set_node_property(int(memory_id), "provisional", False)
    except Exception:
        logger.debug("bucket: could not clear provisional on %s", memory_id, exc_info=True)


def _certify(m: _MemoryCore, memory_id: str | None, by: str) -> None:
    if memory_id is None:
        return
    owned = m._owned_memory(memory_id)
    if owned is None:
        return
    m._db.set_node_property(owned[0], "status", "certified")
    m._db.set_node_property(owned[0], "certified_by", by)


def apply(m: _MemoryCore, od: OpenDecision, answer: str, by: str) -> bool:
    """Write one resolution back. Returns False for an answer outside the options."""
    if answer not in od.options:
        return False
    now = _now()
    p = od.payload or {}
    new_id, target = p.get("new"), p.get("target")
    if od.kind in ("contradiction", "duplicate"):
        target = p.get("resolved_target") or target
        if answer in (REPLACE, INVALIDATE) and target:
            old_text = m._expire_memory(str(target), now, invalid_at=now)
            if answer == REPLACE and new_id:
                try:
                    m._db.create_edge(int(new_id), int(target), SUPERSEDES_EDGE)
                except Exception:
                    logger.debug("bucket: SUPERSEDES %s->%s failed", new_id, target, exc_info=True)
            m._record_history(
                str(target),
                HistoryEntry(event="UPDATE" if answer == REPLACE else "DELETE", old_text=old_text, timestamp=now),
                m._config.user_id,
            )
            _clear_provisional(m, new_id)
        elif answer == KEEP_BOTH:
            _clear_provisional(m, new_id)
        elif answer == DUPLICATE and new_id:
            m._expire_memory(str(new_id), now)
        if by == "user" and answer != DUPLICATE:
            _certify(m, new_id, "user")
    elif od.kind in ("entity_identity", "entity_merge"):
        from grafeo_memory.services.compile import _merge_into, _record_merged_name

        if od.kind == "entity_merge" and answer == MERGE:
            alias, canon = int(p["alias_id"]), int(p["canon_id"])
            _merge_into(m._db, alias, canon)
            _record_merged_name(m._db, canon, str(p.get("alias", "")))
        elif od.kind == "entity_identity" and answer != NEW_ENTITY:
            uid = m._config.user_id
            canon = m._find_entity_by_name(answer, uid)
            alias = m._find_entity_by_name(str(p.get("mention", "")), uid)
            if canon is not None and alias is not None and int(canon) != int(alias):
                _merge_into(m._db, int(alias), int(canon))
                _record_merged_name(m._db, int(canon), str(p.get("mention", "")))
    elif od.kind == "root_migration":
        from grafeo_memory.services import scoping

        scoping.retag(m, str(p.get("retired")), answer)
    elif answer == DISTINCT:
        pass
    _close(m, od, answer, by)
    m._graph_dirty = True
    return True


async def _ask(m: _MemoryCore, od: OpenDecision, context: list[str], on_usage=None) -> str | None:
    """Re-judge one parked question on the configured rung; None = still unsure."""
    cfg = m._config.bucket
    state = od.question + ("\n\nContext:\n" + "\n".join(f"- {c}" for c in context) if context else "")
    if cfg.resolve_engine == "decision" and m._decision_model is not None:
        try:
            ds = await m._decision_model.decide_async(state, [rejudge_question("q", od.options, od.question)])
        except Exception:
            logger.warning("bucket: decision-model re-judge failed", exc_info=True)
            return None
        if not ds or ds[0].confidence < cfg.floor or str(ds[0].value) not in od.options:
            return None
        return str(ds[0].value)
    from pydantic_ai import Agent

    prompt = (
        f"{state}\n\nAnswer with exactly one of these options and nothing else: {', '.join(od.options)}. "
        "If the context does not settle it, answer UNSURE."
    )
    try:
        result = await Agent(m._model, output_type=str).run(prompt)
    except Exception:
        logger.warning("bucket: LLM re-judge failed", exc_info=True)
        return None
    if on_usage is not None:
        on_usage("bucket_rejudge", result.usage)
    text = str(result.output).strip().lower()
    for opt in sorted(od.options, key=len, reverse=True):
        if re.search(rf"\b{re.escape(opt.lower())}\b", text):
            return opt
    return None


def _context_for(m: _MemoryCore, od: OpenDecision) -> list[str]:
    """Current text of the evidence plus the closest memories to it (the new context)."""
    lines: list[str] = []
    for mid in od.evidence:
        owned = m._owned_memory(mid)
        if owned is not None and owned[1].get("expired_at") is None:
            lines.append(f"[{mid}] {owned[1].get('text', '')}")
    p = od.payload or {}
    new_id = p.get("new")
    owned = m._owned_memory(new_id) if new_id else None
    if owned is not None:
        from grafeo_memory.core.search import search_similar

        emb = owned[1].get(m._config.vector_property)
        if emb:
            try:
                near = search_similar(
                    m._db,
                    [list(emb)],
                    user_id=str(owned[1].get("user_id", m._config.user_id)),
                    threshold=m._config.reconciliation_threshold,
                    vector_property=m._config.vector_property,
                )
            except Exception:
                near = []
            seen = set(od.evidence)
            for n in near[:5]:
                if str(n["id"]) not in seen:
                    lines.append(f"[{n['id']}] {n['text']}")
    return lines


async def self_resolve(
    m: _MemoryCore,
    user_id: str,
    touched: set[str] | None = None,
    *,
    on_usage=None,
    skip: set[str] | None = None,
) -> int:
    """Re-judge open entries whose evidence was touched (all of them when ``touched`` is
    None, the compile sweep). ``skip`` holds entries parked by the current write.
    Returns how many resolved."""
    cfg = m._config.bucket
    if not cfg.self_resolution:
        return 0
    pending = [d for d in _all(m, user_id) if d.status in ("open", "asked") and d.id not in (skip or set())]
    if touched is not None:
        pending = [d for d in pending if set(d.evidence) & touched][: max(0, cfg.rejudge_per_add)]
    resolved = 0
    for od in pending:
        if od.kind in ("contradiction", "duplicate"):
            p = od.payload or {}
            new = m._owned_memory(p["new"]) if p.get("new") else None
            if new is None or new[1].get("expired_at") is not None:
                _close(m, od, "unknown", "self", status="expired")  # the evidence is gone
                continue
        answer = await _ask(m, od, _context_for(m, od), on_usage)
        if answer is not None and apply(m, od, answer, "self"):
            resolved += 1
    return resolved


def stats(m: _MemoryCore, user_id: str) -> dict:
    by_status: dict[str, int] = {}
    by_kind: dict[str, int] = {}
    by_resolver: dict[str, int] = {}
    for od in _all(m, user_id):
        by_status[od.status] = by_status.get(od.status, 0) + 1
        by_kind[od.kind] = by_kind.get(od.kind, 0) + 1
        if od.resolved_by:
            by_resolver[od.resolved_by] = by_resolver.get(od.resolved_by, 0) + 1
    closed = sum(by_resolver.values())
    return {
        "by_status": by_status,
        "by_kind": by_kind,
        "by_resolver": by_resolver,
        "self_resolved_share": round(by_resolver.get("self", 0) / closed, 4) if closed else None,
    }
