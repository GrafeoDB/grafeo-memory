"""Layer F wired into the write path: pre rules, post rules, lifecycle, shapes, retention.

Every judgment records the rung that resolved it (``rule:<id>``, ``decision``,
``llm``, ``parked``); ``stats().layers["rungs"]`` reports the counts and the
deterministic share (rules, the temporal pre-check and the no-candidate fast path).
"""

from __future__ import annotations

import logging
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from grafeo_memory.common.types import MEMORY_LABEL, MemoryAction, ReconciliationDecision
from grafeo_memory.core.policy import (
    FactContext,
    ShapeVerdict,
    build_post_rules,
    build_pre_rules,
    check_memory_shape,
    check_relation_shape,
    evaluate,
    find_secret,
    observed_candidate,
)
from grafeo_memory.core.search.vector import _get_props

if TYPE_CHECKING:
    from grafeo_memory.common.types import Fact, MemoryEvent
    from grafeo_memory.services.memory import _MemoryCore

logger = logging.getLogger(__name__)

DETERMINISTIC_RUNGS = ("rule", "temporal", "fast")


def pre_filter(m: _MemoryCore, facts: list[Fact]) -> tuple[list[int], list[tuple[str, str]]]:
    """Rung 1 before any engine: indices of the facts that go on, and (text, rule id) of the rejected."""
    cfg = m._config.policy
    rules = build_pre_rules(cfg.packs, cfg.extra_secret_patterns)
    keep: list[int] = []
    rejected: list[tuple[str, str]] = []
    for i, f in enumerate(facts):
        hit = evaluate(rules, FactContext(text=f.text), "pre")
        if hit is not None and hit.action == "reject":
            rejected.append((f.text, hit.rule_id))
            logger.info("policy %s: rejected a fact before storage", hit.rule_id)
            count(m, f"rule:{hit.rule_id}")
        else:
            keep.append(i)
    return keep, rejected


def scrub(m: _MemoryCore, text: str) -> str | None:
    """Compliance on a raw (infer=False) write: None when the text must not be stored."""
    if "compliance" not in m._config.policy.packs:
        return text
    if find_secret(text, m._config.policy.extra_secret_patterns) is not None:
        count(m, "rule:compliance.secret")
        return None
    return text


def _props(m: _MemoryCore, memory_id: str | None) -> dict | None:
    if not memory_id:
        return None
    owned = m._owned_memory(memory_id)
    return owned[1] if owned is not None else None


def post_apply(
    m: _MemoryCore, decisions: list[ReconciliationDecision], role: str | None
) -> list[ReconciliationDecision]:
    """Rules over the engine's decisions (entrenchment). A hit stands the overwrite down
    (UPDATE becomes ADD, DELETE becomes NONE) or parks it when the bucket is on."""
    cfg = m._config.policy
    rules = build_post_rules(cfg.packs, cfg.conservative)
    if not rules:
        return decisions
    out: list[ReconciliationDecision] = []
    for d in decisions:
        ctx = FactContext(
            text=d.text,
            role=role,
            action=d.action.value,
            engine=d.engine,
            confidence=d.confidence,
            target=_props(m, d.target_memory_id),
        )
        hit = evaluate(rules, ctx, "post")
        if hit is None:
            out.append(d)
            continue
        rung = f"rule:{hit.rule_id}"
        if hit.action == "park" and m._config.bucket_on:
            out.append(replace(d, parked=True, rung=rung, alternatives=[d.action.value]))
        elif d.action == MemoryAction.UPDATE:
            out.append(replace(d, action=MemoryAction.ADD, target_memory_id=None, rung=rung))
        else:  # DELETE stands down
            out.append(replace(d, action=MemoryAction.NONE, target_memory_id=None, rung=rung))
    return out


def observe(m: _MemoryCore, decisions: list[ReconciliationDecision], candidates: list[dict]) -> int:
    """Lifecycle: a NONE fact re-states a memory; ``compilation.promote_after`` such
    observations certify that memory. Returns how many memories were promoted."""
    promoted = 0
    n_needed = m._config.compilation.promote_after
    for d in decisions:
        if d.action != MemoryAction.NONE or d.parked:
            continue
        mid = observed_candidate(d.text, candidates)
        owned = m._owned_memory(mid) if mid else None
        if owned is None:
            continue
        nid, props = owned
        n = int(props.get("observations", 0) or 0) + 1
        m._db.set_node_property(nid, "observations", n)
        if n >= n_needed and props.get("status") != "certified":
            m._db.set_node_property(nid, "status", "certified")
            m._db.set_node_property(nid, "certified_by", "lifecycle")
            promoted += 1
            count(m, "rule:lifecycle.promote")
    return promoted


def relation_verdict(m: _MemoryCore, rel: str, source_type: str, target_type: str) -> ShapeVerdict:
    sev = m._config.policy.shape_severity.get("relation_domain_range", "violation")
    return check_relation_shape(rel, source_type, target_type, sev)


def memory_shapes(m: _MemoryCore, memory_ids: list[str]) -> None:
    """memory_provenance shape on new memories: a "warning" severity flags the memory
    (lowered confidence), "info" only counts; "violation" is treated as a warning here
    because the fact was already accepted by reconciliation."""
    sev = m._config.policy.shape_severity.get("memory_provenance")
    if not sev:
        return
    for mid in memory_ids:
        owned = m._owned_memory(mid)
        if owned is None:
            continue
        verdict = check_memory_shape(owned[1], sev)
        if verdict.ok:
            continue
        count(m, f"shape:memory_provenance:{sev}")
        if sev in ("warning", "violation"):
            m._db.set_node_property(owned[0], "flagged", verdict.shape)


def retention(m: _MemoryCore, uid: str) -> int:
    """Lifecycle TTLs per memory type (``policy.retention_days``): expire, never delete."""
    ttl = m._config.policy.retention_days
    if not ttl:
        return 0
    now = int(time.time() * 1000)
    n = 0
    try:
        nodes = m._db.get_nodes_by_label(MEMORY_LABEL)
    except Exception:
        return 0
    for item in nodes:
        nid, props = (item[0], item[1]) if isinstance(item, tuple) else (getattr(item, "id", None), _get_props(item))
        if nid is None or props.get("user_id") != uid or props.get("expired_at") is not None:
            continue
        if props.get("status") == "certified":
            continue  # certified content outlives retention
        days = ttl.get(str(props.get("memory_type", "semantic")))
        if days is not None and now - int(props.get("created_at") or now) > days * 86_400_000:
            m._db.set_node_property(int(nid), "expired_at", now)
            n += 1
    if n:
        count(m, "rule:lifecycle.retention", n)
    return n


def count(m: _MemoryCore, key: str, n: int = 1) -> None:
    m._layer_counts[key] = m._layer_counts.get(key, 0) + n


def rung_of(engine: str | None, rung: str | None) -> str:
    if rung:
        return rung
    return engine or "llm"


def count_events(m: _MemoryCore, events: list[MemoryEvent], decisions: list[ReconciliationDecision]) -> None:
    for d in decisions:
        r = "parked" if d.parked else rung_of(d.engine, d.rung)
        count(m, f"rung:{r}")


def rung_stats(m: _MemoryCore) -> dict:
    rungs = {k[5:]: v for k, v in m._layer_counts.items() if k.startswith("rung:")}
    total = sum(rungs.values())
    det = sum(v for k, v in rungs.items() if k.startswith(DETERMINISTIC_RUNGS))
    rules = {k[5:]: v for k, v in m._layer_counts.items() if k.startswith("rule:")}
    return {
        "rungs": rungs,
        "rule_hits": rules,
        "deterministic_share": round(det / total, 4) if total else None,
    }
