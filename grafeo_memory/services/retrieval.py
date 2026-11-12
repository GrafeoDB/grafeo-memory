"""Read-path layers: deterministic-first retrieval (Layer D), the Allen filter, and the
ranking knobs of Layers A, E and G (procedural boost, recency, certified boost,
provisional penalty).

Deterministic-first (``retrieval.order="deterministic"``): the query's mentions are
resolved to entities with no model (exact names, compile-merged aliases, plus the
query entities the search already extracted), expanded along relation edges
(``expansion_depth`` hops), and the memories tied to those entities are boosted;
the ones the vector search missed are pulled in and scored by vector similarity
within that set.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import replace
from typing import TYPE_CHECKING

from grafeo_memory.common.intervals import intervals_overlap
from grafeo_memory.common.types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL, RELATION_EDGE, SearchResult
from grafeo_memory.core.ontology.normalize import cosine
from grafeo_memory.core.ontology.scheme import normalize_key
from grafeo_memory.core.reconciliation.temporal import extract_interval
from grafeo_memory.core.search.vector import _get_props

if TYPE_CHECKING:
    from grafeo_memory.services.memory import _MemoryCore

logger = logging.getLogger(__name__)


def fetch_props(m: _MemoryCore, ids: list[str]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for mid in ids:
        try:
            node = m._db.get_node(int(mid))
        except (TypeError, ValueError):
            continue
        if node is not None:
            out[mid] = _get_props(node)
    return out


def _entity_names(m: _MemoryCore, uid: str) -> dict[str, str]:
    """normalized name or merged alias -> canonical entity name (cached per write)."""
    cached = m._entity_name_cache.get(uid)
    if cached is not None and cached[0] == m._write_version:
        return cached[1]
    names: dict[str, str] = {}
    try:
        rows = m._db.execute(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.user_id = $uid RETURN e.name AS name, e.merged_from AS merged",
            {"uid": uid},
        )
        for r in rows:
            if not isinstance(r, dict) or not r.get("name"):
                continue
            canon = str(r["name"])
            for alias in (canon, *[str(a) for a in (r.get("merged") or []) if a]):
                key = normalize_key(alias)
                if len(key) >= 3:
                    names.setdefault(key, canon)
    except Exception:
        logger.debug("retrieval: entity name query failed", exc_info=True)
    m._entity_name_cache[uid] = (m._write_version, names)
    return names


_HUB_MIN_MEMORIES = 20  # below this, shares are too coarse to call anything a hub


def hubs(m: _MemoryCore, uid: str) -> set[str]:
    """Entity names tied to more than ``retrieval.hub_share`` of the user's memories (cached per write)."""
    share = m._config.retrieval.hub_share
    if share >= 1.0:
        return set()
    cached = m._hub_cache.get(uid)
    if cached is not None and cached[0] == (m._write_version, share):
        return cached[1]
    out: set[str] = set()
    try:
        total_rows = m._db.execute(
            f"MATCH (x:{MEMORY_LABEL}) WHERE x.user_id = $uid RETURN count(x) AS n", {"uid": uid}
        )
        total = int(total_rows[0]["n"]) if total_rows else 0
        if total >= _HUB_MIN_MEMORIES:
            rows = m._db.execute(
                f"MATCH (x:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) WHERE e.user_id = $uid "
                "RETURN e.name AS name, count(DISTINCT x) AS n",
                {"uid": uid},
            )
            out = {str(r["name"]) for r in rows if isinstance(r, dict) and r.get("name") and r["n"] > share * total}
    except Exception:
        logger.debug("retrieval: hub query failed", exc_info=True)
    m._hub_cache[uid] = ((m._write_version, share), out)
    return out


def query_entities(m: _MemoryCore, uid: str, query: str, extracted: list[str]) -> list[str]:
    """Entities a query names, with no model: whole-word name and alias matches, plus the
    names the search's entity extraction already produced (resolved through the same map)."""
    table = _entity_names(m, uid)
    qn = f" {normalize_key(query)} "
    found = {canon for key, canon in table.items() if f" {key} " in qn}
    for e in extracted:
        canon = table.get(normalize_key(e))
        if canon is not None:
            found.add(canon)
    return sorted(found)


def expand(m: _MemoryCore, uid: str, names: list[str], depth: int) -> list[str]:
    """Relation-edge neighbours of the named entities, ``depth`` hops (deterministic recall)."""
    seen = set(names)
    frontier = list(names)
    for _ in range(depth):
        if not frontier:
            break
        nxt: list[str] = []
        try:
            rows = m._db.execute(
                f"MATCH (a:{ENTITY_LABEL})-[:{RELATION_EDGE}]-(b:{ENTITY_LABEL}) "
                "WHERE a.user_id = $uid AND a.name IN $names RETURN DISTINCT b.name AS name",
                {"uid": uid, "names": frontier},
            )
        except Exception:
            logger.debug("retrieval: relation expansion failed", exc_info=True)
            break
        for r in rows:
            name = r.get("name") if isinstance(r, dict) else None
            if name and name not in seen:
                seen.add(str(name))
                nxt.append(str(name))
        frontier = nxt
    return sorted(seen)


def _result_from_props(mid: str, props: dict, score: float) -> SearchResult:
    return SearchResult(
        memory_id=mid,
        text=str(props.get("text", "")),
        score=score,
        user_id=str(props.get("user_id", "")),
        actor_id=props.get("actor_id"),
        role=props.get("role"),
        memory_type=props.get("memory_type"),
        source="deterministic",
        created_at=props.get("created_at"),
        learned_at=props.get("learned_at"),
        session_id=props.get("session_id"),
        expired_at=props.get("expired_at"),
        valid_at=props.get("valid_at"),
        invalid_at=props.get("invalid_at"),
    )


def apply_deterministic(
    m: _MemoryCore,
    uid: str,
    results: list[SearchResult],
    member_ids: set[str],
    query_embedding: list[float],
    include_expired: bool,
) -> list[SearchResult]:
    cfg = m._config.retrieval
    boost = 1.0 + cfg.deterministic_boost
    have = {r.memory_id for r in results}
    out = [replace(r, score=r.score * boost) if r.memory_id in member_ids else r for r in results]
    missing = [mid for mid in sorted(member_ids, key=lambda x: int(x) if x.isdigit() else 0) if mid not in have]
    props = fetch_props(m, missing[: cfg.deterministic_cap])
    vp = m._config.vector_property
    gn = m._config.graph_name
    for mid, p in props.items():
        if p.get("user_id") != uid or (gn and p.get("graph_name") != gn):
            continue
        if p.get("expired_at") is not None and not include_expired:
            continue
        vec = p.get(vp)
        sim = cosine(query_embedding, [float(x) for x in vec]) if vec else 0.0
        out.append(_result_from_props(mid, p, max(0.0, sim) * boost))
    out.sort(key=lambda r: r.score, reverse=True)
    return out


def apply_temporal(results: list[SearchResult], props: dict[str, dict], query: str, boost: float) -> list[SearchResult]:
    """Allen filter: when the query names a time, overlap is boosted and disjoint is demoted."""
    q_iv = extract_interval(query)
    if q_iv is None:
        return results
    out: list[SearchResult] = []
    for r in results:
        iv = extract_interval(str(props.get(r.memory_id, {}).get("text", r.text)))
        if iv is None:
            out.append(r)
        elif intervals_overlap(q_iv, iv):
            out.append(replace(r, score=r.score * (1.0 + boost)))
        else:
            out.append(replace(r, score=r.score * (1.0 - boost)))
    out.sort(key=lambda r: r.score, reverse=True)
    return out


def apply_layer_scores(m: _MemoryCore, results: list[SearchResult], props: dict[str, dict]) -> list[SearchResult]:
    """Certified boost (E), provisional penalty (G), procedural boost and recency (A)."""
    cfg = m._config
    certified = cfg.retrieval.certified_boost
    penalty = cfg.bucket.provisional_penalty if cfg.bucket_on else 0.0
    proc = cfg.tiers.procedural.boost_weight if cfg.tiers.procedural.enabled else 0.0
    rw = cfg.tiers.episodic.recency_weight if cfg.tiers.episodic.enabled else 0.0
    half_life_ms = max(1.0, cfg.tiers.episodic.recency_half_life_days) * 86_400_000
    now = time.time() * 1000
    out: list[SearchResult] = []
    for r in results:
        p = props.get(r.memory_id, {})
        s = r.score
        if certified and p.get("status") == "certified":
            s *= 1.0 + certified
        if penalty and p.get("provisional"):
            s *= 1.0 - penalty
        if proc and (r.memory_type or p.get("memory_type")) == "procedural":
            s *= 1.0 + proc
        if rw:
            created = r.created_at or p.get("created_at")
            recency = math.pow(0.5, max(0.0, now - float(created)) / half_life_ms) if created else 0.0
            s = (1.0 - rw) * s + rw * recency
        out.append(replace(r, score=s) if s != r.score else r)
    out.sort(key=lambda r: r.score, reverse=True)
    return out


def needs_layer_pass(m: _MemoryCore) -> bool:
    c = m._config
    return bool(
        c.taxonomy_on
        or c.retrieval.order == "deterministic"
        or c.retrieval.temporal_filter
        or c.retrieval.certified_boost
        or (c.bucket_on and c.bucket.provisional_penalty)
        or c.tiers.procedural.enabled
        or (c.tiers.episodic.enabled and c.tiers.episodic.recency_weight)
    )
