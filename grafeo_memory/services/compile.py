"""Layer E, minimal cut: offline compilation over the entity graph.

``compile_graph()`` is meant to run after write batches (call ``memory.compile()``):

1. Rung 3 of the normalization ladder over the whole entity graph: entities of the
   same type and the same ``graph_name`` scope whose name embeddings reach
   ``merge_threshold`` are merged. The older (lower-id) node survives; the alias's
   edges move onto it with their properties, its name is kept in the survivor's
   ``merged_from`` (read by write-time normalization), and the alias is deleted.
2. Controlled relation-type validation report (``is_violation``): violations are
   reported, never changed; writes were already gated at insert time.

Rungs 1-2 run at write time (``PipelineConfig.ontology``). Certification promotion
and SKOS/Allen precompute come with the full Layer E.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from grafeo_memory.common.types import ENTITY_LABEL, RELATION_EDGE
from grafeo_memory.core._graph import detach_delete_node
from grafeo_memory.core.ontology.normalize import cosine
from grafeo_memory.core.ontology.types import is_violation
from grafeo_memory.core.protocol import GrafeoDBProtocol

logger = logging.getLogger(__name__)


@dataclass
class Merge:
    alias_name: str
    canonical_name: str
    score: float
    rung: int  # 2 = alias table, 3 = embedding similarity


@dataclass
class CompileResult:
    entities_before: int = 0
    entities_after: int = 0
    merges: list[Merge] = field(default_factory=list)
    relation_violations: list[dict] = field(default_factory=list)
    edges_added: int = 0


def _entity_rows(db: GrafeoDBProtocol, user_id: str, graph_name: str | None) -> list[dict]:
    if graph_name:
        rows = db.execute(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.user_id = $uid AND e.graph_name = $gn "
            "RETURN id(e) AS nid, e.name AS name, e.entity_type AS etype, e.graph_name AS gn",
            {"uid": user_id, "gn": graph_name},
        )
    else:
        # Unscoped manager: sees every scope, but merges stay within one scope (see gn below).
        rows = db.execute(
            f"MATCH (e:{ENTITY_LABEL}) WHERE e.user_id = $uid "
            "RETURN id(e) AS nid, e.name AS name, e.entity_type AS etype, e.graph_name AS gn",
            {"uid": user_id},
        )
    return [r for r in rows if isinstance(r, dict) and r.get("name")]


EdgeRow = tuple[str, int, dict]  # (edge_label, other_node_id, edge_properties)


def _edges_of(db: GrafeoDBProtocol, node_id: int) -> tuple[list[EdgeRow], list[EdgeRow]]:
    """(outgoing, incoming) edges of a node, with their properties."""
    out = [
        (str(r["lbl"]), int(r["other"]), dict(r.get("props") or {}))
        for r in db.execute(
            "MATCH (a)-[r]->(x) WHERE id(a) = $id RETURN type(r) AS lbl, id(x) AS other, properties(r) AS props",
            {"id": node_id},
        )
    ]
    inc = [
        (str(r["lbl"]), int(r["other"]), dict(r.get("props") or {}))
        for r in db.execute(
            "MATCH (x)-[r]->(a) WHERE id(a) = $id RETURN type(r) AS lbl, id(x) AS other, properties(r) AS props",
            {"id": node_id},
        )
    ]
    return out, inc


def _has_same_edge(db: GrafeoDBProtocol, src: int, dst: int, label: str, props: dict) -> bool:
    """True when src already has an edge to dst with this label AND these properties
    (a WORKS_AT and a KNOWS relation to the same node are different edges)."""
    out, _ = _edges_of(db, src)
    return any(lbl == label and other == dst and p == props for lbl, other, p in out)


def _merge_into(db: GrafeoDBProtocol, alias_id: int, canon_id: int) -> int:
    """Move alias's edges onto canon with their properties (no exact duplicates, no
    self-loops), then delete alias."""
    added = 0
    out, inc = _edges_of(db, alias_id)
    for lbl, other, props in out:
        if other in (canon_id, alias_id):
            continue
        if not _has_same_edge(db, canon_id, other, lbl, props):
            db.create_edge(canon_id, other, lbl, props or None)
            added += 1
    for lbl, other, props in inc:
        if other in (canon_id, alias_id):
            continue
        if not _has_same_edge(db, other, canon_id, lbl, props):
            db.create_edge(other, canon_id, lbl, props or None)
            added += 1
    detach_delete_node(db, alias_id)
    return added


def _record_merged_name(db: GrafeoDBProtocol, canon_id: int, alias_name: str) -> None:
    """Keep the alias's name on the survivor (``merged_from``), so write-time
    normalization (rung 2) resolves a later mention of it to this entity."""
    rows = db.execute("MATCH (e) WHERE id(e) = $id RETURN e.merged_from AS m", {"id": canon_id})
    current = rows[0].get("m") if rows and isinstance(rows[0], dict) else None
    names = [str(x) for x in current] if isinstance(current, list) else []
    if alias_name not in names:
        db.set_node_property(canon_id, "merged_from", [*names, alias_name])


def compile_graph(
    db: GrafeoDBProtocol,
    user_id: str,
    embedder=None,
    *,
    merge_threshold: float = 0.92,
    graph_name: str | None = None,
) -> CompileResult:
    """One offline compile pass over the user's entity graph. Deterministic rungs
    always run; the embedding rung only when an embedder is supplied (embeddings
    are content-cached, so reruns are free)."""
    res = CompileResult()
    rows = _entity_rows(db, user_id, graph_name)
    res.entities_before = len(rows)

    # Rung 3: type-constrained embedding similarity between distinct entities.
    if embedder is not None and len(rows) > 1:
        names = [str(r["name"]) for r in rows]
        vecs = embedder.embed(names)
        merged_away: set[int] = set()
        # Only entities of one type in one graph_name scope may merge: compare within
        # those groups instead of across every pair.
        groups: dict[tuple[str, object], list[int]] = {}
        for idx, row in enumerate(rows):
            groups.setdefault((str(row.get("etype") or ""), row.get("gn")), []).append(idx)
        for i, r in enumerate(rows):
            if int(r["nid"]) in merged_away:
                continue
            best_j, best_s = -1, 0.0
            for j in groups[(str(r.get("etype") or ""), r.get("gn"))]:
                r2 = rows[j]
                if i == j or int(r2["nid"]) in merged_away:
                    continue
                s = cosine(vecs[i], vecs[j])
                if s > best_s:
                    best_j, best_s = j, s
            if best_j >= 0 and best_s >= merge_threshold:
                # Keep the older (lower-id) node as the canonical survivor.
                if int(r["nid"]) < int(rows[best_j]["nid"]):
                    alias, canon = rows[best_j], r
                else:
                    alias, canon = r, rows[best_j]
                logger.info("compile: merging entity %r -> %r (cos=%.3f)", alias["name"], canon["name"], best_s)
                res.merges.append(Merge(str(alias["name"]), str(canon["name"]), round(best_s, 4), 3))
                res.edges_added += _merge_into(db, int(alias["nid"]), int(canon["nid"]))
                _record_merged_name(db, int(canon["nid"]), str(alias["name"]))
                merged_away.add(int(alias["nid"]))
        rows = [r for r in rows if int(r["nid"]) not in merged_away]

    res.entities_after = len(rows)

    # Controlled relation-type validation report over the remaining graph.
    type_by_id = {int(r["nid"]): str(r.get("etype") or "") for r in rows}
    gn_clause = " AND s.graph_name = $gn" if graph_name else ""
    rel_rows = db.execute(
        f"MATCH (s:{ENTITY_LABEL})-[r:{RELATION_EDGE}]->(t:{ENTITY_LABEL}) "
        f"WHERE s.user_id = $uid{gn_clause} RETURN id(s) AS s, id(t) AS t, r.relation_type AS rel",
        {"uid": user_id, "gn": graph_name} if graph_name else {"uid": user_id},
    )
    for r in rel_rows:
        if not isinstance(r, dict):
            continue
        rel = str(r.get("rel") or "")
        if is_violation(rel, type_by_id.get(int(r["s"]), ""), type_by_id.get(int(r["t"]), "")):
            res.relation_violations.append({"source": int(r["s"]), "target": int(r["t"]), "relation_type": rel})
    if res.relation_violations:
        logger.warning("compile: %d controlled-vocab relation violations found", len(res.relation_violations))
    return res
