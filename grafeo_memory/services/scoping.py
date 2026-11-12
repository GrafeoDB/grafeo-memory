"""Layer B: faceted scoping, wired into the write and read paths.

- Write: every new memory gets subject tags (``subjects``), lexical over the root
  keywords and generated children, with an optional decision-model scorer for the
  facts no label matches. Instance scopes need no storage: a memory's HAS_ENTITY
  edges (and SCOPE_OF edges pinned with ``add(scopes=...)``) tie it to entities, and
  an entity becomes a scope once enough memories link to it.
- Read: the query routes to subject paths and to the prominent entities it names.
  ``soft`` mode boosts in-scope memories; ``filter`` mode keeps only in-scope ones
  plus the general scope. Hard roots never leak into searches that did not route
  to them. Pinned ``subjects=`` / ``scopes=`` skip routing.
- Compile: within each root, clusters of similar memories become provisional
  children with a lexical label; children whose membership holds across compiles
  turn ``stable`` (certification candidates).

The taxonomy state (scheme, statuses, version trail, hard scopes) is kept in one
``TaxonomyState`` node per user and graph scope.
"""

from __future__ import annotations

import json
import logging
import re
from collections import Counter
from typing import TYPE_CHECKING

from grafeo_memory.common.types import (
    ENTITY_LABEL,
    HAS_ENTITY_EDGE,
    MEMORY_LABEL,
    SCOPE_OF_EDGE,
    TAXONOMY_LABEL,
    Entity,
    SearchResult,
)
from grafeo_memory.core.ontology.normalize import cosine
from grafeo_memory.core.ontology.scheme import normalize_key
from grafeo_memory.core.search.vector import _get_props
from grafeo_memory.core.taxonomy import ScopeSet, Taxonomy

if TYPE_CHECKING:
    from grafeo_memory.services.memory import _MemoryCore

logger = logging.getLogger(__name__)

_STOP = frozenset(
    """
a about after also an and another any are as at be been before being but by can could day days did do does
during each first for from get gets getting goes going got had has have he her here his how i in into is it
its just last like made make makes making may me might month months more most my new no not of old on one onto
or other our over really said same says shall she should some such than that the their them then there these
they this those three time times to two under user users very want wanted wants was we week weeks went were
what when where which while who whom whose why will with would year years yes you your
""".split()  # noqa: SIM905  (a word list reads better as text)
)


# --- state --------------------------------------------------------------------


def _state_node(m: _MemoryCore, uid: str) -> tuple[int, dict] | None:
    try:
        nodes = m._db.get_nodes_by_label(TAXONOMY_LABEL)
    except Exception:
        return None
    gn = m._config.graph_name
    for item in nodes:
        nid, props = (item[0], item[1]) if isinstance(item, tuple) else (getattr(item, "id", None), _get_props(item))
        if nid is not None and props.get("user_id") == uid and props.get("graph_name") == gn:
            return int(nid), props
    return None


def load(m: _MemoryCore, uid: str) -> Taxonomy:
    tax = m._taxonomies.get(uid)
    if tax is not None:
        return tax
    cfg = m._config.taxonomy
    if not cfg.enabled:  # pipeline.taxonomy sugar: the layer is on even when the config flag is not
        from dataclasses import replace

        cfg = replace(cfg, enabled=True)
    found = _state_node(m, uid)
    if found is not None and found[1].get("state"):
        try:
            tax = Taxonomy.from_dict(json.loads(found[1]["state"]), cfg)
        except ValueError:
            logger.warning("taxonomy state unreadable, starting from the shipped roots")
    if tax is None:
        tax = Taxonomy(cfg)
        for root in cfg.hard_roots:
            tax.set_hard(root)
    m._taxonomies[uid] = tax
    return tax


def save(m: _MemoryCore, uid: str, tax: Taxonomy) -> None:
    state = tax.to_json()
    found = _state_node(m, uid)
    if found is not None:
        m._db.set_node_property(found[0], "state", state)
        return
    props: dict = {"user_id": uid, "state": state}
    if m._config.graph_name:
        props["graph_name"] = m._config.graph_name
    from grafeo_memory.services.schema import mark

    mark(m._db)
    m._db.create_node([TAXONOMY_LABEL], props)


# --- write path ---------------------------------------------------------------


async def tag_texts(m: _MemoryCore, uid: str, texts: list[str], on_usage=None) -> list[list[str]]:
    """Subject tags per text: lexical first; the decision model scores the leftovers when configured."""
    tax = load(m, uid)
    tags = [tax.tag(t) for t in texts]
    cfg = m._config.taxonomy
    rest = [i for i, t in enumerate(tags) if t == ["general"]]
    if cfg.router_engine == "decision" and m._decision_model is not None and rest:
        from grafeo_memory.providers.decision.types import ScoreQuestion

        roots = [cid for cid in tax.scheme.concepts if tax.scheme.concepts[cid].broader is None and cid != "general"]
        questions = [
            ScoreQuestion(
                question_id=f"{i}|{r}",
                question=f'How much is fact {i} about "{tax.scheme.concepts[r].pref_label}"?',
                criteria=["0.0 = not at all", "0.5 = partly", "1.0 = squarely"],
            )
            for i in rest
            for r in roots
        ]
        state = "Facts:\n" + "\n".join(f"{i}. {texts[i]}" for i in rest)
        try:
            ds = await m._decision_model.decide_async(state, questions)
        except Exception:
            logger.warning("taxonomy: decision-model routing failed, keeping general", exc_info=True)
            ds = []
        best: dict[int, tuple[float, str]] = {}
        for d in ds:
            i_s, r = d.question_id.split("|", 1)
            score = float(d.value) if isinstance(d.value, int | float) else 0.0
            if score >= cfg.router_threshold and score > best.get(int(i_s), (0.0, ""))[0]:
                best[int(i_s)] = (score, r)
        for i, (_s, r) in best.items():
            tags[i] = [r]
    return tags


def tag_memories(m: _MemoryCore, memory_ids: list[str], tags: list[list[str]]) -> None:
    for mid, t in zip(memory_ids, tags, strict=False):
        try:
            m._db.set_node_property(int(mid), "subjects", list(t))
        except Exception:
            logger.debug("taxonomy: tagging memory %s failed", mid, exc_info=True)


def pin_scopes(m: _MemoryCore, uid: str, memory_ids: list[str], scopes: list[str]) -> None:
    """``add(scopes=[...])``: SCOPE_OF edges from the new memories to the named entities."""
    for name in scopes:
        eid = m._find_or_create_entity(Entity(name=name, entity_type="Other"), uid)
        for mid in memory_ids:
            try:
                m._db.create_edge(int(mid), int(eid), SCOPE_OF_EDGE)
            except Exception:
                logger.debug("taxonomy: SCOPE_OF %s->%s failed", mid, eid, exc_info=True)


# --- read path ----------------------------------------------------------------


def prominence(m: _MemoryCore, uid: str) -> dict[str, int]:
    cached = m._prominence_cache.get(uid)
    if cached is not None and cached[0] == m._write_version:
        return cached[1]
    counts: dict[str, int] = {}
    try:
        rows = m._db.execute(
            f"MATCH (x:{MEMORY_LABEL})-[:{HAS_ENTITY_EDGE}]->(e:{ENTITY_LABEL}) WHERE e.user_id = $uid "
            "RETURN e.name AS name, count(x) AS n",
            {"uid": uid},
        )
        for r in rows:
            if isinstance(r, dict) and r.get("name"):
                counts[str(r["name"])] = int(r.get("n") or 0)
    except Exception:
        logger.debug("taxonomy: prominence query failed", exc_info=True)
    m._prominence_cache[uid] = (m._write_version, counts)
    return counts


def route(
    m: _MemoryCore,
    uid: str,
    query: str,
    query_entities: list[str],
    subjects: list[str] | None = None,
    scopes: list[str] | None = None,
) -> ScopeSet:
    tax = load(m, uid)
    if subjects is not None or scopes is not None:
        subj = tax.scheme.expand(list(subjects or []), depth=tax.config.growth.depth_cap, direction="down")
        return ScopeSet(subject_paths=subj, instance_scopes=list(scopes or []))
    tax.set_prominence(prominence(m, uid))
    ss = tax.route(query)
    active = set(tax.active_instance_scopes())
    qkeys = {normalize_key(e) for e in query_entities}
    for name in active:
        if normalize_key(name) in qkeys and name not in ss.instance_scopes:
            ss.instance_scopes.append(name)
    # A hub (a speaker on nearly every memory) scopes nothing: it would admit every memory in filter mode.
    from grafeo_memory.services.retrieval import hubs

    hub_names = hubs(m, uid)
    ss.instance_scopes = [n for n in ss.instance_scopes if n not in hub_names][: tax.config.max_scopes_per_query]
    if ss.subject_paths == ["general"]:
        ss.subject_paths = []  # general alone = no subject routing
    return ss


def memories_of_entities(m: _MemoryCore, uid: str, names: list[str]) -> set[str]:
    out: set[str] = set()
    if not names:
        return out
    for edge in (HAS_ENTITY_EDGE, SCOPE_OF_EDGE):
        try:
            rows = m._db.execute(
                f"MATCH (x:{MEMORY_LABEL})-[:{edge}]->(e:{ENTITY_LABEL}) WHERE e.user_id = $uid AND e.name IN $names "
                "RETURN id(x) AS id",
                {"uid": uid, "names": list(names)},
            )
            out.update(str(r["id"]) for r in rows if isinstance(r, dict) and r.get("id") is not None)
        except Exception:
            logger.debug("taxonomy: scope membership query failed", exc_info=True)
    return out


def _subjects_of(props: dict) -> set[str]:
    return {str(s) for s in (props.get("subjects") or [])} | {str(s) for s in (props.get("subjects_generated") or [])}


def apply_scopes(
    m: _MemoryCore, uid: str, results: list[SearchResult], ss: ScopeSet, props: dict[str, dict]
) -> list[SearchResult]:
    """Boost (soft) or filter (filter mode) by the active scopes; enforce hard walls."""
    from dataclasses import replace

    tax = load(m, uid)
    cfg = tax.config
    subj_active = set(ss.subject_paths)
    in_instance = memories_of_entities(m, uid, ss.instance_scopes)
    hard = {h for h in tax._hard if h in tax.scheme.concepts}
    hard_tree = {h: {h, *tax.scheme.narrower(h, cfg.growth.depth_cap)} for h in hard}
    out: list[SearchResult] = []
    for r in results:
        p = props.get(r.memory_id, {})
        subs = _subjects_of(p)
        if any(subs & tree and not (subj_active & tree) for tree in hard_tree.values()):
            continue  # a hard wall: this root was not routed to
        s_hit = bool(subs & subj_active)
        i_hit = r.memory_id in in_instance
        if cfg.mode == "filter" and (subj_active or ss.instance_scopes):
            general = not subs or subs == {"general"}
            if not (s_hit or i_hit or general):
                continue
        factor = 1.0
        if s_hit:
            factor += cfg.subject_boost
        if i_hit:
            factor += cfg.instance_boost
        if s_hit and i_hit:
            factor += cfg.combination_boost
        out.append(replace(r, score=r.score * factor) if factor != 1.0 else r)
    out.sort(key=lambda x: x.score, reverse=True)
    return out


# --- compile: growth and root editing -------------------------------------------


def _tokens(text: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z\-]{3,}", text.lower()) if w not in _STOP]


def grow(m: _MemoryCore, uid: str) -> dict:
    """Generated children per root: cluster each root's memories, label clusters lexically."""
    tax = load(m, uid)
    g = tax.config.growth
    vp = m._config.vector_property
    gn = m._config.graph_name
    try:
        nodes = m._db.get_nodes_by_label(MEMORY_LABEL)
    except Exception:
        return {}
    rows: list[tuple[int, dict]] = []
    for item in nodes:
        nid, props = (item[0], item[1]) if isinstance(item, tuple) else (getattr(item, "id", None), _get_props(item))
        if nid is None or props.get("user_id") != uid or props.get("expired_at") is not None:
            continue
        if gn and props.get("graph_name") != gn:
            continue
        rows.append((int(nid), props))
    previous = next((h for h in reversed(tax.history) if h.get("op") == "grow"), {}).get("children", {})
    for cid in [c for c, st in tax.status.items() if st in ("provisional", "stable")]:
        tax.remove_concept(cid)
    roots = [c for c, con in tax.scheme.concepts.items() if con.broader is None and c != "general"]
    children: dict[str, int] = {}
    assigned: dict[int, list[str]] = {}
    for root in roots:
        keywords = set(tax.scheme.concepts[root].alt_labels)
        members = [(nid, p) for nid, p in rows if root in {str(s) for s in (p.get("subjects") or [])} and p.get(vp)]
        if len(members) < g.min_cluster_size:
            continue
        clusters: list[tuple[list[float], list[tuple[int, dict]]]] = []
        for nid, p in members:
            vec = [float(x) for x in p[vp]]
            for centroid, mem in clusters:
                if cosine(vec, centroid) >= g.cluster_similarity:
                    mem.append((nid, p))
                    n = len(mem)
                    for k in range(len(centroid)):
                        centroid[k] += (vec[k] - centroid[k]) / n
                    break
            else:
                clusters.append((list(vec), [(nid, p)]))
        for _c, mem in clusters:
            if len(mem) < g.min_cluster_size:
                continue
            counts = Counter(w for _nid, p in mem for w in set(_tokens(str(p.get("text", "")))) if w not in keywords)
            if not counts:
                continue
            top = [w for w, _n in counts.most_common(3)]
            cid = f"{root}.{top[0]}"
            if cid in tax.scheme.concepts:
                continue
            tax.add_child(root, cid, top[0], alt_labels=top)
            prev = previous.get(cid)
            if prev and abs(prev - len(mem)) <= max(1, prev // 10):
                tax.status[cid] = "stable"
            children[cid] = len(mem)
            for nid, _p in mem:
                assigned.setdefault(nid, []).append(cid)
    for nid, p in rows:
        new = assigned.get(nid, [])
        if new or p.get("subjects_generated"):
            m._db.set_node_property(nid, "subjects_generated", new)
    tax.version += 1
    tax.history.append({"v": tax.version, "op": "grow", "children": children})
    save(m, uid, tax)
    return {"children": children, "stable": [c for c in children if tax.status.get(c) == "stable"]}


def retag(m: _MemoryCore, retired: str, new_root: str) -> int:
    """Move memories tagged with a retired root (or its migrated children) to ``new_root``."""
    n = 0
    try:
        nodes = m._db.get_nodes_by_label(MEMORY_LABEL)
    except Exception:
        return 0
    for item in nodes:
        nid, props = (item[0], item[1]) if isinstance(item, tuple) else (getattr(item, "id", None), _get_props(item))
        subs = [str(s) for s in (props.get("subjects") or [])]
        if nid is not None and retired in subs:
            m._db.set_node_property(int(nid), "subjects", [new_root if s == retired else s for s in subs])
            n += 1
    return n


def set_root_scheme(
    m: _MemoryCore,
    uid: str,
    *,
    add: dict[str, str] | None = None,
    rename: dict[str, str] | None = None,
    retire: list[str] | None = None,
    hard: dict[str, bool] | None = None,
) -> dict:
    """Edit the designed roots at runtime. Retiring never deletes memories: they stay
    tagged and a ``root_migration`` open decision asks where they belong (bucket on),
    or they move to ``general`` (bucket off)."""
    from grafeo_memory.services import bucket

    tax = load(m, uid)
    entries: list[dict] = []
    for rid, label in (add or {}).items():
        tax.add_root(rid, label)
        entries.append({"op": "add_root", "id": rid})
    for rid, label in (rename or {}).items():
        tax.rename_root(rid, label)
        entries.append({"op": "rename_root", "id": rid, "label": label})
    for rid in retire or []:
        entry = tax.retire_root(rid)
        entries.append(entry)
        if m._config.bucket_on:
            options = [c for c, con in tax.scheme.concepts.items() if con.broader is None]
            bucket.create(
                m,
                uid,
                kind="root_migration",
                question=f'The subject root "{rid}" was retired. Where do its memories belong?',
                options=options,
                evidence=[rid],
                rung="taxonomy",
                importance=0.7,
                payload={"retired": rid, "fallback": entry.get("fallback", "general")},
            )
        else:
            retag(m, rid, entry.get("fallback", "general"))
    for rid, flag in (hard or {}).items():
        tax.set_hard(rid, flag)
        entries.append({"op": "set_hard", "id": rid, "hard": flag})
    save(m, uid, tax)
    return {"version": tax.version, "changes": entries}


def scopes(m: _MemoryCore, uid: str) -> dict:
    """The subject tree (roots, generated children, statuses) and the active instance scopes."""
    tax = load(m, uid)
    tax.set_prominence(prominence(m, uid))

    def node(cid: str) -> dict:
        c = tax.scheme.concepts[cid]
        return {
            "id": cid,
            "label": c.pref_label,
            "status": tax.status.get(cid, "shipped"),
            "hard": tax.is_hard(cid),
            "children": [node(ch) for ch in tax.scheme._children.get(cid, []) if ch in tax.scheme.concepts],
        }

    roots = [c for c, con in tax.scheme.concepts.items() if con.broader is None]
    return {
        "version": tax.version,
        "subjects": [node(r) for r in roots],
        "instance_scopes": tax.active_instance_scopes(),
    }
