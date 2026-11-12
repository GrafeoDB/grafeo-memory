"""Layer B: dynamic, faceted taxonomy for scoping.

Two post-coordinated facets:
- Facet S (subject): a SKOS concept tree, designed, shipped, versioned roots;
  deeper levels generated at compile time (Layer E). Lives in a ConceptScheme.
- Facet I (instance): entity nodes act as scopes via prominence (frequency +
  centrality); all entity types are eligible, only the threshold gates.

Nothing here calls a model: query routing is lexical first (labels/altLabels),
with an optional engine callback for the residue. Scopes are soft by default.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field

from grafeo_memory.common.types import (
    DEFAULT_ROOTS,
    GrowthConfig,
    InstanceScopeConfig,
    RootSpec,
    TaxonomyConfig,
)
from grafeo_memory.core.ontology.scheme import Concept, ConceptScheme, normalize_key

__all__ = [
    "DEFAULT_ROOTS",
    "GrowthConfig",
    "InstanceScopeConfig",
    "RootSpec",
    "ScopeSet",
    "Taxonomy",
    "TaxonomyConfig",
]


@dataclass
class ScopeSet:
    """Active scopes for a query: subject paths x instance scopes, post-coordinated."""

    subject_paths: list[str] = field(default_factory=list)  # concept ids (facet S)
    instance_scopes: list[str] = field(default_factory=list)  # entity ids (facet I)

    @property
    def is_global(self) -> bool:
        return not self.subject_paths and not self.instance_scopes

    def combinations(self) -> list[tuple[str, str | None]]:
        """S x I post-coordination; general is always included by the search layer."""
        if not self.instance_scopes:
            return [(s, None) for s in self.subject_paths]
        return [(s, i) for s in self.subject_paths for i in self.instance_scopes]


Scorer = Callable[[str, list[str]], list[tuple[float, str]]]  # (query, concept_ids) -> [(score, id)]


class Taxonomy:
    """The subject facet (roots + generated tree) plus instance-scope prominence."""

    def __init__(self, config: TaxonomyConfig | None = None):
        self.config = config or TaxonomyConfig()
        self.scheme = ConceptScheme(scheme_id="subjects")
        self.status: dict[str, str] = {}
        self.version: int = 0
        self.history: list[dict] = []
        self._prominence: dict[str, int] = {}
        self._hard: set[str] = set()
        for root in self.config.roots:
            self._add_root(root)

    def _add_root(self, root: RootSpec) -> Concept:
        c = self.scheme.add_concept(root.id, root.label, "Scope", alt_labels=root.keywords, broader=None)
        self.status[root.id] = root.status
        return c

    def add_child(
        self,
        parent: str,
        id: str,
        label: str,
        alt_labels: tuple[str, ...] | list[str] = (),
        status: str = "provisional",
    ) -> Concept:
        """A generated (or user) concept under ``parent``; its labels route like a root's."""
        c = self.scheme.add_concept(id, label, "Scope", alt_labels=tuple(alt_labels), broader=parent)
        self.status[id] = status
        return c

    def remove_concept(self, id: str) -> None:
        """Drop a generated concept (compile regenerates children; roots are retired, not removed)."""
        if self.scheme.remove_concept(id) is None:
            return
        self.scheme._children.pop(id, None)
        self.status.pop(id, None)

    def root_of(self, cid: str) -> str:
        path = self.scheme.broader_path(cid)
        return path[-1] if path else cid

    # -- write path: subject tags for a fact (deterministic) ----

    def tag(self, text: str, scorer: Scorer | None = None) -> list[str]:
        """Subject-path tags for a stored fact: the deepest lexical matches, else the
        scorer's best root above the threshold, else ``general``."""
        hits = self.scheme.alias_hits(text)
        hits.discard("general")
        # keep the deepest hits: a child hit makes its ancestors redundant
        deep = sorted(h for h in hits if not any(h in self.scheme.broader_path(o) for o in hits if o != h))
        if not deep and scorer is not None:
            scored = sorted(scorer(text, [c for c in self.scheme.concepts if c != "general"]), reverse=True)
            if scored and scored[0][0] >= self.config.router_threshold:
                deep = [scored[0][1]]
        return deep or ["general"]

    # -- root scheme editing (config / set_root_scheme / MCP all land here) ----

    def add_root(self, id: str, label: str, status: str = "user") -> Concept:
        if id in self.scheme.concepts:
            raise ValueError(f"concept {id!r} already exists")
        c = self.scheme.add_concept(id, label, "Scope")
        self.status[id] = status
        self.version += 1
        self.history.append({"v": self.version, "op": "add_root", "id": id})
        return c

    def rename_root(self, id: str, label: str) -> None:
        """The new label becomes the prefLabel and routes; the old one stays an
        altLabel (dual-label grace period), so tags and queries using it keep working."""
        c = self.scheme.concepts[id]
        alt = c.alt_labels if c.pref_label in c.alt_labels else (*c.alt_labels, c.pref_label)
        self.scheme.put(
            Concept(
                id=c.id,
                pref_label=label,
                entity_type=c.entity_type,
                alt_labels=alt,
                notation=c.notation,
                broader=c.broader,
                related=c.related,
                exact_match=c.exact_match,
            )
        )
        self.version += 1
        self.history.append({"v": self.version, "op": "rename_root", "id": id, "label": label})

    def retire_root(self, id: str, fallback: str = "general") -> dict:
        """Retiring never deletes memories: the root stops routing and stops being a
        top concept, its subtree re-parents to ``fallback``, and a root_migration entry
        is recorded (routed to the open-decisions bucket)."""
        if id not in self.scheme.concepts:
            raise KeyError(id)
        # the fallback must outlive the retired subtree, or migrated children dangle or loop
        if (
            fallback not in self.scheme.concepts
            or fallback == id
            or fallback in self.scheme.narrower(id, len(self.scheme.concepts))
        ):
            raise ValueError(f"fallback {fallback!r} must be an existing concept outside the subtree of {id!r}")
        migrated = list(self.scheme._children.get(id, ()))  # direct children move; their subtrees follow
        self.scheme.remove_concept(id)
        for cid in migrated:
            c = self.scheme.concepts[cid]
            self.scheme.put(
                Concept(
                    id=c.id,
                    pref_label=c.pref_label,
                    entity_type=c.entity_type,
                    alt_labels=c.alt_labels,
                    notation=c.notation,
                    broader=fallback,
                    related=c.related,
                    exact_match=c.exact_match,
                )
            )
        self.scheme._children.pop(id, None)
        self.status.pop(id, None)
        self.version += 1
        entry = {
            "v": self.version,
            "op": "retire_root",
            "id": id,
            "fallback": fallback,
            "migrated": [
                *migrated,
                *[d for c in migrated for d in self.scheme.narrower(c, self.config.growth.depth_cap)],
            ],
        }
        self.history.append(entry)
        return entry

    # -- instance facet: prominence-gated scopes, all entity types eligible ----

    def record_mention(self, entity_id: str, edges: int = 0) -> int:
        self._prominence[entity_id] = self._prominence.get(entity_id, 0) + 1 + edges
        return self._prominence[entity_id]

    def set_hard(self, entity_id: str, hard: bool = True) -> None:
        if hard:
            self._hard.add(entity_id)
        else:
            self._hard.discard(entity_id)

    def is_hard(self, entity_id: str) -> bool:
        return entity_id in self._hard

    def active_instance_scopes(self) -> list[str]:
        t = self.config.instance_scopes.prominence_threshold
        return sorted(e for e, n in self._prominence.items() if n >= t)

    # -- query routing, deterministic first ----

    def route_subject(self, query: str, scorer: Scorer | None = None) -> list[str]:
        """Subject facet: lexical/altLabel matches first; optional engine scores the
        residue parent-first (children only when the parent clears router_threshold)."""
        if not self.config.enabled:
            return ["general"]
        # whole words only: "homework" does not route to "home"
        hits = self.scheme.alias_hits(query)
        paths = self.scheme.expand(sorted(hits), depth=1, direction="down") if hits else []
        if scorer is not None and len(paths) < self.config.max_scopes_per_query:
            remaining = [cid for cid in self.scheme.concepts if cid not in paths]
            scored = scorer(query, remaining)
            for score, cid in sorted(scored, reverse=True):
                if score >= self.config.router_threshold and len(paths) < self.config.max_scopes_per_query:
                    paths.extend(d for d in self.scheme.expand([cid], depth=1, direction="down") if d not in paths)
                elif score < self.config.router_threshold:
                    break
        if not paths:
            return ["general"]
        return paths[: self.config.max_scopes_per_query]

    def route(self, query: str, scorer: Scorer | None = None) -> ScopeSet:
        """Full routing: subject paths + prominence-activated instance scopes."""
        if not self.config.enabled:
            return ScopeSet(subject_paths=["general"])
        subjects = self.route_subject(query, scorer)
        # instance facet: normalize query mentions through the scheme aliases is the
        # Layer D ladder's job; here we activate prominent entities mentioned verbatim
        qn = f" {normalize_key(query)} "
        instances = [e for e in self.active_instance_scopes() if f" {normalize_key(e)} " in qn]
        return ScopeSet(subject_paths=subjects, instance_scopes=instances)

    def set_prominence(self, counts: dict[str, int]) -> None:
        self._prominence = dict(counts)

    @classmethod
    def from_dict(cls, d: dict, config: TaxonomyConfig | None = None) -> Taxonomy:
        t = cls(config)
        if d.get("scheme"):
            t.scheme = ConceptScheme.from_dict(d["scheme"])
        t.status.update(d.get("status", {}))
        t.version = int(d.get("version", 0))
        t.history = list(d.get("history", []))
        t._prominence = dict(d.get("prominence", {}))
        t._hard = set(d.get("hard", []))
        return t

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "status": self.status,
            "scheme": self.scheme.to_dict(),
            "prominence": self._prominence,
            "hard": sorted(self._hard),
            "history": self.history,
        }
