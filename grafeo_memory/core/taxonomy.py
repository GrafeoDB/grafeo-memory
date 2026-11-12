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

from collections.abc import Callable
from dataclasses import dataclass, field

from grafeo_memory.core.ontology.scheme import Concept, ConceptScheme, normalize_key


@dataclass(frozen=True)
class RootSpec:
    id: str
    label: str
    status: str = "shipped"  # shipped | user | certified


DEFAULT_ROOTS: tuple[RootSpec, ...] = (
    RootSpec("general", "General"),
    RootSpec("career", "Career & work"),
    RootSpec("technology", "Technology & tools"),
    RootSpec("skills", "Skills & learning"),
    RootSpec("projects", "Projects & goals"),
    RootSpec("health", "Health & wellness"),
    RootSpec("finance", "Finance & legal"),
    RootSpec("relationships", "Relationships & social"),
    RootSpec("home", "Home & daily life"),
    RootSpec("interests", "Interests & leisure"),
)


@dataclass
class GrowthConfig:
    min_cluster_size: int = 8
    depth_cap: int = 4
    merge_similarity: float = 0.9
    demote_ttl_days: int = 30
    min_fact_count: int = 3


@dataclass
class InstanceScopeConfig:
    prominence_threshold: int = 5  # mentions+edges before an entity becomes a scope
    hard_default: bool = False


@dataclass
class TaxonomyConfig:
    enabled: bool = False  # default off: single general scope = today's behavior
    roots: tuple[RootSpec, ...] = DEFAULT_ROOTS
    growth: GrowthConfig = field(default_factory=GrowthConfig)
    instance_scopes: InstanceScopeConfig = field(default_factory=InstanceScopeConfig)
    router_threshold: float = 0.5
    max_scopes_per_query: int = 4


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
        c = self.scheme.add_concept(root.id, root.label, "Scope", broader=None)
        self.status[root.id] = root.status
        return c

    # -- root scheme editing (config / set_root_scheme / MCP all land here) ----

    def add_root(self, id: str, label: str, status: str = "user") -> Concept:
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
        self.scheme.concepts[id] = Concept(
            id=c.id,
            pref_label=label,
            entity_type=c.entity_type,
            alt_labels=alt,
            notation=c.notation,
            broader=c.broader,
            related=c.related,
            exact_match=c.exact_match,
        )
        self.scheme._pref_index.pop((c.pref_label, c.entity_type), None)
        self.scheme._pref_index[(label, c.entity_type)] = id
        self.scheme._alias_index.setdefault(normalize_key(label), id)
        self.version += 1
        self.history.append({"v": self.version, "op": "rename_root", "id": id, "label": label})

    def retire_root(self, id: str, fallback: str = "general") -> dict:
        """Retiring never deletes memories: the root stops routing and stops being a
        top concept, its subtree re-parents to ``fallback``, and a root_migration entry
        is recorded (routed to the open-decisions bucket)."""
        if id not in self.scheme.concepts:
            raise KeyError(id)
        migrated = list(self.scheme._children.get(id, ()))  # direct children move; their subtrees follow
        retired = self.scheme.concepts.pop(id)
        self.scheme._pref_index.pop((retired.pref_label, retired.entity_type), None)
        for alias in [a for a, cid in self.scheme._alias_index.items() if cid == id]:
            del self.scheme._alias_index[alias]
        for siblings in self.scheme._children.values():
            if id in siblings:
                siblings.remove(id)
        for cid in migrated:
            c = self.scheme.concepts[cid]
            self.scheme.concepts[cid] = Concept(
                id=c.id,
                pref_label=c.pref_label,
                entity_type=c.entity_type,
                alt_labels=c.alt_labels,
                notation=c.notation,
                broader=fallback if cid != fallback else None,
                related=c.related,
                exact_match=c.exact_match,
            )
        self.scheme._children.setdefault(fallback, []).extend(migrated)
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
        qn = f" {normalize_key(query)} "
        # whole words only: "homework" does not route to "home"
        hits = {cid for alias, cid in self.scheme._alias_index.items() if alias and f" {alias} " in qn}
        paths = self.scheme.expand(sorted(hits), depth=1, direction="down") if hits else []
        if scorer is not None and len(paths) < self.config.max_scopes_per_query:
            remaining = [cid for cid in self.scheme.concepts if cid not in paths]
            scored = scorer(query, remaining)
            for score, cid in sorted(scored, reverse=True):
                if score >= self.config.router_threshold and len(paths) < self.config.max_scopes_per_query:
                    paths.extend(self.scheme.expand([cid], depth=1, direction="down"))
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

    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "status": self.status,
            "scheme": self.scheme.to_dict(),
            "prominence": self._prominence,
            "hard": sorted(self._hard),
            "history": self.history,
        }
