"""SKOS-style concept scheme: the subject taxonomy for scoping (Layer B/D).

Concepts carry ``prefLabel`` (canonical), ``altLabel`` synonyms and ``notation``
codes; hierarchy via broader/narrower with ``topConceptOf`` roots. A
deterministic alias index is consulted before any model call (normalization
ladder rungs 1-2). Router choices are generated from the scheme, so concept
routing is deterministic code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def normalize_key(text: str) -> str:
    """Deterministic alias key: lowercase, non-alphanumerics to single spaces."""
    return _NON_ALNUM.sub(" ", text.lower()).strip()


@dataclass
class Concept:
    id: str
    pref_label: str
    entity_type: str
    alt_labels: tuple[str, ...] = ()
    notation: str | None = None
    broader: str | None = None  # parent concept id (skos:broader)
    related: tuple[str, ...] = ()  # skos:related
    exact_match: tuple[str, ...] = ()  # external identities kept by aliasing

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "pref_label": self.pref_label,
            "entity_type": self.entity_type,
            "alt_labels": list(self.alt_labels),
            "notation": self.notation,
            "broader": self.broader,
            "related": list(self.related),
            "exact_match": list(self.exact_match),
        }

    @classmethod
    def from_dict(cls, d: dict) -> Concept:
        return cls(
            id=d["id"],
            pref_label=d["pref_label"],
            entity_type=d["entity_type"],
            alt_labels=tuple(d.get("alt_labels", ())),
            notation=d.get("notation"),
            broader=d.get("broader"),
            related=tuple(d.get("related", ())),
            exact_match=tuple(d.get("exact_match", ())),
        )


@dataclass
class ConceptScheme:
    """A skos:ConceptScheme with deterministic lookup and capped-depth expansion."""

    scheme_id: str = "memory"
    concepts: dict[str, Concept] = field(default_factory=dict)
    _children: dict[str | None, list[str]] = field(default_factory=dict)
    # normalized alias -> concept ids in insertion order (one alias may name concepts of several types)
    _alias_index: dict[str, list[str]] = field(default_factory=dict)
    _pref_index: dict[tuple[str, str], str] = field(default_factory=dict)  # (pref_label, entity_type) -> id

    def add_concept(
        self,
        id: str,
        pref_label: str,
        entity_type: str,
        alt_labels: tuple[str, ...] | list[str] = (),
        notation: str | None = None,
        broader: str | None = None,
        related: tuple[str, ...] | list[str] = (),
        exact_match: tuple[str, ...] | list[str] = (),
    ) -> Concept:
        """Add a concept; re-adding an existing id replaces it and its index entries."""
        if broader is not None and broader not in self.concepts:
            raise KeyError(f"broader concept {broader!r} not in scheme")
        c = Concept(
            id=id,
            pref_label=pref_label,
            entity_type=entity_type,
            alt_labels=tuple(alt_labels),
            notation=notation,
            broader=broader,
            related=tuple(related),
            exact_match=tuple(exact_match),
        )
        self.put(c)
        return c

    def put(self, c: Concept) -> None:
        """Insert or replace ``c`` with all its index entries (no broader check)."""
        if c.id in self.concepts:
            self._unindex(c.id)
        self.concepts[c.id] = c
        self._children.setdefault(c.broader, []).append(c.id)
        self._pref_index[(c.pref_label, c.entity_type)] = c.id
        for alias in (c.id, *c.alt_labels, c.notation or "", *c.exact_match, c.pref_label):
            self.add_alias(c.id, alias)

    def add_alias(self, cid: str, alias: str) -> None:
        """Index ``alias`` for ``cid``. Aliases without a letter or digit a key can hold are skipped."""
        key = normalize_key(alias) if alias else ""
        if not key:
            return
        ids = self._alias_index.setdefault(key, [])
        if cid not in ids:
            ids.append(cid)

    def remove_concept(self, cid: str) -> Concept | None:
        """Drop ``cid`` and its index entries. Its own children list is left to the caller."""
        if cid not in self.concepts:
            return None
        self._unindex(cid)
        return self.concepts.pop(cid)

    def _unindex(self, cid: str) -> None:
        old = self.concepts[cid]
        siblings = self._children.get(old.broader)
        if siblings is not None and cid in siblings:
            siblings.remove(cid)
        if self._pref_index.get((old.pref_label, old.entity_type)) == cid:
            del self._pref_index[(old.pref_label, old.entity_type)]
        for key in [k for k, ids in self._alias_index.items() if cid in ids]:
            self._alias_index[key].remove(cid)
            if not self._alias_index[key]:
                del self._alias_index[key]

    def alias_hits(self, text: str) -> set[str]:
        """Concepts whose alias occurs in ``text`` as whole words (first concept per alias)."""
        qn = f" {normalize_key(text)} "
        return {ids[0] for alias, ids in self._alias_index.items() if f" {alias} " in qn}

    def lookup_pref(self, label: str, entity_type: str | None = None) -> str | None:
        """Rung 1: exact prefLabel match, optionally typed by entity_type."""
        if entity_type is not None:
            hit = self._pref_index.get((label, entity_type))
            if hit is not None:
                return hit
        for (pl, _et), cid in self._pref_index.items():
            if pl == label and entity_type is None:
                return cid
        return None

    def lookup_alias(self, text: str, entity_type: str | None = None) -> str | None:
        """Rung 2: normalized altLabel / notation / exactMatch / prefLabel. With
        ``entity_type``, only a concept of that type matches ("Apple" the company
        never resolves to "apple" the preference)."""
        ids = self._alias_index.get(normalize_key(text), [])
        if entity_type is None:
            return ids[0] if ids else None
        for cid in ids:
            c = self.concepts.get(cid)
            if c is not None and (not c.entity_type or c.entity_type == entity_type):
                return cid
        return None

    def narrower(self, cid: str, depth: int = 2) -> list[str]:
        """Descendants of cid, capped at depth levels (deterministic order)."""
        out: list[str] = []
        frontier = [cid]
        for _ in range(max(0, depth)):
            if not frontier:
                break
            nxt: list[str] = []
            for node in frontier:
                for child in self._children.get(node, ()):
                    out.append(child)
                    nxt.append(child)
            frontier = nxt
        return out

    def broader_path(self, cid: str) -> list[str]:
        """Ancestors of cid, nearest first, up to the root (topConceptOf)."""
        out: list[str] = []
        node = self.concepts.get(cid)
        while node is not None and node.broader is not None:
            out.append(node.broader)
            node = self.concepts.get(node.broader)
        return out

    def expand(self, cids: list[str], depth: int = 2, direction: str = "down") -> list[str]:
        """SKOS expansion for retrieval: self + narrower (down) or + broader (up), ``depth`` levels."""
        out: list[str] = []
        seen: set[str] = set()
        for cid in cids:
            if cid in seen:
                continue
            seen.add(cid)
            out.append(cid)
            if direction == "down":
                for d in self.narrower(cid, depth):
                    if d not in seen:
                        seen.add(d)
                        out.append(d)
            elif direction == "up":
                for a in self.broader_path(cid)[: max(0, depth)]:
                    if a not in seen:
                        seen.add(a)
                        out.append(a)
            else:
                raise ValueError(f"direction must be 'down' or 'up', got {direction!r}")
        return out

    def to_dict(self) -> dict:
        return {"scheme_id": self.scheme_id, "concepts": [c.to_dict() for c in self.concepts.values()]}

    @classmethod
    def from_dict(cls, d: dict) -> ConceptScheme:
        scheme = cls(scheme_id=d.get("scheme_id", "memory"))
        for cd in d.get("concepts", []):
            scheme.put(Concept.from_dict(cd))
        return scheme
