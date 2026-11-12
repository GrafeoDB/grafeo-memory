"""Tests for the Layer D ontology package: scheme, controlled types, ladder, temporal rules."""

from __future__ import annotations

from datetime import UTC, datetime

from grafeo_memory.common.intervals import Interval, parse_iso
from grafeo_memory.core.ontology import (
    ConceptScheme,
    EntityType,
    RelationType,
    cosine,
    normalize,
    normalize_key,
    point_in_time_filter,
    pre_model_conflict,
    validate_relation,
)


def _scheme() -> ConceptScheme:
    s = ConceptScheme()
    s.add_concept("career", "Career", EntityType.CONCEPT)
    s.add_concept("career.interviewing", "Interviewing", EntityType.CONCEPT, broader="career")
    s.add_concept("career.engineering", "Engineering", EntityType.CONCEPT, broader="career")
    s.add_concept("js", "JavaScript", EntityType.TECHNOLOGY, alt_labels=("JS", "ECMAScript"), notation="JS")
    s.add_concept("ts", "TypeScript", EntityType.TECHNOLOGY, alt_labels=("TS",), broader=None)
    s.add_concept("acme", "Acme Corp", EntityType.ORGANIZATION, alt_labels=("Acme",))
    s.add_concept("alice", "Alice", EntityType.PERSON)
    return s


class TestScheme:
    def test_rung1_exact_pref_typed(self):
        s = _scheme()
        assert s.lookup_pref("JavaScript", "Technology") == "js"
        assert s.lookup_pref("JavaScript", "Person") is None
        assert s.lookup_pref("javascript") is None  # exact is case-sensitive

    def test_rung2_alias(self):
        s = _scheme()
        assert s.lookup_alias("JS") == "js"
        assert s.lookup_alias("ecmascript") == "js"
        assert s.lookup_alias("acme corp") == "acme"
        assert s.lookup_alias("unknown thing") is None

    def test_normalize_key(self):
        assert normalize_key("  Acme,   Corp! ") == "acme corp"
        assert normalize_key("JS") == "js"

    def test_hierarchy_expansion(self):
        s = _scheme()
        assert set(s.narrower("career", depth=1)) == {"career.interviewing", "career.engineering"}
        assert s.narrower("career", depth=0) == []
        assert s.broader_path("career.interviewing") == ["career"]
        assert set(s.expand(["career"], depth=1, direction="down")) == {
            "career",
            "career.interviewing",
            "career.engineering",
        }
        assert s.expand(["career.interviewing"], direction="up") == ["career.interviewing", "career"]

    def test_broader_must_exist(self):
        s = ConceptScheme()
        try:
            s.add_concept("x", "X", EntityType.CONCEPT, broader="missing")
        except KeyError:
            return
        raise AssertionError("expected KeyError")

    def test_serialization_roundtrip(self):
        s = _scheme()
        back = ConceptScheme.from_dict(s.to_dict())
        assert back.lookup_alias("JS") == "js"
        assert back.lookup_pref("JavaScript", "Technology") == "js"
        assert set(back.narrower("career")) == {"career.interviewing", "career.engineering"}


class TestControlledTypes:
    def test_domain_range_ok(self):
        assert validate_relation(RelationType.WORKS_AT, EntityType.PERSON, EntityType.ORGANIZATION)
        assert validate_relation("WORKS_AT", "Person", "Organization")
        assert validate_relation(RelationType.PREFERS, "Person", "Location")
        assert validate_relation(RelationType.RELATED_TO, "Event", "Project")

    def test_domain_range_rejects(self):
        assert not validate_relation(RelationType.WORKS_AT, EntityType.ORGANIZATION, EntityType.PERSON)
        assert not validate_relation(RelationType.LIVES_IN, "Person", "Organization")
        assert not validate_relation(RelationType.ATTENDED, "Organization", "Event")
        assert not validate_relation(RelationType.KNOWS, "Person", "Technology")

    def test_unknown_type_fails_closed(self):
        assert not validate_relation(RelationType.WORKS_AT, "Robot", "Organization")


class TestNormalizationLadder:
    def test_rung1_and_rung2_zero_model(self):
        s = _scheme()
        r = normalize("JavaScript", EntityType.TECHNOLOGY, s)
        assert r.rung == 1 and r.canonical_id == "js" and r.resolved
        r = normalize("JS", EntityType.TECHNOLOGY, s)
        assert r.rung == 2 and r.canonical_id == "js"

    def test_rung3_embedding_above_threshold(self):
        s = _scheme()
        # Fake embedder: hash-free deterministic toy, "java script" close to "JavaScript".
        vectors = {"JavaScript": [1.0, 0.0], "TypeScript": [0.0, 1.0]}
        r = normalize(
            "java script",
            EntityType.TECHNOLOGY,
            s,
            embed=lambda t: vectors.get(t, [0.9, 0.1]),
            concept_vectors={"js": [1.0, 0.0], "ts": [0.0, 1.0]},
            threshold=0.9,
        )
        assert r.rung == 3 and r.canonical_id == "js" and r.confidence >= 0.9

    def test_rung4_residue_returns_candidates(self):
        s = _scheme()
        vectors = {"javascrpt": [0.5, 0.5]}
        r = normalize(
            "javascrpt",
            EntityType.TECHNOLOGY,
            s,
            embed=lambda t: vectors.get(t, [0.5, 0.5]),
            concept_vectors={"js": [1.0, 0.0], "ts": [0.0, 1.0]},
            threshold=0.99,
        )
        assert r.rung == 4 and not r.resolved and set(r.candidates) == {"js", "ts"}

    def test_rung4_without_embedder(self):
        s = _scheme()
        r = normalize("something new", EntityType.TECHNOLOGY, s)
        assert r.rung == 4 and r.candidates == ["js", "ts"]

    def test_cosine(self):
        assert cosine([1, 0], [1, 0]) == 1.0
        assert cosine([1, 0], [0, 1]) == 0.0
        assert cosine([1, 0], [2, 0]) == 1.0
        assert cosine([], []) == 0.0


class TestTemporalRules:
    def _iv(self, s: str, e: str | None) -> Interval:
        b = parse_iso(s)
        assert b is not None
        end = parse_iso(e) if e else None
        return Interval(start=b, end=end)

    def test_disjoint_never_conflict(self):
        acme = self._iv("2020-01-01", "2022-01-01")
        beta = self._iv("2022-01-01", None)
        assert not pre_model_conflict(acme, beta)

    def test_overlapping_may_conflict(self):
        a = self._iv("2020-01-01", "2023-01-01")
        b = self._iv("2022-01-01", None)
        assert pre_model_conflict(a, b)

    def test_unknown_time_reaches_model(self):
        a = self._iv("2020-01-01", "2021-01-01")
        assert pre_model_conflict(a, None)
        assert pre_model_conflict(None, None)

    def test_point_in_time_filter(self):
        items = [
            ("acme-job", self._iv("2020-01-01", "2022-01-01")),
            ("beta-job", self._iv("2022-01-01", None)),
            ("unknown-job", None),
        ]
        at_2021 = point_in_time_filter(items, datetime(2021, 6, 1, tzinfo=UTC))
        assert {"acme-job", "unknown-job"} == set(at_2021)
        at_2023 = point_in_time_filter(items, datetime(2023, 6, 1, tzinfo=UTC))
        assert {"beta-job", "unknown-job"} == set(at_2023)


class TestIsViolation:
    """Lenient violation predicate for the executor gate: only known-vocabulary
    relations with known endpoint types can violate; free text passes through."""

    def test_known_violation(self):
        from grafeo_memory.core.ontology import is_violation

        assert is_violation("WORKS_AT", "Organization", "Person")
        assert is_violation("LIVES_IN", "Person", "Organization")
        assert not is_violation("WORKS_AT", "Person", "Organization")
        assert not is_violation("PREFERS", "Person", "Location")

    def test_free_text_relation_is_not_a_violation(self):
        from grafeo_memory.core.ontology import is_violation

        assert not is_violation("likes", "Person", "Person")
        assert not is_violation("works at", "Person", "Organization")

    def test_unknown_entity_type_is_not_a_violation(self):
        from grafeo_memory.core.ontology import is_violation

        assert not is_violation("WORKS_AT", "Robot", "Organization")
        assert not is_violation("WORKS_AT", "", "")


def test_manager_normalization_renames_relation_endpoints_too():
    """0.3.0 review: a normalized entity name must also rename the relation endpoints,
    or the relation points at a name no entity carries and is never written."""
    from conftest import make_manager

    from grafeo_memory import PipelineConfig
    from grafeo_memory.common.types import Entity, ExtractionResult, Relation

    manager = make_manager([], pipeline=PipelineConfig(ontology=True))
    try:
        manager._db.create_node(["Entity"], {"name": "Alice", "entity_type": "Person", "user_id": "test_user"})
        extraction = ExtractionResult(
            facts=[],
            entities=[Entity(name="alice", entity_type="Person"), Entity(name="Acme", entity_type="Organization")],
            relations=[Relation(source="alice", target="Acme", relation_type="WORKS_AT")],
        )
        manager._normalize_entity_names(extraction, "test_user")
        assert extraction.entities[0].name == "Alice"
        assert (extraction.relations[0].source, extraction.relations[0].target) == ("Alice", "Acme")
    finally:
        manager.close()


def test_types_are_case_insensitive():
    """0.3.0 review: LLM output such as "person" / "works_at" is checked like "Person" / "WORKS_AT"."""
    from grafeo_memory.core.ontology.types import (
        EntityType,
        RelationType,
        coerce_entity_type,
        coerce_relation_type,
        is_violation,
    )

    assert coerce_entity_type("person") is EntityType.PERSON
    assert coerce_relation_type("works at") is RelationType.WORKS_AT
    assert coerce_relation_type("likes_hiking") is None
    assert is_violation("works_at", "person", "location")
    assert not is_violation("works_at", "person", "organization")


def test_alias_lookup_respects_entity_type():
    from grafeo_memory.core.ontology import ConceptScheme, normalize

    scheme = ConceptScheme()
    scheme.add_concept(id="apple", pref_label="apple", entity_type="Preference")
    assert normalize("Apple", "Organization", scheme).canonical_id is None
    assert normalize("APPLE", "preference", scheme).canonical_id == "apple"


def test_compile_merge_is_remembered_by_write_time_normalization():
    """After compile() merges "alice anderson" into "Alice Anderson", a later mention of the
    alias resolves to the survivor instead of recreating the duplicate."""
    from conftest import make_manager

    from grafeo_memory import PipelineConfig
    from grafeo_memory.common.types import Entity, ExtractionResult
    from grafeo_memory.services.compile import compile_graph

    class _SameVec:
        dimensions = 4

        def embed(self, texts):
            return [[1.0, 0.0, 0.0, 0.0] for _ in texts]

    manager = make_manager([], pipeline=PipelineConfig(ontology=True))
    try:
        db = manager._db
        db.create_node(["Entity"], {"name": "Alice Anderson", "entity_type": "Person", "user_id": "test_user"})
        db.create_node(["Entity"], {"name": "A. Anderson", "entity_type": "Person", "user_id": "test_user"})
        res = compile_graph(db, "test_user", embedder=_SameVec(), merge_threshold=0.9)
        assert len(res.merges) == 1
        extraction = ExtractionResult(facts=[], entities=[Entity(name="a. anderson", entity_type="person")])
        manager._normalize_entity_names(extraction, "test_user")
        assert extraction.entities[0].name == "Alice Anderson"
    finally:
        manager.close()
