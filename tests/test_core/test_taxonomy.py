"""Tests for Layer B taxonomy: roots, versioning, prominence scopes, routing."""

from __future__ import annotations

import pytest

from grafeo_memory.core.taxonomy import DEFAULT_ROOTS, InstanceScopeConfig, Taxonomy, TaxonomyConfig


def _tax(**kw) -> Taxonomy:
    cfg = TaxonomyConfig(enabled=True, **kw)
    return Taxonomy(cfg)


class TestRoots:
    def test_default_roots_shipped(self):
        t = _tax()
        ids = [r.id for r in DEFAULT_ROOTS]
        assert ids[0] == "general" and len(ids) == 10
        assert all(t.status[r.id] == "shipped" for r in DEFAULT_ROOTS)

    def test_disabled_is_single_general_scope(self):
        t = Taxonomy(TaxonomyConfig(enabled=False))
        assert t.route("anything at all").subject_paths == ["general"]
        assert t.route("anything").is_global is False  # general is the single scope

    def test_add_rename_retire_versioned(self):
        t = _tax()
        t.add_root("work", "Work stuff")
        assert t.status["work"] == "user" and t.version == 1
        t.rename_root("work", "Job stuff")
        assert t.scheme.lookup_pref("Job stuff", "Scope") == "work"
        assert t.scheme.lookup_pref("Work stuff", "Scope") is None
        # attach a child, then retire: child re-parents to general, nothing deleted
        t.scheme.add_concept("work.apollo", "Project Apollo", "Scope", broader="work")
        entry = t.retire_root("work", fallback="general")
        assert "work.apollo" in entry["migrated"]
        assert t.scheme.concepts["work.apollo"].broader == "general"
        assert "work.apollo" in t.scheme.concepts  # never deleted
        assert t.version == 3 and len(t.history) == 3


class TestInstanceScopes:
    def test_prominence_gates_activation(self):
        t = _tax(instance_scopes=InstanceScopeConfig(prominence_threshold=3))
        assert t.active_instance_scopes() == []
        t.record_mention("acme")
        t.record_mention("acme")
        assert t.active_instance_scopes() == []
        t.record_mention("acme")
        assert t.active_instance_scopes() == ["acme"]

    def test_edges_count_toward_prominence(self):
        t = _tax(instance_scopes=InstanceScopeConfig(prominence_threshold=3))
        t.record_mention("acme", edges=2)
        assert t.active_instance_scopes() == ["acme"]

    def test_hard_soft(self):
        t = _tax()
        assert not t.is_hard("acme")
        t.set_hard("acme")
        assert t.is_hard("acme")
        t.set_hard("acme", False)
        assert not t.is_hard("acme")


class TestRouting:
    def test_lexical_subject_match(self):
        t = _tax()
        t.scheme.add_concept("career.interviewing", "Interviewing", "Scope", broader="career")
        ss = t.route("any career advice?")
        assert "career" in ss.subject_paths
        assert "career.interviewing" in ss.subject_paths  # expanded down from the hit

    def test_instance_scope_activates_on_mention(self):
        t = _tax(instance_scopes=InstanceScopeConfig(prominence_threshold=2))
        t.record_mention("acme")
        t.record_mention("acme")
        ss = t.route("what did we decide at acme")
        assert "acme" in ss.instance_scopes

    def test_post_coordination_combinations(self):
        t = _tax(instance_scopes=InstanceScopeConfig(prominence_threshold=2))
        t.scheme.add_concept("career.interviewing", "Interviewing", "Scope", broader="career")
        t.record_mention("acme")
        t.record_mention("acme")
        ss = t.route("career interview prep at acme")
        combos = ss.combinations()
        assert any(s.startswith("career") and i == "acme" for s, i in combos)

    def test_scorer_fallback_parent_first(self):
        t = _tax()
        t.scheme.add_concept("health.running", "Running", "Scope", broader="health")

        def scorer(query, ids):
            return [(0.9 if i == "health" else 0.1, i) for i in ids]

        ss = t.route("unseen phrasing about wellness", scorer=scorer)
        assert "health" in ss.subject_paths
        # child included via expansion of the clearing parent
        assert "health.running" in ss.subject_paths

    def test_no_match_is_general(self):
        t = _tax()
        assert t.route("zzz qqq").subject_paths == ["general"]

    def test_max_scopes_cap(self):
        t = _tax()
        for i in range(10):
            t.scheme.add_concept(f"r{i}", f"thing {i}", "Scope")

        def scorer(query, ids):
            return [(0.99, i) for i in ids]

        ss = t.route("anything", scorer=scorer)
        assert len(ss.subject_paths) <= t.config.max_scopes_per_query

    def test_serialization(self):
        t = _tax()
        t.record_mention("acme")
        d = t.to_dict()
        assert d["prominence"] == {"acme": 1} and d["scheme"]["scheme_id"] == "subjects"


class TestRootEditingAndRouting:
    """0.3.0 review regressions."""

    def test_routing_matches_whole_words(self):
        t = _tax()
        assert "home" not in t.route_subject("help with my homework")
        assert "home" in t.route_subject("tidying my home")

    def test_renamed_root_routes_by_new_and_old_label(self):
        t = _tax()
        t.add_root("work", "Work stuff")
        t.rename_root("work", "Job stuff")
        assert "work" in t.route_subject("some job stuff question")
        assert "work" in t.route_subject("some work stuff question")  # grace period

    def test_retired_root_stops_routing_and_is_no_top_concept(self):
        t = _tax()
        t.add_root("work", "Work stuff")
        t.retire_root("work", fallback="general")
        assert "work" not in t.route_subject("work stuff")
        assert "work" not in t.scheme._children.get(None, [])

    def test_add_root_rejects_an_existing_id(self):
        t = _tax()
        t.add_root("work", "Work stuff")
        with pytest.raises(ValueError):
            t.add_root("work", "Other label")
        assert t.scheme.concepts["work"].pref_label == "Work stuff"
        assert t.scheme._children[None].count("work") == 1 and t.version == 1

    def test_retire_root_rejects_a_bad_fallback(self):
        t = _tax()
        t.add_root("work", "Work stuff")
        t.add_child("work", "work.apollo", "Apollo")
        for fallback in ("work", "missing", "work.apollo"):
            with pytest.raises(ValueError):
                t.retire_root("work", fallback=fallback)
        with pytest.raises(ValueError):
            t.retire_root("general")  # default fallback is general itself
        assert "work" in t.scheme.concepts and t.version == 1  # nothing changed

    def test_scorer_expansion_does_not_repeat_lexical_hits(self):
        t = _tax(max_scopes_per_query=10)
        t.add_child("career", "career.apollo", "Apollo")
        paths = t.route_subject("apollo update", scorer=lambda q, ids: [(0.9, "career")])
        assert len(paths) == len(set(paths)) and set(paths) >= {"career", "career.apollo"}
