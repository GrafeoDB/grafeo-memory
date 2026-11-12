"""Tests for the reconciliation pipeline."""

from mock_llm import make_error_model, make_test_model

from grafeo_memory.common.types import Fact, MemoryAction, Relation
from grafeo_memory.core.reconciliation import reconcile, reconcile_relations


def test_reconcile_no_existing():
    """With no existing memories, all facts should be ADD."""
    model = make_test_model([])  # No LLM call needed, fast path
    facts = [Fact("alice works at acme"), Fact("alice is a data scientist")]
    decisions = reconcile(model, facts, [])
    assert len(decisions) == 2
    assert all(d.action == MemoryAction.ADD for d in decisions)


def test_reconcile_with_updates():
    model = make_test_model(
        [
            {
                "decisions": [
                    {"action": "UPDATE", "target_memory_id": "100", "text": "alice works at beta inc"},
                    {"action": "NONE"},
                ]
            },
        ]
    )
    facts = [Fact("alice works at beta inc"), Fact("alice likes hiking")]
    existing = [
        {"id": "100", "text": "alice works at acme corp", "score": 0.15},
    ]
    decisions = reconcile(model, facts, existing)
    assert len(decisions) == 2
    assert decisions[0].action == MemoryAction.UPDATE
    assert decisions[0].target_memory_id == "100"
    assert decisions[0].text == "alice works at beta inc"
    assert decisions[1].action == MemoryAction.NONE


def test_reconcile_with_delete():
    model = make_test_model(
        [
            {
                "decisions": [
                    {"action": "DELETE", "target_memory_id": "200"},
                    {"action": "ADD", "text": "alice works at beta inc"},
                ]
            },
        ]
    )
    facts = [Fact("alice left acme"), Fact("alice joined beta inc")]
    existing = [
        {"id": "200", "text": "alice works at acme corp", "score": 0.12},
    ]
    decisions = reconcile(model, facts, existing)
    assert len(decisions) == 2
    assert decisions[0].action == MemoryAction.DELETE
    assert decisions[0].target_memory_id == "200"
    assert decisions[1].action == MemoryAction.ADD


def test_reconcile_error_falls_back_to_add():
    model = make_error_model()
    facts = [Fact("test fact")]
    existing = [{"id": "1", "text": "something", "score": 0.3}]
    decisions = reconcile(model, facts, existing)
    assert len(decisions) == 1
    assert decisions[0].action == MemoryAction.ADD


def test_reconcile_empty_facts():
    model = make_test_model([])
    decisions = reconcile(model, [], [])
    assert decisions == []


# --- Relation reconciliation ---


def test_reconcile_relations_contradiction():
    """A job change should delete the old works_at relation."""
    model = make_test_model(
        [
            {
                "delete": [
                    {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
                ]
            },
        ]
    )
    new_rels = [Relation(source="alice", target="beta_inc", relation_type="works_at")]
    existing = [
        {"source": "alice", "target": "acme_corp", "relation_type": "works_at", "edge_id": 10},
    ]
    to_delete = reconcile_relations(model, new_rels, existing)
    assert len(to_delete) == 1
    assert to_delete[0]["source"] == "alice"
    assert to_delete[0]["target"] == "acme_corp"


def test_reconcile_relations_coexistence():
    """Likes pizza + likes sushi should both survive."""
    model = make_test_model(
        [
            {"delete": []},
        ]
    )
    new_rels = [Relation(source="alice", target="sushi", relation_type="likes")]
    existing = [
        {"source": "alice", "target": "pizza", "relation_type": "likes", "edge_id": 20},
    ]
    to_delete = reconcile_relations(model, new_rels, existing)
    assert to_delete == []


def test_reconcile_relations_no_existing():
    """No existing relations -> nothing to delete, no LLM call."""
    model = make_test_model([])  # No responses, would fail if called
    new_rels = [Relation(source="alice", target="acme", relation_type="works_at")]
    to_delete = reconcile_relations(model, new_rels, [])
    assert to_delete == []


def test_reconcile_relations_no_new():
    """No new relations -> nothing to reconcile."""
    model = make_test_model([])
    existing = [{"source": "a", "target": "b", "relation_type": "x", "edge_id": 1}]
    to_delete = reconcile_relations(model, [], existing)
    assert to_delete == []


def test_reconcile_relations_error():
    """Error response -> empty list, no crash."""
    model = make_error_model()
    new_rels = [Relation(source="alice", target="beta", relation_type="works_at")]
    existing = [{"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 5}]
    to_delete = reconcile_relations(model, new_rels, existing)
    assert to_delete == []


# --- T6: Reconciliation boundary (threshold behavior) ---


class TestReconciliationBoundary:
    """T6: verify that similar memories trigger UPDATE when above threshold."""

    def test_similar_fact_triggers_update(self):
        """When existing memory is found above threshold, reconciler should produce UPDATE."""
        model = make_test_model(
            [
                {
                    "decisions": [
                        {"action": "UPDATE", "target_memory_id": "42", "text": "alice works from home now"},
                    ]
                },
            ]
        )
        facts = [Fact("alice works from home now")]
        existing = [{"id": "42", "text": "alice loves remote work", "score": 0.85}]
        decisions = reconcile(model, facts, existing)
        assert len(decisions) == 1
        assert decisions[0].action == MemoryAction.UPDATE
        assert decisions[0].target_memory_id == "42"
        assert "home" in decisions[0].text

    def test_no_existing_always_adds(self):
        """With no existing memories above threshold, all facts become ADDs."""
        model = make_test_model([])
        facts = [Fact("alice works from home")]
        decisions = reconcile(model, facts, [])
        assert len(decisions) == 1
        assert decisions[0].action == MemoryAction.ADD

    def test_mixed_add_and_update(self):
        """Some facts update existing memories, others are new ADDs."""
        model = make_test_model(
            [
                {
                    "decisions": [
                        {"action": "UPDATE", "target_memory_id": "10", "text": "alice works from home"},
                        {"action": "ADD", "text": "alice has a cat named whiskers"},
                    ]
                },
            ]
        )
        facts = [Fact("alice works from home"), Fact("alice has a cat named whiskers")]
        existing = [{"id": "10", "text": "alice works at acme office", "score": 0.7}]
        decisions = reconcile(model, facts, existing)
        assert len(decisions) == 2
        assert decisions[0].action == MemoryAction.UPDATE
        assert decisions[1].action == MemoryAction.ADD
