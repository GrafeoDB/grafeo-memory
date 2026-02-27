"""Tests for the extraction pipeline."""

from mock_llm import make_error_model, make_test_model

from grafeo_memory.extraction import extract, extract_entities, extract_facts
from grafeo_memory.types import Fact


def test_extract_facts():
    model = make_test_model(
        [
            {"facts": ["alice works at acme corp", "alice is a data scientist"]},
        ]
    )
    facts = extract_facts(model, "I work at Acme Corp as a data scientist", "alice")
    assert len(facts) == 2
    assert facts[0].text == "alice works at acme corp"
    assert facts[1].text == "alice is a data scientist"


def test_extract_facts_empty():
    model = make_test_model([{"facts": []}])
    facts = extract_facts(model, "Hello there", "alice")
    assert facts == []


def test_extract_facts_error():
    model = make_error_model()
    facts = extract_facts(model, "test", "alice")
    assert facts == []


def test_extract_entities():
    model = make_test_model(
        [
            {
                "entities": [
                    {"name": "alice", "entity_type": "person"},
                    {"name": "acme_corp", "entity_type": "organization"},
                ],
                "relations": [
                    {"source": "alice", "target": "acme_corp", "relation_type": "works_at"},
                ],
            },
        ]
    )
    facts = [Fact("alice works at acme corp")]
    entities, relations = extract_entities(model, facts, "alice")
    assert len(entities) == 2
    assert entities[0].name == "alice"
    assert entities[1].entity_type == "organization"
    assert len(relations) == 1
    assert relations[0].relation_type == "works_at"


def test_extract_entities_empty_facts():
    model = make_test_model([])  # No responses needed — short-circuits
    entities, relations = extract_entities(model, [], "alice")
    assert entities == []
    assert relations == []


def test_full_extract():
    model = make_test_model(
        [
            # Single combined call: facts + entities + relations
            {
                "facts": ["bob lives in new york"],
                "entities": [
                    {"name": "bob", "entity_type": "person"},
                    {"name": "new_york", "entity_type": "location"},
                ],
                "relations": [
                    {"source": "bob", "target": "new_york", "relation_type": "lives_in"},
                ],
            },
        ]
    )
    result = extract(model, "Bob lives in New York", "bob")
    assert len(result.facts) == 1
    assert len(result.entities) == 2
    assert len(result.relations) == 1


def test_full_extract_error():
    model = make_error_model()
    result = extract(model, "test", "alice")
    assert result.facts == []
    assert result.entities == []
    assert result.relations == []
