"""Tests for the extraction pipeline."""

from typing import ClassVar

from mock_llm import make_error_model, make_test_model

from grafeo_memory.common.types import Fact
from grafeo_memory.core.extraction import extract, extract_entities, extract_facts


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
    entities, relations = extract_entities(model, facts)
    assert len(entities) == 2
    assert entities[0].name == "alice"
    assert entities[1].entity_type == "organization"
    assert len(relations) == 1
    assert relations[0].relation_type == "works_at"


def test_extract_entities_empty_facts():
    model = make_test_model([])  # No responses needed, short-circuits
    entities, relations = extract_entities(model, [])
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


def test_full_extract_fallback_to_separate_calls():
    """When combined extraction fails, fall back to separate fact + entity extraction."""
    from mock_llm import make_error_then_succeed_model

    model = make_error_then_succeed_model(
        [
            # Call 1 (index 0) errors (combined extraction)
            # Call 2 (index 1) → fact extraction
            {"facts": ["alice works at acme"]},
            # Call 3 (index 2) → entity extraction
            {
                "entities": [{"name": "alice", "entity_type": "person"}],
                "relations": [],
            },
        ]
    )
    result = extract(model, "Alice works at Acme Corp", "alice")
    assert len(result.facts) == 1
    assert result.facts[0].text == "alice works at acme"
    assert len(result.entities) == 1
    assert result.entities[0].name == "alice"


def test_mistral_fallback_no_stderr_traceback(capsys):
    """T10: combined extraction failure should not print tracebacks to stderr.

    Mistral (and other providers) may fail on the combined extraction schema.
    The fallback to separate calls should work silently without noisy output.
    """
    from mock_llm import make_error_then_succeed_model

    model = make_error_then_succeed_model(
        [
            # Call 1 (index 0) errors (combined extraction)
            # Call 2 (index 1) -> fact extraction
            {"facts": ["bob likes swimming"]},
            # Call 3 (index 2) -> entity extraction
            {
                "entities": [{"name": "bob", "entity_type": "person"}],
                "relations": [],
            },
        ]
    )
    result = extract(model, "Bob likes swimming", "bob")

    # Verify fallback succeeded
    assert len(result.facts) == 1
    assert result.facts[0].text == "bob likes swimming"

    # Verify no traceback was printed to stderr
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "Error" not in captured.err


def _capturing_model(output: dict, seen: list[str]):
    from pydantic_ai.messages import ModelResponse, ToolCallPart
    from pydantic_ai.models.function import FunctionModel

    def handler(messages, info):
        seen.append(" ".join(str(getattr(p, "content", "")) for m in messages for p in m.parts))
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=output)])

    return FunctionModel(function=handler)


class TestIdentityInPrompts:
    """The user_id is a tenancy key: it never reaches a prompt, and the speaker comes from the input."""

    OUT: ClassVar[dict] = {"facts": ["the user likes hiking"], "entities": [], "relations": []}

    def test_default_speaker_is_the_user(self):
        seen: list[str] = []
        extract(_capturing_model(self.OUT, seen), "I like hiking")
        assert '"the user"' in seen[0]
        assert "user_id" not in seen[0]

    def test_named_speaker(self):
        seen: list[str] = []
        extract(_capturing_model(self.OUT, seen), "I like hiking", "Caroline")
        assert "The speaker is Caroline" in seen[0]

    def test_notes_have_no_speaker(self):
        seen: list[str] = []
        extract(_capturing_model(self.OUT, seen), "grafeo commits need a signed tag", source="notes")
        assert "notes or a document" in seen[0]
        assert "the user" not in seen[0]

    def test_query_entity_extraction_has_no_identity(self):
        seen: list[str] = []
        extract_entities(_capturing_model({"entities": [], "relations": []}, seen), [Fact("where is pied piper")])
        assert "user_id" not in seen[0]


class TestEntityKeys:
    def test_names_as_written_become_keys_with_a_display_name(self):
        model = make_test_model(
            [
                {
                    "facts": ["Gavin Belson runs Hooli"],
                    "entities": [
                        {"name": "Gavin Belson", "entity_type": "person"},
                        {"name": "hooli", "entity_type": "organization"},
                    ],
                    "relations": [{"source": "Gavin Belson", "target": "Hooli", "relation_type": "runs"}],
                }
            ]
        )
        result = extract(model, "Gavin Belson runs Hooli")
        assert [(e.name, e.display_name) for e in result.entities] == [
            ("gavin_belson", "Gavin Belson"),
            ("hooli", None),
        ]
        assert (result.relations[0].source, result.relations[0].target) == ("gavin_belson", "hooli")

    def test_entity_key_is_idempotent(self):
        from grafeo_memory.core.extraction.entities import entity_key

        assert entity_key("Acme Corp") == entity_key("acme_corp") == entity_key(" Acme  Corp. ") == "acme_corp"
        assert entity_key(entity_key("Gavin Belson")) == "gavin_belson"

    def test_distinct_names_keep_distinct_keys(self):
        from grafeo_memory.core.extraction.entities import entity_key

        assert len({entity_key(n) for n in ("C++", "C#", "C", "node.js", "AT&T")}) == 5
