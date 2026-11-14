"""The user_id is a tenancy key, never a speaker: what reaches the prompts and the graph."""

from __future__ import annotations

import pytest
from mock_llm import MockEmbedder
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.core.prompts import PROMPTS_VERSION


def _manager(outputs: list[dict], prompts: list[str]) -> MemoryManager:
    queue = list(outputs)

    def handler(messages, info):
        prompts.append(" ".join(str(getattr(p, "content", "")) for m in messages for p in m.parts))
        return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=queue.pop(0))])

    config = MemoryConfig(user_id="grafeo", embedding_dimensions=16)
    return MemoryManager(FunctionModel(function=handler), config, embedder=MockEmbedder(16))


EXTRACTION = {
    "facts": ["Alice works at Acme Corp"],
    "entities": [
        {"name": "Alice", "entity_type": "person"},
        {"name": "Acme Corp", "entity_type": "organization"},
    ],
    "relations": [{"source": "Alice", "target": "Acme Corp", "relation_type": "works_at"}],
}


def test_the_user_id_never_reaches_a_prompt():
    prompts: list[str] = []
    with _manager([EXTRACTION], prompts) as memory:
        memory.add("Alice works at Acme Corp")
    assert prompts and all('"grafeo"' not in p for p in prompts)


def test_speaker_and_source_reach_the_extraction_prompt():
    prompts: list[str] = []
    with _manager([EXTRACTION, EXTRACTION], prompts) as memory:
        memory.add("I work at Acme Corp", speaker="Alice")
        memory.add("Alice works at Acme Corp", user_id="other", source="notes")
    assert "The speaker is Alice" in prompts[0]
    assert "notes or a document" in prompts[1]


def test_an_unknown_source_is_refused():
    with _manager([], []) as memory, pytest.raises(ValueError, match="source"):
        memory.add("text", source="email")


def test_entities_keep_the_key_as_name_and_gain_a_display_name():
    with _manager([EXTRACTION], []) as memory:
        memory.add("Alice works at Acme Corp")
        nodes = {p["name"]: p for _, p in memory._db.get_nodes_by_label("Entity")}
    assert nodes["acme_corp"]["display_name"] == "Acme Corp"
    assert nodes["alice"]["display_name"] == "Alice"


def test_stats_report_the_prompt_version():
    with _manager([], []) as memory:
        assert memory.stats().prompts_version == PROMPTS_VERSION
