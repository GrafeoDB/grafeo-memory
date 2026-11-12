"""Write memory databases with grafeo 0.5.44, for the upgrade tests (tests/test_services/test_upgrade_from_05.py).

grafeo 0.6 migrates a 0.5.x database on its first read-write open. The upgrade
tests open the databases this script writes with the current grafeo-memory and
grafeo 0.6, and compare what they read with the manifest written next to them.

It must run with grafeo 0.5.44 and a grafeo-memory 0.2.x, never with this
checkout's package. From the repository root:

- ``grafeo-memory-0.2.2/``, the release on PyPI (it keeps no history on grafeo 0.5.44)::

    uv run --no-project --with grafeo==0.5.44 --with "pydantic-ai-slim<2" --with grafeo-memory==0.2.2 \\
        python tests/fixtures/grafeo_0_5/make_fixtures.py tests/fixtures/grafeo_0_5/grafeo-memory-0.2.2

- ``grafeo-memory-0.2.3/``, the 0.2.3 code at commit 44e950d (``:History`` nodes)::

    git archive --format=tar --prefix=gm023/ 44e950d | tar -x -C <scratch>
    uv run --no-project --with grafeo==0.5.44 --with "pydantic-ai-slim<2" --with <scratch>/gm023 \\
        python tests/fixtures/grafeo_0_5/make_fixtures.py tests/fixtures/grafeo_0_5/grafeo-memory-0.2.3

Delete the ``*.spill`` directories grafeo 0.5.44 leaves next to the databases.

Writes, into the output directory:

- ``memory.grafeo``: a single-file database
- ``memory_wal/``: a WAL-directory database (a path without the ``.grafeo`` extension)
- ``manifest.json``: what each database holds, read back through grafeo-memory 0.2.x

Both hold the same data: memories of two users (added with LLM extraction and
without), entities and relations, an update and a delete with their history,
importance scores, PageRank/Louvain results and a community summary, and one
memory with history in the grafeo-memory 0.1.x layout (a ``HAS_HISTORY`` edge,
no ``memory_id``).
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import grafeo
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # tests/, for mock_llm

from mock_llm import MockEmbedder

import grafeo_memory
from grafeo_memory import MemoryConfig, MemoryManager

DIMS = 16

# What the mock LLM extracts from each message added with infer=True.
EXTRACTIONS = {
    "Alix works at Acme in Amsterdam": {
        "facts": ["alix works at acme in amsterdam"],
        "entities": [
            {"name": "alix", "entity_type": "person"},
            {"name": "acme", "entity_type": "organization"},
            {"name": "amsterdam", "entity_type": "city"},
        ],
        "relations": [
            {"source": "alix", "target": "acme", "relation_type": "works_at"},
            {"source": "acme", "target": "amsterdam", "relation_type": "located_in"},
        ],
    },
    "Alix goes hiking with Gus": {
        "facts": ["alix goes hiking with gus"],
        "entities": [
            {"name": "alix", "entity_type": "person"},
            {"name": "gus", "entity_type": "person"},
            {"name": "hiking", "entity_type": "activity"},
        ],
        "relations": [
            {"source": "alix", "target": "gus", "relation_type": "friend_of"},
            {"source": "alix", "target": "hiking", "relation_type": "enjoys"},
        ],
    },
    "Gus plays chess in Utrecht": {
        "facts": ["gus plays chess in utrecht"],
        "entities": [
            {"name": "gus", "entity_type": "person"},
            {"name": "chess", "entity_type": "activity"},
            {"name": "utrecht", "entity_type": "city"},
        ],
        "relations": [
            {"source": "gus", "target": "chess", "relation_type": "plays"},
            {"source": "gus", "target": "utrecht", "relation_type": "lives_in"},
        ],
    },
}


def _model() -> FunctionModel:
    """A mock LLM that answers by the output schema it is asked for."""
    pending: list[str] = []

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        fields = set(tool.parameters_json_schema.get("properties", {}))
        prompt = str(messages)
        if {"facts", "entities"} <= fields:
            output = next((out for text, out in EXTRACTIONS.items() if text in prompt), {"facts": []})
            pending[:] = output["facts"]
        elif "decisions" in fields:
            output = {"decisions": [{"action": "add", "text": fact} for fact in pending]}
        elif {"name", "summary"} <= fields:
            output = {"name": "Alix and friends", "summary": "Alix, Gus and the places they go."}
        elif "entities" in fields:
            output = {"entities": [], "relations": []}
        elif "delete" in fields:
            output = {"delete": []}
        elif "annotations" in fields:
            output = {"annotations": []}
        else:
            raise AssertionError(f"unexpected output schema {sorted(fields)}")
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args=output)])

    return FunctionModel(function=handler)


def _rows(db, query: str, params: dict | None = None) -> list[dict]:
    return [row for row in db.execute(query, params or {}) if isinstance(row, dict)]


def _write(path: Path) -> dict:
    embedder = MockEmbedder(DIMS)
    config = MemoryConfig(
        db_path=str(path),
        user_id="alix",
        embedding_dimensions=DIMS,
        enable_importance=True,
        enable_graph_algorithms=True,
        enable_community_summaries=True,
    )
    manager = MemoryManager(_model(), config, embedder=embedder)

    for text in EXTRACTIONS:
        manager.add(text)
    tea = manager.add("alix drinks black tea", infer=False)[0].memory_id
    manager.update(tea, "alix drinks green tea")
    gone = manager.add("alix owns a red bicycle", infer=False)[0].memory_id
    manager.delete(gone)
    rated = manager.add("alix speaks dutch and english", infer=False)[0].memory_id
    if hasattr(manager, "set_importance"):
        manager.set_importance(rated, 0.75)
    manager.add("gus has a cat named Vincent", user_id="gus", infer=False)

    # A memory in the grafeo-memory 0.1.x history layout: HAS_HISTORY edge, no memory_id.
    db = manager._db
    legacy_text = "alix grew up in leiden"
    legacy = db.create_node(
        ["Memory"],
        {"text": legacy_text, "user_id": "alix", "created_at": 1_600_000_000_000, "memory_type": "semantic"},
    )
    db.set_node_property(legacy.id, "embedding", [float(x) for x in embedder.embed([legacy_text])[0]])
    entry = db.create_node(["History"], {"event": "ADD", "new_text": legacy_text, "timestamp": 1_600_000_000_000})
    db.create_edge(legacy.id, entry.id, "HAS_HISTORY")

    # A search runs PageRank and Louvain and materializes the community summaries.
    manager.search("alix hiking", user_id="alix")

    memories = {}
    for user in ("alix", "gus"):
        for memory in manager.get_all(user_id=user):
            props = db.get_node(int(memory.memory_id)).properties
            props = dict(props() if callable(props) else props)
            memories[memory.memory_id] = {
                "user_id": user,
                "text": memory.text,
                "importance": props.get("importance"),
                "created_at": props.get("created_at"),
                "embedding": props.get("embedding"),
            }
    history = {
        mid: [
            {"event": h.event, "old_text": h.old_text, "new_text": h.new_text, "timestamp": h.timestamp}
            for h in manager.history(mid)
        ]
        for mid in [*memories, gone]
    }
    graph_props = {
        str(row["id"]): {"pagerank": row["pr"], "betweenness": row["bc"], "community_id": row["cid"]}
        for row in _rows(
            db,
            "MATCH (e:Entity) RETURN id(e) AS id, e._pagerank AS pr, e._betweenness AS bc, e._community AS cid",
        )
    }
    manifest = {
        "memories": memories,
        "deleted": gone,
        "updated": tea,
        "legacy": str(legacy.id),
        "history": history,
        "entities": sorted(row["name"] for row in _rows(db, "MATCH (e:Entity) RETURN e.name AS name")),
        "relations": sorted(
            [row["s"], row["r"], row["t"]]
            for row in _rows(
                db,
                "MATCH (s:Entity)-[r:RELATION]->(t:Entity) RETURN s.name AS s, r.relation_type AS r, t.name AS t",
            )
        ),
        "entity_graph_props": graph_props,
        "communities": [
            {"name": c.name, "summary": c.summary, "members": sorted(c.member_entities)}
            for c in (manager.get_communities() if hasattr(manager, "get_communities") else [])
        ],
        "label_counts": {
            label: len(db.get_nodes_by_label(label))
            for label in ("Memory", "Entity", "History", "Community", "Episode")
        },
    }
    manager.close()
    return manifest


def main(out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "written_with": {"grafeo": grafeo.__version__, "grafeo_memory": grafeo_memory.__version__},
        "dimensions": DIMS,
        "databases": {},
    }
    for name in ("memory.grafeo", "memory_wal"):
        path = out / name
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()
        result["databases"][name] = _write(path)
    (out / "manifest.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(Path(__file__).parent))
