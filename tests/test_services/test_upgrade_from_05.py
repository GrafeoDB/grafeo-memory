"""Upgrade path: memory databases written on grafeo 0.5.44, opened on grafeo 0.6.

The databases in tests/fixtures/grafeo_0_5 were written by grafeo-memory 0.2.2
and 0.2.3 on grafeo 0.5.44 (see make_fixtures.py there), each as a single
``.grafeo`` file and as a 0.5.x WAL directory. grafeo 0.6 migrates them on the
first read-write open and keeps the old file or directory under a ``.pre-0.6``
name. These tests open copies through the manager, the CLI and the MCP server
and compare everything with the manifest written next to the databases.
"""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

import grafeo
import pytest
from mock_llm import MockEmbedder
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from grafeo_memory import AsyncMemoryManager, MemoryConfig, MemoryManager, cli
from grafeo_memory.core.search.vector import _get_props

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures" / "grafeo_0_5"
WRITERS = ("grafeo-memory-0.2.2", "grafeo-memory-0.2.3")
LAYOUTS = ("memory.grafeo", "memory_wal")
CASES = [pytest.param(w, layout, id=f"{w}-{layout}") for w in WRITERS for layout in LAYOUTS]


def _model() -> FunctionModel:
    """Mock LLM for searches: the query yields no entities."""

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        fields = set(tool.parameters_json_schema.get("properties", {}))
        assert "entities" in fields, f"unexpected LLM call with output fields {sorted(fields)}"
        return ModelResponse(parts=[ToolCallPart(tool_name=tool.name, args={"entities": [], "relations": []})])

    return FunctionModel(function=handler)


def _manifest(writer: str, layout: str) -> dict:
    data = json.loads((FIXTURES / writer / "manifest.json").read_text(encoding="utf-8"))
    return {"dimensions": data["dimensions"], "writer": data["written_with"], **data["databases"][layout]}


def _copy(writer: str, layout: str, tmp_path: Path) -> Path:
    source = FIXTURES / writer / layout
    target = tmp_path / layout
    if source.is_dir():
        shutil.copytree(source, target)
    else:
        shutil.copy2(source, target)
    return target


def _manager(path: Path) -> MemoryManager:
    config = MemoryConfig(db_path=str(path), user_id="alix", embedding_dimensions=16, enable_importance=True)
    return MemoryManager(_model(), config, embedder=MockEmbedder(16))


def _expected_history(manifest: dict, memory_id: str) -> list[tuple]:
    if manifest["writer"]["grafeo_memory"] == "0.2.2":
        # 0.2.2 kept its history in engine CDC, which grafeo 0.5 does not store:
        # only the 0.1.x entry (HAS_HISTORY) is on disk.
        return [("ADD", None, "alix grew up in leiden")] if memory_id == manifest["legacy"] else []
    return [(h["event"], h["old_text"], h["new_text"]) for h in manifest["history"][memory_id]]


def _snapshot(path: Path) -> dict[str, bytes]:
    if path.is_file():
        return {"": path.read_bytes()}
    return {str(p.relative_to(path)): p.read_bytes() for p in sorted(path.rglob("*")) if p.is_file()}


@pytest.mark.parametrize(("writer", "layout"), CASES)
class TestManagerOpensA05Database:
    def test_migrates_once_and_keeps_the_old_database(self, writer, layout, tmp_path):
        path = _copy(writer, layout, tmp_path)
        original = _snapshot(path)

        manager = _manager(path)
        manager.close()

        backup = tmp_path / f"{layout}.pre-0.6"
        assert path.is_file(), "a migrated database is a single file at the same path"
        assert _snapshot(backup) == original, "the 0.5.x database is kept unchanged under .pre-0.6"
        assert not (tmp_path / f"{layout}.wal").exists(), "close() leaves no WAL behind"

        # The second open finds a 0.6 file: nothing migrates again, the backup stays as it is.
        manager = _manager(path)
        manager.close()
        assert sorted(p.name for p in tmp_path.iterdir()) == sorted([layout, f"{layout}.pre-0.6"])
        assert _snapshot(backup) == original

    def test_memories_history_and_graph_read_back(self, writer, layout, tmp_path):
        manifest = _manifest(writer, layout)
        manager = _manager(_copy(writer, layout, tmp_path))
        db = manager._db
        try:
            for user in ("alix", "gus"):
                expected = {mid: m for mid, m in manifest["memories"].items() if m["user_id"] == user}
                found = {m.memory_id: m for m in manager.get_all(user_id=user)}
                assert set(found) == set(expected)
                for mid, memory in expected.items():
                    props = _get_props(db.get_node(int(mid)))
                    assert found[mid].text == memory["text"]
                    assert props["created_at"] == memory["created_at"]
                    assert type(props["created_at"]) is int
                    assert props.get("importance") == memory["importance"]
                    if memory["importance"] is not None:
                        assert type(props["importance"]) is float
                    assert props["embedding"] == pytest.approx(memory["embedding"])
                    assert all(type(x) is float for x in props["embedding"])

            for mid in [*manifest["memories"], manifest["deleted"]]:
                history = [(h.event, h.old_text, h.new_text) for h in manager.history(mid)]
                assert history == _expected_history(manifest, mid), mid
            assert db.get_node(int(manifest["deleted"])) is None

            entities = sorted(r["name"] for r in db.execute("MATCH (e:Entity) RETURN e.name AS name"))
            relations = sorted(
                [r["s"], r["r"], r["t"]]
                for r in db.execute(
                    "MATCH (s:Entity)-[r:RELATION]->(t:Entity) RETURN s.name AS s, r.relation_type AS r, t.name AS t"
                )
            )
            assert entities == manifest["entities"]
            assert relations == manifest["relations"]
            for nid, expected in manifest["entity_graph_props"].items():
                props = _get_props(db.get_node(int(nid)))
                assert props["_pagerank"] == pytest.approx(expected["pagerank"])
                assert type(props["_pagerank"]) is float
                assert props["_betweenness"] == pytest.approx(expected["betweenness"])
                assert props["_community"] == expected["community_id"]

            communities = sorted(
                (c.name, c.summary, tuple(sorted(c.member_entities))) for c in manager.get_communities("alix")
            )
            assert communities == sorted(
                (c["name"], c["summary"], tuple(c["members"])) for c in manifest["communities"]
            )
            for label, count in manifest["label_counts"].items():
                assert len(db.get_nodes_by_label(label)) == count, label
        finally:
            manager.close()

    def test_vector_search_finds_every_memory(self, writer, layout, tmp_path):
        manifest = _manifest(writer, layout)
        manager = _manager(_copy(writer, layout, tmp_path))
        embedder = MockEmbedder(16)
        try:
            for mid, memory in manifest["memories"].items():
                hits = manager._db.vector_search("Memory", "embedding", embedder.embed([memory["text"]])[0], 1)
                assert [str(h[0]) for h in hits] == [mid]
                results = manager.search(memory["text"], user_id=memory["user_id"], k=1)
                assert [r.memory_id for r in results] == [mid]
                assert type(results[0].score) is float
        finally:
            manager.close()

    def test_writes_after_the_migration_persist(self, writer, layout, tmp_path):
        path = _copy(writer, layout, tmp_path)
        manifest = _manifest(writer, layout)
        manager = _manager(path)
        new_id = manager.add("alix learns the cello", infer=False)[0].memory_id
        manager.update(manifest["updated"], "alix drinks rooibos")
        manager.delete(manifest["legacy"])
        manager.close()

        manager = _manager(path)
        try:
            texts = {m.memory_id: m.text for m in manager.get_all()}
            assert texts[new_id] == "alix learns the cello"
            assert texts[manifest["updated"]] == "alix drinks rooibos"
            assert manifest["legacy"] not in texts
            assert [h.event for h in manager.history(manifest["legacy"])] == ["ADD", "DELETE"]
            assert [h.event for h in manager.history(manifest["updated"])][-1] == "UPDATE"
            assert [r.memory_id for r in manager.search("alix learns the cello", k=1)] == [new_id]
        finally:
            manager.close()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_cli_opens_a_05_database(layout, tmp_path, monkeypatch, capsys):
    writer = "grafeo-memory-0.2.3"
    manifest = _manifest(writer, layout)
    path = _copy(writer, layout, tmp_path)
    monkeypatch.setattr(cli.app, "_create_embedder", lambda model: MockEmbedder(16))
    for var in ("GRAFEO_MEMORY_DB", "GRAFEO_MEMORY_USER", "GRAFEO_MEMORY_MODEL"):
        monkeypatch.delenv(var, raising=False)

    def run(*args: str) -> dict:
        cli.main(["--model", "test", "--db", str(path), "--user", "alix", "--json", *args])
        return json.loads(capsys.readouterr().out)

    listed = {m["memory_id"]: m["text"] for m in run("list")["memories"]}
    assert listed == {mid: m["text"] for mid, m in manifest["memories"].items() if m["user_id"] == "alix"}
    history = run("history", manifest["updated"])["history"]
    assert [h["event"] for h in history] == ["ADD", "UPDATE"]
    assert path.is_file()
    assert (tmp_path / f"{layout}.pre-0.6").exists()


@pytest.mark.parametrize("layout", LAYOUTS)
def test_mcp_server_opens_a_05_database(layout, tmp_path, monkeypatch):
    from grafeo_memory.mcp import server

    writer = "grafeo-memory-0.2.3"
    manifest = _manifest(writer, layout)
    path = _copy(writer, layout, tmp_path)
    monkeypatch.setenv("GRAFEO_MEMORY_MODEL", "test")
    monkeypatch.setenv("GRAFEO_MEMORY_DB", str(path))
    monkeypatch.setenv("GRAFEO_MEMORY_USER", "alix")
    monkeypatch.delenv("GRAFEO_MEMORY_YOLO", raising=False)
    monkeypatch.setattr(server, "_create_embedder", lambda model: MockEmbedder(16))

    async def run_server() -> tuple[set[str], list[str]]:
        async with server.app_lifespan(server.mcp) as app:
            manager: AsyncMemoryManager = app.manager
            ids = {m.memory_id for m in await manager.get_all()}
            events = [h.event for h in await manager.history(manifest["updated"])]
            return ids, events

    ids, events = asyncio.run(run_server())
    assert ids == {mid for mid, m in manifest["memories"].items() if m["user_id"] == "alix"}
    assert events == ["ADD", "UPDATE"]
    assert path.is_file()
    assert (tmp_path / f"{layout}.pre-0.6").exists()


def test_an_empty_directory_is_not_a_database(tmp_path):
    """grafeo 0.6 refuses to open a directory that is not a 0.5.x database (0.5 created one in it)."""
    folder = tmp_path / "memories"
    folder.mkdir()
    with pytest.raises(grafeo.GrafeoError):
        _manager(folder)
    _manager(folder / "memory.grafeo").close()
