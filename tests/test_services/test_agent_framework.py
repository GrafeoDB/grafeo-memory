"""Microsoft Agent Framework integration: GrafeoVectorStore over a MemoryManager.

The framework's own abstractions, BaseVectorCollection CRUD/search and the
agent-usable FunctionTools (create_upsert_tool, create_vector_search_tool,
...), run against grafeo: records persist as graph nodes and semantic search
uses grafeo's vector index.

Skipped when agent-framework-core is not installed (it is a dev dependency).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Annotated

import pytest

af = pytest.importorskip("agent_framework")

from agent_framework import (  # noqa: E402
    BaseEmbeddingClient,
    Embedding,
    Filter,
    FilterGroup,
    GeneratedEmbeddings,
    VectorStoreField,
    create_get_tool,
    create_upsert_tool,
    create_vector_search_tool,
    vectorstoremodel,
)
from conftest import make_manager  # noqa: E402
from mock_llm import MockEmbedder  # noqa: E402

from grafeo_memory.agent_framework import GrafeoVectorStore, label_for  # noqa: E402

DIMS = 8


@vectorstoremodel(collection_name="memories")
@dataclass
class MemoryRecord:
    id: Annotated[str, VectorStoreField(field_type="key", type_="str")] = ""
    embedding: Annotated[list[float], VectorStoreField(field_type="vector", type_="float", dimensions=DIMS)] = field(
        default_factory=list
    )
    text: Annotated[str, VectorStoreField(field_type="data")] = ""
    kind: Annotated[str, VectorStoreField(field_type="data")] = ""


class AFEmbedder(BaseEmbeddingClient):
    """Agent Framework embedding client over the test embedder."""

    def __init__(self, embedder: MockEmbedder) -> None:
        super().__init__()
        self._embedder = embedder

    async def get_embeddings(self, values, *, options=None):
        vectors = self._embedder.embed(list(values))
        return GeneratedEmbeddings([Embedding(vector=v) for v in vectors])


def _vec(seed: float) -> list[float]:
    return [seed + i * 0.01 for i in range(DIMS)]


def _store() -> GrafeoVectorStore:
    manager = make_manager([])
    return GrafeoVectorStore.from_manager(manager, embedding_generator=AFEmbedder(MockEmbedder(DIMS)))


def test_crud_and_search_roundtrip():
    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        assert collection.collection_name == "memories"
        assert not await collection.collection_exists()
        await collection.ensure_collection_exists()
        assert await collection.collection_exists()

        keys = await collection.upsert(
            [
                MemoryRecord(id="a", embedding=_vec(0.0), text="likes green tea", kind="fact"),
                MemoryRecord(id="b", embedding=_vec(0.5), text="works at Acme", kind="fact"),
                MemoryRecord(id="c", embedding=_vec(0.9), text="enjoys hiking", kind="hobby"),
            ],
            generate_vectors=False,
        )
        assert keys == ["a", "b", "c"]

        got = await collection.get(["a", "missing"])
        assert len(got) == 1 and got[0].text == "likes green tea" and got[0].id == "a"

        results = await collection.search(vector=_vec(0.0), top=2)
        all_results = [r async for r in results]
        assert [r["record"].id for r in all_results] == ["a", "b"], "cosine distance, nearest first"
        assert all_results[0]["score"] <= all_results[1]["score"]

        filtered = await collection.search(vector=_vec(0.0), top=3, filter=Filter("kind", "eq", "hobby"))
        assert [r["record"].id async for r in filtered] == ["c"]

        await collection.delete(["b"])
        assert [r.id for r in await collection.get(["a", "b", "c"])] == ["a", "c"]
        assert "memories" in await store.list_collection_names()

        await collection.ensure_collection_deleted()
        assert not await collection.collection_exists()

    asyncio.run(_run())


def test_records_persist_as_graph_nodes():
    async def _run():
        manager = make_manager([])
        store = GrafeoVectorStore.from_manager(manager)
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()
        await collection.upsert(
            [MemoryRecord(id="a", embedding=_vec(0.0), text="persisted", kind="fact")], generate_vectors=False
        )

        label = label_for("memories")
        rows = list(manager._db.execute(f"MATCH (n:{label}) RETURN n.text AS text, n.kind AS kind"))
        assert rows == [{"text": "persisted", "kind": "fact"}]
        # the vector index exists for the collection's vector field
        names = [i["name"] for i in manager._db.memory_usage()["indexes"]["vector_indexes"]]
        assert any(n.startswith(f"{label}:") for n in names)

    asyncio.run(_run())


def test_upsert_is_idempotent_by_key():
    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()
        await collection.upsert(
            [MemoryRecord(id="a", embedding=_vec(0.0), text="v1", kind="fact")], generate_vectors=False
        )
        await collection.upsert(
            [MemoryRecord(id="a", embedding=_vec(0.1), text="v2", kind="fact")], generate_vectors=False
        )
        got = await collection.get(["a"])
        assert len(got) == 1 and got[0].text == "v2"
        assert store.db.node_count == 1

    asyncio.run(_run())


def test_filter_group_and_unsupported_operators():
    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()
        await collection.upsert(
            [
                MemoryRecord(id="a", embedding=_vec(0.0), text="x", kind="fact"),
                MemoryRecord(id="b", embedding=_vec(0.1), text="y", kind="hobby"),
            ],
            generate_vectors=False,
        )
        group = FilterGroup("and", (Filter("kind", "eq", "hobby"),))
        results = await collection.search(vector=_vec(0.0), top=5, filter=group)
        assert [r["record"].id async for r in results] == ["b"]
        with pytest.raises(NotImplementedError):
            await collection.search(vector=_vec(0.0), top=5, filter=Filter("kind", "ne", "fact"))
        with pytest.raises(NotImplementedError):
            await collection.search(vector=_vec(0.0), top=5, filter=FilterGroup("or", (Filter("kind", "eq", "fact"),)))

    asyncio.run(_run())


def test_agent_framework_function_tools_use_grafeo():
    """The framework's agent-usable tools read and write the grafeo database."""

    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()

        upsert_tool = create_upsert_tool(collection, generate_vectors=False)
        await upsert_tool.invoke(
            arguments={"records": [{"id": "a", "embedding": _vec(0.0), "text": "likes green tea", "kind": "fact"}]}
        )
        rows = list(store.db.execute(f"MATCH (n:{label_for('memories')}) RETURN n.text AS text"))
        assert rows == [{"text": "likes green tea"}]

        search_tool = create_vector_search_tool(collection)
        result = await search_tool.invoke(arguments={"query": "green tea"})
        text = "".join(getattr(c, "text", "") or "" for c in result)
        assert "likes green tea" in text

        get_tool = create_get_tool(collection)
        result = await get_tool.invoke(arguments={"keys": ["a"]})
        text = "".join(getattr(c, "text", "") or "" for c in result) if isinstance(result, list) else str(result)
        assert "likes green tea" in text

    asyncio.run(_run())


def test_score_threshold_and_skip():
    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()
        one_hot = {
            "a": [1.0] + [0.0] * (DIMS - 1),
            "b": [0.0, 1.0] + [0.0] * (DIMS - 2),
            "c": [0.0, 0.0, 1.0] + [0.0] * (DIMS - 3),
        }
        await collection.upsert(
            [MemoryRecord(id=c, embedding=v, text=c, kind="fact") for c, v in one_hot.items()],
            generate_vectors=False,
        )
        query = [1.0] + [0.0] * (DIMS - 1)
        results = await collection.search(vector=query, top=10, score_threshold=0.5)
        assert {r["record"].id async for r in results} == {"a"}, (
            "cosine distance: lower is better, threshold drops the far ones"
        )
        skipped = await collection.search(vector=query, top=1, skip=1)
        assert {r["record"].id async for r in skipped} in ({"b"}, {"c"})

    asyncio.run(_run())


@vectorstoremodel(collection_name="notes")
@dataclass
class NoteRecord:
    id: Annotated[str, VectorStoreField(field_type="key", type_="str")] = ""
    embedding: Annotated[list[float], VectorStoreField(field_type="vector", type_="float", dimensions=DIMS)] = field(
        default_factory=list
    )
    title: Annotated[list[float], VectorStoreField(field_type="vector", type_="float", dimensions=DIMS)] = field(
        default_factory=list
    )
    text: Annotated[str | None, VectorStoreField(field_type="data")] = None


def test_keys_are_scoped_to_their_collection():
    async def _run():
        store = _store()
        memories = store.get_collection(MemoryRecord)
        notes = store.get_collection(NoteRecord)
        await memories.ensure_collection_exists()
        await notes.ensure_collection_exists()
        await memories.upsert([MemoryRecord(id="a", embedding=_vec(0.0), text="memory")], generate_vectors=False)
        await notes.upsert(
            [NoteRecord(id="a", embedding=_vec(0.1), title=_vec(0.2), text="note")], generate_vectors=False
        )
        assert [r.text for r in await memories.get(["a"])] == ["memory"]
        assert [r.text for r in await notes.get(["a"])] == ["note"]
        await notes.delete(["a"])
        assert [r.text for r in await memories.get(["a"])] == ["memory"]

    asyncio.run(_run())


def test_collection_names_do_not_collide():
    assert label_for("a-b") != label_for("a_b")

    async def _run():
        store = _store()
        for name in ("a-b", "a_b"):
            collection = store.get_collection(MemoryRecord, collection_name=name)
            await collection.ensure_collection_exists()
            await collection.upsert([MemoryRecord(id=name, embedding=_vec(0.0), text=name)], generate_vectors=False)
        assert {"a-b", "a_b"} <= set(await store.list_collection_names())
        only = store.get_collection(MemoryRecord, collection_name="a-b")
        assert [r.id for r in await only.get(top=10)] == ["a-b"]

    asyncio.run(_run())


def test_conflicting_and_filter_matches_nothing():
    async def _run():
        store = _store()
        collection = store.get_collection(MemoryRecord)
        await collection.ensure_collection_exists()
        await collection.upsert(
            [MemoryRecord(id="b", embedding=_vec(0.1), text="y", kind="hobby")], generate_vectors=False
        )
        group = FilterGroup("and", (Filter("kind", "eq", "fact"), Filter("kind", "eq", "hobby")))
        assert [r async for r in await collection.search(vector=_vec(0.0), top=5, filter=group)] == []
        assert await collection.get(filter=group, top=5) == []

    asyncio.run(_run())


def test_upsert_replaces_a_field_set_to_none_and_indexes_every_vector_field():
    async def _run():
        store = _store()
        notes = store.get_collection(NoteRecord)
        await notes.ensure_collection_exists()
        names = [i["name"] for i in store.db.memory_usage()["indexes"]["vector_indexes"]]
        label = label_for("notes")
        assert {f"{label}:embedding", f"{label}:title"} <= set(names)

        await notes.upsert(
            [NoteRecord(id="n", embedding=_vec(0.0), title=_vec(0.5), text="v1")], generate_vectors=False
        )
        await notes.upsert(
            [NoteRecord(id="n", embedding=_vec(0.0), title=_vec(0.5), text=None)], generate_vectors=False
        )
        assert [r.text for r in await notes.get(["n"])] == [None]

        hits = await notes.search(vector=_vec(0.5), top=1, vector_property_name="title")
        assert [r["record"].id async for r in hits] == ["n"]

    asyncio.run(_run())
