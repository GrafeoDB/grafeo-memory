"""Microsoft Agent Framework vector-store adapter backed by GrafeoDB.

Lets an ``agent_framework`` ``Agent`` keep its memory in grafeo: collections
are graph labels, records are nodes, and the vector field is a grafeo vector
index, so semantic search, exact filters and persistence all come from the
database.

Optional extra::

    pip install "grafeo-memory[agent-framework]"

Usage::

    from dataclasses import dataclass, field
    from typing import Annotated
    from agent_framework import VectorStoreField, vectorstoremodel
    from grafeo_memory.agent_framework import GrafeoVectorStore

    @vectorstoremodel(collection_name="memories")
    @dataclass
    class MemoryRecord:
        id: Annotated[str, VectorStoreField(field_type="key", type_="str")] = ""
        embedding: Annotated[
            list[float], VectorStoreField(field_type="vector", type_="float", dimensions=1536)
        ] = field(default_factory=list)
        text: Annotated[str, VectorStoreField(field_type="data")] = ""

    store = GrafeoVectorStore.from_manager(manager)
    collection = store.get_collection(MemoryRecord)
    await collection.ensure_collection_exists()
    await collection.upsert(MemoryRecord(id="1", embedding=[...], text="likes tea"))
    results = await collection.search("tea", top=5)
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast
from uuid import uuid4

try:
    from agent_framework import (
        BaseVectorCollection,
        BaseVectorSearch,
        BaseVectorStore,
        Filter,
        FilterGroup,
        SearchResults,
        SearchType,
    )
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "grafeo_memory.agent_framework needs Microsoft Agent Framework: "
        "pip install 'agent-framework-core>=1.25,<2' (or grafeo-memory[agent-framework])"
    ) from exc

if TYPE_CHECKING:
    from grafeo_memory.services.memory import MemoryManager

KeyT = TypeVar("KeyT")
ModelT = TypeVar("ModelT")

#: Graph label prefix for Agent Framework collections.
LABEL_PREFIX = "AF_"

#: Cosine distance is what grafeo's vector index returns; lower is better.
_DISTANCE_FUNCTIONS = {"cosine_distance", "DEFAULT", None}


def label_for(collection_name: str) -> str:
    """The grafeo label holding one collection's records."""
    return LABEL_PREFIX + re.sub(r"[^0-9A-Za-z_]", "_", collection_name)


class GrafeoVectorCollection(BaseVectorCollection[KeyT, ModelT], BaseVectorSearch[KeyT, ModelT]):
    """One Agent Framework collection as a grafeo label of nodes."""

    supported_search_types: ClassVar[set[SearchType]] = {"vector"}
    supported_key_types: ClassVar[set[str] | None] = {"str", "int", "float", "bool"}
    supported_vector_types: ClassVar[set[str] | None] = {
        "float",
        "float32",
        "float64",
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
    }

    def __init__(
        self,
        store: GrafeoVectorStore,
        record_type: type[ModelT],
        *,
        definition: Any = None,
        collection_name: str | None = None,
        embedding_generator: Any = None,
    ) -> None:
        super().__init__(
            record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator,
        )
        self._store = store

    @property
    def _db(self) -> Any:
        # The full grafeo API (schema, memory_usage, drop_vector_index, ...), not
        # just the minimal GrafeoDBProtocol the manager accepts.
        return self._store.db

    @property
    def _label(self) -> str:
        return label_for(self.collection_name)

    @property
    def _key_storage(self) -> str:
        return self.definition.key_field_storage_name

    def _vector_field(self, vector_property_name: str | None = None) -> Any:
        field = self.definition.try_get_vector_field(vector_property_name)
        if field is None:
            raise ValueError(f"Collection '{self.collection_name}' has no vector field")
        return field

    async def ensure_collection_exists(self, *, operation_options: Mapping[str, Any] | None = None) -> None:
        self._db.create_property_index(self._key_storage)
        field = self._vector_field()
        if field.dimensions:
            self._db.create_vector_index(
                self._label, field.storage_name or field.name, dimensions=field.dimensions, metric="cosine"
            )

    async def collection_exists(self, *, operation_options: Mapping[str, Any] | None = None) -> bool:
        if self._db.get_nodes_by_label(self._label, limit=1):
            return True
        prefix = f"{self._label}:"
        usage = self._db.memory_usage() if hasattr(self._db, "memory_usage") else {}
        indexes = usage.get("indexes", {}).get("vector_indexes", [])
        return any(str(idx.get("name", "")).startswith(prefix) for idx in indexes)

    async def ensure_collection_deleted(self, *, operation_options: Mapping[str, Any] | None = None) -> None:
        self._db.execute(f"MATCH (n:{self._label}) DETACH DELETE n", {})
        field = self._vector_field()
        drop = getattr(self._db, "drop_vector_index", None)
        if callable(drop):
            drop(self._label, field.storage_name or field.name)

    async def _inner_upsert(
        self, records: Sequence[Any], *, operation_options: Mapping[str, Any] | None = None
    ) -> Sequence[KeyT]:
        keys: list[KeyT] = []
        for record in records:
            if not isinstance(record, Mapping):
                raise TypeError("GrafeoVectorCollection records must serialize to mappings.")
            rec = dict(record)
            key = rec.get(self._key_storage)
            if key is None:
                if self.definition.key_field.is_auto_generated and self.definition.key_field.type_ == "str":
                    key = rec[self._key_storage] = str(uuid4())
                else:
                    raise ValueError(f"Record is missing key field '{self.definition.key_field.name}'.")
            matches = self._db.find_nodes_by_property(self._key_storage, key)
            if matches:
                node_id = matches[0]
                for name, value in rec.items():
                    if value is None:
                        continue  # grafeo stores no null properties; a missing key reads as absent
                    self._db.set_node_property(node_id, name, value)
            else:
                self._db.create_node([self._label], {k: v for k, v in rec.items() if v is not None})
            keys.append(cast(KeyT, key))
        return keys

    async def _inner_get(
        self,
        *,
        keys: Sequence[KeyT] | None = None,
        filter: Any = None,
        top: int = 10,
        skip: int = 0,
        order_by: Mapping[str, bool] | None = None,
        include_vectors: bool = False,
        operation_options: Mapping[str, Any] | None = None,
    ) -> Sequence[Any] | None:
        if keys is not None:
            out: list[dict[str, Any]] = []
            for key in keys:
                for node_id in self._db.find_nodes_by_property(self._key_storage, key):
                    node = self._db.get_node(node_id)
                    if node is not None:
                        out.append(dict(node.properties()))
            return out
        records = [dict(props) for _node_id, props in self._db.get_nodes_by_label(self._label)]
        translated = self._translate_filter(filter)
        if translated:
            records = [r for r in records if all(r.get(k) == v for k, v in translated.items())]
        if order_by:
            for field_name, ascending in reversed(tuple(order_by.items())):
                records.sort(key=lambda r: (r.get(field_name) is None, r.get(field_name)), reverse=not ascending)
        return records[skip : skip + top]

    async def _inner_delete(self, keys: Sequence[KeyT], *, operation_options: Mapping[str, Any] | None = None) -> None:
        for key in keys:
            for node_id in self._db.find_nodes_by_property(self._key_storage, key):
                self._db.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": node_id})

    async def _inner_search(
        self,
        *,
        search_type: str = "vector",
        filter: Any = None,
        values: Any | None = None,
        vector: Any | None = None,
        top: int = 3,
        skip: int = 0,
        include_vectors: bool = False,
        vector_property_name: str | None = None,
        additional_property_name: str | None = None,
        score_threshold: float | None = None,
        operation_options: Mapping[str, Any] | None = None,
    ) -> SearchResults[Any]:
        if search_type != "vector":
            raise NotImplementedError("GrafeoVectorCollection supports only vector search.")
        if vector is None:
            raise ValueError(
                "vector search requires a query vector (the collection's embedding generator supplies one)"
            )
        field = self._vector_field(vector_property_name)
        if field.distance_function not in _DISTANCE_FUNCTIONS:
            raise NotImplementedError(f"Distance function '{field.distance_function}' is not supported; use cosine.")
        storage_name = field.storage_name or field.name
        query_vector = [float(x) for x in vector]
        translated = self._translate_filter(filter)
        hits = self._db.vector_search(self._label, storage_name, query_vector, top + skip, filters=translated)
        results: list[dict[str, Any]] = []
        for node_id, distance in hits:
            if score_threshold is not None and distance > score_threshold:
                continue
            node = self._db.get_node(node_id)
            if node is None:
                continue
            props = dict(node.properties())
            if not include_vectors:
                props.pop(storage_name, None)
            results.append({"record": props, "score": distance})
        results.sort(key=lambda r: r["score"])
        return SearchResults(results[skip : skip + top], metadata={"grafeo_total_count": len(hits)})

    def _get_record_from_result(self, result: Any) -> Any:
        return result["record"]

    def _get_score_from_result(self, result: Any) -> float | None:
        return result.get("score")

    def _translate_filter(self, filter: Any) -> dict[str, Any] | None:
        """Translate an Agent Framework filter into grafeo's exact-match ``filters`` dict."""
        if filter is None:
            return None
        if isinstance(filter, Filter):
            if filter.operator != "eq":
                raise NotImplementedError(f"GrafeoVectorCollection supports only 'eq' filters, got '{filter.operator}'")
            field = self.definition.try_get_field(filter.field_name)
            name = (field.storage_name or field.name) if field is not None else filter.field_name
            return {name: filter.value}
        if isinstance(filter, FilterGroup):
            if filter.operator != "and":
                raise NotImplementedError(
                    f"GrafeoVectorCollection supports only 'and' filter groups, got '{filter.operator}'"
                )
            merged: dict[str, Any] = {}
            for child in filter.filters:
                part = self._translate_filter(child)
                if part:
                    merged.update(part)
            return merged or None
        raise NotImplementedError(f"Unsupported filter type {type(filter).__name__}")


class GrafeoVectorStore(BaseVectorStore):
    """An Agent Framework vector store whose collections live in a grafeo database."""

    def __init__(self, db: Any, *, embedding_generator: Any = None, managed_client: bool = True) -> None:
        super().__init__(embedding_generator=embedding_generator, managed_client=managed_client)
        self.db = db

    @classmethod
    def from_manager(cls, manager: MemoryManager, **kwargs: Any) -> GrafeoVectorStore:
        """A store over a ``MemoryManager``'s database."""
        return cls(manager._db, **kwargs)

    def get_collection(
        self,
        record_type: type[ModelT],
        *,
        definition: Any = None,
        collection_name: str | None = None,
        embedding_generator: Any = None,
    ) -> GrafeoVectorCollection[Any, ModelT]:
        return GrafeoVectorCollection(
            self,
            record_type,
            definition=definition,
            collection_name=collection_name,
            embedding_generator=embedding_generator or self.embedding_generator,
        )

    async def list_collection_names(self, *, operation_options: Mapping[str, Any] | None = None) -> Sequence[str]:
        names = [e["name"] for e in self.db.schema().get("labels", []) if e["name"].startswith(LABEL_PREFIX)]
        usage = self.db.memory_usage() if hasattr(self.db, "memory_usage") else {}
        for idx in usage.get("indexes", {}).get("vector_indexes", []):
            name = str(idx.get("name", ""))
            if name.startswith(LABEL_PREFIX) and ":" in name:
                label = name.split(":", 1)[0]
                if label not in names:
                    names.append(label)
        return sorted(n[len(LABEL_PREFIX) :] for n in names)

    async def _inner_ensure_collection_deleted(
        self, collection_name: str, *, operation_options: Mapping[str, Any] | None = None
    ) -> None:
        label = label_for(collection_name)
        self.db.execute(f"MATCH (n:{label}) DETACH DELETE n", {})
        usage = self.db.memory_usage() if hasattr(self.db, "memory_usage") else {}
        for idx in usage.get("indexes", {}).get("vector_indexes", []):
            name = str(idx.get("name", ""))
            if name.startswith(f"{label}:"):
                drop = getattr(self.db, "drop_vector_index", None)
                if callable(drop):
                    drop(label, name.split(":", 1)[1])
