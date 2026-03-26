"""Structural typing protocols for the GrafeoDB database interface.

Defines the subset of the GrafeoDB API that grafeo-memory actually uses.
Using Protocol means any object that implements these methods works,
no inheritance required (structural subtyping).
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

# ---------------------------------------------------------------------------
# Node protocol
# ---------------------------------------------------------------------------


class GrafeoNode(Protocol):
    """A graph node returned by the database."""

    @property
    def id(self) -> int: ...

    @property
    def labels(self) -> list[str]: ...

    @property
    def properties(self) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Algorithms sub-object protocol
# ---------------------------------------------------------------------------


class AlgorithmsProtocol(Protocol):
    """Graph algorithm suite accessible via db.algorithms."""

    def pagerank(
        self, damping: float = 0.85, max_iterations: int = 100, tolerance: float = 1e-6
    ) -> dict[int, float]: ...

    def betweenness_centrality(self, normalized: bool = True) -> dict[int, float]: ...

    def louvain(self, resolution: float = 1.0) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Core database protocol (required methods only)
# ---------------------------------------------------------------------------


@runtime_checkable
class GrafeoDBProtocol(Protocol):
    """Minimal database interface that grafeo-memory requires.

    All methods listed here are called without hasattr guards, so any
    conforming database must implement them.

    Optional methods (hybrid_search, mmr_search, node_history,
    batch_create_nodes_with_props, algorithms, close, detailed_stats)
    are checked via hasattr at call sites and intentionally excluded.
    """

    # -- Node CRUD --

    def create_node(self, labels: list[str], properties: dict[str, Any]) -> GrafeoNode: ...

    def get_node(self, node_id: int) -> GrafeoNode | None: ...

    def delete_node(self, node_id: int) -> bool: ...

    def set_node_property(self, node_id: int, key: str, value: Any) -> None: ...

    def get_nodes_by_label(self, label: str) -> list[tuple[int, dict[str, Any]]]: ...

    def find_nodes_by_property(self, property_name: str, value: Any) -> list[int]: ...

    # -- Edge CRUD --

    def create_edge(
        self,
        source_id: int,
        target_id: int,
        edge_type: str,
        properties: dict[str, Any] | None = None,
    ) -> Any: ...

    def delete_edge(self, edge_id: int) -> bool: ...

    # -- Queries --

    def execute(self, query: str, params: dict[str, Any]) -> list[dict[str, Any]]: ...

    # -- Vector search --

    def vector_search(
        self,
        label: str,
        property: str,
        query: list[float],
        k: int,
        *,
        ef: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[tuple[int, float]]: ...

    # -- Indexes --

    def create_vector_index(
        self,
        label: str,
        property: str,
        *,
        dimensions: int,
        metric: str = "cosine",
    ) -> None: ...

    def create_text_index(self, label: str, property: str) -> None: ...

    def create_property_index(self, property: str) -> None: ...

    # -- Info --

    def info(self) -> dict[str, Any]: ...
