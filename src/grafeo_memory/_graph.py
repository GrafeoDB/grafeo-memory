"""Helpers over the GrafeoDB direct API."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import GrafeoDBProtocol


def node_labels(node: object) -> list[str]:
    """Return the labels of a node returned by ``db.get_node()``."""
    labels = getattr(node, "labels", None)
    if callable(labels):
        labels = labels()
    return list(labels or [])


def detach_delete_node(db: GrafeoDBProtocol, node_id: int) -> bool:
    """Delete a node together with all of its edges. Returns True when the node existed.

    Since grafeo 0.5.44, ``db.delete_node()`` raises for a node that still has
    edges, and the Python direct API has no detach variant, so this runs a
    ``DETACH DELETE`` query: one statement, so no edge can be added between
    removing the edges and removing the node.
    """
    node_id = int(node_id)
    if db.get_node(node_id) is None:
        return False
    db.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": node_id})
    return db.get_node(node_id) is None
