"""Durable, graph-native history of memory changes.

Every change grafeo-memory makes to a memory (ADD, UPDATE, DELETE) is stored
as a ``:History`` node in the same database as the memory. The node carries the
memory's node id in its ``memory_id`` property and is not connected to the
memory by an edge, so:

- history survives closing and reopening a file-backed database, because it is
  ordinary graph data;
- history survives a hard ``delete()`` of the memory (the audit trail keeps its
  DELETE entry);
- history nodes do not change the topology that graph algorithms (PageRank,
  betweenness, Louvain) and topology scoring see.

Engine change data capture (``db.node_history()``) is deliberately not used:
it is opt-in since grafeo 0.5.32 (``db.cdc_enabled`` is False by default), it
is kept in memory only (lost on close and reopen), it is pruned by retention
limits, and its timestamps are hybrid logical clock values rather than epoch
milliseconds. Relying on it made ``history()`` silently empty.

grafeo-memory 0.1.x linked ``:History`` nodes to memories with ``HAS_HISTORY``
edges. Those entries are still read.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .protocol import GrafeoDBProtocol

HISTORY_LABEL = "History"
#: Edge type used by grafeo-memory 0.1.x to link a memory to its history. Read only.
HAS_HISTORY_EDGE = "HAS_HISTORY"
#: Property on ``:History`` nodes holding the node id of the memory they describe.
MEMORY_ID_PROPERTY = "memory_id"

_RETURN_FIELDS = (
    "RETURN id(h) AS hid, h.event AS event, h.old_text AS old_text, h.new_text AS new_text, "
    "h.timestamp AS timestamp, h.actor_id AS actor_id, h.role AS role"
)


@dataclass
class HistoryEntry:
    """A single history event for a memory."""

    event: str  # "ADD", "UPDATE", "DELETE"
    old_text: str | None = None
    new_text: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    actor_id: str | None = None
    role: str | None = None


def _history_props(
    memory_node_id: int, entry: HistoryEntry, user_id: str | None, graph_name: str | None
) -> dict[str, Any]:
    props: dict[str, Any] = {
        MEMORY_ID_PROPERTY: int(memory_node_id),
        "event": entry.event,
        "timestamp": int(entry.timestamp),
    }
    if entry.old_text is not None:
        props["old_text"] = entry.old_text
    if entry.new_text is not None:
        props["new_text"] = entry.new_text
    if entry.actor_id is not None:
        props["actor_id"] = entry.actor_id
    if entry.role is not None:
        props["role"] = entry.role
    if user_id is not None:
        props["user_id"] = user_id
    if graph_name is not None:
        props["graph_name"] = graph_name
    return props


def record_history(
    db: GrafeoDBProtocol,
    memory_node_id: int,
    entry: HistoryEntry,
    *,
    user_id: str | None = None,
    graph_name: str | None = None,
) -> int:
    """Store a history entry for a memory and return the id of the new ``:History`` node.

    Errors from the database are raised, never swallowed, so a change can not
    lose its history silently.
    """
    node = db.create_node([HISTORY_LABEL], _history_props(memory_node_id, entry, user_id, graph_name))
    return int(node.id if hasattr(node, "id") else node)


def record_history_batch(
    db: GrafeoDBProtocol,
    entries: list[tuple[int, HistoryEntry]],
    *,
    user_id: str | None = None,
    graph_name: str | None = None,
) -> list[int]:
    """Store several ``(memory_node_id, entry)`` history entries; returns the new node ids.

    Uses one ``batch_create_nodes_with_props`` call when the database has it.
    """
    props_list = [_history_props(mid, entry, user_id, graph_name) for mid, entry in entries]
    if not props_list:
        return []
    batch_create = getattr(db, "batch_create_nodes_with_props", None)
    if batch_create is not None:
        return [int(i) for i in batch_create(HISTORY_LABEL, props_list)]
    ids: list[int] = []
    for props in props_list:
        node = db.create_node([HISTORY_LABEL], props)
        ids.append(int(node.id if hasattr(node, "id") else node))
    return ids


def adopt_legacy_history(db: GrafeoDBProtocol, memory_node_id: int) -> None:
    """Stamp ``memory_id`` on 0.1.x ``:History`` nodes linked to a memory by ``HAS_HISTORY``.

    Call this before the memory node is deleted: deleting it removes the
    ``HAS_HISTORY`` edges, and the stamp keeps those entries reachable.
    """
    db.execute(
        f"MATCH (m)-[:{HAS_HISTORY_EDGE}]->(h:{HISTORY_LABEL}) WHERE id(m) = $mid SET h.{MEMORY_ID_PROPERTY} = $mid",
        {"mid": int(memory_node_id)},
    )


def get_history(
    db: GrafeoDBProtocol,
    memory_node_id: int,
    *,
    graph_name: str | None = None,
) -> list[HistoryEntry]:
    """Return the history of a memory, oldest first.

    Args:
        db: The database.
        memory_node_id: Node id of the memory (it may have been deleted since).
        graph_name: When set, only entries recorded under this ``graph_name``
            scope are returned.

    Database errors are raised.
    """
    mid = int(memory_node_id)
    if graph_name is not None:
        rows = list(
            db.execute(
                f"MATCH (h:{HISTORY_LABEL}) WHERE h.{MEMORY_ID_PROPERTY} = $mid AND h.graph_name = $gn "
                f"{_RETURN_FIELDS}",
                {"mid": mid, "gn": graph_name},
            )
        )
    else:
        rows = list(
            db.execute(
                f"MATCH (h:{HISTORY_LABEL}) WHERE h.{MEMORY_ID_PROPERTY} = $mid {_RETURN_FIELDS}",
                {"mid": mid},
            )
        )
        # Entries written by grafeo-memory 0.1.x, linked by an edge instead of memory_id.
        rows.extend(
            db.execute(
                f"MATCH (m)-[:{HAS_HISTORY_EDGE}]->(h:{HISTORY_LABEL}) WHERE id(m) = $mid {_RETURN_FIELDS}",
                {"mid": mid},
            )
        )

    by_id: dict[int, dict] = {}
    for row in rows:
        if isinstance(row, dict) and row.get("hid") is not None:
            by_id[int(row["hid"])] = row

    ordered = sorted(by_id.items(), key=lambda item: (int(item[1].get("timestamp") or 0), item[0]))
    return [
        HistoryEntry(
            event=str(row.get("event") or "UNKNOWN"),
            old_text=row.get("old_text"),
            new_text=row.get("new_text"),
            timestamp=int(row.get("timestamp") or 0),
            actor_id=row.get("actor_id"),
            role=row.get("role"),
        )
        for _, row in ordered
    ]
