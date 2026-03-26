"""Graph-native history tracking via :History nodes and HAS_HISTORY edges."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .protocol import GrafeoDBProtocol

HISTORY_LABEL = "History"
HAS_HISTORY_EDGE = "HAS_HISTORY"


@dataclass
class HistoryEntry:
    """A single history event for a memory."""

    event: str  # "ADD", "UPDATE", "DELETE"
    old_text: str | None = None
    new_text: str | None = None
    timestamp: int = field(default_factory=lambda: int(time.time() * 1000))
    actor_id: str | None = None
    role: str | None = None


def record_history(db: GrafeoDBProtocol, memory_node_id: int, entry: HistoryEntry) -> int | None:
    """Record a history entry.

    With native CDC, this is a no-op (the engine tracks changes automatically).
    Returns the node ID of the new History node, or None when CDC handles it.
    """
    if hasattr(db, "node_history"):
        # Native CDC available: engine tracks changes automatically
        return None

    # Fallback: create History node (legacy behavior)
    props: dict = {
        "event": entry.event,
        "timestamp": entry.timestamp,
    }
    if entry.old_text is not None:
        props["old_text"] = entry.old_text
    if entry.new_text is not None:
        props["new_text"] = entry.new_text
    if entry.actor_id is not None:
        props["actor_id"] = entry.actor_id
    if entry.role is not None:
        props["role"] = entry.role

    node = db.create_node([HISTORY_LABEL], props)
    node_id = node.id if hasattr(node, "id") else node
    db.create_edge(memory_node_id, node_id, HAS_HISTORY_EDGE)
    return node_id


def get_history(db: GrafeoDBProtocol, memory_node_id: int) -> list[HistoryEntry]:
    """Retrieve history. Prefers native CDC when available."""
    if hasattr(db, "node_history"):
        try:
            events = db.node_history(memory_node_id)  # ty: ignore[call-non-callable]
            entries: list[HistoryEntry] = []
            for ev in events:
                kind = (ev.get("kind") or "UNKNOWN").upper()
                before = ev.get("before") or {}
                after = ev.get("after") or {}
                ts = ev.get("timestamp", 0)

                if "CREATE" in kind:
                    entries.append(
                        HistoryEntry(
                            event="ADD",
                            new_text=after.get("text"),
                            timestamp=ts,
                        )
                    )
                elif "DELETE" in kind:
                    entries.append(
                        HistoryEntry(
                            event="DELETE",
                            old_text=before.get("text"),
                            timestamp=ts,
                        )
                    )
                elif "UPDATE" in kind or "SET" in kind:
                    old_text = before.get("text")
                    new_text = after.get("text")
                    # Only include if the text property actually changed
                    if old_text != new_text and (old_text is not None or new_text is not None):
                        entries.append(
                            HistoryEntry(
                                event="UPDATE",
                                old_text=old_text,
                                new_text=new_text,
                                timestamp=ts,
                            )
                        )
            return entries
        except Exception:
            pass  # Fall through to legacy

    # Legacy: query History nodes
    try:
        query = (
            f"MATCH (m)-[:{HAS_HISTORY_EDGE}]->(h:{HISTORY_LABEL}) "
            f"WHERE id(m) = $mid "
            f"RETURN h.event, h.old_text, h.new_text, h.timestamp, h.actor_id, h.role "
            f"ORDER BY h.timestamp ASC"
        )
        result = db.execute(query, {"mid": memory_node_id})
    except Exception:
        import logging

        logging.getLogger(__name__).warning(
            "get_history: query failed for memory node %s",
            memory_node_id,
            exc_info=True,
        )
        return []

    entries: list[HistoryEntry] = []
    for row in result:
        if not isinstance(row, dict):
            continue
        vals = list(row.values())
        if len(vals) < 4:
            continue
        entries.append(
            HistoryEntry(
                event=str(vals[0]) if vals[0] else "UNKNOWN",
                old_text=vals[1],
                new_text=vals[2],
                timestamp=int(vals[3]) if vals[3] else 0,
                actor_id=vals[4] if len(vals) > 4 else None,
                role=vals[5] if len(vals) > 5 else None,
            )
        )
    return entries
