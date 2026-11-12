"""Store schema version: checked on open, migrations run before any other operation.

Version 1 is a store without layer structures (every store before 0.3, and any
store that never turned a layer on). Version 2 adds them (OpenDecision and
TaxonomyState nodes, ``subjects`` / ``provisional`` / ``status`` properties,
SCOPE_OF edges). They are backfilled lazily, so the 1 -> 2 migration only records
the version, and the marker node is written only when the first layer structure is
(``mark``): a store that never uses a layer stays byte-identical.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from grafeo_memory.common.types import META_LABEL, SCHEMA_VERSION
from grafeo_memory.core.protocol import GrafeoDBProtocol
from grafeo_memory.core.search.vector import _get_props

logger = logging.getLogger(__name__)

# from_version -> migration that brings the store to from_version + 1
MIGRATIONS: dict[int, Callable[[GrafeoDBProtocol], None]] = {
    1: lambda db: None,  # lazy backfill: nothing to rewrite
}


def _meta(db: GrafeoDBProtocol) -> tuple[int, dict] | None:
    try:
        nodes = db.get_nodes_by_label(META_LABEL)
    except Exception:
        return None
    for item in nodes:
        if isinstance(item, tuple):
            return int(item[0]), dict(item[1])
        nid = getattr(item, "id", None)
        if nid is not None:
            return int(nid), _get_props(item)
    return None


def schema_version(db: GrafeoDBProtocol) -> int:
    found = _meta(db)
    return int(found[1].get("schema_version", 1)) if found else 1


def mark(db: GrafeoDBProtocol) -> None:
    """Record SCHEMA_VERSION before the first layer structure is written."""
    ensure_schema(db, create=True)


def ensure_schema(db: GrafeoDBProtocol, *, create: bool = False) -> int:
    """Bring the store to SCHEMA_VERSION; returns the version found.

    Without ``create``, a store with no marker is left as it is (version 1). A store
    newer than this library is left untouched (logged), never downgraded.
    """
    try:
        found = _meta(db)
        current = int(found[1].get("schema_version", 1)) if found else 1
        if current > SCHEMA_VERSION:
            logger.warning(
                "store schema_version %d is newer than this grafeo-memory (%d); upgrade grafeo-memory",
                current,
                SCHEMA_VERSION,
            )
            return current
        if found is None and not create:
            return current
        version = current
        while version < SCHEMA_VERSION:
            MIGRATIONS[version](db)
            version += 1
        if found is None:
            if create:
                db.create_node([META_LABEL], {"schema_version": SCHEMA_VERSION})
        elif current != SCHEMA_VERSION:
            db.set_node_property(found[0], "schema_version", SCHEMA_VERSION)
        return current
    except Exception:
        logger.warning("schema version check failed; continuing without it", exc_info=True)
        return SCHEMA_VERSION
