"""Community lifecycle: detect, summarize, and query entity clusters."""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

from pydantic_ai import Agent

from .prompts import COMMUNITY_SUMMARY_SYSTEM, COMMUNITY_SUMMARY_USER
from .schemas import CommunitySummaryOutput
from .types import (
    COMMUNITY_LABEL,
    ENTITY_LABEL,
    HAS_MEMBER_EDGE,
    RELATION_EDGE,
    CommunityInfo,
    ModelType,
)

if TYPE_CHECKING:
    from pydantic_ai.usage import RunUsage

    from .protocol import GrafeoDBProtocol

logger = logging.getLogger(__name__)


async def materialize_communities_async(
    db: GrafeoDBProtocol,
    model: ModelType,
    user_id: str,
    louvain_result: dict,
    *,
    _on_usage: Callable[[str, RunUsage], None] | None = None,
) -> list[CommunityInfo]:
    """Create/update Community nodes from Louvain partition results.

    Only communities with >= 2 members are materialized.
    Summaries are regenerated only when membership changes.
    """
    communities_map = louvain_result.get("communities", {})
    if not communities_map:
        return []

    # Group entity node IDs by community_id
    clusters: dict[int, list[int]] = {}
    for node_id, community_id in communities_map.items():
        clusters.setdefault(community_id, []).append(node_id)

    # Only process entities belonging to this user
    entity_nodes: dict[int, dict] = {}
    try:
        for node_id, props in db.get_nodes_by_label(ENTITY_LABEL):
            if props.get("user_id") == user_id:
                entity_nodes[node_id] = props
    except Exception:
        return []

    # Existing community nodes
    existing_communities: dict[int, tuple[int, int]] = {}  # community_id -> (node_id, member_count)
    try:
        for node_id, props in db.get_nodes_by_label(COMMUNITY_LABEL):
            if props.get("user_id") == user_id:
                cid = props.get("community_id")
                if cid is not None:
                    existing_communities[int(cid)] = (node_id, int(props.get("member_count", 0)))
    except Exception:
        pass

    results: list[CommunityInfo] = []
    active_cids: set[int] = set()

    for cid, member_ids in clusters.items():
        # Filter to only entity nodes for this user
        user_members = [nid for nid in member_ids if nid in entity_nodes]
        if len(user_members) < 2:
            continue

        active_cids.add(cid)
        member_names = [entity_nodes[nid].get("name", "") for nid in user_members]

        # Check if community already exists with same membership
        if cid in existing_communities:
            existing_nid, existing_count = existing_communities[cid]
            if existing_count == len(user_members):
                # Membership unchanged, read existing summary
                node = db.get_node(existing_nid)
                if node is not None:
                    props = node.properties if hasattr(node, "properties") else {}
                    results.append(
                        CommunityInfo(
                            community_id=str(cid),
                            name=props.get("name", ""),
                            summary=props.get("summary", ""),
                            member_count=existing_count,
                            member_entities=member_names,
                        )
                    )
                continue

        # Collect relations between members for context
        relation_texts: list[str] = []
        for nid in user_members:
            try:
                query = (
                    f"MATCH (a:{ENTITY_LABEL})-[r:{RELATION_EDGE}]->(b:{ENTITY_LABEL}) "
                    f"WHERE id(a) = $nid RETURN a.name, r.relation_type, b.name"
                )
                for row in db.execute(query, {"nid": nid}):
                    if isinstance(row, dict):
                        vals = list(row.values())
                        if len(vals) >= 3:
                            relation_texts.append(f"{vals[0]} -> {vals[1]} -> {vals[2]}")
            except Exception:
                pass

        # Generate summary via LLM
        entities_text = "\n".join(f"- {n}" for n in member_names)
        relations_text = "\n".join(f"- {r}" for r in relation_texts) if relation_texts else "(no relationships)"

        agent = Agent(model, system_prompt=COMMUNITY_SUMMARY_SYSTEM, output_type=CommunitySummaryOutput)
        try:
            result = await agent.run(COMMUNITY_SUMMARY_USER.format(entities=entities_text, relations=relations_text))
        except Exception:
            logger.warning("Community summary generation failed for cid=%s", cid, exc_info=True)
            continue

        if _on_usage is not None:
            _on_usage("community_summary", result.usage())

        name = result.output.name
        summary = result.output.summary

        import time

        now_ms = int(time.time() * 1000)

        # Create or update community node
        if cid in existing_communities:
            existing_nid, _ = existing_communities[cid]
            db.set_node_property(existing_nid, "name", name)
            db.set_node_property(existing_nid, "summary", summary)
            db.set_node_property(existing_nid, "member_count", len(user_members))
            db.set_node_property(existing_nid, "updated_at", now_ms)
            community_nid = existing_nid
            # Delete old HAS_MEMBER edges
            try:
                del_query = (
                    f"MATCH (c:{COMMUNITY_LABEL})-[r:{HAS_MEMBER_EDGE}]->(:{ENTITY_LABEL}) "
                    f"WHERE id(c) = $cid RETURN id(r)"
                )
                for row in db.execute(del_query, {"cid": existing_nid}):
                    if isinstance(row, dict):
                        vals = list(row.values())
                        if vals:
                            with contextlib.suppress(Exception):
                                db.delete_edge(int(vals[0]))
            except Exception:
                pass
        else:
            node = db.create_node(
                [COMMUNITY_LABEL],
                {
                    "name": name,
                    "summary": summary,
                    "community_id": cid,
                    "member_count": len(user_members),
                    "user_id": user_id,
                    "created_at": now_ms,
                    "updated_at": now_ms,
                },
            )
            community_nid = node.id if hasattr(node, "id") else node

        # Create HAS_MEMBER edges
        for nid in user_members:
            with contextlib.suppress(Exception):
                db.create_edge(community_nid, nid, HAS_MEMBER_EDGE)

        results.append(
            CommunityInfo(
                community_id=str(cid),
                name=name,
                summary=summary,
                member_count=len(user_members),
                member_entities=member_names,
            )
        )

    # Delete dissolved communities (existing but no longer active)
    for cid, (existing_nid, _) in existing_communities.items():
        if cid not in active_cids:
            with contextlib.suppress(Exception):
                db.delete_node(existing_nid)

    return results


def get_communities(db: GrafeoDBProtocol, user_id: str) -> list[CommunityInfo]:
    """Retrieve all Community nodes for a user."""
    try:
        nodes = db.get_nodes_by_label(COMMUNITY_LABEL)
    except Exception:
        return []

    results: list[CommunityInfo] = []
    for node_id, props in nodes:
        if props.get("user_id") != user_id:
            continue

        # Collect member entity names
        member_names: list[str] = []
        try:
            query = (
                f"MATCH (c:{COMMUNITY_LABEL})-[:{HAS_MEMBER_EDGE}]->(e:{ENTITY_LABEL}) WHERE id(c) = $cid RETURN e.name"
            )
            for row in db.execute(query, {"cid": node_id}):
                if isinstance(row, dict):
                    vals = list(row.values())
                    if vals and vals[0]:
                        member_names.append(str(vals[0]))
        except Exception:
            pass

        results.append(
            CommunityInfo(
                community_id=str(props.get("community_id", node_id)),
                name=props.get("name", ""),
                summary=props.get("summary", ""),
                member_count=int(props.get("member_count", 0)),
                member_entities=member_names,
            )
        )
    return results


def get_community_context(db: GrafeoDBProtocol, entity_names: list[str], user_id: str) -> list[CommunityInfo]:
    """Find communities that contain any of the given entities."""
    if not entity_names:
        return []

    try:
        nodes = db.get_nodes_by_label(COMMUNITY_LABEL)
    except Exception:
        return []

    name_set = {n.lower() for n in entity_names}
    results: list[CommunityInfo] = []

    for node_id, props in nodes:
        if props.get("user_id") != user_id:
            continue

        # Check if any member entity matches
        member_names: list[str] = []
        try:
            query = (
                f"MATCH (c:{COMMUNITY_LABEL})-[:{HAS_MEMBER_EDGE}]->(e:{ENTITY_LABEL}) WHERE id(c) = $cid RETURN e.name"
            )
            for row in db.execute(query, {"cid": node_id}):
                if isinstance(row, dict):
                    vals = list(row.values())
                    if vals and vals[0]:
                        member_names.append(str(vals[0]))
        except Exception:
            pass

        if any(m.lower() in name_set for m in member_names):
            results.append(
                CommunityInfo(
                    community_id=str(props.get("community_id", node_id)),
                    name=props.get("name", ""),
                    summary=props.get("summary", ""),
                    member_count=int(props.get("member_count", 0)),
                    member_entities=member_names,
                )
            )

    return results
