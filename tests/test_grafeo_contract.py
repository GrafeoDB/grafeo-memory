"""Engine contract: the grafeo behaviors grafeo-memory relies on.

Each test pins one behavior of the ``grafeo`` Python package that this package
depends on, using the same calls and query shapes as the source. When an engine
release changes one of them, the failure names the behavior and what in
grafeo-memory depends on it, so the fix starts in the right place.

The scheduled "Engine compatibility" workflow runs this module against the
newest grafeo on PyPI (pre-releases included). Do not match on error codes:
they may change between engine releases (for example "GRAFEO-X001").
"""

from __future__ import annotations

import grafeo
import pytest

from grafeo_memory._graph import detach_delete_node, node_labels
from grafeo_memory.search.vector import _get_props

VEC_A = [1.0, 0.0, 0.0, 0.0]
VEC_B = [0.0, 1.0, 0.0, 0.0]
VEC_AB = [0.7071, 0.7071, 0.0, 0.0]


def _changed(what: str, used_by: str) -> str:
    return f"grafeo {grafeo.__version__} changed: {what}. grafeo-memory relies on it in {used_by}."


@pytest.fixture
def db():
    database = grafeo.GrafeoDB()
    yield database
    database.close()


def _rows(result) -> list[dict]:
    return [row for row in result if isinstance(row, dict)]


# ---------------------------------------------------------------------------
# Package surface
# ---------------------------------------------------------------------------


class TestPackageSurface:
    def test_required_features_compiled_in(self):
        features = set(grafeo.features())
        missing = {"vector-index", "text-index", "hybrid-search", "algos"} - features
        assert not missing, _changed(f"the wheel lacks features {sorted(missing)}", "search and graph algorithms")

    def test_errors_are_grafeo_error_and_runtime_error(self, db):
        assert issubclass(grafeo.GrafeoError, RuntimeError), _changed(
            "GrafeoError is no longer a RuntimeError", "search/vector.py (except RuntimeError around vector_search)"
        )
        with pytest.raises(grafeo.GrafeoError):
            db.create_edge(0, 1, "X")

    def test_build_info_reports_version(self):
        info = grafeo.build_info()
        assert info["version"] == grafeo.__version__, _changed("build_info()['version']", "bug reports")


# ---------------------------------------------------------------------------
# Direct API: nodes
# ---------------------------------------------------------------------------


class TestNodes:
    def test_create_node_returns_node_with_id_labels_properties(self, db):
        node = db.create_node(["Memory"], {"text": "a", "user_id": "u"})
        assert isinstance(node.id, int), _changed("create_node().id is not an int", "every memory id")
        assert node_labels(node) == ["Memory"], _changed("Node.labels", "_owned_memory() label check")
        assert _get_props(node) == {"text": "a", "user_id": "u"}, _changed("Node.properties", "_get_props()")

    def test_get_node_round_trip_and_missing(self, db):
        node = db.create_node(["Memory"], {"text": "a"})
        fetched = db.get_node(node.id)
        assert fetched is not None
        assert _get_props(fetched)["text"] == "a"
        assert node_labels(fetched) == ["Memory"]
        assert db.get_node(999_999) is None, _changed("get_node(missing) no longer returns None", "every id lookup")

    def test_missing_property_is_absent(self, db):
        node = db.create_node(["Memory"], {"text": "a"})
        props = _get_props(db.get_node(node.id))
        assert "expired_at" not in props, _changed(
            "a property never set is reported", "props.get('expired_at') is None (active memories)"
        )

    def test_get_nodes_by_label_returns_id_props_tuples(self, db):
        a = db.create_node(["Memory"], {"text": "a"})
        db.create_node(["Entity"], {"name": "e"})
        rows = db.get_nodes_by_label("Memory")
        assert rows == [(a.id, {"text": "a"})], _changed(
            "get_nodes_by_label() shape or label filtering", "get_all(), stats(), session chains"
        )

    def test_set_node_property_overwrites(self, db):
        node = db.create_node(["Memory"], {"text": "v1"})
        db.set_node_property(node.id, "text", "v2")
        db.set_node_property(node.id, "expired_at", 123)
        props = _get_props(db.get_node(node.id))
        assert props["text"] == "v2"
        assert props["expired_at"] == 123

    def test_set_node_property_on_missing_node_raises(self, db):
        with pytest.raises(grafeo.GrafeoError):
            db.set_node_property(999_999, "text", "x")

    def test_delete_node_without_edges(self, db):
        node = db.create_node(["Memory"], {"text": "a"})
        assert db.delete_node(node.id) is True
        assert db.delete_node(node.id) is False, _changed("delete_node(missing) no longer returns False", "delete()")

    def test_delete_node_with_edges_raises(self, db):
        a = db.create_node(["Memory"], {})
        b = db.create_node(["Entity"], {})
        db.create_edge(a.id, b.id, "HAS_ENTITY")
        with pytest.raises(grafeo.GrafeoError):
            db.delete_node(a.id)
        assert db.get_node(a.id) is not None

    def test_node_ids_are_not_reused(self, db):
        nodes = [db.create_node(["Memory"], {}) for _ in range(3)]
        top = nodes[-1].id
        assert db.delete_node(top) is True
        fresh = db.create_node(["Memory"], {})
        assert fresh.id > top, _changed(
            "node ids are reused after a delete", "history.py (History nodes are keyed by memory_id)"
        )

    def test_batch_create_nodes_with_props(self, db):
        ids = db.batch_create_nodes_with_props("Memory", [{"text": "a"}, {"text": "b"}])
        assert len(ids) == 2
        assert all(isinstance(i, int) for i in ids)
        assert [_get_props(db.get_node(i))["text"] for i in ids] == ["a", "b"], _changed(
            "batch_create_nodes_with_props() id order", "add_batch(infer=False)"
        )


# ---------------------------------------------------------------------------
# Direct API: edges
# ---------------------------------------------------------------------------


class TestEdges:
    def test_create_and_delete_edge(self, db):
        a = db.create_node(["Entity"], {"name": "a"})
        b = db.create_node(["Entity"], {"name": "b"})
        edge = db.create_edge(a.id, b.id, "RELATION", {"relation_type": "knows"})
        rows = _rows(db.execute("MATCH (s)-[r:RELATION]->(t) RETURN id(r), r.relation_type"))
        assert len(rows) == 1
        edge_id = next(iter(rows[0].values()))
        assert edge.id == edge_id
        assert db.delete_edge(edge_id) is True
        assert db.delete_edge(edge_id) is False

    def test_create_edge_to_missing_node_raises(self, db):
        a = db.create_node(["Memory"], {})
        with pytest.raises(grafeo.GrafeoError):
            db.create_edge(a.id, 999_999, "SUPERSEDES")


# ---------------------------------------------------------------------------
# Queries: result format and the query shapes the package runs
# ---------------------------------------------------------------------------


class TestQueryResults:
    def test_rows_are_dicts_in_return_order(self, db):
        db.create_node(["Entity"], {"name": "e", "user_id": "u"})
        rows = _rows(db.execute("MATCH (e:Entity) RETURN id(e), e.name, e.user_id"))
        assert len(rows) == 1
        assert list(rows[0].keys()) == ["id(e)", "e.name", "e.user_id"], _changed(
            "column order or names of unaliased RETURN items", "every list(row.values()) in the package"
        )

    def test_aliases_name_columns(self, db):
        db.create_node(["History"], {"memory_id": 1, "event": "ADD"})
        rows = _rows(db.execute("MATCH (h:History) RETURN id(h) AS hid, h.event AS event, h.old_text AS old_text"))
        assert rows == [{"hid": rows[0]["hid"], "event": "ADD", "old_text": None}], _changed(
            "aliased columns or missing properties in RETURN", "history.get_history()"
        )

    def test_result_truthiness_and_length(self, db):
        empty = db.execute("MATCH (n:Nothing) RETURN id(n)")
        assert not empty
        assert len(empty) == 0
        db.create_node(["Memory"], {})
        assert db.execute("MATCH (n:Memory) RETURN id(n)"), _changed(
            "QueryResult truthiness", "scoring._batch_topology_scores (if shared_rows)"
        )

    def test_unsupplied_parameter_raises(self, db):
        with pytest.raises(grafeo.GrafeoError):
            db.execute("MATCH (n) WHERE n.x = $missing RETURN n")

    def test_entity_lookup_shapes(self, db):
        e = db.create_node(["Entity"], {"name": "Alix", "user_id": "u", "graph_name": "g"})
        exact = _rows(
            db.execute(
                "MATCH (e:Entity) WHERE e.name = $name AND e.user_id = $uid AND e.graph_name = $gn RETURN id(e)",
                {"name": "Alix", "uid": "u", "gn": "g"},
            )
        )
        lowered = _rows(
            db.execute(
                "MATCH (e:Entity) WHERE toLower(e.name) = $name AND e.user_id = $uid RETURN id(e)",
                {"name": "alix", "uid": "u"},
            )
        )
        listed = _rows(db.execute("MATCH (e:Entity) WHERE e.name IN $names RETURN e.name", {"names": ["Alix", "Gus"]}))
        assert exact == [{"id(e)": e.id}]
        assert lowered == [{"id(e)": e.id}], _changed("toLower()", "graph_search case-insensitive fallback")
        assert listed == [{"e.name": "Alix"}]

    def test_memory_entity_traversals(self, db):
        m1 = db.create_node(["Memory"], {"text": "m1", "user_id": "u"})
        m2 = db.create_node(["Memory"], {"text": "m2", "user_id": "u"})
        alix = db.create_node(["Entity"], {"name": "alix", "user_id": "u"})
        acme = db.create_node(["Entity"], {"name": "acme", "user_id": "u"})
        db.create_edge(m1.id, alix.id, "HAS_ENTITY")
        db.create_edge(m2.id, acme.id, "HAS_ENTITY")
        db.create_edge(alix.id, acme.id, "RELATION", {"relation_type": "works_at"})

        relations = _rows(
            db.execute(
                "MATCH (m:Memory)-[:HAS_ENTITY]->(e:Entity)-[r:RELATION]->(t:Entity) "
                "WHERE id(m) = $nid RETURN e.name, r.relation_type, t.name",
                {"nid": m1.id},
            )
        )
        assert [list(r.values()) for r in relations] == [["alix", "works_at", "acme"]]

        two_hop = _rows(
            db.execute(
                "MATCH (e1:Entity)-[:RELATION]->(e2:Entity)<-[:HAS_ENTITY]-(m:Memory) "
                "WHERE e1.name IN $names AND e1.user_id = $uid AND m.user_id = $uid "
                "RETURN DISTINCT id(m), m.text",
                {"names": ["alix"], "uid": "u"},
            )
        )
        assert [list(r.values()) for r in two_hop] == [[m2.id, "m2"]], _changed(
            "2-hop pattern or RETURN DISTINCT", "graph_search(search_depth=2)"
        )

        others = _rows(
            db.execute(
                "MATCH (other:Memory)-[:HAS_ENTITY]->(e:Entity) WHERE id(e) = $eid AND id(other) <> $mid "
                "RETURN id(other) LIMIT 1",
                {"eid": alix.id, "mid": m1.id},
            )
        )
        assert others == []
        counted = _rows(db.execute("MATCH (:Entity)-[r:RELATION]->(:Entity) RETURN count(r)", {}))
        assert list(counted[0].values()) == [1], _changed("count() result", "stats().relation_count")

    def test_variable_length_chain_ordered(self, db):
        ids = [db.create_node(["Memory"], {"text": t, "created_at": i, "user_id": "u"}).id for i, t in enumerate("abc")]
        db.create_edge(ids[0], ids[1], "LEADS_TO", {"sequence": 0})
        db.create_edge(ids[1], ids[2], "LEADS_TO", {"sequence": 1})
        rows = _rows(
            db.execute(
                "MATCH (m:Memory)-[:LEADS_TO*1..5]->(rest:Memory) WHERE id(m) = $mid AND rest.user_id = $uid "
                "RETURN id(rest), rest.text, rest.created_at, rest.session_id ORDER BY rest.created_at",
                {"mid": ids[0], "uid": "u"},
            )
        )
        assert [list(r.values()) for r in rows] == [[ids[1], "b", 1, None], [ids[2], "c", 2, None]], _changed(
            "variable-length expansion or ORDER BY", "temporal_chain()"
        )

    def test_detach_delete_query(self, db):
        m = db.create_node(["Memory"], {"text": "m"})
        e = db.create_node(["Entity"], {"name": "e"})
        db.create_edge(m.id, e.id, "HAS_ENTITY")
        db.create_edge(e.id, m.id, "MENTIONS")
        result = db.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": m.id})
        assert db.get_node(m.id) is None, _changed("DETACH DELETE by id", "_graph.detach_delete_node()")
        assert db.get_node(e.id) is not None
        assert result.counters["nodes_deleted"] == 1
        assert result.counters["edges_deleted"] == 2
        # Deleting an id that does not exist is not an error.
        db.execute("MATCH (n) WHERE id(n) = $id DETACH DELETE n", {"id": m.id})

    def test_detach_delete_helper(self, db):
        m = db.create_node(["Memory"], {})
        e = db.create_node(["Entity"], {})
        db.create_edge(m.id, e.id, "HAS_ENTITY")
        assert detach_delete_node(db, m.id) is True
        assert detach_delete_node(db, m.id) is False

    def test_set_through_pattern(self, db):
        m = db.create_node(["Memory"], {})
        h = db.create_node(["History"], {"event": "ADD"})
        db.create_edge(m.id, h.id, "HAS_HISTORY")
        db.execute(
            "MATCH (m)-[:HAS_HISTORY]->(h:History) WHERE id(m) = $mid SET h.memory_id = $mid",
            {"mid": m.id},
        )
        assert _get_props(db.get_node(h.id))["memory_id"] == m.id, _changed(
            "SET through a matched pattern", "history.adopt_legacy_history()"
        )


# ---------------------------------------------------------------------------
# Indexes and search
# ---------------------------------------------------------------------------


def _vector_db(db):
    db.create_vector_index("Memory", "embedding", dimensions=4, metric="cosine")
    db.create_text_index("Memory", "text")
    a = db.create_node(["Memory"], {"text": "alix likes hiking", "user_id": "u1"})
    db.set_node_property(a.id, "embedding", VEC_A)
    ids = db.batch_create_nodes_with_props("Memory", [{"text": "gus likes chess", "user_id": "u2", "embedding": VEC_B}])
    return a.id, ids[0]


class TestIndexesAndSearch:
    def test_index_creation_is_idempotent(self, db):
        for _ in range(2):
            db.create_vector_index("Memory", "embedding", dimensions=4, metric="cosine")
            db.create_text_index("Memory", "text")
            for prop in ("user_id", "created_at", "memory_type", "graph_name", "memory_id", "name"):
                db.create_property_index(prop)
        assert db.has_property_index("memory_id")

    def test_vector_search_shape_order_and_filters(self, db):
        a, b = _vector_db(db)
        hits = db.vector_search("Memory", "embedding", VEC_A, 5)
        assert [h[0] for h in hits] == [a, b], _changed("vector_search ordering", "search_similar()")
        assert all(isinstance(nid, int) and isinstance(dist, float) for nid, dist in hits)
        assert hits[0][1] < 0.01, _changed("cosine distance of identical vectors", "1 - distance scoring")
        assert abs(hits[1][1] - 1.0) < 0.01
        filtered = db.vector_search("Memory", "embedding", VEC_A, 5, filters={"user_id": "u2"})
        assert [h[0] for h in filtered] == [b], _changed("vector_search equality filters", "per-user scoping")

    def test_vector_property_set_after_index_is_searchable(self, db):
        db.create_vector_index("Memory", "embedding", dimensions=4, metric="cosine")
        node = db.create_node(["Memory"], {"text": "late"})
        db.set_node_property(node.id, "embedding", VEC_AB)
        assert [h[0] for h in db.vector_search("Memory", "embedding", VEC_AB, 1)] == [node.id], _changed(
            "set_node_property() of a float list no longer feeds the vector index", "_create_memory()"
        )

    def test_hybrid_search(self, db):
        a, _ = _vector_db(db)
        hits = db.hybrid_search("Memory", "text", "embedding", "hiking", 5, query_vector=VEC_A, fusion="rrf", rrf_k=1)
        assert hits, _changed("hybrid_search() returned nothing", "hybrid_search()")
        assert all(isinstance(nid, int) for nid, _ in hits)
        assert a in [nid for nid, _ in hits]

    def test_mmr_search(self, db):
        a, _ = _vector_db(db)
        hits = db.mmr_search("Memory", "embedding", VEC_A, 1, fetch_k=4, lambda_mult=0.5, filters={"user_id": "u1"})
        assert [nid for nid, _ in hits] == [a], _changed("mmr_search() signature or filters", "diverse_search()")

    def test_deleted_node_leaves_vector_search(self, db):
        a, b = _vector_db(db)
        detach_delete_node(db, a)
        assert [h[0] for h in db.vector_search("Memory", "embedding", VEC_A, 5)] == [b]


# ---------------------------------------------------------------------------
# Graph algorithms
# ---------------------------------------------------------------------------


def _two_triangles(db) -> list[int]:
    ids = [db.create_node(["Entity"], {"name": f"e{i}", "user_id": "u"}).id for i in range(6)]
    for a, b in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (2, 3)]:
        db.create_edge(ids[a], ids[b], "RELATION", {"relation_type": "r"})
    return ids


class TestAlgorithms:
    def test_pagerank_and_betweenness_shapes(self, db):
        ids = _two_triangles(db)
        ranks = db.algorithms.pagerank(0.85, 100, 1e-6)
        centrality = db.algorithms.betweenness_centrality(True)
        assert set(ranks) == set(ids)
        assert set(centrality) == set(ids)
        assert all(isinstance(v, float) for v in ranks.values())
        assert abs(sum(ranks.values()) - 1.0) < 1e-3
        # The bridge nodes (2 and 3) are the only ones on shortest paths between the triangles.
        assert centrality[ids[2]] > 0 and centrality[ids[3]] > 0
        assert centrality[ids[0]] == 0.0

    def test_louvain_output(self, db):
        ids = _two_triangles(db)
        result = db.algorithms.louvain(1.0)
        assert isinstance(result, dict) and "communities" in result, _changed(
            "louvain() result shape", "_recompute_graph_metrics() and communities.py"
        )
        communities = result["communities"]
        assert set(communities) == set(ids)
        assert communities[ids[0]] == communities[ids[1]] == communities[ids[2]]
        assert communities[ids[3]] == communities[ids[4]] == communities[ids[5]]
        assert communities[ids[0]] != communities[ids[3]]
        # Deterministic since 0.5.44: numbered from 0 by smallest node id, same every run.
        assert communities[ids[0]] == 0 and communities[ids[3]] == 1
        assert db.algorithms.louvain(1.0)["communities"] == communities, _changed(
            "louvain() is not deterministic", "community_id matching in materialize_communities_async()"
        )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_single_file_database_round_trip(self, tmp_path):
        path = tmp_path / "memory.grafeo"
        db = grafeo.GrafeoDB(str(path))
        db.create_vector_index("Memory", "embedding", dimensions=4, metric="cosine")
        m = db.create_node(["Memory"], {"text": "kept", "user_id": "u"})
        db.set_node_property(m.id, "embedding", VEC_A)
        h = db.create_node(["History"], {"memory_id": m.id, "event": "ADD", "timestamp": 1})
        db.close()
        assert path.is_file(), _changed("a .grafeo path is not a single file", "README (db_path)")

        db = grafeo.GrafeoDB(str(path))
        try:
            assert _get_props(db.get_node(m.id))["text"] == "kept"
            assert _get_props(db.get_node(h.id))["memory_id"] == m.id
            db.create_vector_index("Memory", "embedding", dimensions=4, metric="cosine")
            assert [x[0] for x in db.vector_search("Memory", "embedding", VEC_A, 1)] == [m.id]
        finally:
            db.close()

    def test_directory_database_round_trip(self, tmp_path):
        path = tmp_path / "memory.db"
        db = grafeo.GrafeoDB(str(path))
        m = db.create_node(["Memory"], {"text": "kept"})
        db.close()
        db = grafeo.GrafeoDB(str(path))
        try:
            assert _get_props(db.get_node(m.id))["text"] == "kept"
        finally:
            db.close()


# ---------------------------------------------------------------------------
# Change data capture: why history.py does not use it
# ---------------------------------------------------------------------------


class TestChangeDataCapture:
    def test_cdc_is_off_by_default(self, db):
        assert db.cdc_enabled is False, _changed(
            "CDC is on by default", "the history design notes in history.py (re-check them)"
        )

    def test_cdc_history_does_not_survive_reopen(self, tmp_path):
        path = str(tmp_path / "cdc.grafeo")
        db = grafeo.GrafeoDB(path)
        db.enable_cdc()
        node = db.create_node(["Memory"], {"text": "v1"})
        assert db.node_history(node.id), "CDC recorded nothing for a direct write"
        db.close()
        db = grafeo.GrafeoDB(path)
        try:
            assert db.node_history(node.id) == [], _changed(
                "CDC history now survives reopen",
                "history.py, which keeps its own :History nodes because it did not (the design can be re-checked)",
            )
        finally:
            db.close()
