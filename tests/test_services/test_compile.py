"""Tests for Layer E minimal compile: services/compile.compile_graph."""

from __future__ import annotations

import grafeo

from grafeo_memory.services.compile import compile_graph


class StubEmbedder:
    """Embedder that returns the same vector for the given names (merge candidates)."""

    def __init__(self, same: set[str], dims: int = 8):
        self._same = same
        self._dims = dims

    def embed(self, texts: list[str]) -> list[list[float]]:
        out = []
        for t in texts:
            v = [1.0 if t in self._same else 0.0] + [0.0] * (self._dims - 1)
            out.append(v)
        return out

    @property
    def dimensions(self) -> int:
        return self._dims


def _mk_entity(db, name: str, etype: str, uid: str = "u"):
    node = db.create_node(["Entity"], {"name": name, "entity_type": etype, "user_id": uid})
    return {"id": node.id if hasattr(node, "id") else int(node)}


def test_compile_merges_similar_entities_and_rewires_edges():
    db = grafeo.GrafeoDB()
    canon = _mk_entity(db, "Alice Anderson", "Person")
    alias = _mk_entity(db, "alice anderson", "Person")  # same-type, embeds identical
    other = _mk_entity(db, "Acme", "Organization")
    db.create_edge(alias["id"], other["id"], "RELATION", {"relation_type": "WORKS_AT"})
    db.create_edge(other["id"], alias["id"], "RELATION", {"relation_type": "LOCATED_IN"})

    res = compile_graph(db, "u", embedder=StubEmbedder({"Alice Anderson", "alice anderson"}), merge_threshold=0.9)

    assert res.entities_before == 3
    assert res.entities_after == 2
    assert len(res.merges) == 1
    assert res.merges[0].alias_name == "alice anderson"
    assert res.merges[0].canonical_name == "Alice Anderson"
    assert res.merges[0].rung == 3
    # alias node deleted, its edges rewired onto the canonical node
    assert db.get_node(alias["id"]) is None
    assert db.get_node(canon["id"]) is not None
    rows = db.execute(
        "MATCH (a)-[r]->(b) WHERE id(a) = $c RETURN type(r) AS lbl, id(b) AS other",
        {"c": canon["id"]},
    )
    assert {r["lbl"] for r in rows} == {"RELATION"}
    assert res.edges_added == 2


def test_compile_without_embedder_only_reports():
    db = grafeo.GrafeoDB()
    a = _mk_entity(db, "Acme", "Organization")
    p = _mk_entity(db, "Bob", "Person")
    # WORKS_AT Organization -> Person violates domain/range
    db.create_edge(a["id"], p["id"], "RELATION", {"relation_type": "WORKS_AT"})

    res = compile_graph(db, "u", embedder=None)
    assert res.merges == []
    assert res.entities_after == 2
    assert len(res.relation_violations) == 1
    assert res.relation_violations[0]["relation_type"] == "WORKS_AT"


def test_compile_noop_on_empty_graph():
    db = grafeo.GrafeoDB()
    res = compile_graph(db, "nobody", embedder=StubEmbedder(set()))
    assert res.entities_before == 0 and res.entities_after == 0
    assert res.merges == [] and res.relation_violations == []


def test_manager_compile_runs_one_pass():
    """MemoryManager.compile() runs one compile pass and returns its CompileResult."""
    from mock_llm import MockEmbedder, make_test_model

    from grafeo_memory import MemoryConfig, MemoryManager, PipelineConfig

    out = {
        "facts": ["alice works at acme"],
        "entities": [{"name": "alice", "entity_type": "person"}, {"name": "acme", "entity_type": "organization"}],
        "relations": [{"source": "alice", "target": "acme", "relation_type": "works_at"}],
    }
    model = make_test_model([out])
    cfg = MemoryConfig(db_path=None, user_id="u", embedding_dimensions=16, pipeline=PipelineConfig(compile=True))
    m = MemoryManager(model, cfg, embedder=MockEmbedder(16))
    m.add("Alice works at Acme")
    res = m.compile()
    assert res.entities_after >= 1
    m.close()


# --- 0.3.0 release review regressions -------------------------------------------


def test_compile_keeps_edge_properties_and_distinct_relations():
    """Rewired edges keep relation_type and validity; a WORKS_AT is not dropped because
    the survivor already has a KNOWS edge to the same node."""
    db = grafeo.GrafeoDB()
    canon = _mk_entity(db, "Alice", "Person")
    alias = _mk_entity(db, "alice", "Person")
    acme = _mk_entity(db, "Acme", "Organization")
    db.create_edge(canon["id"], acme["id"], "RELATION", {"relation_type": "KNOWS"})
    db.create_edge(alias["id"], acme["id"], "RELATION", {"relation_type": "WORKS_AT", "valid_from": 5})

    res = compile_graph(db, "u", embedder=StubEmbedder({"Alice", "alice"}), merge_threshold=0.9)

    assert res.edges_added == 1
    rows = db.execute(
        "MATCH (a)-[r]->(b) WHERE id(a) = $c RETURN properties(r) AS p",
        {"c": canon["id"]},
    )
    props = sorted((r["p"] for r in rows), key=lambda p: p["relation_type"])
    assert props == [{"relation_type": "KNOWS"}, {"relation_type": "WORKS_AT", "valid_from": 5}]


def test_compile_never_merges_across_graph_name_scopes():
    db = grafeo.GrafeoDB()
    work = db.create_node(["Entity"], {"name": "Alice", "entity_type": "Person", "user_id": "u", "graph_name": "work"})
    home = db.create_node(["Entity"], {"name": "alice", "entity_type": "Person", "user_id": "u", "graph_name": "home"})

    res = compile_graph(db, "u", embedder=StubEmbedder({"Alice", "alice"}), merge_threshold=0.9)

    assert res.merges == []
    assert db.get_node(work.id) is not None and db.get_node(home.id) is not None


def test_compile_carries_names_merged_into_the_alias():
    db = grafeo.GrafeoDB()
    canon = _mk_entity(db, "Alice", "Person")
    alias = _mk_entity(db, "alice", "Person")
    db.set_node_property(alias["id"], "merged_from", ["Ally"])
    compile_graph(db, "u", embedder=StubEmbedder({"Alice", "alice"}), merge_threshold=0.9)
    assert list(db.get_node(canon["id"]).properties()["merged_from"]) == ["Ally", "alice"]


def test_compile_compares_type_spellings_as_one_type():
    db = grafeo.GrafeoDB()
    _mk_entity(db, "Alice", "Person")
    _mk_entity(db, "alice", "person")
    res = compile_graph(db, "u", embedder=StubEmbedder({"Alice", "alice"}), merge_threshold=0.9)
    assert len(res.merges) == 1 and res.entities_after == 1
