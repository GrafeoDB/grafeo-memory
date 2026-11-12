"""Layers A, B, D (rungs 3-4, interval closing, deterministic-first), E, F and G end to end.

Mock LLM (queued outputs), FakeDecisionModel and a controllable embedder: no network.
"""

from __future__ import annotations

import pytest
from pydantic_ai.messages import ModelResponse, TextPart, ToolCallPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from grafeo_memory import MemoryConfig, MemoryManager
from grafeo_memory.common.types import (
    BucketConfig,
    CompileConfig,
    Engine,
    GrowthConfig,
    InstanceScopeConfig,
    MemoryAction,
    OntologyConfig,
    PipelineConfig,
    PolicyConfig,
    ProceduralConfig,
    RetrievalConfig,
    StageConfig,
    TaxonomyConfig,
    TiersConfig,
    WorkingConfig,
)
from grafeo_memory.providers.decision import FakeDecisionModel
from grafeo_memory.providers.decision.types import CategoricalQuestion, Decision


class QueueModel:
    """Queued outputs: dicts answer structured calls, strings answer text calls."""

    def __init__(self, outputs: list):
        self.outputs = list(outputs)
        self.calls = 0

    def model(self) -> FunctionModel:
        def handler(messages: list, info: AgentInfo) -> ModelResponse:
            self.calls += 1
            out = self.outputs.pop(0)
            if info.output_tools:
                return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args=out)])
            return ModelResponse(parts=[TextPart(content=str(out))])

        return FunctionModel(function=handler)


class VecEmbedder:
    """Same vector for every text unless a text is mapped explicitly."""

    def __init__(self, dims: int = 4, table: dict[str, list[float]] | None = None):
        self._dims = dims
        self.table = table or {}

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self.table.get(t, [1.0] + [0.0] * (self._dims - 1)) for t in texts]

    @property
    def dimensions(self) -> int:
        return self._dims


def _facts(*facts, entities=(), relations=()):
    return {"facts": list(facts), "entities": list(entities), "relations": list(relations)}


NO_ENTITIES = {"entities": [], "relations": []}


def _p(m, memory_id) -> dict:
    from grafeo_memory.core.search.vector import _get_props

    return _get_props(m._db.get_node(int(memory_id)))


def _mgr(outputs, embedder=None, **cfg) -> tuple[MemoryManager, QueueModel]:
    q = QueueModel(outputs)
    config = MemoryConfig(user_id="u", embedding_dimensions=4, **cfg)
    return MemoryManager(q.model(), config, embedder=embedder or VecEmbedder()), q


def _unsure_update(conf: float = 0.3):
    def handler(state, questions):
        out = []
        for q in questions:
            assert isinstance(q, CategoricalQuestion)
            upd = next((c for c in q.choices if c.startswith("UPDATE_")), "ADD")
            out.append(Decision(q.question_id, upd, conf, conf, runner_up=0.2))
        return out

    return FakeDecisionModel(handler)


# --- Layer G ---------------------------------------------------------------------


class TestBucket:
    def _parked(self, outputs, **extra):
        cfg = {
            "decision_model": _unsure_update(),
            "pipeline": PipelineConfig(memories=StageConfig(engine=Engine.DECISION)),
            "bucket": BucketConfig(enabled=True, floor=0.5),
        }
        cfg.update(extra)
        m, q = _mgr(outputs, **cfg)
        m.add("Alice works at Acme")
        res = m.add("Alice works at Beta")
        return m, q, res

    def test_low_confidence_update_parks_and_keeps_both(self):
        m, _q, res = self._parked([_facts("alice works at acme"), _facts("alice works at beta")])
        ev = res[0]
        assert ev.action == MemoryAction.ADD and ev.open_decision_id and ev.rung == "parked"
        ods = m.list_open_decisions()
        assert len(ods) == 1 and ods[0].kind == "contradiction" and ods[0].proposed == "replace"
        assert len(m.get_all()) == 2  # nothing forced: both facts stand, the new one provisional
        assert m.stats().layers["provisional"] == 1

    def test_user_resolution_writes_back_and_certifies(self):
        m, _q, res = self._parked([_facts("alice works at acme"), _facts("alice works at beta")])
        od = m.list_open_decisions()[0]
        assert m.resolve_open_decision(od.id, "replace")
        texts = [r.text for r in m.get_all()]
        assert texts == ["alice works at beta"]  # the old memory expired
        node = _p(m, res[0].memory_id)
        assert node["status"] == "certified" and not node["provisional"]
        assert m.list_open_decisions() == []
        assert not m.resolve_open_decision(od.id, "replace")  # already resolved

    def test_compile_sweep_self_resolves(self):
        m, _q, _res = self._parked([_facts("alice works at acme"), _facts("alice works at beta"), "keep_both"])
        out = m.compile()
        assert out.bucket["self_resolved"] == 1
        assert m.list_open_decisions(status="resolved")[0].resolved_by == "self"
        assert m.stats().layers["bucket"]["self_resolved_share"] == 1.0

    def test_dedup_and_expiry(self):
        m, _q, _res = self._parked(
            [_facts("alice works at acme"), _facts("alice works at beta"), _facts("alice works at beta")],
            bucket=BucketConfig(enabled=True, floor=0.5, self_resolution=False, max_open=1),
        )
        m.add("Alice works at Beta")
        assert len(m.list_open_decisions()) == 1  # second parked question either re-abstained or overflowed

    def test_hybrid_parks_instead_of_escalating(self):
        m, q = _mgr(
            [_facts("alice works at acme"), _facts("alice works at beta")],
            decision_model=_unsure_update(0.3),
            pipeline=PipelineConfig(memories=StageConfig(engine=Engine.HYBRID)),
            bucket=BucketConfig(enabled=True, floor=0.5, self_resolution=False),
        )
        m.add("a")
        m.add("b")
        assert q.calls == 2  # two extractions, no escalation call
        assert len(m.list_open_decisions()) == 1


# --- Layer F ---------------------------------------------------------------------


class TestPolicy:
    def test_compliance_rejects_secrets(self):
        m, _q = _mgr(
            [_facts("the api key is sk-abcdefghijklmnopqrstuv", "alice likes tea")], policy=PolicyConfig(enabled=True)
        )
        res = m.add("x")
        assert [e.text for e in res] == ["alice likes tea"]
        assert m.stats().layers["rule_hits"]["compliance.secret"] == 1
        assert not m.add("my password is hunter22", infer=False)

    def test_entrenchment_rule_on_real_target(self):
        q = QueueModel([_facts("alice works at acme"), _facts("alice works at beta")])
        m = MemoryManager(
            q.model(),
            MemoryConfig(user_id="u", embedding_dimensions=4, policy=PolicyConfig(enabled=True, conservative=False)),
            embedder=VecEmbedder(),
        )
        first = m.add("a")[0]
        m.certify(first.memory_id)
        q.outputs.append(
            {"decisions": [{"action": "UPDATE", "text": "alice works at beta", "target_memory_id": first.memory_id}]}
        )
        res = m.add("b")
        assert res[0].action == MemoryAction.ADD and res[0].rung == "rule:entrenchment.certified"
        assert len(m.get_all()) == 2

    def test_lifecycle_promotes_after_observations(self):
        q = QueueModel([_facts("alice likes tea")])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                policy=PolicyConfig(enabled=True),
                compilation=CompileConfig(promote_after=2),
            ),
            embedder=VecEmbedder(),
        )
        first = m.add("a")[0]
        for _ in range(2):
            q.outputs += [_facts("alice likes tea"), {"decisions": [{"action": "NONE", "text": "alice likes tea"}]}]
            m.add("again")
        assert _p(m, first.memory_id)["status"] == "certified"
        assert m.stats().layers["deterministic_share"] is not None

    def test_relation_shape_warning_flags(self):
        q = QueueModel(
            [
                _facts(
                    "acme works at alice",
                    entities=[
                        {"name": "Acme", "entity_type": "Organization"},
                        {"name": "Alice", "entity_type": "Person"},
                    ],
                    relations=[{"source": "Acme", "target": "Alice", "relation_type": "WORKS_AT"}],
                )
            ]
        )
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                policy=PolicyConfig(enabled=True, shape_severity={"relation_domain_range": "warning"}),
            ),
            embedder=VecEmbedder(),
        )
        m.add("x")
        assert m._layer_counts.get("shape:relation_domain_range:flagged") == 1


# --- Layer B ---------------------------------------------------------------------


class TestScoping:
    def _two(self, **tax):
        q = QueueModel([_facts("alice has a job interview at acme"), _facts("alice adopted a dog")])
        m = MemoryManager(
            q.model(),
            MemoryConfig(user_id="u", embedding_dimensions=4, taxonomy=TaxonomyConfig(enabled=True, **tax)),
            embedder=VecEmbedder(),
        )
        a = m.add("x")[0]
        q.outputs.append({"decisions": [{"action": "ADD", "text": "alice adopted a dog"}]})
        b = m.add("y")[0]
        return m, q, a, b

    def test_facts_get_subject_tags(self):
        m, _q, a, b = self._two()
        assert _p(m, a.memory_id)["subjects"] == ["career"]
        assert _p(m, b.memory_id)["subjects"] == ["home"]

    def test_pinned_subjects_filter(self):
        m, q, a, _b = self._two(mode="filter")
        q.outputs.append(NO_ENTITIES)
        res = m.search("anything", subjects=["career"])
        assert [r.memory_id for r in res] == [a.memory_id]

    def test_hard_root_walls(self):
        m, q, _a, b = self._two(hard_roots=("home",))
        q.outputs.append(NO_ENTITIES)
        assert b.memory_id not in [r.memory_id for r in m.search("tell me things")]
        q.outputs.append(NO_ENTITIES)
        assert b.memory_id in [r.memory_id for r in m.search("my dog at home")]

    def test_scopes_and_root_editing_with_bucket(self):
        q = QueueModel([_facts("alice adopted a dog")])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                taxonomy=TaxonomyConfig(enabled=True),
                bucket=BucketConfig(enabled=True),
            ),
            embedder=VecEmbedder(),
        )
        mem = m.add("x")[0]
        tree = m.scopes()
        assert any(n["id"] == "home" for n in tree["subjects"])
        m.set_root_scheme(add={"pets": "Pets"}, retire=["home"])
        od = m.list_open_decisions(kind="root_migration")[0]
        assert "pets" in od.options
        assert m.resolve_open_decision(od.id, "pets")
        assert _p(m, mem.memory_id)["subjects"] == ["pets"]
        # state persists across managers on the same db
        m2 = MemoryManager(QueueModel([]).model(), m._config, embedder=VecEmbedder(), db=m._db)
        assert "pets" in [n["id"] for n in m2.scopes()["subjects"]]

    def test_compile_grows_children(self):
        facts = [f"alice went hiking near the lake {i}" for i in range(4)]
        q = QueueModel([_facts(*facts)])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                taxonomy=TaxonomyConfig(enabled=True, growth=GrowthConfig(min_cluster_size=3)),
            ),
            embedder=VecEmbedder(),
        )
        m.add("x")
        out = m.compile()
        assert any(c.startswith("interests.") for c in out.taxonomy["children"])
        out2 = m.compile()
        assert out2.taxonomy["stable"]  # same membership across compiles

    def test_pinned_instance_scopes(self):
        q = QueueModel([_facts("kickoff went well"), _facts("lunch was fine")])
        m = MemoryManager(q.model(), MemoryConfig(user_id="u", embedding_dimensions=4), embedder=VecEmbedder())
        a = m.add("x", scopes=["Apollo"])[0]
        q.outputs.append({"decisions": [{"action": "ADD", "text": "lunch was fine"}]})
        m.add("y")
        q.outputs.append(NO_ENTITIES)
        res = m.search("anything", scopes=["Apollo"])
        assert res[0].memory_id == a.memory_id


# --- Layer D ---------------------------------------------------------------------


class TestOntologyAndRetrieval:
    def test_rung4_asks_the_decision_model(self):
        emb = VecEmbedder(table={"JS": [0.8, 0.6, 0.0, 0.0], "JavaScript": [1.0, 0.0, 0.0, 0.0]})

        def handler(state, questions):
            return [Decision(q.question_id, "JavaScript", 0.9, 0.9) for q in questions]

        q = QueueModel(
            [
                _facts("alice codes", entities=[{"name": "JavaScript", "entity_type": "Technology"}]),
                _facts("alice likes js", entities=[{"name": "JS", "entity_type": "Technology"}]),
                {"decisions": [{"action": "ADD", "text": "alice likes js"}]},
            ]
        )
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                pipeline=PipelineConfig(ontology=True),
                ontology=OntologyConfig(write_embedding_rung=True, rung4=True, rung4_floor=0.7),
                decision_model=FakeDecisionModel(handler),
            ),
            embedder=emb,
        )
        m.add("a")
        m.add("b")
        names = {r["name"] for r in m._db.execute("MATCH (e:Entity) RETURN e.name AS name")}
        assert names == {"JavaScript"}

    def test_interval_closing_on_update(self):
        q = QueueModel(
            [
                _facts("alice works at acme"),
                _facts("alice works at beta since 2021"),
            ]
        )
        m = MemoryManager(
            q.model(),
            MemoryConfig(user_id="u", embedding_dimensions=4, ontology=OntologyConfig(interval_closing=True)),
            embedder=VecEmbedder(),
        )
        first = m.add("a")[0]
        q.outputs.append(
            {
                "decisions": [
                    {"action": "UPDATE", "text": "alice works at beta since 2021", "target_memory_id": first.memory_id}
                ]
            }
        )
        m.add("b")
        old = _p(m, first.memory_id)
        assert old["invalid_at"] == 1609459200000  # 2021-01-01T00:00Z

    def test_deterministic_first_and_layer_scores(self):
        q = QueueModel(
            [
                _facts("alice works at acme", entities=[{"name": "Acme", "entity_type": "Organization"}]),
                {"entities": [], "relations": []},
            ]
        )
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                retrieval=RetrievalConfig(order="deterministic", temporal_filter=True, certified_boost=0.5),
            ),
            embedder=VecEmbedder(),
        )
        mem = m.add("x")[0]
        m.certify(mem.memory_id)
        ex = m.explain("what about acme in 2023")
        step = next(s for s in ex.steps if s.name == "deterministic_first")
        assert step.detail["entities"] == ["Acme"] and step.detail["memories"] == 1

    @staticmethod
    def _speaker_graph(m: MemoryManager, n_memories: int) -> None:
        from grafeo_memory.common.types import ENTITY_LABEL, HAS_ENTITY_EDGE, MEMORY_LABEL

        db = m._db
        alice = db.create_node([ENTITY_LABEL], {"name": "Alice", "user_id": "u"})
        acme = db.create_node([ENTITY_LABEL], {"name": "Acme", "user_id": "u"})
        for i in range(n_memories):
            mem = db.create_node([MEMORY_LABEL], {"text": f"alice fact {i}", "user_id": "u"})
            db.create_edge(mem.id, alice.id, HAS_ENTITY_EDGE)
            if i < 3:
                db.create_edge(mem.id, acme.id, HAS_ENTITY_EDGE)

    @pytest.mark.parametrize(
        ("n_memories", "hub_share", "expected"),
        [(25, 0.3, {"Alice"}), (25, 1.0, set()), (10, 0.3, set())],
    )
    def test_hubs_are_entities_on_most_memories(self, n_memories, hub_share, expected):
        from grafeo_memory.services import retrieval

        m = MemoryManager(
            QueueModel([]).model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                retrieval=RetrievalConfig(order="deterministic", hub_share=hub_share),
            ),
            embedder=VecEmbedder(),
        )
        self._speaker_graph(m, n_memories)
        assert retrieval.hubs(m, "u") == expected

    def test_hub_is_no_instance_scope(self):
        from grafeo_memory.services import scoping

        m = MemoryManager(
            QueueModel([]).model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                taxonomy=TaxonomyConfig(enabled=True, instance_scopes=InstanceScopeConfig(prominence_threshold=2)),
            ),
            embedder=VecEmbedder(),
        )
        self._speaker_graph(m, 25)
        ss = scoping.route(m, "u", "what does alice do at acme", ["Alice", "Acme"])
        assert "Acme" in ss.instance_scopes and "Alice" not in ss.instance_scopes

    def test_hub_share_is_validated(self):
        with pytest.raises(ValueError, match="hub_share"):
            MemoryConfig(retrieval=RetrievalConfig(hub_share=0.0))


# --- Layer A ---------------------------------------------------------------------


class TestTiers:
    def test_instruction_assemble_and_window(self):
        q = QueueModel([_facts("alice likes tea"), NO_ENTITIES, NO_ENTITIES])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                tiers=TiersConfig(working=WorkingConfig(enabled=True), procedural=ProceduralConfig(enabled=True)),
            ),
            embedder=VecEmbedder(),
        )
        ev = m.add_instruction("Always answer in Dutch")
        assert _p(m, ev.memory_id)["status"] == "certified"
        m.add([{"role": "user", "content": "I like tea"}], session_id="s1")
        block = m.assemble("drinks?", session_id="s1")
        assert block.sections["working"][0].text == "user: I like tea"
        assert block.sections["procedural"][0].text == "Always answer in Dutch"
        text = block.render()
        assert "## Instructions" in text and "status=certified" in text

    def test_deferred_extraction_and_end_session(self):
        q = QueueModel([_facts("alice likes tea"), "Alice talked about tea."])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                tiers=TiersConfig(working=WorkingConfig(enabled=True, defer_extraction=True, window_messages=1)),
            ),
            embedder=VecEmbedder(),
        )
        assert len(m.add([{"role": "user", "content": "I like tea"}], session_id="s")) == 0
        m.add([{"role": "user", "content": "really"}], session_id="s")
        res = m.end_session("s")
        texts = [e.text for e in res]
        assert "alice likes tea" in texts and any(t.startswith("Session summary") for t in texts)
        assert q.calls == 2  # one extraction over the window, one summary

    def test_procedural_auto_triggers(self):
        q = QueueModel([_facts("user prefers uv")])
        m = MemoryManager(
            q.model(),
            MemoryConfig(
                user_id="u",
                embedding_dimensions=4,
                tiers=TiersConfig(procedural=ProceduralConfig(enabled=True, auto_triggers=True)),
            ),
            embedder=VecEmbedder(),
        )
        res = m.add([{"role": "user", "content": "No, use uv instead of pip."}])
        assert any(e.memory_type == "procedural" for e in res)


# --- config and schema -----------------------------------------------------------


def test_layer_config_validation():
    with pytest.raises(ValueError, match=r"retrieval.order"):
        MemoryConfig(retrieval=RetrievalConfig(order="sideways"))
    with pytest.raises(ValueError, match=r"bucket.floor"):
        MemoryConfig(bucket=BucketConfig(floor=2.0))
    with pytest.raises(ValueError, match="decision_model"):
        MemoryConfig(ontology=OntologyConfig(rung4=True))
    assert MemoryConfig(pipeline=PipelineConfig(bucket=True)).bucket_on


def test_schema_marker_is_lazy():
    from grafeo_memory.services.schema import SCHEMA_VERSION, schema_version

    m, _q = _mgr([_facts("a b")])
    m.add("x")
    assert schema_version(m._db) == 1  # no layer structure written: no marker
    m2, _q2 = _mgr([], taxonomy=TaxonomyConfig(enabled=True))
    m2.set_root_scheme(add={"pets": "Pets"})
    assert schema_version(m2._db) == SCHEMA_VERSION
