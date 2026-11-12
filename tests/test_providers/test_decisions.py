"""Tests for the decisions/ package and per-stage engine dispatch (LLM / DECISION / HYBRID)."""

import pytest
from conftest import make_manager
from mock_llm import make_error_model, make_test_model

from grafeo_memory import Engine, EscalationSignal, Granularity, MemoryAction, MemoryConfig, PipelineConfig, StageConfig
from grafeo_memory.common.types import Fact, Relation
from grafeo_memory.core.reconciliation import reconcile, reconcile_async, reconcile_relations
from grafeo_memory.providers.decision import (
    CategoricalQuestion,
    Decision,
    FakeDecisionModel,
    decisions_to_deletes,
    get_decision_model,
    relations_pack,
    should_escalate,
)
from grafeo_memory.providers.decision.packs import PACKS

FACTS = [Fact("alice works at beta inc"), Fact("alice likes hiking")]
EXISTING = [{"id": "100", "text": "alice works at acme corp", "score": 0.42}]


# --- config validation ---------------------------------------------------------


def test_default_pipeline_is_llm_everywhere():
    c = MemoryConfig()
    assert c.decision_model is None
    for name in ("facts", "entities", "temporal", "memories", "relations", "search"):
        assert getattr(c.pipeline, name).engine == Engine.LLM


def test_non_llm_engine_requires_decision_model():
    with pytest.raises(ValueError, match="decision_model"):
        MemoryConfig(pipeline=PipelineConfig(memories=StageConfig(engine=Engine.DECISION)))


@pytest.mark.parametrize("stage_name", ["facts", "entities", "temporal"])
def test_generation_only_stages_reject_non_llm(stage_name):
    pipeline = PipelineConfig(**{stage_name: StageConfig(engine=Engine.HYBRID)})
    with pytest.raises(ValueError, match="generation-only"):
        MemoryConfig(decision_model="fake", pipeline=pipeline)


def test_confidence_threshold_bounds():
    with pytest.raises(ValueError, match="confidence_threshold"):
        MemoryConfig(
            decision_model="fake",
            pipeline=PipelineConfig(memories=StageConfig(engine=Engine.HYBRID, confidence_threshold=1.5)),
        )


def test_unknown_question_pack_rejected():
    # config accepts any str; the name is validated against the registry when the
    # manager resolves the decision model (common cannot import providers)
    with pytest.raises(ValueError, match="question_pack"):
        make_manager([], decision_model="fake", pipeline=PipelineConfig(question_pack="nope"))


def test_valid_decision_config():
    c = MemoryConfig(
        decision_model=FakeDecisionModel(),
        pipeline=PipelineConfig(
            memories=StageConfig(engine=Engine.HYBRID, granularity=Granularity.PAIRWISE),
            relations=StageConfig(engine=Engine.DECISION),
            search=StageConfig(engine=Engine.DECISION, rerank_blend_weight=0.7),
            question_pack="pairwise",
        ),
    )
    assert c.pipeline.memories.engine == Engine.HYBRID
    assert c.pipeline.search.rerank_blend_weight == 0.7


def test_registry_resolves_fake_and_filters_kwargs():
    dm = get_decision_model("fake", cache=object(), api_key="secret")
    assert isinstance(dm, FakeDecisionModel)
    with pytest.raises(ValueError, match="Unknown decision model"):
        get_decision_model("does-not-exist")


def test_registry_instantiates_a_class_spec():
    dm = get_decision_model(FakeDecisionModel, api_key="secret")
    assert isinstance(dm, FakeDecisionModel)


def test_registry_skips_a_broken_entry_point(monkeypatch, caplog):
    import importlib.metadata

    from grafeo_memory.providers.decision import registry

    class _EP:
        def __init__(self, name, target):
            self.name, self._target = name, target

        def load(self):
            if isinstance(self._target, Exception):
                raise self._target
            return self._target

    eps = [_EP("broken-plugin", ImportError("missing dep")), _EP("good-plugin", FakeDecisionModel)]
    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: eps)
    monkeypatch.setattr(registry, "_REGISTRY", dict(registry._REGISTRY))
    with caplog.at_level("WARNING", logger=registry.__name__):
        assert isinstance(get_decision_model("good-plugin"), FakeDecisionModel)
    assert "broken-plugin" in caplog.text


# --- reconcile_async: DECISION engine -------------------------------------------


def test_decision_engine_per_fact_add():
    fake = FakeDecisionModel()  # default: picks ADD, confidence 1.0
    decisions = reconcile(
        make_test_model([]),
        FACTS,
        EXISTING,
        engine=Engine.DECISION,
        decision_model=fake,
        stage=StageConfig(engine=Engine.DECISION),
    )
    assert len(fake.calls) == 1
    state, questions = fake.calls[0]
    assert len(questions) == len(FACTS)  # per-fact pack: one question per fact
    assert "alice works at beta inc" in state and "100" in state
    assert all(d.action == MemoryAction.ADD for d in decisions)
    assert all(d.engine == "decision" and d.confidence == 1.0 for d in decisions)


def test_decision_engine_targets_update_delete():
    def handler(state, questions):
        out = []
        for q in questions:
            assert isinstance(q, CategoricalQuestion)
            if q.question_id == "fact_0":
                out.append(Decision(q.question_id, "UPDATE_100", 0.9, 0.95, runner_up=0.1))
            else:
                out.append(Decision(q.question_id, "DELETE_100", 0.8, 0.9, runner_up=0.2))
        return out

    fake = FakeDecisionModel(handler)
    decisions = reconcile(
        make_test_model([]),
        FACTS,
        EXISTING,
        engine=Engine.DECISION,
        decision_model=fake,
        stage=StageConfig(engine=Engine.DECISION),
    )
    assert decisions[0].action == MemoryAction.UPDATE
    assert decisions[0].target_memory_id == "100"
    assert decisions[1].action == MemoryAction.DELETE
    assert decisions[1].target_memory_id == "100"


def test_decision_engine_pairwise_granularity():
    def handler(state, questions):
        out = []
        for q in questions:
            # one question per (fact, candidate)
            if q.question_id == "fact_0|mem_100":
                out.append(Decision(q.question_id, "INVALIDATE", 0.9, 0.9, runner_up=0.1))
            else:
                out.append(Decision(q.question_id, "UNRELATED", 0.95, 0.95, runner_up=0.05))
        return out

    fake = FakeDecisionModel(handler)
    stage = StageConfig(engine=Engine.DECISION, granularity=Granularity.PAIRWISE)
    decisions = reconcile(
        make_test_model([]), FACTS, EXISTING, engine=Engine.DECISION, decision_model=fake, stage=stage
    )
    assert len(fake.calls[0][1]) == len(FACTS) * len(EXISTING)  # pairwise: fact x candidate
    assert decisions[0].action == MemoryAction.DELETE
    assert decisions[0].target_memory_id == "100"
    assert decisions[1].action == MemoryAction.ADD  # all-UNRELATED row resolves to ADD


def test_question_pack_override_wins():
    fake = FakeDecisionModel()
    stage = StageConfig(engine=Engine.DECISION)  # per_fact default
    decisions = reconcile(
        make_test_model([]),
        FACTS,
        EXISTING,
        engine=Engine.DECISION,
        decision_model=fake,
        stage=stage,
        question_pack="pairwise",
    )
    assert len(fake.calls[0][1]) == len(FACTS) * len(EXISTING)
    assert decisions[0].engine == "decision"


# --- reconcile_async: HYBRID engine ----------------------------------------------


def test_hybrid_escalates_low_confidence_to_llm():
    def handler(state, questions):
        return [Decision(q.question_id, "ADD", 0.6, confidence=0.5, runner_up=0.4) for q in questions]

    fake = FakeDecisionModel(handler)
    llm = make_test_model(
        [{"decisions": [{"action": "ADD", "text": FACTS[0].text}, {"action": "ADD", "text": FACTS[1].text}]}]
    )
    ops: list[str] = []
    decisions = reconcile(
        llm,
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=fake,
        stage=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
        _on_usage=lambda op, usage: ops.append(op),
    )
    assert "reconcile_decision" in ops
    assert "reconcile_escalated" in ops  # one LLM batch for the uncertain facts
    assert all(d.engine == "llm" for d in decisions)  # LLM made the final call


def test_hybrid_confident_decisions_skip_llm():
    fake = FakeDecisionModel()  # confidence 1.0 >= 0.75 -> no escalation
    llm = make_error_model()  # must never be called
    decisions = reconcile(
        llm,
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=fake,
        stage=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
    )
    assert len(fake.calls) == 1
    assert all(d.engine == "decision" for d in decisions)


def test_hybrid_prob_gap_signal_escalates():
    # High confidence but a narrow margin: PROB_GAP must still escalate.
    def handler(state, questions):
        return [Decision(q.question_id, "ADD", 0.55, confidence=0.99, runner_up=0.45) for q in questions]

    fake = FakeDecisionModel(handler)
    llm = make_test_model([{"decisions": [{"action": "ADD", "text": f.text} for f in FACTS]}])
    decisions = reconcile(
        llm,
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=fake,
        stage=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.2, escalation_signal=EscalationSignal.PROB_GAP),
    )
    assert all(d.engine == "llm" for d in decisions)


# --- fallback semantics ------------------------------------------------------------


def test_decision_model_error_falls_back_to_llm():
    class Boom:
        async def decide_async(self, state, questions):
            raise RuntimeError("provider down")

    llm = make_test_model([{"decisions": [{"action": "ADD", "text": f.text} for f in FACTS]}])
    decisions = reconcile(llm, FACTS, EXISTING, engine=Engine.DECISION, decision_model=Boom())
    assert all(d.action == MemoryAction.ADD for d in decisions)
    assert all(d.engine == "llm" for d in decisions)


def test_llm_error_falls_back_to_fast_path():
    decisions = reconcile(make_error_model(), FACTS, EXISTING)
    assert all(d.action == MemoryAction.ADD and d.engine == "fast" for d in decisions)


def test_no_candidates_is_fast_path_regardless_of_engine():
    fake = FakeDecisionModel()
    decisions = reconcile(
        make_test_model([]),
        FACTS,
        [],
        engine=Engine.DECISION,
        decision_model=fake,
        stage=StageConfig(engine=Engine.DECISION),
    )
    assert len(fake.calls) == 0
    assert all(d.engine == "fast" for d in decisions)


# --- should_escalate unit ----------------------------------------------------------


def test_should_escalate_signals():
    d = Decision("q", "ADD", probability=0.8, confidence=0.9, runner_up=0.15)  # gap 0.65
    assert not should_escalate(d, 0.75, EscalationSignal.CONFIDENCE)
    assert not should_escalate(d, 0.5, EscalationSignal.PROB_GAP)
    assert should_escalate(d, 0.7, EscalationSignal.PROB_GAP)  # gap 0.65 < 0.7
    assert not should_escalate(d, 0.6, EscalationSignal.COMBINED)  # both signals clear 0.6
    assert should_escalate(d, 0.7, EscalationSignal.COMBINED)  # gap fails
    low_conf = Decision("q", "ADD", probability=0.9, confidence=0.5, runner_up=0.1)
    assert should_escalate(low_conf, 0.75, EscalationSignal.CONFIDENCE)
    assert should_escalate(low_conf, 0.75, EscalationSignal.COMBINED)
    assert not should_escalate(low_conf, 0.4, EscalationSignal.PROB_GAP)  # gap 0.8 clears
    # runner_up None -> two-class estimate 1 - p: gap = 2p - 1 = 0.6 here
    no_dist = Decision("q", "ADD", probability=0.8, confidence=0.99, runner_up=None)
    assert not should_escalate(no_dist, 0.5, EscalationSignal.PROB_GAP)
    assert should_escalate(no_dist, 0.7, EscalationSignal.PROB_GAP)


# --- relations reconciliation ------------------------------------------------------


def test_relations_decision_engine_delete():
    def handler(state, questions):
        out = []
        for q in questions:
            if q.question_id == "new_0|existing_0":
                out.append(Decision(q.question_id, "DELETE", 0.9, 0.9, runner_up=0.1))
            else:
                out.append(Decision(q.question_id, "KEEP", 0.95, 0.95, runner_up=0.05))
        return out

    fake = FakeDecisionModel(handler)
    new = [Relation(source="alice", target="beta", relation_type="works_at")]
    existing = [
        {"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 7},
        {"source": "alice", "target": "acme", "relation_type": "lives_in", "edge_id": 8},
    ]
    to_delete = reconcile_relations(
        make_test_model([]),
        new,
        existing,
        engine=Engine.DECISION,
        decision_model=fake,
        stage=StageConfig(engine=Engine.DECISION),
    )
    assert len(fake.calls) == 1
    assert len(fake.calls[0][1]) == 2  # one question per (new, existing) pair
    assert to_delete == [{"source": "alice", "target": "acme", "relation_type": "works_at"}]


def test_relations_decision_fallback_to_llm():
    class Boom:
        async def decide_async(self, state, questions):
            raise RuntimeError("down")

    llm = make_test_model([{"delete": [{"source": "alice", "target": "acme", "relation_type": "works_at"}]}])
    new = [Relation(source="alice", target="beta", relation_type="works_at")]
    existing = [{"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 7}]
    to_delete = reconcile_relations(llm, new, existing, engine=Engine.DECISION, decision_model=Boom())
    assert to_delete == [{"source": "alice", "target": "acme", "relation_type": "works_at"}]


def test_filter_relation_candidates_overlap_and_cap():
    from grafeo_memory.core.reconciliation import filter_relation_candidates

    new = [Relation(source="alice", target="beta", relation_type="works_at")]
    existing = [
        {"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 7},
        {"source": "bob", "target": "acme", "relation_type": "works_at", "edge_id": 9},  # no overlap
        {"source": "beta", "target": "carol", "relation_type": "knows", "edge_id": 8},
    ]
    out = filter_relation_candidates(new, existing)
    assert {r["edge_id"] for r in out} == {7, 8}  # bob->acme dropped (no shared endpoint)
    # cap keeps most recent (highest edge_id) first
    many = [{"source": "alice", "target": f"t{i}", "relation_type": "x", "edge_id": i} for i in range(100)]
    capped = filter_relation_candidates(new, many, cap=50)
    assert len(capped) == 50 and capped[0]["edge_id"] == 99


def test_relations_pack_and_mapper_directly():
    new = [Relation(source="a", target="b", relation_type="knows")]
    existing = [{"source": "a", "target": "c", "relation_type": "knows"}]
    questions = relations_pack(new, existing)
    assert questions[0].choices == ["KEEP", "DELETE"]
    decisions = [Decision("new_0|existing_0", "DELETE", 0.8, 0.8, runner_up=0.2)]
    assert decisions_to_deletes(new, existing, decisions) == [{"source": "a", "target": "c", "relation_type": "knows"}]
    keep = [Decision("new_0|existing_0", "KEEP", 0.8, 0.8, runner_up=0.2)]
    assert decisions_to_deletes(new, existing, keep) == []


def test_unrelated_confident_keep_does_not_cancel_a_contradiction():
    new = [
        Relation(source="a", target="b", relation_type="knows"),
        Relation(source="x", target="y", relation_type="uses"),
    ]
    existing = [{"source": "a", "target": "c", "relation_type": "knows"}]
    decisions = [
        Decision("new_0|existing_0", "DELETE", 0.7, 0.7, runner_up=0.3),
        Decision("new_1|existing_0", "KEEP", 0.99, 0.99, runner_up=0.01),  # unrelated new relation
    ]
    assert decisions_to_deletes(new, existing, decisions) == [{"source": "a", "target": "c", "relation_type": "knows"}]


# --- manager integration -----------------------------------------------------------


def test_manager_resolves_decision_model_from_registry():
    manager = make_manager([], decision_model="fake")
    assert isinstance(manager._decision_model, FakeDecisionModel)
    manager.close()


def test_manager_add_uses_decision_engine():
    fake = FakeDecisionModel()
    manager = make_manager(
        [
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
            # second add: same fact text -> identical embedding -> guaranteed candidate
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
        ],
        decision_model=fake,
        pipeline=PipelineConfig(memories=StageConfig(engine=Engine.DECISION)),
        reconciliation_threshold=0.0,
    )
    ev1 = manager.add("Alice works at Acme.")
    assert ev1[0].engine == "fast"  # no candidates on first add
    assert len(fake.calls) == 0

    ev2 = manager.add("Alice still works at Acme.")
    assert len(fake.calls) == 1  # one decision call for the second add
    assert ev2[0].engine == "decision"
    assert ev2[0].confidence == 1.0
    manager.close()


def test_manager_search_decision_rerank_blends_scores():
    fake = FakeDecisionModel()  # score questions answer with ceiling = 1.0
    manager = make_manager(
        [
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
            {"entities": [], "relations": []},  # query entity extraction in search
        ],
        decision_model=fake,
        pipeline=PipelineConfig(search=StageConfig(engine=Engine.DECISION, rerank_blend_weight=1.0)),
    )
    manager.add("Alice works at Acme.")
    results = manager.search("where does alice work?", k=5)
    assert len(results) >= 1
    assert len(fake.calls) == 1  # one ScoreQuestion batch for the rerank stage
    assert all(r.score == 1.0 for r in results)  # blend weight 1.0 -> pure decision score
    manager.close()


def test_manager_hybrid_add_escalates_and_tags_events():
    def handler(state, questions):
        return [Decision(q.question_id, "ADD", 0.5, confidence=0.4, runner_up=0.5) for q in questions]

    fake = FakeDecisionModel(handler)
    manager = make_manager(
        [
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
            {"facts": ["alice works at acme"], "entities": [], "relations": []},
            # escalated LLM reconciliation for the second add
            {"decisions": [{"action": "ADD", "text": "alice works at acme"}]},
        ],
        decision_model=fake,
        pipeline=PipelineConfig(memories=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75)),
        reconciliation_threshold=0.0,
    )
    ops: list[str] = []
    manager._config.usage_callback = lambda op, usage: ops.append(op)
    manager.add("Alice works at Acme.")
    ev2 = manager.add("Alice still works at Acme.")
    assert "reconcile_decision" in ops
    assert "reconcile_escalated" in ops
    assert ev2[0].engine == "llm"  # escalated fact was decided by the LLM
    manager.close()


def test_build_state_bounded_for_jev_context():
    from grafeo_memory.providers.decision.packs import build_relation_state, build_state

    facts = [Fact("f" * 5000) for _ in range(50)]
    cands = [{"id": str(i), "text": "m" * 5000, "score": i / 100} for i in range(100)]
    state = build_state(facts, cands, max_chars=100_000)
    assert len(state) <= 100_000 + 60  # truncation marker
    assert "truncated" in state
    # highest-similarity candidate (ID 99) is kept within the budget
    assert "ID 99" in state
    rel_state = build_relation_state(
        [Relation(source="a" * 3000, target="b", relation_type="x")],
        [{"source": "s", "target": "t", "relation_type": "r", "edge_id": i} for i in range(500)],
        max_chars=10_000,
    )
    assert len(rel_state) <= 10_060


def test_jev_parse_answer_normalizes_score_and_runner_up():
    from grafeo_memory.providers.decision.jev import JevDecisionModel
    from grafeo_memory.providers.decision.types import ScoreQuestion

    j = JevDecisionModel(api_key="test-only-not-a-real-key")
    cat = CategoricalQuestion("c", "pick", ["A", "B"], {"A": "a", "B": "b"})
    d = j._parse_answer(cat, {"choice": "A", "probabilities": {"A": 0.7, "B": 0.3}, "confidence": 0.9})
    assert (d.value, d.probability, d.runner_up, d.confidence) == ("A", 0.7, 0.3, 0.9)
    # 3 rubric levels: positional score 2 == ceiling, 1 == midpoint
    sq = ScoreQuestion("s", "rate", 0.0, 1.0, ["low", "mid", "high"])
    assert j._parse_answer(sq, {"score": 2.0, "confidence": 0.8}).value == 1.0
    assert abs(j._parse_answer(sq, {"score": 1.0, "confidence": 0.8}).value - 0.5) < 1e-9
    # default 2 levels
    sq2 = ScoreQuestion("s2", "rate")
    assert j._parse_answer(sq2, {"score": 1.0, "confidence": 1.0}).value == 1.0
    # payload shapes match the published OpenAPI schema
    p = j._build_payload("state", [cat, sq])
    assert p["questions"]["c"] == {"type": "choice", "instructions": "pick", "criteria": {"A": "a", "B": "b"}}
    assert p["questions"]["s"]["type"] == "score" and p["questions"]["s"]["criteria"] == ["low", "mid", "high"]


def test_reconcile_async_direct():
    import asyncio

    fake = FakeDecisionModel()
    decisions = asyncio.run(
        reconcile_async(
            make_test_model([]),
            FACTS,
            EXISTING,
            engine="decision",
            decision_model=fake,
            stage=StageConfig(engine=Engine.DECISION),
        )
    )
    assert all(d.engine == "decision" for d in decisions)
    assert "per_fact" in PACKS and "pairwise" in PACKS


# --- 0.3.0 release review regressions -------------------------------------------


def test_entry_point_provider_resolves_to_instance(monkeypatch):
    """An entry-point provider is instantiated, not returned as its class."""
    import importlib.metadata

    from grafeo_memory.providers.decision import registry

    class _EP:
        name = "ep-fake"

        def load(self):
            return FakeDecisionModel

    monkeypatch.setattr(importlib.metadata, "entry_points", lambda group: [_EP()])
    monkeypatch.delitem(registry._REGISTRY, "ep-fake", raising=False)
    try:
        dm = get_decision_model("ep-fake")
        assert isinstance(dm, FakeDecisionModel)
    finally:
        registry._REGISTRY.pop("ep-fake", None)


def test_factory_error_is_not_masked():
    """A provider's own constructor error surfaces as is, not as a kwargs retry error."""
    from grafeo_memory.providers.decision import register_decision_model, registry

    def _broken(cache=None):
        raise ValueError("bad provider config")

    register_decision_model("broken", _broken)
    try:
        with pytest.raises(ValueError, match="bad provider config"):
            get_decision_model("broken", cache=object(), api_key="secret")
    finally:
        registry._REGISTRY.pop("broken", None)


def test_jev_zero_confidence_and_probability_are_kept():
    from grafeo_memory.providers.decision.jev import JevDecisionModel

    q = CategoricalQuestion(question_id="q", question="?", choices=["ADD", "NONE"])
    d = JevDecisionModel._parse_answer(q, {"choice": "ADD", "confidence": 0.0, "probability": 0.0})
    assert d.confidence == 0.0
    assert d.probability == 0.0
    d = JevDecisionModel._parse_answer(q, {"choice": "ADD"})
    assert d.confidence == 1.0 and d.probability == 1.0


def test_typesafe_key_goes_to_jev_only(monkeypatch):
    """A third-party provider never receives the TypeSafe key or the Jev model name."""
    from grafeo_memory.providers.decision import register_decision_model, registry

    seen: dict = {}

    def _provider(**kwargs):
        seen.update(kwargs)
        return FakeDecisionModel()

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-secret")
    register_decision_model("third-party", _provider)
    try:
        manager = make_manager([], decision_model="third-party")
        manager.close()
    finally:
        registry._REGISTRY.pop("third-party", None)
    assert "api_key" not in seen
    assert "model" not in seen


def test_filter_relation_candidates_ranks_likely_conflicts_first():
    """An old same-type relation of the source survives the cap ahead of newer unrelated ones."""
    from grafeo_memory.core.reconciliation import filter_relation_candidates

    new = [Relation(source="alice", target="beta", relation_type="works_at")]
    old_conflict = {"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 1}
    noise = [{"source": "alice", "target": f"t{i}", "relation_type": "likes", "edge_id": 100 + i} for i in range(60)]
    out = filter_relation_candidates(new, [old_conflict, *noise], cap=50)
    assert out[0] is old_conflict and len(out) == 50


def test_hybrid_pairwise_confident_unrelated_skips_llm():
    """A fact UNRELATED to every candidate, with confident answers, is an ADD without an LLM call."""

    def handler(state, questions):
        return [Decision(q.question_id, "UNRELATED", 0.95, confidence=0.95, runner_up=0.05) for q in questions]

    decisions = reconcile(
        make_error_model(),  # must never be called
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=FakeDecisionModel(handler),
        stage=StageConfig(engine=Engine.HYBRID, granularity=Granularity.PAIRWISE, confidence_threshold=0.75),
    )
    assert [d.action for d in decisions] == [MemoryAction.ADD, MemoryAction.ADD]
    assert all(d.engine == "decision" for d in decisions)


def test_hybrid_pairwise_unsure_unrelated_escalates():
    def handler(state, questions):
        return [Decision(q.question_id, "UNRELATED", 0.55, confidence=0.4, runner_up=0.45) for q in questions]

    llm = make_test_model([{"decisions": [{"action": "add", "text": f.text} for f in FACTS]}])
    decisions = reconcile(
        llm,
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=FakeDecisionModel(handler),
        stage=StageConfig(engine=Engine.HYBRID, granularity=Granularity.PAIRWISE, confidence_threshold=0.75),
    )
    assert all(d.engine != "decision" for d in decisions)


def test_relations_hybrid_escalates_only_unsure_verdicts():
    def handler(state, questions):
        out = []
        for q in questions:
            if q.question_id == "new_0|existing_0":
                out.append(Decision(q.question_id, "DELETE", 0.95, 0.95, runner_up=0.05))  # sure
            else:
                out.append(Decision(q.question_id, "KEEP", 0.55, 0.3, runner_up=0.45))  # unsure
        return out

    new = [Relation(source="alice", target="beta", relation_type="works_at")]
    existing = [
        {"source": "alice", "target": "acme", "relation_type": "works_at", "edge_id": 7},
        {"source": "alice", "target": "paris", "relation_type": "lives_in", "edge_id": 8},
    ]
    llm = make_test_model([{"delete": [{"source": "alice", "target": "paris", "relation_type": "lives_in"}]}])
    to_delete = reconcile_relations(
        llm,
        new,
        existing,
        engine=Engine.HYBRID,
        decision_model=FakeDecisionModel(handler),
        stage=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
    )
    assert {r["target"] for r in to_delete} == {"acme", "paris"}


def test_hybrid_escalation_failure_keeps_decision_answers():
    """When the escalation LLM call fails, the decision model's answers stand (no blanket ADD)."""

    def handler(state, questions):
        return [Decision(q.question_id, "NONE", 0.6, confidence=0.5, runner_up=0.4) for q in questions]

    decisions = reconcile(
        make_error_model(),
        FACTS,
        EXISTING,
        engine=Engine.HYBRID,
        decision_model=FakeDecisionModel(handler),
        stage=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
    )
    assert [d.action for d in decisions] == [MemoryAction.NONE, MemoryAction.NONE]
    assert all(d.engine == "decision" for d in decisions)


def test_partial_pairwise_pack_uses_pairwise_mapper():
    import functools

    from grafeo_memory.core.reconciliation.memories import _resolve_pack
    from grafeo_memory.providers.decision.packs import pairwise_pack

    _fn, pairwise = _resolve_pack(StageConfig(), functools.partial(pairwise_pack))
    assert pairwise


def test_jev_opens_circuit_on_payment_required(monkeypatch):
    """After a 402 the API is not called again during the cooldown, and the body is not logged."""
    import httpx

    from grafeo_memory.providers.decision.jev import JevDecisionModel

    posts: list[int] = []

    class _Client:
        def __init__(self, *a, **k):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def post(self, url, json, headers):
            posts.append(1)
            req = httpx.Request("POST", url)
            return httpx.Response(
                402, json={"error": {"code": "insufficient_funds"}, "echo": "SECRET MEMORY"}, request=req
            )

    monkeypatch.setattr(httpx, "AsyncClient", _Client)
    jev = JevDecisionModel(api_key="k", max_retries=1)
    q = [CategoricalQuestion(question_id="q", question="?", choices=["ADD", "NONE"])]
    from grafeo_memory.common.compat import run_sync

    with pytest.raises(httpx.HTTPStatusError):
        run_sync(jev.decide_async("state", q))
    with pytest.raises(RuntimeError, match="402"):
        run_sync(jev.decide_async("state", q))
    assert len(posts) == 1
