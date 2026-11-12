"""Tests for core types."""

from grafeo_memory import (
    Entity,
    ExtractionResult,
    Fact,
    MemoryAction,
    MemoryConfig,
    MemoryEvent,
    Relation,
    SearchResult,
)


def test_memory_action_values():
    assert MemoryAction.ADD.value == "add"
    assert MemoryAction.UPDATE.value == "update"
    assert MemoryAction.DELETE.value == "delete"
    assert MemoryAction.NONE.value == "none"


def test_memory_config_defaults():
    config = MemoryConfig()
    assert config.db_path is None
    assert config.user_id == "default"
    assert config.session_id is None
    assert config.reconciliation_threshold == 0.3
    assert config.embedding_dimensions == 1536


def test_memory_config_custom():
    config = MemoryConfig(
        db_path="./test.db",
        user_id="alice",
        session_id="s1",
        embedding_dimensions=384,
    )
    assert config.db_path == "./test.db"
    assert config.user_id == "alice"
    assert config.embedding_dimensions == 384


def test_memory_event():
    event = MemoryEvent(action=MemoryAction.ADD, memory_id="42", text="test fact")
    assert event.action == MemoryAction.ADD
    assert event.memory_id == "42"
    assert event.text == "test fact"
    assert event.old_text is None


def test_search_result():
    result = SearchResult(memory_id="1", text="fact", score=0.95, user_id="alice")
    assert result.score == 0.95
    assert result.metadata is None
    assert result.relations is None


def test_memory_config_yolo():
    config = MemoryConfig.yolo()
    assert config.enable_importance is True
    assert config.enable_vision is True
    assert config.usage_callback is not None
    # defaults still hold
    assert config.user_id == "default"
    assert config.db_path is None


def test_memory_config_yolo_overrides():
    config = MemoryConfig.yolo(user_id="alice", db_path="./yolo.db")
    assert config.enable_importance is True
    assert config.enable_vision is True
    assert config.user_id == "alice"
    assert config.db_path == "./yolo.db"


def test_memory_config_yolo_custom_callback():
    def custom_cb(op, usage):
        pass

    config = MemoryConfig.yolo(usage_callback=custom_cb)
    assert config.usage_callback is custom_cb


def test_extraction_result():
    er = ExtractionResult(
        facts=[Fact("alice works at acme")],
        entities=[Entity("alice", "person"), Entity("acme", "organization")],
        relations=[Relation("alice", "acme", "works_at")],
    )
    assert len(er.facts) == 1
    assert len(er.entities) == 2
    assert er.relations[0].relation_type == "works_at"


def test_pipeline_layer_toggles_default_off():
    """Composable layer toggles (design doc §14 item 20) all default to False,
    so a default PipelineConfig is exactly the baseline behavior."""
    from grafeo_memory.common.types import PipelineConfig

    pipe = PipelineConfig()
    for flag in ("temporal_precheck", "ontology", "bitemporal", "episodes", "compile", "taxonomy", "policy", "bucket"):
        assert getattr(pipe, flag) is False, flag


def test_legacy_flags_are_sugar_over_pipeline_toggles():
    """Either MemoryConfig.enable_bitemporal / enable_episodes or the PipelineConfig toggle
    turns the layer on; neither is written into the other (0.3.0 review)."""
    import dataclasses

    from grafeo_memory.common.types import PipelineConfig

    assert MemoryConfig(enable_bitemporal=True).bitemporal_on
    assert MemoryConfig(pipeline=PipelineConfig(bitemporal=True)).bitemporal_on
    assert MemoryConfig(enable_episodes=True).episodes_on
    assert MemoryConfig(pipeline=PipelineConfig(episodes=True)).episodes_on
    c = MemoryConfig()
    assert not c.bitemporal_on and not c.episodes_on
    # set after construction: takes effect
    c.pipeline.bitemporal = True
    assert c.bitemporal_on
    # replace() can turn the legacy flag off again, and a shared pipeline is not mutated
    shared = PipelineConfig()
    on = MemoryConfig(enable_bitemporal=True, pipeline=shared)
    assert shared.bitemporal is False
    assert not dataclasses.replace(on, enable_bitemporal=False).bitemporal_on
