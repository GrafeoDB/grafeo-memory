"""Tests for OpenTelemetry instrumentation (opt-in via MemoryConfig.instrument)."""

from __future__ import annotations

from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import InstrumentationSettings, MemoryConfig, MemoryManager
from grafeo_memory._tracing import trace_operation


class TestInstrumentConfig:
    def test_instrument_default_false(self):
        config = MemoryConfig()
        assert config.instrument is False

    def test_instrument_true(self):
        config = MemoryConfig(instrument=True)
        assert config.instrument is True

    def test_instrument_settings_instance(self):
        settings = InstrumentationSettings()
        config = MemoryConfig(instrument=settings)
        assert isinstance(config.instrument, InstrumentationSettings)

    def test_yolo_enables_instrument(self):
        config = MemoryConfig.yolo()
        assert config.instrument is True


class TestInstrumentInit:
    def test_instrument_false_no_side_effects(self):
        """instrument=False (default) should not call Agent.instrument_all."""
        from pydantic_ai import Agent

        # Save original value
        original = Agent._instrument_default

        model = make_test_model([{"facts": ["test"], "entities": [], "relations": []}])
        embedder = MockEmbedder(16)
        config = MemoryConfig(db_path=None, instrument=False)
        manager = MemoryManager(model, config, embedder=embedder)
        manager.close()

        # Should not have changed the class-level default
        assert Agent._instrument_default == original

    def test_instrument_true_calls_instrument_all(self):
        """instrument=True should call Agent.instrument_all(True)."""
        from pydantic_ai import Agent

        original = Agent._instrument_default
        try:
            model = make_test_model([{"facts": ["test"], "entities": [], "relations": []}])
            embedder = MockEmbedder(16)
            config = MemoryConfig(db_path=None, instrument=True)
            manager = MemoryManager(model, config, embedder=embedder)
            manager.close()

            # instrument_all(True) sets _instrument_default to True
            assert Agent._instrument_default is True
        finally:
            # Restore original to avoid affecting other tests
            Agent._instrument_default = original
