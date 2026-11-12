"""Tests for the pydantic-settings process layer (GRAFEO_MEMORY_* + .env)."""

from __future__ import annotations

import pytest

from grafeo_memory import CacheConfig, GrafeoMemorySettings, MemoryConfig


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("GRAFEO_MEMORY_") or k.endswith("_API_KEY"):
            monkeypatch.delenv(k, raising=False)
    yield


class TestSettings:
    def test_defaults(self):
        s = GrafeoMemorySettings()
        assert s.model == "openai:gpt-4o-mini"
        assert s.user == "default"
        assert s.cache_enabled is False
        assert s.cache_max_entries == 10_000
        assert s.cache_max_memory_mb == 128
        assert s.cache_max_storage_mb == 512
        assert s.db_memory_limit_mb is None
        assert s.cache_config() is None

    def test_env_prefix_and_mb_coercion(self, monkeypatch):
        monkeypatch.setenv("GRAFEO_MEMORY_CACHE_ENABLED", "true")
        monkeypatch.setenv("GRAFEO_MEMORY_CACHE_MAX_MEMORY_MB", "256")
        monkeypatch.setenv("GRAFEO_MEMORY_CACHE_MAX_ENTRIES", "500")
        monkeypatch.setenv("GRAFEO_MEMORY_DB_MEMORY_LIMIT_MB", "2048")
        monkeypatch.setenv("GRAFEO_MEMORY_DB_SPILL_LIMIT_MB", "4096")
        monkeypatch.setenv("GRAFEO_MEMORY_YOLO", "1")
        s = GrafeoMemorySettings()
        assert s.cache_enabled is True and s.yolo is True
        cc = s.cache_config()
        assert isinstance(cc, CacheConfig)
        assert cc.max_memory_bytes == 256 * 1024 * 1024
        assert cc.max_entries == 500
        cfg = s.apply_to(MemoryConfig())
        assert cfg.db_memory_limit_bytes == 2048 * 1024 * 1024
        assert cfg.db_spill_limit_bytes == 4096 * 1024 * 1024
        assert isinstance(cfg.cache, CacheConfig)

    def test_secrets_are_secretstr_and_not_leaked(self, monkeypatch):
        monkeypatch.setenv("TYPESAFE_API_KEY", "super-secret-value")
        monkeypatch.setenv("MISTRAL_API_KEY", "mistral-secret")
        s = GrafeoMemorySettings()
        assert s.typesafe_api_key is not None
        assert s.typesafe_api_key.get_secret_value() == "super-secret-value"
        assert "super-secret-value" not in repr(s)
        assert "super-secret-value" not in str(s)
        assert "mistral-secret" not in repr(s.model_dump())

    def test_apply_to_does_not_override_explicit(self, monkeypatch):
        monkeypatch.setenv("GRAFEO_MEMORY_DB_MEMORY_LIMIT_MB", "2048")
        s = GrafeoMemorySettings()
        cfg = MemoryConfig(db_memory_limit_bytes=1 << 20)
        s.apply_to(cfg)
        assert cfg.db_memory_limit_bytes == 1 << 20

    def test_dotenv_file(self, tmp_path, monkeypatch):
        env = tmp_path / ".env"
        env.write_text("GRAFEO_MEMORY_USER=alice\nGRAFEO_MEMORY_CACHE_ENABLED=true\nOPENAI_API_KEY=sk-test-123\n")
        monkeypatch.chdir(tmp_path)
        s = GrafeoMemorySettings()
        assert s.user == "alice"
        assert s.cache_enabled is True
        assert s.openai_api_key is not None and s.openai_api_key.get_secret_value() == "sk-test-123"


def test_redaction_filter_scrubs_child_logger_records(monkeypatch):
    """0.3.0 review: the filter sits on handlers, so records of child loggers are scrubbed too."""
    import io
    import logging

    from grafeo_memory.common.logging import install_redaction

    monkeypatch.setenv("TYPESAFE_API_KEY", "ts-very-secret-value")
    root = logging.getLogger("redaction_test_root")
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    root.addHandler(handler)
    root.propagate = False
    try:
        install_redaction(root)
        logging.getLogger("redaction_test_root.child").warning("key=%s", "ts-very-secret-value")
    finally:
        root.removeHandler(handler)
    assert "ts-very-secret-value" not in stream.getvalue()
    assert "***REDACTED***" in stream.getvalue()
