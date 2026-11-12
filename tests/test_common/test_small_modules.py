"""Secrets, model registry, telemetry and the retry helpers."""

from __future__ import annotations

import pytest

from grafeo_memory.common import timing
from grafeo_memory.common.exceptions import SecretNotFoundError
from grafeo_memory.common.secrets import get_secret
from grafeo_memory.common.telemetry import UsageTracker, stderr_usage
from grafeo_memory.providers.llm.model_registry import provider_of, resolve_model


class TestSecrets:
    def test_process_env_wins_and_value_is_masked(self, monkeypatch):
        monkeypatch.setenv("GM_TEST_API_KEY", "from-env-123456")
        s = get_secret("GM_TEST_API_KEY")
        assert s is not None and s.get_secret_value() == "from-env-123456"
        assert "from-env" not in repr(s) and "from-env" not in str(s)

    def test_dotenv_fallback(self, monkeypatch, tmp_path):
        env = tmp_path / ".env"
        env.write_text("GM_TEST_API_KEY=from-file-123456\n", encoding="utf-8")
        monkeypatch.delenv("GM_TEST_API_KEY", raising=False)
        monkeypatch.setenv("GRAFEO_MEMORY_ENV_FILE", str(env))
        s = get_secret("GM_TEST_API_KEY")
        assert s is not None and s.get_secret_value() == "from-file-123456"

    def test_missing(self, monkeypatch, tmp_path):
        monkeypatch.delenv("GM_TEST_API_KEY", raising=False)
        monkeypatch.setenv("GRAFEO_MEMORY_ENV_FILE", str(tmp_path / "none.env"))
        assert get_secret("GM_TEST_API_KEY") is None
        with pytest.raises(SecretNotFoundError, match="GM_TEST_API_KEY"):
            get_secret("GM_TEST_API_KEY", required=True)


class TestModelRegistry:
    def test_provider_of(self):
        assert provider_of("mistral:mistral-small-latest") == "mistral"
        assert provider_of("gpt-4o-mini") is None
        assert provider_of("claude-sonnet-4-5") is None

    def test_resolve_passes_through(self):
        assert resolve_model("openai:gpt-4o-mini") == "openai:gpt-4o-mini"
        sentinel = object()
        assert resolve_model(sentinel) is sentinel  # type: ignore[arg-type]


class TestTelemetry:
    def test_usage_tracker_totals(self):
        class _U:
            def __init__(self, i, o):
                self.input_tokens, self.output_tokens = i, o

        t = UsageTracker()
        t("extract", _U(10, 2))
        t("extract", _U(5, 1))
        t("reconcile", _U(7, None))
        s = t.stats()
        assert s["ops"]["extract"] == {"calls": 2, "input_tokens": 15, "output_tokens": 3}
        assert (s["total_input_tokens"], s["total_output_tokens"]) == (22, 3)

    def test_stderr_usage(self, capsys):
        stderr_usage("extract", "u")
        assert "[usage] extract: u" in capsys.readouterr().err


class TestRetrySync:
    def test_transient_then_success(self, monkeypatch):
        monkeypatch.setattr(timing.time, "sleep", lambda _s: None)
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            if calls["n"] < 3:
                raise TimeoutError("ReadTimeout")
            return "ok"

        assert timing.retry_sync(fn) == "ok" and calls["n"] == 3

    def test_permanent_error_raises_at_once(self):
        calls = {"n": 0}

        def fn():
            calls["n"] += 1
            raise ValueError("bad")

        with pytest.raises(ValueError):
            timing.retry_sync(fn)
        assert calls["n"] == 1

    def test_transient_until_exhausted(self, monkeypatch):
        monkeypatch.setattr(timing.time, "sleep", lambda _s: None)

        def fn():
            raise TimeoutError("ConnectTimeout")

        with pytest.raises(TimeoutError):
            timing.retry_sync(fn, retries=2)


class TestTransient:
    class _HTTPError(Exception):
        def __init__(self, status: int, msg: str = ""):
            super().__init__(msg)
            self.status_code = status

    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504, 529])
    def test_transient_statuses(self, status):
        assert timing.is_transient(self._HTTPError(status))

    def test_status_wins_over_message_text(self):
        # a 400 whose message echoes "500" or "Timeout" from the prompt is not retried
        assert not timing.is_transient(self._HTTPError(400, "max 500 tokens; ConnectTimeout in prompt"))
        assert not timing.is_transient(ValueError("429 Timeout Connect"))

    def test_class_names_and_wrapped_causes(self):
        assert timing.is_transient(TimeoutError())
        assert timing.is_transient(ConnectionResetError())
        try:
            try:
                raise TimeoutError()
            except TimeoutError as inner:
                raise RuntimeError("wrapped") from inner
        except RuntimeError as outer:
            assert timing.is_transient(outer)

    def test_zero_retries_still_calls_once(self):
        assert timing.retry_sync(lambda: "ok", retries=0) == "ok"

        async def fn():
            return "ok"

        import asyncio

        assert asyncio.run(timing.retry_async(fn, retries=0)) == "ok"


def test_embedding_client_protocol_is_runtime_checkable():
    from grafeo_memory.common.types import EmbeddingClient

    class _Emb:
        dimensions = 2

        def embed(self, texts):
            return []

    assert isinstance(_Emb(), EmbeddingClient)
    assert not isinstance(object(), EmbeddingClient)
