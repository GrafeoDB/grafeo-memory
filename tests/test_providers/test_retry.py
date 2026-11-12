"""Transient LLM provider errors retry inside the model (0.3.0 benchmark finding:
ConnectTimeouts made entity extraction fail and stored facts without entities)."""

from __future__ import annotations

from pydantic_ai.messages import ModelResponse, TextPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from grafeo_memory.common import timing
from grafeo_memory.common.cache import _chat_cache_model
from grafeo_memory.common.compat import run_sync
from grafeo_memory.providers.llm.retry import RetryingModel, with_retries


def _flaky(failures: int, exc: Exception):
    calls = {"n": 0}

    def handler(messages: list, info: AgentInfo) -> ModelResponse:
        calls["n"] += 1
        if calls["n"] <= failures:
            raise exc
        return ModelResponse(parts=[TextPart("ok")])

    return FunctionModel(function=handler), calls


def _run(model) -> str:
    from pydantic_ai import Agent

    return run_sync(Agent(model).run("hi")).output


def test_transient_error_is_retried(monkeypatch):
    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(timing.asyncio, "sleep", _no_sleep)
    model, calls = _flaky(2, TimeoutError("ConnectTimeout"))
    assert _run(with_retries(model, 3)) == "ok"
    assert calls["n"] == 3


def test_other_errors_fail_at_once():
    import pytest

    model, calls = _flaky(1, ValueError("bad request"))
    with pytest.raises(ValueError):
        _run(with_retries(model, 3))
    assert calls["n"] == 1


def test_zero_retries_leaves_the_model_alone():
    model, _ = _flaky(0, ValueError("x"))
    assert with_retries(model, 0) is model


def test_cache_key_is_unchanged_by_the_wrapper():
    model, _ = _flaky(0, ValueError("x"))
    assert _chat_cache_model(RetryingModel(model)) == _chat_cache_model(model)
