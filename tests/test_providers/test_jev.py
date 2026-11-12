"""JevDecisionModel request paths against a stubbed HTTP client (no network)."""

from __future__ import annotations

from typing import ClassVar

import httpx
import pytest

from grafeo_memory.common.cache import MemoryCache
from grafeo_memory.common.compat import run_sync
from grafeo_memory.common.types import CacheConfig
from grafeo_memory.providers.decision import jev as jev_mod
from grafeo_memory.providers.decision.jev import JevDecisionModel
from grafeo_memory.providers.decision.types import CategoricalQuestion, ScoreQuestion


class _Client:
    """httpx.AsyncClient stand-in: answers from a queue of (status, json) or a callable."""

    responses: ClassVar[list] = []
    posts: ClassVar[list] = []

    def __init__(self, *a, **k):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def post(self, url, json, headers):
        _Client.posts.append(json)
        item = _Client.responses.pop(0)
        status, body = item(json) if callable(item) else item
        return httpx.Response(status, json=body, request=httpx.Request("POST", url))


@pytest.fixture
def client(monkeypatch):
    _Client.responses, _Client.posts = [], []
    monkeypatch.setattr(httpx, "AsyncClient", _Client)

    async def _no_sleep(_s):
        return None

    monkeypatch.setattr(jev_mod.asyncio, "sleep", _no_sleep)
    return _Client


def _answers(questions, choice="ADD"):
    return {"answers": {q: {"choice": choice, "confidence": 0.9} for q in questions}, "usage": {"input_tokens": 7}}


Q = [CategoricalQuestion(question_id="q1", question="?", choices=["ADD", "NONE"])]


def test_success_parses_answers_and_usage(client):
    client.responses.append((200, _answers(["q1"])))
    d = run_sync(JevDecisionModel(api_key="k").decide_async("state", Q))
    assert d[0].value == "ADD" and d[0].confidence == 0.9
    payload = client.posts[0]
    assert payload["questions"]["q1"]["type"] == "choice" and payload["state"] == "state"


def test_cache_hit_skips_the_api(client):
    cache = MemoryCache(CacheConfig(persist_path=None))
    client.responses.append((200, _answers(["q1"])))
    jev = JevDecisionModel(api_key="k", cache=cache)
    run_sync(jev.decide_async("state", Q))
    again = run_sync(jev.decide_async("state", Q))
    assert again[0].value == "ADD" and len(client.posts) == 1


def test_rate_limit_is_retried(client):
    client.responses.extend([(429, {"error": {"code": "rate_limited"}}), (200, _answers(["q1"]))])
    d = run_sync(JevDecisionModel(api_key="k").decide_async("state", Q))
    assert d[0].value == "ADD" and len(client.posts) == 2


def test_too_large_batch_is_split(client):
    qs = [CategoricalQuestion(question_id=f"q{i}", question="?", choices=["ADD", "NONE"]) for i in range(2)]

    def respond(payload):
        if len(payload["questions"]) > 1:
            return 400, {"error": {"code": "max_tokens_exceeded"}}
        return 200, _answers(payload["questions"])

    client.responses.extend([respond, respond, respond])
    d = run_sync(JevDecisionModel(api_key="k").decide_async("state", qs))
    assert [x.question_id for x in d] == ["q0", "q1"]


def test_score_question_maps_onto_its_range(client):
    sq = [ScoreQuestion(question_id="s", question="?", floor=0.0, ceiling=1.0, criteria=["low", "mid", "high"])]
    client.responses.append((200, {"answers": {"s": {"score": 2, "confidence": 1.0}}}))
    d = run_sync(JevDecisionModel(api_key="k").decide_async("state", sq))
    assert d[0].value == 1.0


def test_needs_a_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="API key"):
        JevDecisionModel()
