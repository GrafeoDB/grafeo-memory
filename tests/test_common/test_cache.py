"""Tests for the in-memory LRU call cache (grafeo_memory.cache) and its wiring."""

from __future__ import annotations

import asyncio
import itertools
import os

import pytest
from mock_llm import MockEmbedder, make_test_model

from grafeo_memory import CacheConfig, CachingEmbedder, CachingModel, MemoryCache, MemoryConfig, MemoryManager
from grafeo_memory.common.cache import make_key


def _cfg(**kw) -> CacheConfig:
    defaults = {"persist_path": None}
    defaults.update(kw)
    return CacheConfig(**defaults)


class TestMemoryCache:
    def test_put_get_roundtrip(self):
        c = MemoryCache(_cfg())
        c.put("chat", "m", {"q": 1}, {"a": "b"})
        assert c.get("chat", "m", {"q": 1}) == {"a": "b"}
        assert c.get("chat", "m", {"q": 2}) is None
        s = c.stats()
        assert s["hits"] == 1 and s["misses"] == 1 and s["entries"] == 1

    def test_key_is_order_insensitive_and_secret_free(self):
        k1 = make_key("chat", "mistral-small", {"a": 1, "b": {"c": 2, "d": 3}})
        k2 = make_key("chat", "mistral-small", {"b": {"d": 3, "c": 2}, "a": 1})
        assert k1 == k2
        assert "secret" not in k1

    def test_lru_eviction_by_entries(self):
        c = MemoryCache(_cfg(max_entries=3))
        for i in range(5):
            c.put("chat", "m", {"i": i}, {"v": i})
        assert c.stats()["entries"] == 3
        assert c.stats()["evictions"] == 2
        assert c.get("chat", "m", {"i": 0}) is None  # evicted
        assert c.get("chat", "m", {"i": 4}) == {"v": 4}  # kept

    def test_lru_eviction_by_bytes(self):
        c = MemoryCache(_cfg(max_entries=1000, max_memory_bytes=200))
        big = "x" * 150
        c.put("chat", "m", {"i": 0}, {"v": big})
        c.put("chat", "m", {"i": 1}, {"v": "small"})
        s = c.stats()
        assert s["bytes"] <= 200
        assert s["entries"] >= 1

    def test_ttl_expiry(self):
        c = MemoryCache(_cfg(ttl_seconds=0.01))
        c.put("chat", "m", {"i": 1}, {"v": 1})
        import time

        time.sleep(0.02)
        assert c.get("chat", "m", {"i": 1}) is None

    def test_jsonl_persistence_roundtrip_and_compaction(self, tmp_path):
        p = str(tmp_path / "c.jsonl")
        cfg = _cfg(persist_path=p, max_storage_bytes=400)
        c = MemoryCache(cfg)
        for i in range(10):
            c.put("chat", "m", {"i": i}, {"v": "x" * 60})
        c.close()
        assert os.path.getsize(p) <= 400 + 200  # compacted to live entries (allow slack for last append)
        c2 = MemoryCache(cfg)
        assert c2.get("chat", "m", {"i": 9}) == {"v": "x" * 60}
        assert c2.stats()["persisted"] is True

    def test_stats_shape(self):
        c = MemoryCache(_cfg())
        c.put("embed", "E:1", {"text": "hi"}, [0.1, 0.2])
        c.get("embed", "E:1", {"text": "hi"})
        s = c.stats()
        for k in ("entries", "bytes", "hits", "misses", "hit_rate", "evictions", "per_kind", "storage_bytes"):
            assert k in s
        assert s["per_kind"]["embed"]["hits"] == 1


class TestCachingWrappers:
    def test_caching_embedder_calls_inner_once(self):
        inner = MockEmbedder(8)
        orig = inner.embed
        calls = []

        def counting(texts):
            calls.append(len(texts))
            return orig(texts)

        inner.embed = counting
        ce = CachingEmbedder(inner, MemoryCache(_cfg()))
        a = ce.embed(["hello", "world"])
        b = ce.embed(["hello", "world"])
        assert a == b
        assert len(calls) == 1 and calls[0] == 2  # second call fully cached
        assert ce.dimensions == 8

    def test_caching_model_serves_second_identical_run_from_cache(self):
        from pydantic import BaseModel
        from pydantic_ai import Agent
        from pydantic_ai.messages import ModelResponse, ToolCallPart
        from pydantic_ai.models.function import AgentInfo, FunctionModel

        class Out(BaseModel):
            answer: str

        calls = []

        def handler(messages: list, info: AgentInfo) -> ModelResponse:
            calls.append(1)
            return ModelResponse(parts=[ToolCallPart(tool_name=info.output_tools[0].name, args={"answer": "42"})])

        cache = MemoryCache(_cfg())
        agent = Agent(model=CachingModel(FunctionModel(handler), cache), output_type=Out)

        async def run():
            r1 = await agent.run("what is the answer?")
            r2 = await agent.run("what is the answer?")
            r3 = await agent.run("different question")
            return r1.output.answer, r2.output.answer, r3.output.answer

        a1, a2, a3 = asyncio.run(run())
        assert (a1, a2, a3) == ("42", "42", "42")
        assert len(calls) == 2  # third run was a cache miss
        s = cache.stats()
        assert s["per_kind"]["chat"]["hits"] == 1 and s["per_kind"]["chat"]["misses"] == 2


class TestManagerIntegration:
    def test_cache_wraps_model_and_embedder(self):
        model = make_test_model([{"facts": ["x"], "entities": [], "relations": []}])
        cfg = MemoryConfig(cache=CacheConfig(persist_path=None), embedding_dimensions=16)
        mgr = MemoryManager(model, cfg, embedder=MockEmbedder(16))
        assert isinstance(mgr._model, CachingModel)
        assert isinstance(mgr._embedder, CachingEmbedder)
        assert mgr._cache is not None
        st = mgr.stats()
        assert "cache" in st.__dataclass_fields__ or hasattr(st, "cache")
        assert st.cache["enabled"] is True
        assert "db_limits" in st.db_limits or st.db_limits == {} or "memory_bytes" in st.db_limits

    def test_db_limits_passed_to_grafeo(self, tmp_path):
        model = make_test_model([])
        cfg = MemoryConfig(
            db_path=str(tmp_path / "m.grafeo"),
            embedding_dimensions=16,
            db_memory_limit_bytes=1 << 28,
            db_spill_limit_bytes=1 << 29,
        )
        mgr = MemoryManager(model, cfg, embedder=MockEmbedder(16))
        usage = mgr._db.memory_usage()
        assert usage["limit_bytes"] == 1 << 28
        assert usage["spill"]["limit_bytes"] == 1 << 29
        st = mgr.stats()
        assert st.db_limits["memory_limit_bytes"] == 1 << 28
        mgr.close()

    def test_persist_path_derived_from_db_path(self, tmp_path):
        model = make_test_model([])
        cfg = MemoryConfig(db_path=str(tmp_path / "m.grafeo"), embedding_dimensions=16, cache=True)
        mgr = MemoryManager(model, cfg, embedder=MockEmbedder(16))
        assert mgr._cache._persist_path == str(tmp_path / "m.grafeo") + ".llmcache.jsonl"
        mgr.close()

    def test_memory_limit_warning_at_95pct(self, tmp_path, caplog):
        import logging

        model = make_test_model([])
        cfg = MemoryConfig(db_path=str(tmp_path / "m.grafeo"), embedding_dimensions=16)
        mgr = MemoryManager(model, cfg, embedder=MockEmbedder(16))

        class FakeDB:
            def memory_usage(self):
                return {"total_bytes": 960, "limit_bytes": 1000, "spill": {}}

        mgr._db = FakeDB()
        with caplog.at_level(logging.WARNING, logger="grafeo_memory.manager"):
            mgr._check_db_memory_limit()
            mgr._check_db_memory_limit()  # second call must not re-warn at same pct
        warnings = [r for r in caplog.records if "95%" in r.getMessage() or "96%" in r.getMessage()]
        assert len(warnings) == 1
        assert "db_memory_limit_bytes" in warnings[0].getMessage()

    def test_cache_validation(self):
        with pytest.raises(ValueError):
            MemoryConfig(cache=CacheConfig(max_entries=0))
        with pytest.raises(ValueError):
            MemoryConfig(db_memory_limit_bytes=0)


class TestPersistenceDurability:
    """0.3.0 release review: the persisted file must never shrink to the in-memory LRU view."""

    def test_close_keeps_entries_beyond_max_entries(self, tmp_path):
        path = str(tmp_path / "c.jsonl")
        a = MemoryCache(_cfg(persist_path=path, max_entries=3), persist_path=path)
        for i in range(10):
            a.put("chat", "m", {"q": i}, {"a": i})
        a.close()
        b = MemoryCache(_cfg(persist_path=path, max_entries=3), persist_path=path)
        b.close()  # a reload plus close must not truncate either
        big = MemoryCache(_cfg(persist_path=path, max_entries=100), persist_path=path)
        assert all(big.get("chat", "m", {"q": i}) == {"a": i} for i in range(10))

    def test_reload_keeps_newest_entries(self, tmp_path):
        path = str(tmp_path / "c.jsonl")
        a = MemoryCache(_cfg(persist_path=path, max_storage_bytes=250), persist_path=path)
        for i in range(5):
            a.put("chat", "m", {"q": i}, {"a": i})
        a.close()
        b = MemoryCache(_cfg(persist_path=path, max_entries=2), persist_path=path)
        assert b.get("chat", "m", {"q": 4}) == {"a": 4}

    def test_reload_keeps_newest_when_the_clock_does_not_tick(self, tmp_path, monkeypatch):
        # Windows: back-to-back writes can share one time.time() value
        import grafeo_memory.common.cache as cache_mod

        monkeypatch.setattr(cache_mod.time, "time", lambda: 1_000_000.0)
        path = str(tmp_path / "c.jsonl")
        a = MemoryCache(_cfg(persist_path=path, max_storage_bytes=250), persist_path=path)
        for i in range(5):
            a.put("chat", "m", {"q": i}, {"a": i})
        a.close()
        b = MemoryCache(_cfg(persist_path=path, max_entries=2), persist_path=path)
        assert b.get("chat", "m", {"q": 4}) == {"a": 4}

    def test_two_writers_on_one_file_keep_both(self, tmp_path):
        path = str(tmp_path / "c.jsonl")
        a = MemoryCache(_cfg(persist_path=path), persist_path=path)
        b = MemoryCache(_cfg(persist_path=path), persist_path=path)
        a.put("chat", "m", {"q": "a"}, {"a": 1})
        b.put("chat", "m", {"q": "b"}, {"a": 2})
        b.close()
        a.close()
        c = MemoryCache(_cfg(persist_path=path), persist_path=path)
        assert c.get("chat", "m", {"q": "a"}) == {"a": 1}
        assert c.get("chat", "m", {"q": "b"}) == {"a": 2}


class TestEmbedderKey:
    def test_models_of_one_class_do_not_share_entries(self):
        class _Emb:
            dimensions = 2

            def __init__(self, model: str = "default-model", value: float = 0.0):
                self._model = model
                self._value = value

            def embed(self, texts):
                return [[self._value, self._value] for _ in texts]

        cache = MemoryCache(_cfg())
        assert CachingEmbedder(_Emb(), cache).embed(["x"]) == [[0.0, 0.0]]
        assert CachingEmbedder(_Emb("other-model", 1.0), cache).embed(["x"]) == [[1.0, 1.0]]
        # the default model keeps the pre-fix key, so existing caches still hit
        assert cache.get("embed", "_Emb:2", {"text": "x"}) == [0.0, 0.0]

    def test_array_like_embeddings_are_cached_as_floats(self):
        class _Scalar:
            """Stands in for a numpy float: not JSON-serializable, converts with float()."""

            def __init__(self, v: float):
                self._v = v

            def __float__(self) -> float:
                return self._v

        class _Vec:
            """Stands in for a numpy array: iterable, but not a list."""

            def __init__(self, vals):
                self._vals = [_Scalar(v) for v in vals]

            def __iter__(self):
                return iter(self._vals)

            def __len__(self):
                return len(self._vals)

        class _Emb:
            dimensions = 2

            def embed(self, texts):
                return [_Vec([0.25, 0.5]) for _ in texts]

        cache = MemoryCache(_cfg())
        CachingEmbedder(_Emb(), cache).embed(["x"])
        hit = cache.get("embed", "_Emb:2", {"text": "x"})
        assert hit == [0.25, 0.5] and all(type(x) is float for x in hit)
        assert CachingEmbedder(_Emb(), cache).embed(["x"]) == [[0.25, 0.5]]

    def test_custom_endpoint_gets_its_own_key(self):
        from grafeo_memory.common.cache import _embed_cache_model

        class _Client:
            def __init__(self, base_url):
                self.base_url = base_url

        class _Emb:
            def __init__(self, client):
                self._client = client

        assert _embed_cache_model(_Emb(_Client("https://api.openai.com/v1/")), 8) == "_Emb:8"
        assert _embed_cache_model(_Emb(object()), 8) == "_Emb:8"
        assert _embed_cache_model(_Emb(_Client("http://localhost:1234/v1/")), 8) == "_Emb:8@http://localhost:1234/v1"


def test_disabled_cache_config_creates_no_cache(tmp_path):
    db = str(tmp_path / "m.grafeo")
    cfg = MemoryConfig(db_path=db, embedding_dimensions=16, cache=CacheConfig(enabled=False))
    with MemoryManager(make_test_model([]), cfg, embedder=MockEmbedder(16)) as m:
        assert m._cache is None
    assert not os.path.exists(db + ".llmcache.jsonl")


def test_chat_key_separates_custom_endpoints_and_keeps_official_ones():
    from grafeo_memory.common.cache import _chat_cache_model

    class _M:
        def __init__(self, name, system, base_url):
            self.model_name, self.system, self.base_url = name, system, base_url

    assert _chat_cache_model(_M("mistral-small-latest", "mistral", "https://api.mistral.ai")) == "mistral-small-latest"
    assert _chat_cache_model(_M("gpt-4o-mini", "openai", "https://api.openai.com/v1/")) == "gpt-4o-mini"
    local = _chat_cache_model(_M("gpt-4o-mini", "openai", "http://localhost:11434/v1/"))
    assert local != "gpt-4o-mini" and "localhost:11434" in local
    # a lookalike host is not the official one
    spoof = _chat_cache_model(_M("gpt-4o-mini", "openai", "https://api.openai.com.attacker.invalid"))
    assert spoof != "gpt-4o-mini" and "attacker.invalid" in spoof


def test_clear_removes_entries_and_file(tmp_path):
    path = str(tmp_path / "c.jsonl")
    c = MemoryCache(_cfg(persist_path=path), persist_path=path)
    c.put("chat", "m", {"q": 1}, {"a": 1})
    c.clear()
    assert c.get("chat", "m", {"q": 1}) is None
    assert not os.path.exists(path)


def test_volatile_keys_are_kept_in_values_but_not_in_keys():
    c = MemoryCache(_cfg())
    c.put("chat", "m", {"q": 1, "timestamp": "t1"}, {"args": {"timestamp": "2024-01-01", "run_id": "r"}})
    assert c.get("chat", "m", {"q": 1, "timestamp": "t2"}) == {"args": {"timestamp": "2024-01-01", "run_id": "r"}}


def test_reload_keeps_volatile_keys_in_values(tmp_path):
    path = str(tmp_path / "c.jsonl")
    MemoryCache(_cfg(persist_path=path), persist_path=path).put("chat", "m", {"q": 1}, {"timestamp": 5})
    assert MemoryCache(_cfg(persist_path=path), persist_path=path).get("chat", "m", {"q": 1}) == {"timestamp": 5}


def test_unreadable_persist_path_does_not_abort_startup(tmp_path):
    path = tmp_path / "dir.jsonl"
    path.mkdir()
    c = MemoryCache(_cfg(persist_path=str(path)), persist_path=str(path))
    assert c.stats()["entries"] == 0


def test_load_keeps_newest_without_counting_evictions(tmp_path):
    path = str(tmp_path / "c.jsonl")
    a = MemoryCache(_cfg(persist_path=path), persist_path=path)
    for i in range(5):
        a.put("chat", "m", {"q": i}, {"a": i})
    b = MemoryCache(_cfg(persist_path=path, max_entries=2), persist_path=path)
    s = b.stats()
    assert s["entries"] == 2 and s["evictions"] == 0
    assert b.get("chat", "m", {"q": 4}) == {"a": 4} and b.get("chat", "m", {"q": 2}) is None


def test_compaction_stays_under_cap_and_leaves_headroom(tmp_path):
    path = str(tmp_path / "c.jsonl")
    cap = 2000
    c = MemoryCache(_cfg(persist_path=path, max_storage_bytes=cap), persist_path=path)
    sizes = []
    for i in range(60):
        c.put("chat", "m", {"q": i}, {"a": "x" * 40})
        sizes.append(os.path.getsize(path))
    assert max(sizes) <= cap
    # after a compaction the file sits at the 80% watermark, so the next one is several writes away
    drops = [i for i in range(1, len(sizes)) if sizes[i] < sizes[i - 1]]
    assert len(drops) >= 2 and all(b - a > 1 for a, b in itertools.pairwise(drops))


def test_record_larger_than_cap_is_not_kept(tmp_path):
    path = str(tmp_path / "c.jsonl")
    c = MemoryCache(_cfg(persist_path=path, max_storage_bytes=100), persist_path=path)
    c.put("chat", "m", {"q": 1}, {"a": "x" * 500})
    assert os.path.getsize(path) <= 100


def test_caching_model_request_stream_is_an_async_context_manager():
    from pydantic_ai.messages import ModelRequest, UserPromptPart
    from pydantic_ai.models import ModelRequestParameters
    from pydantic_ai.models.test import TestModel

    model = CachingModel(TestModel(custom_output_text="streamed"), MemoryCache(_cfg()))

    async def run():
        msgs = [ModelRequest(parts=[UserPromptPart("hi")])]
        async with model.request_stream(msgs, None, ModelRequestParameters()) as stream:
            async for _ in stream:
                pass
            return stream.get()

    resp = asyncio.run(run())
    assert "streamed" in str(resp.parts)
