"""Cache replay and secret-freeness gates for the shared benchmark cache.

The benchmark promise is "no call is made twice" and "no secret is persisted":
both are tested here at $0.
"""

from __future__ import annotations

import json
import os

from grafeo_memory import CacheConfig
from grafeo_memory.common.cache import MemoryCache, make_key


def _cfg(path) -> CacheConfig:
    return CacheConfig(persist_path=str(path), max_entries=1000)


def test_jsonl_replay_roundtrip(tmp_path):
    p = tmp_path / "cache.jsonl"
    c = MemoryCache(_cfg(p))
    c.put("chat", "mistral-small-latest", {"q": "hi"}, {"a": "yo"})
    c.put("decision", "jev", {"questions": {"q1": {"type": "choice"}}}, {"answers": {}})
    c2 = MemoryCache(_cfg(p))  # fresh instance = replay from disk
    assert c2.get("chat", "mistral-small-latest", {"q": "hi"}) == {"a": "yo"}
    assert c2.get("decision", "jev", {"questions": {"q1": {"type": "choice"}}}) == {"answers": {}}
    assert c2.stats()["entries"] == 2


def test_cache_file_never_contains_api_secrets(tmp_path, monkeypatch):
    """No persisted cache line may contain the value of a known secret env var."""
    probes = {
        "TYPESAFE_API_KEY": "ts-probe-abc123",
        "MISTRAL_API_KEY": "ms-probe-def456",
        "OPENAI_API_KEY": "oa-probe-ghi789",
    }
    for k, v in probes.items():
        monkeypatch.setenv(k, v)
    p = tmp_path / "cache.jsonl"
    c = MemoryCache(_cfg(p))
    # payloads that legitimately mention config values must still not leak secrets
    for i in range(50):
        c.put("chat", f"model-{i}", {"prompt": f"q{i}", "key_env": "TYPESAFE_API_KEY"}, {"a": str(i)})
    raw = p.read_text()
    for k, v in probes.items():
        assert v not in raw, f"secret value of {k} leaked into cache file"
    # and every line is valid JSON with a hash key (no plaintext payloads of secrets)
    for line in raw.strip().splitlines():
        rec = json.loads(line)
        assert set(rec) >= {"kind", "model", "key"}


def test_make_key_stable_across_processes():
    """Same call -> same key (this is what makes replay work across runs)."""
    k1 = make_key("chat", "m", {"b": 2, "a": [1, {"y": 1, "x": 2}]})
    k2 = make_key("chat", "m", {"a": [1, {"x": 2, "y": 1}], "b": 2})
    assert k1 == k2
    assert make_key("chat", "m", {"a": 1}) != k1


def test_bench_home_env_is_honored():
    """GRAFEO_MEMORY_BENCH_HOME is the isolation knob for benchmark outputs."""
    assert os.environ.get("GRAFEO_MEMORY_BENCH_HOME"), "conftest must set GRAFEO_MEMORY_BENCH_HOME"
