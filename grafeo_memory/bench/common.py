"""Shared plumbing for the bench harnesses (locomo/lme/beam/matrix).

Paths derive from GRAFEO_MEMORY_BENCH_HOME (default <repo>/.benchmarks) via
common.paths, so CI can isolate runs; datasets fall back to the checkout.
LLM calls are cached through the shared JSONL cache, no call is made twice.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

from grafeo_memory.common.paths import bench_home, data_dir, default_cache, repo_root, suite_runs
from grafeo_memory.common.telemetry import UsageTracker  # noqa: F401  (re-exported for harnesses)
from grafeo_memory.common.timing import is_transient, retry_async

REPO = repo_root()
PY = sys.executable  # the interpreter running the harness, on any OS

__all__ = [
    "BENCH_HOME",
    "DIRECT_TOKENS",
    "PY",
    "REPO",
    "Cache",
    "bench_home",
    "call_mistral",
    "data_dir",
    "default_cache",
    "load_env",
    "require_bench_deps",
    "suite_runs",
    "with_retry",
]

BENCH_HOME = bench_home()


def load_env() -> None:
    """Load <repo>/.env with override, the file wins over ambient env (stale keys)."""
    from dotenv import load_dotenv

    load_dotenv(override=True)


def require_bench_deps() -> None:
    """Clear error when the `bench` extra is missing (mistralai, nltk, numpy, pyarrow)."""
    missing = []
    for mod in ("mistralai", "nltk", "numpy", "pyarrow"):
        try:
            __import__(mod)
        except ImportError:
            missing.append(mod)
    if missing:
        raise ImportError(
            f"grafeo_memory.bench needs the bench extra: uv add 'grafeo-memory[bench]' (missing: {', '.join(missing)})"
        )


class Cache:
    """JSON answer/judge cache: {key: value}, rewritten on each put."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict = json.load(open(path)) if path.exists() else {}

    def get(self, key: str):
        return self.data.get(key)

    def put(self, key: str, value) -> None:
        self.data[key] = value
        with open(self.path, "w") as f:
            json.dump(self.data, f)


# Direct Mistral chat calls (answer/judge) bypass the pydantic-ai usage callback;
# track their tokens here for cost accounting.
DIRECT_TOKENS: dict[str, dict[str, int]] = {}


def call_mistral(
    client, model: str, prompt: str, max_tokens: int = 300, temperature: float = 0.0, _retries: int = 8, cache=None
) -> str:
    req = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    if cache is not None:
        hit = cache.get("chat", model, req)
        if hit is not None:
            return hit["content"]
    for attempt in range(_retries):
        try:
            r = client.chat.complete(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                temperature=temperature,
            )
            text = r.choices[0].message.content.strip()
            u = getattr(r, "usage", None)
            if u is not None:
                t = DIRECT_TOKENS.setdefault(model, {"input_tokens": 0, "output_tokens": 0})
                t["input_tokens"] += int(getattr(u, "prompt_tokens", 0) or 0)
                t["output_tokens"] += int(getattr(u, "completion_tokens", 0) or 0)
            if cache is not None:
                cache.put("chat", model, req, {"content": text})
            return text
        except Exception as e:
            if is_transient(e) and attempt < _retries - 1:
                time.sleep(min(2**attempt, 30))
                continue
            raise
    raise RuntimeError("call_mistral: retries exhausted")


async def with_retry(coro_fn, _retries: int = 5):
    """Await coro_fn() with backoff on transient errors (429/timeouts)."""
    return await retry_async(coro_fn, retries=_retries)
