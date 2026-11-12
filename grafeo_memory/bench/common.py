"""Shared plumbing for the bench harnesses (locomo/lme/beam/matrix).

Paths derive from GRAFEO_MEMORY_BENCH_HOME (default <repo>/.benchmarks) via
common.paths, so CI can isolate runs; datasets fall back to the checkout.
LLM calls are cached through the shared JSONL cache, no call is made twice.
"""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import sys
import threading
import time
from pathlib import Path
from typing import Any

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
    "add_set_argument",
    "apply_overrides",
    "bench_home",
    "call_mistral",
    "data_dir",
    "default_cache",
    "ingest_complete",
    "load_env",
    "remove_db",
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
        raise ModuleNotFoundError(
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
        # write-then-rename: an interrupted put never leaves a truncated file behind
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w") as f:
            json.dump(self.data, f)
        os.replace(tmp, self.path)


def ingest_complete(log_path: Path) -> bool:
    """True when an ingest log exists and recorded no session errors (safe to skip on resume)."""
    if not log_path.exists():
        return False
    try:
        log = json.loads(log_path.read_text())
    except ValueError:
        return False
    return not log.get("errors")


def remove_db(db_path: Path) -> None:
    """Delete a partial database (the file and its sidecars) so a re-ingest starts fresh."""
    if not db_path.parent.exists():
        return
    for p in db_path.parent.iterdir():
        if p.name == db_path.name or p.name.startswith(db_path.name + "."):
            if p.is_dir():
                shutil.rmtree(p)
            else:
                p.unlink()


# Direct Mistral chat calls (answer/judge) bypass the pydantic-ai usage callback;
# track their tokens here for cost accounting.
DIRECT_TOKENS: dict[str, dict[str, int]] = {}

# Single-flight: concurrent identical requests wait for the first one and read its cached reply.
_INFLIGHT: dict[str, threading.Lock] = {}
_INFLIGHT_GUARD = threading.Lock()


def call_mistral(
    client, model: str, prompt: str, max_tokens: int = 300, temperature: float = 0.0, _retries: int = 8, cache=None
) -> str:
    req = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens, "temperature": temperature}
    if cache is None:
        return _call_mistral(client, model, prompt, max_tokens, temperature, _retries, None, req)
    hit = cache.get("chat", model, req)
    if hit is not None:
        return hit["content"]
    key = model + "|" + json.dumps(req, sort_keys=True)
    with _INFLIGHT_GUARD:
        lock = _INFLIGHT.setdefault(key, threading.Lock())
    with lock:
        hit = cache.get("chat", model, req)
        if hit is not None:
            return hit["content"]
        return _call_mistral(client, model, prompt, max_tokens, temperature, _retries, cache, req)


def _call_mistral(client, model: str, prompt: str, max_tokens: int, temperature: float, _retries: int, cache, req):
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


def add_set_argument(ap) -> None:
    """``--set KEY=VALUE`` (repeatable): any MemoryConfig field by dotted path."""
    ap.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help=(
            "override any MemoryConfig field by dotted path, repeatable; e.g. --set bucket.enabled=true "
            "--set bucket.floor=0.4 --set retrieval.order=deterministic --set taxonomy.mode=filter "
            "--set policy.packs=compliance,entrenchment --set tiers.procedural.enabled=true"
        ),
    )


def _coerce(current: Any, raw: str) -> Any:
    if isinstance(current, bool):
        if raw.lower() in ("1", "true", "yes", "on"):
            return True
        if raw.lower() in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"expected a boolean, got {raw!r}")
    if isinstance(current, int):
        return int(raw)
    if isinstance(current, float):
        return float(raw)
    if isinstance(current, tuple):
        return tuple(x.strip() for x in raw.split(",") if x.strip())
    if isinstance(current, list):
        return [x.strip() for x in raw.split(",") if x.strip()]
    if isinstance(current, dict):
        return json.loads(raw)
    if current is None:
        try:
            return json.loads(raw)
        except ValueError:
            return raw
    return raw


def apply_overrides(config, sets: list[str] | None):
    """Apply ``--set`` overrides to a MemoryConfig and re-validate it. Unknown paths fail loudly."""
    for item in sets or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"--set expects KEY=VALUE, got {item!r}")
        parts = key.strip().split(".")
        obj = config
        for part in parts[:-1]:
            if not dataclasses.is_dataclass(obj) or not hasattr(obj, part):
                raise ValueError(f"--set {key}: no field {part!r}")
            obj = getattr(obj, part)
        leaf = parts[-1]
        if not dataclasses.is_dataclass(obj) or leaf not in {f.name for f in dataclasses.fields(obj)}:
            raise ValueError(f"--set {key}: no field {leaf!r}")
        value = _coerce(getattr(obj, leaf), raw.strip())
        if dataclasses.is_dataclass(obj) and type(obj).__dataclass_params__.frozen:
            raise ValueError(f"--set {key}: {type(obj).__name__} is frozen")
        setattr(obj, leaf, value)
    if sets:
        config.__post_init__()
    return config
