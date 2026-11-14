"""In-memory LRU cache for LLM, embedding and decision-model calls.

The cache is an integral, opt-in part of grafeo-memory: enable it with
``MemoryConfig(cache=CacheConfig())`` (or ``cache=True`` for the defaults), or
process-wide with ``GRAFEO_MEMORY_CACHE_ENABLED=true``.

Design:

- Hot store is a plain in-memory ``OrderedDict`` (LRU), bounded by both
  ``max_entries`` and ``max_memory_bytes``, no SQLite, no external service.
- Optional persistence is an append-only JSONL sidecar (``persist_path``),
  bounded by ``max_storage_bytes``; when the file exceeds the cap it is
  compacted to the live entries (write tmp + atomic replace). The manager
  derives ``<db_path>.llmcache.jsonl`` automatically for file-backed
  databases, so benchmark reruns only pay for calls never made before.
- Keys are content-addressed: ``sha256(kind | model | canonical_json(request)
  | temperature | max_tokens)``. Canonical JSON (sorted keys) makes hits robust
  to dict ordering. Credentials are never part of the key or the stored
  request, model *name* only.
- Three kinds: ``chat`` (pydantic-ai model calls), ``embed`` (per-text
  embeddings), ``decision`` (decision-model calls, wired when the decisions
  package lands).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import logging
import os
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Any
from urllib.parse import urlparse

from pydantic_ai.models import Model

from grafeo_memory.common.types import CacheConfig, EmbeddingClient

logger = logging.getLogger(__name__)


def canonical_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, compact separators, dataclasses/pydantic handled."""
    return json.dumps(_to_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)


# Per-run volatile fields: excluding them keeps identical requests cacheable
# across runs (timestamps, per-run UUIDs).
_VOLATILE_FIELDS = {"timestamp", "conversation_id", "run_id"}


def _to_jsonable(obj: Any, *, strip_keys: bool = True) -> Any:
    """JSON-ready form. Volatile dataclass fields (message metadata) are always
    dropped; volatile dict keys only when ``strip_keys`` (request keys, never
    stored values, where they may be real data such as tool-call arguments)."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: _to_jsonable(getattr(obj, f.name), strip_keys=strip_keys)
            for f in dataclasses.fields(obj)
            if f.name not in _VOLATILE_FIELDS
        }
    if hasattr(obj, "model_dump"):
        return _to_jsonable(obj.model_dump(mode="json"), strip_keys=strip_keys)
    if isinstance(obj, dict):
        return {
            str(k): _to_jsonable(v, strip_keys=strip_keys)
            for k, v in obj.items()
            if not (strip_keys and k in _VOLATILE_FIELDS)
        }
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v, strip_keys=strip_keys) for v in obj]
    return obj


def _value_jsonable(obj: Any) -> Any:
    return _to_jsonable(obj, strip_keys=False)


def _value_json(obj: Any) -> str:
    """Canonical JSON of a stored value (volatile dict keys kept)."""
    return json.dumps(_value_jsonable(obj), sort_keys=True, separators=(",", ":"), default=str)


def make_key(kind: str, model: str, request: Any) -> str:
    h = hashlib.sha256()
    h.update(kind.encode())
    h.update(b"\0")
    h.update(str(model).encode())
    h.update(b"\0")
    h.update(canonical_json(request).encode())
    return h.hexdigest()


_PART_CLASSES: dict[str, type] | None = None


def _part_classes() -> dict[str, Any]:
    global _PART_CLASSES
    if _PART_CLASSES is None:
        import pydantic_ai.messages as M

        _PART_CLASSES = {}
        for n, c in vars(M).items():
            if dataclasses.is_dataclass(c) and n.endswith("Part"):
                pk = c.__dataclass_fields__.get("part_kind")
                if pk is not None and pk.default:
                    _PART_CLASSES[pk.default] = c
    return _PART_CLASSES


def _revive(obj: Any) -> Any:
    """Rebuild pydantic-ai dataclasses (messages/parts/usage) from cached JSON."""
    import typing
    from datetime import datetime

    if isinstance(obj, list):
        return [_revive(v) for v in obj]
    if not isinstance(obj, dict):
        return obj
    pk = obj.get("part_kind")
    if pk is not None and pk in _part_classes():
        cls = _part_classes()[pk]
        hints = typing.get_type_hints(cls)
        kwargs = {}
        for f in dataclasses.fields(cls):
            if f.name not in obj:
                continue
            v = _revive(obj[f.name])
            t = hints.get(f.name)
            if t is datetime and isinstance(v, str):
                v = datetime.fromisoformat(v.replace("Z", "+00:00"))
            elif dataclasses.is_dataclass(t) and isinstance(v, dict):
                v = _revive_dataclass(t, v)
            kwargs[f.name] = v
        return cls(**kwargs)
    kind = obj.get("kind")
    if kind in ("request", "response"):
        import pydantic_ai.messages as M

        cls = M.ModelRequest if kind == "request" else M.ModelResponse
        return _revive_dataclass(cls, obj)
    return {k: _revive(v) for k, v in obj.items()}


def _revive_dataclass(cls: Any, obj: dict) -> Any:
    import typing
    from datetime import datetime

    hints = typing.get_type_hints(cls)
    kwargs = {}
    for f in dataclasses.fields(cls):
        if f.name not in obj:
            continue
        v = _revive(obj[f.name])
        t = hints.get(f.name)
        if t is datetime and isinstance(v, str):
            v = datetime.fromisoformat(v.replace("Z", "+00:00"))
        elif dataclasses.is_dataclass(t) and isinstance(v, dict):
            v = _revive_dataclass(t, v)
        kwargs[f.name] = v
    return cls(**kwargs)


# --- OpenTelemetry counters (zero-cost when disabled) ------------------------

_instrument = False
_counters: dict[str, Any] = {}


def set_instrumentation(enabled: bool) -> None:
    global _instrument
    _instrument = enabled


def _counter(name: str) -> Any:
    if not _instrument:
        return None
    if name in _counters:
        return _counters[name]
    try:
        from opentelemetry.metrics import get_meter

        c = get_meter("grafeo-memory").create_counter(name)
    except ImportError:
        c = None
    _counters[name] = c
    return c


# Compaction shrinks the file to this share of max_storage_bytes.
_COMPACT_WATERMARK = 0.8


class MemoryCache:
    """Thread-safe in-memory LRU cache with optional JSONL persistence."""

    def __init__(self, config: CacheConfig, persist_path: str | None = None):
        self._cfg = config
        self._persist_path = persist_path if persist_path is not None else config.persist_path
        self._lock = threading.Lock()
        self._entries: OrderedDict[str, tuple[float, str]] = OrderedDict()
        self._bytes = 0
        self._hits = 0
        self._misses = 0
        self._evictions = 0
        self._per_kind: dict[str, dict[str, int]] = {}
        self._last_ts = 0.0
        if self._persist_path:
            self._load()

    # -- core API ------------------------------------------------------------

    def get(self, kind: str, model: str, request: Any) -> Any | None:
        key = make_key(kind, model, request)
        now = time.time()
        with self._lock:
            item = self._entries.get(key)
            if item is None:
                self._misses += 1
                self._kind(kind)["misses"] += 1
                c = _counter("grafeo_memory.cache.misses")
                if c:
                    c.add(1, {"kind": kind})
                return None
            ts, val = item
            ttl = self._cfg.ttl_seconds
            if ttl is not None and now - ts > ttl:
                self._remove_locked(key)
                self._misses += 1
                self._kind(kind)["misses"] += 1
                return None
            self._entries.move_to_end(key)
            self._hits += 1
            self._kind(kind)["hits"] += 1
            c = _counter("grafeo_memory.cache.hits")
            if c:
                c.add(1, {"kind": kind})
            return json.loads(val)

    def put(self, kind: str, model: str, request: Any, value: Any) -> None:
        key = make_key(kind, model, request)
        val = _value_json(value)
        record = {  # request already stripped; the value keeps its volatile keys
            "key": key,
            "kind": kind,
            "model": str(model),
            "request": _to_jsonable(request),
            "value": _value_jsonable(value),
        }
        with self._lock:
            # strictly increasing per instance: reload and compaction rank entries by ts, and
            # time.time() repeats within a clock tick on Windows, which made "newest" arbitrary
            ts = self._last_ts = max(time.time(), self._last_ts + 1e-6)
            line = _value_json({**record, "ts": ts})
            if key in self._entries:
                self._remove_locked(key)
            self._entries[key] = (ts, val)
            self._bytes += len(key) + len(val)
            self._evict_locked()
            if self._persist_path:
                self._append_locked(line)

    def stats(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "enabled": True,
                "entries": len(self._entries),
                "bytes": self._bytes,
                "max_entries": self._cfg.max_entries,
                "max_memory_bytes": self._cfg.max_memory_bytes,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else None,
                "evictions": self._evictions,
                "per_kind": {k: dict(v) for k, v in self._per_kind.items()},
                "persisted": bool(self._persist_path),
                "persist_path": self._persist_path,
                "storage_bytes": self._storage_bytes(),
                "max_storage_bytes": self._cfg.max_storage_bytes,
                "ttl_seconds": self._cfg.ttl_seconds,
            }

    def clear(self) -> None:
        """Drop every entry, in memory and in the persistence file. The file holds
        prompts and responses, which include memory text: clear it when deleting a
        user's data."""
        with self._lock:
            self._entries.clear()
            self._bytes = 0
            if self._persist_path and os.path.exists(self._persist_path):
                os.remove(self._persist_path)

    def close(self) -> None:
        """Compact the persistence file if it outgrew ``max_storage_bytes``.

        The file is the durable store and may hold far more entries than the
        in-memory LRU (``max_entries``) or other instances' writes, so closing
        never rewrites it from memory."""
        if not self._persist_path:
            return
        with self._lock:
            if self._storage_bytes() > self._cfg.max_storage_bytes:
                self._compact_locked()

    # -- internals ------------------------------------------------------------

    def _kind(self, kind: str) -> dict[str, int]:
        d = self._per_kind.setdefault(kind, {"hits": 0, "misses": 0})
        return d

    def _remove_locked(self, key: str) -> None:
        item = self._entries.pop(key, None)
        if item is not None:
            self._bytes -= len(key) + len(item[1])

    def _evict_locked(self) -> None:
        while len(self._entries) > self._cfg.max_entries and self._entries:
            self._remove_locked(next(iter(self._entries)))
            self._evictions += 1
        while self._bytes > self._cfg.max_memory_bytes and self._entries:
            self._remove_locked(next(iter(self._entries)))
            self._evictions += 1

    def _storage_bytes(self) -> int:
        if not self._persist_path:
            return 0
        try:
            return os.path.getsize(self._persist_path)
        except OSError:
            return 0

    def _load(self) -> None:
        path = self._persist_path
        if not path or not os.path.exists(path):
            return
        now = time.time()
        ttl = self._cfg.ttl_seconds
        try:
            records = self._read_file(path)
        except OSError:  # unreadable file or a directory: run without the persisted entries
            logger.warning("cache: cannot read %s", path, exc_info=True)
            return
        # Keep only the newest entries that fit the hot-cache limits (the file may be
        # in append order or newest-first after a compaction), then insert them oldest
        # first so LRU eviction drops the oldest.
        ordered = sorted(records.items(), key=lambda kv: kv[1][0])
        del records
        keep: list[tuple[str, float, str]] = []
        nbytes = 0
        for key, (ts, val) in reversed(ordered):
            if ttl is not None and now - ts > ttl:
                continue
            size = len(key) + len(val)
            if len(keep) >= self._cfg.max_entries or nbytes + size > self._cfg.max_memory_bytes:
                break
            keep.append((key, ts, val))
            nbytes += size
        del ordered
        for key, ts, val in reversed(keep):
            self._entries[key] = (ts, val)
            self._last_ts = max(self._last_ts, ts)
        self._bytes += nbytes
        logger.debug("cache: loaded %d entries from %s", len(keep), path)

    def _append_locked(self, line: str) -> None:
        path = self._persist_path
        assert path
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
            if self._storage_bytes() > self._cfg.max_storage_bytes:
                self._compact_locked()
        except OSError:
            logger.warning("cache: cannot persist to %s", path, exc_info=True)

    @staticmethod
    def _read_file(path: str) -> dict[str, tuple[float, str]]:
        """Every entry in a persistence file, newest record per key."""
        records: dict[str, tuple[float, str]] = {}
        with open(path, encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    key, val, ts = rec["key"], _value_json(rec["value"]), float(rec.get("ts", 0) or 0)
                except (KeyError, TypeError, ValueError):  # torn or foreign line
                    continue
                if key not in records or ts >= records[key][0]:
                    records[key] = (ts, val)
        return records

    def _compact_locked(self) -> None:
        """Rewrite the persistence file newest-first, down to a low watermark of
        max_storage_bytes so the next compaction is many writes away. A record larger
        than that on its own is dropped, so the file always ends under the cap.

        Built from the file itself plus this instance's entries, never from memory
        alone: the in-memory LRU is bounded and other instances may share the file."""
        path = self._persist_path
        if not path:
            return
        tmp = f"{path}.{os.getpid()}.{id(self)}.tmp"
        try:
            records = self._read_file(path) if os.path.exists(path) else {}
            for key, (ts, val) in self._entries.items():
                if key not in records or ts >= records[key][0]:
                    records[key] = (ts, val)
            target = int(self._cfg.max_storage_bytes * _COMPACT_WATERMARK)
            written = 0
            with open(tmp, "w", encoding="utf-8") as f:
                for key, (ts, val) in sorted(records.items(), key=lambda kv: kv[1][0], reverse=True):  # newest first
                    line = json.dumps(
                        {"key": key, "ts": ts, "value": json.loads(val)}, sort_keys=True, separators=(",", ":")
                    )
                    if len(line) + 1 > target:
                        continue
                    if written + len(line) + 1 > target:
                        break
                    f.write(line + "\n")
                    written += len(line) + 1
            os.replace(tmp, path)
        except OSError:
            logger.warning("cache: compaction of %s failed", path, exc_info=True)


# Providers' own API hosts. A model served from one of these keeps the plain
# ``model_name`` cache key (so existing caches stay valid); any other endpoint
# (Azure, Ollama, a gateway) adds ``system@base_url`` so two endpoints serving the
# same model name never share entries.
_OFFICIAL_HOSTS = frozenset(
    {
        "api.openai.com",
        "api.mistral.ai",
        "api.anthropic.com",
        "api.groq.com",
        "generativelanguage.googleapis.com",
        "api.cohere.com",
        "api.deepseek.com",
        "api.x.ai",
    }
)


def _is_official(base_url: Any) -> bool:
    """True when ``base_url`` is unset or its hostname is exactly a provider's own API host."""
    if not base_url:
        return True
    url = str(base_url).strip()
    try:
        host = urlparse(url if "//" in url else f"//{url}").hostname
    except ValueError:
        return False
    return host in _OFFICIAL_HOSTS


def _chat_cache_model(inner: Any) -> str:
    name = str(getattr(inner, "model_name", "") or type(inner).__name__)
    base_url = getattr(inner, "base_url", None)
    if _is_official(base_url):
        return name
    return f"{name}|{getattr(inner, 'system', '')}@{str(base_url).rstrip('/')}"


class CachingModel(Model):
    """Wrap any pydantic-ai ``Model`` (or model string) so identical requests are
    served from the :class:`MemoryCache`, only cache misses hit the provider."""

    def __init__(self, model: Any, cache: MemoryCache):
        self._spec = model
        self._inner: Any = model if not isinstance(model, str) else None
        self._cache = cache

    def _resolve(self) -> Any:
        if self._inner is None:
            from pydantic_ai.models import infer_model

            self._inner = infer_model(self._spec)
        return self._inner

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._resolve(), name)

    @property
    def model_name(self) -> str:
        return self._inner.model_name if self._inner is not None else str(self._spec)

    @property
    def system(self) -> str | None:  # ty: ignore[incompatible-property-override]
        return self._inner.system if self._inner is not None else None

    @property
    def provider(self) -> Any:
        return self._inner.provider if self._inner is not None else None

    @property
    def base_url(self) -> str | None:
        return self._inner.base_url if self._inner is not None else None

    async def count_tokens(self, messages, model_settings, model_request_parameters):
        return await self._resolve().count_tokens(messages, model_settings, model_request_parameters)

    async def request(self, messages, model_settings, model_request_parameters):
        inner = self._resolve()
        req = {
            "messages": [_to_jsonable(m) for m in messages],
            "settings": _to_jsonable(model_settings) if model_settings is not None else None,
            "params": _to_jsonable(model_request_parameters),
        }
        key_model = _chat_cache_model(inner)
        hit = self._cache.get("chat", key_model, req)
        if hit is not None:
            return _revive(hit)
        resp = await inner.request(messages, model_settings, model_request_parameters)
        self._cache.put("chat", key_model, req, _value_jsonable(resp))
        return resp

    @asynccontextmanager
    async def request_stream(self, *args, **kwargs):
        # Streaming responses are not cached; an async context manager like Model.request_stream.
        async with self._resolve().request_stream(*args, **kwargs) as stream:
            yield stream


def _embed_endpoint(inner: Any) -> str | None:
    """The embedder's API endpoint, from itself or its SDK client (``base_url`` for
    OpenAI-style clients, ``sdk_configuration`` for Mistral); None when unknown."""
    for src in (inner, getattr(inner, "_client", None), getattr(inner, "client", None)):
        if src is None:
            continue
        url = getattr(src, "base_url", None)
        if isinstance(url, str) or type(url).__name__ == "URL":  # str or httpx.URL, never a mock
            return str(url)
        get_details = getattr(getattr(src, "sdk_configuration", None), "get_server_details", None)
        if callable(get_details):
            try:
                details = get_details()
            except Exception:
                continue
            if isinstance(details, tuple) and details and isinstance(details[0], str):
                return details[0]
    return None


def _embed_cache_model(inner: Any, dimensions: int) -> str:
    """Cache-key model string for an embedder: ``<Class>:<dims>``, plus ``:<model>``
    when the embedder runs a model other than its class default, plus ``@<endpoint>``
    when it calls an endpoint other than the provider's own API (so two models or
    endpoints never share entries, while default keys stay as before)."""
    import inspect

    key = f"{type(inner).__name__}:{dimensions}"
    model = getattr(inner, "model", None) or getattr(inner, "_model", None)
    if isinstance(model, str):
        try:
            default = inspect.signature(type(inner)).parameters["model"].default
        except (KeyError, TypeError, ValueError):
            default = inspect.Parameter.empty
        if model != default:
            key = f"{key}:{model}"
    endpoint = _embed_endpoint(inner)
    if not _is_official(endpoint):
        key = f"{key}@{str(endpoint).rstrip('/')}"
    return key


class CachingEmbedder:
    """Wrap an :class:`EmbeddingClient` so per-text embeddings are cached."""

    def __init__(self, inner: EmbeddingClient, cache: MemoryCache):
        self._inner = inner
        self._cache = cache

    @property
    def dimensions(self) -> int:
        return self._inner.dimensions

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._inner, name)

    def embed(self, texts: list[str]) -> list[list[float]]:
        model = _embed_cache_model(self._inner, self.dimensions)
        out: list[list[float] | None] = [None] * len(texts)
        missing: list[int] = []
        for i, t in enumerate(texts):
            hit = self._cache.get("embed", model, {"text": t})
            if hit is not None:
                out[i] = hit
            else:
                missing.append(i)
        if missing:
            vecs = self._inner.embed([texts[i] for i in missing])
            for i, v in zip(missing, vecs, strict=True):
                v = [float(x) for x in v]  # numpy arrays would serialize as strings
                out[i] = v
                self._cache.put("embed", model, {"text": texts[i]}, v)
        return [v for v in out if v is not None]  # type: ignore[misc]


def cached_model(model: Any, cache: MemoryCache) -> CachingModel:
    return CachingModel(model, cache)


def cached_embedder(embedder: EmbeddingClient, cache: MemoryCache) -> CachingEmbedder:
    return CachingEmbedder(embedder, cache)
