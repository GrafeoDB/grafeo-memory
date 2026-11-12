# common/: base layer

Standard library and third-party imports only; every other layer builds on this (enforced by `ruff.toml`).

- `logging.py`: structured logging with a secret-redaction filter
- `settings.py`: env and `.env` configuration (`GRAFEO_MEMORY_*` prefix)
- `secrets.py`: API keys as `SecretStr` (never logged or printed)
- `telemetry.py`: `UsageTracker`, a sink for the manager's `usage_callback`
- `cache.py`: in-memory LRU and JSONL persistent call cache
- `types.py`, `schemas.py`: core dataclasses and pydantic output schemas
- `intervals.py`: ISO 8601 / EDTF intervals and Allen's 13 relations (pure code)
- `compat.py`: sync/async runner; `paths.py`: repo, bench home and env paths; `timing.py`: retry and backoff;
  `exceptions.py`: error hierarchy
