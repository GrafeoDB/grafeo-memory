# core/reconciliation/: ADD / UPDATE / DELETE decisions

`memories.py` and `relations.py` decide over extracted facts and relations; `temporal.py` holds the deterministic
interval extraction, the pre-model conflict partition and interval-closing on supersede. Each stage runs on the
LLM, a decision model, or hybrid with escalation (`providers/decision/packs.py`); relation candidates are
pre-filtered by endpoint overlap.

May import `common/`, `providers/` and other core modules, never `core/search/`, `services/` or entry points.
Enforced by `ruff.toml` and `tests/test_common/test_architecture.py`. See `docs/architecture.md`.
