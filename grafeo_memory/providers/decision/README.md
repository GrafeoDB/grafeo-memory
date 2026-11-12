# providers/decision/: Layer C decision engines

One protocol (`protocol.py`) for typed, calibrated decisions over categorical and score questions
(`types.py`). Providers: `jev.py` (TypeSafe System One API, cached, batched, auto-splitting) and `fake.py`
(offline tests and benchmarks). `packs.py` builds question packs per stage (per-fact or pairwise) and the hybrid
escalation signals. `registry.py` resolves a configured model by name or `grafeo_memory.decision_models` entry
point. The engine enums and `StageConfig` live in `common/types.py`.

Imports `common/` only, never `core/`, `services/` or entry points. Enforced by `ruff.toml` and
`tests/test_common/test_architecture.py`. See `docs/architecture.md`.
