# core/ontology/: Layer D core

SKOS-style concept scheme (`scheme.py`), controlled entity and relation vocabularies with domain/range
validation (`types.py`), the deterministic entity normalization ladder (`normalize.py`: rungs 1 to 3 in pure code,
rung 4 emits categorical candidates), and pre-model temporal rules (`temporal_rules.py`).

Imports `common/` and core base modules only, never `providers/`, `core/extraction/`, `core/reconciliation/`,
`core/search/`, `services/` or entry points. Enforced by `ruff.toml` (absolute imports) and
`tests/test_common/test_architecture.py` (relative imports). See `docs/architecture.md`.
