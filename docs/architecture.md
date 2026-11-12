# Architecture

`grafeo-memory` is a layered package (flat layout, deriva-style). Layers may
only import layers below them, and the DAG is **enforced, not documented**:
each layer directory ships a `ruff.toml` whose `banned-api` table (rule
`TID251`, selected in `pyproject.toml`) fails lint on imports of higher layers,
and `tests/test_common/test_architecture.py` applies the same ban tables (read
from the `ruff.toml` files, the single source of truth) to every module via the
AST, checking each file against its nearest `ruff.toml` as ruff does. Both run in CI.

```
entry points (import everything)     cli/  mcp/  agent_framework.py
bench/                               harnesses; only loaded explicitly via
                                     `python -m grafeo_memory.bench.*` (ruff-banned elsewhere)
─────────────────────────────────────────────────────────────────────────────
services/                            facades: memory.py = MemoryManager / AsyncMemoryManager
─────────────────────────────────────────────────────────────────────────────
core/                                business logic (modules): extraction/ reconciliation/
                                     search/ ontology/ policy/ + taxonomy.py bucket.py tiers.py temporal.py prompts.py …
                                     (peer bans between subpackages in their own ruff.toml)
─────────────────────────────────────────────────────────────────────────────
providers/                           adapters for external services, import common only:
   llm/          model_registry.py (model-string -> pydantic-ai Model), retry.py
   decision/     Layer C engines: protocol, registry, packs, jev, fake
   embedding/    EmbeddingClient protocol + Mistral/OpenAI embedders
─────────────────────────────────────────────────────────────────────────────
common/                              base layer, stdlib + 3rd-party only:
   logging.py (secret-redaction filter)   settings.py (env/.env config)
   secrets.py (API keys as SecretStr)     telemetry.py (UsageTracker)
   cache.py (in-memory LRU + JSONL)       types.py schemas.py
   intervals.py (ISO/EDTF + Allen)        compat.py paths.py timing.py exceptions.py
```

Layer letters in the code (A to G) name the design layers: A memory tiers, B scoping, C decision engines,
D the ontological and temporal core, E compilation, F the policy cascade, G open decisions. C and D ship in
0.3.0, E as a first cut (`compile()`); the rest are reserved toggles on `PipelineConfig`.

## Package map

| Path | Layer | Contents |
|---|---|---|
| `grafeo_memory/common/` | base | logging, settings, secrets, telemetry, cache, types, schemas, intervals, compat, paths, timing, exceptions |
| `grafeo_memory/providers/llm/` | adapter | model registry, retry policy |
| `grafeo_memory/providers/decision/` | adapter (Layer C) | decision-model protocol/registry/packs + jev, fake |
| `grafeo_memory/providers/embedding/` | adapter | embedders |
| `grafeo_memory/core/` | modules | extraction/, reconciliation/, search/, ontology/, policy/ (Layer F rule packs and shapes), taxonomy.py, bucket.py (Layer G questions), tiers.py (Layer A budgets and triggers), temporal.py, prompts.py, schemas helpers, scoring, reranker, history, communities, vision, _graph, _tracing |
| `grafeo_memory/services/memory.py` | facade | `MemoryManager` / `AsyncMemoryManager` wiring all layers |
| `grafeo_memory/services/` (layer services) | wiring | `bucket.py` (G: park, resolve, expire), `policy.py` (F: rules in the write path, rung counts), `scoping.py` (B: tags, routing, growth, roots), `retrieval.py` (D/E/A read-path knobs), `tiers.py` (A: window, `assemble()`, `end_session()`), `compile.py` (E), `schema.py` (store schema version) |
| `grafeo_memory/cli/` | entry | CLI (`grafeo-memory = grafeo_memory.cli:main`) |
| `grafeo_memory/mcp/` | entry | MCP server |
| `grafeo_memory/bench/` | harnesses | locomo_eval, lme_eval, beam_eval, matrix, common; data/runs under `GRAFEO_MEMORY_BENCH_HOME` (default `.benchmarks/`) |

## Compatibility

The public API is unchanged: `from grafeo_memory import MemoryManager, MemoryConfig, ...` works as before.
Every pre-0.3 module path (`grafeo_memory.manager`, `.types`, `.embedding`, `.history`, `.scoring`,
`.search.vector`, `.extraction.facts`, ...) still imports and resolves to the module at its new home, with a
`DeprecationWarning`; `grafeo_memory/_deprecated.py` holds the map. The old paths are removed in 0.7.

## Rules of the house

- Models propose, standards validate: stores hold only canonical content.
- Deterministic first, model last: normalization ladder, Allen algebra, and
  pre-model temporal rules run in pure code; only residue reaches a model.
- With the cache on, every LLM, embedding and decision call goes through it, so no identical call is made twice.
- See `docs/benchmarks.md` for the evaluation harness and gates.
