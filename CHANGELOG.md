# Changelog

All notable changes to grafeo-memory are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.2] - 2026-02-27

Performance and quality release: fewer LLM calls per operation, smarter memory extraction, and topology-aware scoring.

### Added

- **Combined extraction**: fact + entity + relation extraction now runs in a single LLM call instead of two sequential calls, saving ~1 LLM call per `add()`. New `ExtractionOutput` schema and `COMBINED_EXTRACTION_SYSTEM` / `COMBINED_PROCEDURAL_EXTRACTION_SYSTEM` prompts
- **Topology-aware scoring** (opt-in): graph-connectivity score based on entity sharing between memories. Enable with `weight_topology > 0` in `MemoryConfig`. Inspired by VimRAG
- **Structural decay modulation** (opt-in): foundational memories (those reinforced by newer related memories) resist temporal decay. Enable with `enable_structural_decay=True` and tune `structural_feedback_gamma`. Inspired by VimRAG Eq. 7
- **New `MemoryConfig` options**: `weight_topology` (default 0.0), `enable_structural_decay` (default False), `structural_feedback_gamma` (default 0.3)
- 20 new topology scoring tests (`test_topology_scoring.py`)
- 2 new extraction coverage tests (combined extraction error path, vector_search embedding fallback)

### Improved

- **1 fewer embedding call per search**: query embedding is now computed once in `_search()` and shared across both `hybrid_search()` and `graph_search()`, via new `query_embedding` parameter on both functions
- **Fact grouping prompt**: extraction prompts now instruct the LLM to group closely related details into a single fact (e.g., "marcus plays guitar, is learning jazz, and focuses on Wes Montgomery's style" instead of three separate facts), producing fewer but richer memories
- **Reconciliation temporal reasoning**: reconciliation prompt now includes explicit guidance for temporal updates ("now works at X" → UPDATE), state changes ("car is fixed" → UPDATE "car is broken"), and accumulative facts ("also likes sushi" → ADD alongside "likes pizza")
- **Type annotations**: `run_sync()` now uses generic `[T]` syntax with proper `Coroutine` typing instead of `object -> object`
- **Windows safety net**: `_ProactorBasePipeTransport.__del__` monkey-patch uses `contextlib.suppress(RuntimeError)` instead of bare try/except

### Fixed

- **Search deadlock on Windows**: `_search()` no longer triggers nested `run_sync()` calls. Entity extraction for graph search is now performed asynchronously within the already-running event loop, then passed to `graph_search()` via the new `_entities` parameter. This fixes the `RuntimeError: Event loop is closed` / hang when calling `search()` on Windows with Python 3.13+
- **`graph_search()` nested `run_sync()`**: accepts pre-extracted `_entities` to avoid calling `extract_entities()` (which internally calls `run_sync()`) from within an async context

### Changed

- `extract_async()` now makes 1 LLM call (combined) instead of 2 (facts → entities). The standalone `extract_facts_async()` and `extract_entities_async()` functions remain unchanged for independent use (e.g., search query entity extraction)
- `graph_search()` signature: added `_entities` and `query_embedding` keyword-only parameters (backward compatible, both default to `None`)
- `vector_search()` and `hybrid_search()` signatures: added `query_embedding` keyword-only parameter (backward compatible, defaults to `None`)
- `compute_composite_score()` signature: added `topology` and `reinforcement` keyword-only parameters (backward compatible, both default to 0.0)
- Removed local `grafeo` path dependency from `pyproject.toml` (`[tool.uv.sources]` section)
- Configured `ty` checker: added `extra-paths = ["tests"]` and downgraded rules that produce false positives from Rust-extension deps

## [0.1.1] - 2026-02-12

### Fixed

- CI configuration and failing tests
- Type checking fixes for `ty`

### Changed

- Documentation pass on README
- Lock file updates

## [0.1.0] - 2026-02-12

Initial release.

### Added

- **`MemoryManager`** (sync) and **`AsyncMemoryManager`** (async): full memory CRUD with `add()`, `search()`, `update()`, `delete()`, `get_all()`, `history()`
- **LLM-driven extraction pipeline**: fact extraction, entity/relation extraction via pydantic-ai structured output
- **LLM-driven reconciliation**: ADD / UPDATE / DELETE / NONE decisions for new facts against existing memories, plus relation reconciliation for graph edges
- **Hybrid search**: BM25 + vector similarity with RRF fusion, falling back to vector-only when hybrid is unavailable
- **Graph search**: entity extraction from queries, graph traversal via HAS_ENTITY edges, cosine similarity scoring
- **Importance scoring** (opt-in): composite scoring with configurable weights for similarity, recency, frequency, and importance
- **Memory summarization**: LLM-driven consolidation of old memories into fewer, richer entries
- **Procedural memory**: separate memory type for instructions, preferences, and behavioral rules with dedicated extraction prompts
- **Vision / multimodal** (opt-in): describe-first approach for image inputs via LLM vision
- **Actor tracking**: optional `actor_id` and `role` on messages for multi-agent scenarios
- **Scoping**: `user_id`, `agent_id`, `run_id` for multi-tenant memory isolation
- **Usage tracking** (opt-in): per-step LLM usage callbacks via `usage_callback`
- **CLI**: `grafeo-memory add`, `search`, `list`, `update`, `delete`, `history`, `summarize` with JSON output mode
- **Graph-native history**: HAS_HISTORY edges tracking all memory mutations with actor and timestamp
- **Windows async compatibility**: persistent `asyncio.Runner` and `ProactorEventLoop` safety net for Python 3.13+
- 230 tests, 83% coverage

[0.1.2]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.1...v0.1.2
[0.1.1]: https://github.com/GrafeoDB/grafeo-memory/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/GrafeoDB/grafeo-memory/releases/tag/v0.1.0
