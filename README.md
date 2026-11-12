[![CI](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-memory/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-memory)
[![PyPI](https://img.shields.io/pypi/v/grafeo-memory.svg)](https://pypi.org/project/grafeo-memory/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENCE)

# grafeo-memory

AI memory layer powered by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

No servers, no Docker, no Neo4j, no Qdrant. One `.grafeo` file + one LLM.

```
Typical memory stack: Containers with Neo4j + Qdrant, Embedding API + LLM
grafeo-memory stack:  grafeo (single file) + LLM
```

## Install

```bash
uv add grafeo-memory                   # base (bring your own LLM + embedder)
uv add grafeo-memory[mistral]          # + Mistral embeddings
uv add grafeo-memory[openai]           # + OpenAI embeddings
uv add grafeo-memory[anthropic]        # + Anthropic models
uv add grafeo-memory[mcp,openai]       # + MCP server for AI agents (with the provider extra of its model)
uv add grafeo-memory[jev]              # + TypeSafe Jev decision model (hosted, opt-in)
uv add grafeo-memory[agent-framework]  # + Microsoft Agent Framework vector store
uv add grafeo-memory[bench]            # + LOCOMO / LongMemEval / BEAM harnesses
uv add grafeo-memory[all]              # all LLM providers
```

## Quick Start

### OpenAI

```python
from openai import OpenAI
from grafeo_memory import MemoryManager, MemoryConfig, OpenAIEmbedder

embedder = OpenAIEmbedder(OpenAI())
config = MemoryConfig(db_path="./memory.grafeo", user_id="alice")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Add memories from conversation
    events = memory.add("I just started a new job at Acme Corp as a data scientist")
    # -> [ADD "alice works at acme_corp", ADD "alice is a data_scientist"]

    events = memory.add("I've been promoted to senior data scientist at Acme")
    # -> [UPDATE "alice is a senior data scientist at acme_corp"]

    events = memory.add("I left Acme and joined Beta Inc")
    # -> [DELETE "alice works at acme_corp", ADD "alice works at beta_inc"]

    # Search
    results = memory.search("Where does Alice work?")
    # -> [SearchResult(text="alice works at beta_inc", score=0.92, ...)]
```

### Mistral

```python
from mistralai.client import Mistral
from grafeo_memory import MemoryManager, MemoryConfig, MistralEmbedder

embedder = MistralEmbedder(Mistral())
config = MemoryConfig(db_path="./memory.grafeo", user_id="alice")

with MemoryManager("mistral:mistral-small-latest", config, embedder=embedder) as memory:
    events = memory.add("I just started a new job at Acme Corp as a data scientist")
    results = memory.search("Where does Alice work?")
```

## How It Works

grafeo-memory implements the **reconciliation loop**, the intelligence layer that decides what to remember:

1. **Extract** facts from conversation text (LLM call)
2. **Extract** entities and relationships (LLM tool call)
3. **Search** existing memory for related facts (vector + graph)
4. **Reconcile** new facts against existing memory (LLM decides ADD/UPDATE/DELETE/NONE)
5. **Execute** the decisions against GrafeoDB

```
┌──────────────────────────────────────────┐
│             grafeo-memory                │
│                                          │
│  Extractor -> Reconciler -> Executor     │
│  (LLM)       (LLM)        (GrafeoDB)     │
└──────────────────┬───────────────────────┘
                   │
         ┌─────────┴──────────┐
         │      GrafeoDB      │
         │  Graph + Vector    │
         │  + Text (optional) │
         │ single .grafeo file│
         └────────────────────┘
```

## Documentation

- [docs/architecture.md](docs/architecture.md): the package layers and the import rules between them
- [docs/benchmarks.md](docs/benchmarks.md): LOCOMO, LongMemEval and BEAM harnesses, matrix rounds, test gates

Since 0.3.0 the modules live in layers (`grafeo_memory.common`, `.providers`, `.core`, `.services`).
`from grafeo_memory import ...` is unchanged. Pre-0.3 module paths such as `grafeo_memory.manager` or
`grafeo_memory.search.vector` still import, with a `DeprecationWarning` naming the new path; they are removed in 0.7.

## Multi-User Isolation

```python
config = MemoryConfig(db_path="./chat_memory.grafeo")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Each user's memories are isolated
    memory.add("I love hiking in the mountains", user_id="bob")
    memory.add("I prefer beach vacations", user_id="carol")

    bob_results = memory.search("vacation preferences", user_id="bob")
    # -> hiking, mountains

    carol_results = memory.search("vacation preferences", user_id="carol")
    # -> beach vacations
```

## Persistence

```python
config = MemoryConfig(db_path="./memory.grafeo")  # a database file
config = MemoryConfig()  # in memory, gone when the process ends
```

- A `db_path` is always a single database file, whatever its extension. While it is open, Grafeo keeps its
  write-ahead log in a `memory.grafeo.wal/` directory next to it, and removes it at close; a `memory.grafeo.spill/`
  directory may appear when a query spills to disk. The directory that holds the file must be writable.
- A `db_path` must not be an existing directory: Grafeo 0.6 refuses to open one (an empty one included) unless it is
  a Grafeo 0.5 database (see below). Name a file inside it instead, such as `./memories/memory.grafeo`.
- One process at a time can open a database for writing. Close your app, the CLI or the MCP server before another
  one opens the same path.
- Indexes are (re)created whenever a manager opens a database.

### Upgrading Grafeo (0.5 to 0.6)

grafeo-memory 0.3.0 requires Grafeo 0.6.2 or later (`grafeo>=0.6.2,<0.7`); for Grafeo 0.5, stay on grafeo-memory 0.2.x.
Grafeo 0.6 writes a new database format, and the switch is one way:

1. **Before the first open with 0.6**, stop every process that uses the database (your app, `grafeo-memory`,
   `grafeo-memory-mcp`). The migration needs free disk space for a copy of the database.
2. **The first open with Grafeo 0.6** (any `MemoryManager`, the CLI or the MCP server; they all open the database
   for writing) migrates it, at the same path:
   - a database file written by 0.5.x (`memory.grafeo`) is rewritten in the new format, and the old file is kept
     next to it, unchanged, as `memory.grafeo.pre-0.6` (with `memory.grafeo.pre-0.6.wal/` if it had a
     write-ahead log left);
   - a 0.5.x WAL-directory database (a `db_path` without `.grafeo`, such as `./memory.db/`) becomes a database
     file at that path, and the old directory is kept, unchanged, as `memory.db.pre-0.6/`. Its commits are replayed
     into the file. Grafeo 0.5 did not log index definitions in that directory; grafeo-memory creates its indexes
     again on open.
3. **Later opens** find a 0.6 file: nothing migrates again and the `.pre-0.6` copy is left alone. Keep it until you
   are sure you will not go back.
4. **Grafeo 0.5 cannot open the migrated database.** To go back, stop the 0.6 processes, move the migrated file
   away, rename the `.pre-0.6` file (or directory) to the original name and run grafeo-memory 0.2.x with
   Grafeo 0.5.44. Changes made after the migration are only in the migrated file.
5. Grafeo 0.7 will not read 0.5.x databases, so open each one once with 0.6 before upgrading further.

Your memories, embeddings, entities, relations, communities and history are ordinary graph data, so they move with
the database: grafeo-memory needs no migration step of its own, and vector search works on the migrated database
right away. History written by grafeo-memory 0.2.3 (`:History` nodes) and by 0.1.x (`HAS_HISTORY` edges) carries
over. grafeo-memory 0.2.2 read history from Grafeo's change data capture, which Grafeo 0.5 keeps in memory only, so
a 0.2.2 database holds no history to carry over.

## Memory History

`history(memory_id)` returns every change grafeo-memory made to a memory (ADD, UPDATE, DELETE), oldest first:

```python
mid = memory.add("I like tea", infer=False)[0].memory_id
memory.update(mid, "I like green tea")
memory.delete(mid)

for entry in memory.history(mid):
    print(entry.event, entry.old_text, entry.new_text, entry.timestamp)
# ADD None I like tea ...
# UPDATE I like tea I like green tea ...
# DELETE I like green tea None ...
```

- History is stored as `:History` nodes in the same database, keyed by the memory id and not connected by edges,
  so it survives closing and reopening the database and a `delete()` of the memory, and it does not change graph
  algorithm scores.
- grafeo-memory does not need Grafeo's change data capture (CDC is off by default and kept in memory only). A
  database you pass in with `db=` keeps its CDC setting.
- Only changes made through grafeo-memory are history; writes made directly with the Grafeo API are not.
- grafeo-memory 0.2.0 to 0.2.2 read history from CDC and never stored it, so memories written by those versions
  have no entries from before the upgrade.
- `delete()` and `delete_all()` keep the history as an audit trail. Entries carry the `user_id`, so with the
  database at hand (pass it in with `db=`) you can remove a user's history as well:

```python
import grafeo

db = grafeo.GrafeoDB("./memory.grafeo")
memory = MemoryManager("openai:gpt-4o-mini", config, embedder=embedder, db=db)

memory.delete_all(user_id="alice")
db.execute("MATCH (h:History) WHERE h.user_id = $uid DELETE h", {"uid": "alice"})
```

## Supported LLM Providers

grafeo-memory uses [pydantic-ai](https://ai.pydantic.dev) model strings, so any provider pydantic-ai supports works out of the box:

```python
# OpenAI: use OpenAIEmbedder for embeddings
MemoryManager("openai:gpt-4o-mini", config, embedder=OpenAIEmbedder(OpenAI()))

# Anthropic: pair with OpenAI or custom embedder
MemoryManager("anthropic:claude-sonnet-4-5-20250929", config, embedder=embedder)

# Groq: pair with OpenAI or custom embedder
MemoryManager("groq:llama-3.3-70b-versatile", config, embedder=embedder)

# Mistral: use MistralEmbedder for embeddings
MemoryManager("mistral:mistral-small-latest", config, embedder=MistralEmbedder(Mistral()))

# Google: pair with OpenAI or custom embedder
MemoryManager("google-gla:gemini-2.0-flash", config, embedder=embedder)
```

### Built-in Embedders

| Class | Provider | Default Model | Install Extra |
|---|---|---|---|
| `OpenAIEmbedder` | OpenAI | `text-embedding-3-small` | `[openai]` |
| `MistralEmbedder` | Mistral | `mistral-embed` | `[mistral]` |

Both accept an optional `model` parameter to override the default.

### Which model works best?

| Provider | Combined extraction | Individual extraction | Notes |
| --- | --- | --- | --- |
| OpenAI (gpt-4o-mini) | Works | Works | Recommended default |
| Anthropic (claude-sonnet) | Works | Works | Untested in CI |
| Mistral (mistral-small) | Falls back | Works | 3 API calls instead of 1 |
| Groq | Unknown | Unknown | Needs testing |
| Google | Unknown | Unknown | Needs testing |

grafeo-memory tries a single "combined" LLM call first (facts + entities + relations in one schema). If the model cannot handle the combined schema, it falls back to separate calls automatically. Mistral models typically trigger this fallback, which means 3 API calls per `add()` instead of 1. The fallback is silent (logged at DEBUG level, no traceback printed).

## Custom Embeddings

Implement the `EmbeddingClient` protocol to use any embedding provider:

```python
from grafeo_memory import EmbeddingClient


class MyEmbedder:
    def embed(self, texts: list[str]) -> list[list[float]]:
        # Call your embedding API
        return [...]

    @property
    def dimensions(self) -> int:
        return 1024  # your model's output dimensions


memory = MemoryManager("openai:gpt-4o-mini", config, embedder=MyEmbedder())
```

## Decision Engines and Pipeline Toggles

Every judgment in the write path can run on the LLM (the default), on a decision model that answers typed
questions with a calibrated confidence, or on both: `hybrid` asks the decision model first and sends only the
facts it is unsure of to one LLM call. Extraction always uses the LLM.

```python
from grafeo_memory import Engine, MemoryConfig, PipelineConfig, StageConfig

config = MemoryConfig(
    db_path="./memory.grafeo",
    decision_model="jev",  # or "fake" (offline), or a provider registered under "grafeo_memory.decision_models"
    pipeline=PipelineConfig(
        memories=StageConfig(engine=Engine.HYBRID, confidence_threshold=0.75),
        relations=StageConfig(engine=Engine.HYBRID),
        temporal_precheck=True,  # a fact whose stated validity overlaps no candidate is added without a model call
        ontology=True,  # entity names resolve to existing entities; relations that break domain/range are not written
    ),
)
```

| Stage | Engines | What it decides |
| --- | --- | --- |
| `memories` | `llm`, `decision`, `hybrid` | ADD / UPDATE / DELETE / NONE for each new fact |
| `relations` | `llm`, `decision`, `hybrid` | Which existing relations a new relation invalidates |
| `search` | `llm` (no decision rerank), `decision`, `hybrid` | Rerank score blended by `rerank_blend_weight`; `hybrid` keeps the retrieval score where the model is unsure |
| `facts`, `entities`, `temporal` | `llm` only | Extraction (generates text) |

Custom providers implement `DecisionModel.decide_async(state, questions)` and register with
`register_decision_model(name, factory)`; `FakeDecisionModel` is a deterministic stand-in for tests.

**Data leaving the device.** The default pipeline sends text only to the LLM and embedder you configure. With
`decision_model="jev"` (extra `jev`, key in `TYPESAFE_API_KEY`), the reconciliation state (the new facts and the
candidate memories they are compared with) is also sent to TypeSafe's hosted API. `temporal_precheck`, `ontology`
and `compile()` run locally and send nothing.

All `PipelineConfig` toggles default to off, which is the 0.2 pipeline:

| Toggle | Effect |
| --- | --- |
| `temporal_precheck` | Pure-code interval check before reconciliation |
| `ontology` | Deterministic entity normalization and relation-type checks before graph writes |
| `bitemporal`, `episodes` | Same as `MemoryConfig.enable_bitemporal` / `enable_episodes` |
| `compile` | Read by the benchmark harnesses only; applications call `memory.compile()` |
| `taxonomy`, `policy`, `bucket` | Switch on the scoping, policy and open-decisions layers below (same as their `enabled` flags) |

`memory.compile()` (also `await amemory.acompile()`) is an offline pass to run after write batches: entities of the
same type and scope whose names embed nearly identically (cosine >= 0.92) merge into the older node, which keeps
their edges, and relations that break the controlled vocabulary are reported. With the layers below on, it also
proposes near-merges, grows the subject taxonomy, applies retention and re-judges open decisions.

### Memory layers

Each layer has its own config object on `MemoryConfig`, and every one is off by default.

| Config | What it adds |
| --- | --- |
| `bucket` (`BucketConfig`) | A decision-model answer below `floor` is parked, not forced: the fact is kept as `provisional` and an open decision records the question. `list_open_decisions()` and `resolve_open_decision(id, answer)` let a person answer; a later write that touches the evidence, or `compile()`, re-judges it (`resolve_engine`); stale entries expire to an explicit unknown. |
| `policy` (`PolicyConfig`) | Rule packs that decide before any model: `compliance` (credentials and card numbers are never stored), `temporal` (the pre-check), `entrenchment` (certified and user-stated memories are not overwritten by inferred decisions), `lifecycle` (re-stated facts certify a memory, retention TTLs). Relation domain/range checks run as shapes with a severity per shape. `stats().layers` reports which rung decided (rule, decision model, LLM) and the deterministic share. |
| `taxonomy` (`TaxonomyConfig`) | Subject scopes: ten designed roots (career, health, ...) tag every new memory by keyword; prominent entities act as instance scopes. Searches route to the scopes a query names and boost (`mode="soft"`) or keep only (`"filter"`) those memories; `hard_roots` wall a root off. `search(subjects=..., scopes=...)` and `add(scopes=...)` pin scopes; `scopes()` and `set_root_scheme()` show and edit the tree; `compile()` grows generated children. |
| `retrieval` (`RetrievalConfig`) | `order="deterministic"`: memories of the entities a query names (and their relation neighbours) are boosted and pulled in before vector ranking; entities on more than `hub_share` (0.3) of a user's memories, such as the speakers, are skipped. `temporal_filter`: a query that names a time favors memories whose interval overlaps it. `certified_boost` ranks certified memories higher. |
| `ontology` (`OntologyConfig`) | Write-time embedding match of new entity names (`write_embedding_rung`), one question to the decision model for the ambiguous rest (`rung4`), and UPDATE closing the old fact's interval at the new fact's start (`interval_closing`). |
| `compilation` (`CompileConfig`) | Merge and proposal thresholds, automatic compile every N adds, and the observations that certify a memory. |
| `tiers` (`TiersConfig`) | `working`: a per-session window and rolling summary; `assemble(query, session_id, budget)` returns a `ContextBlock` (session, instructions, facts, events, each line with its provenance) and `end_session()` promotes the session. `procedural`: `add_instruction()` stores certified instructions, `auto_triggers` picks up "always ...", "no, use ..." turns, and procedural memories rank higher. `episodic`: recency-weighted ranking. |

```python
from grafeo_memory import BucketConfig, MemoryConfig, PolicyConfig, RetrievalConfig, TaxonomyConfig

config = MemoryConfig(
    decision_model="jev",
    bucket=BucketConfig(enabled=True, floor=0.5),
    policy=PolicyConfig(enabled=True),
    taxonomy=TaxonomyConfig(enabled=True, mode="soft"),
    retrieval=RetrievalConfig(order="deterministic"),
)
```

The rules, the scoping and the retrieval order run locally. Re-judging an open decision sends its question and
evidence to the LLM (or to the decision model with `resolve_engine="decision"`); `rung4` and
`taxonomy.router_engine="decision"` send entity names or fact text to the decision model. The benchmark harnesses
set any of these fields with `--set`, for example `--set bucket.enabled=true --set bucket.floor=0.4`.

### LLM call cache

`MemoryConfig(cache=True)` (or `CacheConfig(...)`; for the CLI and MCP server, `GRAFEO_MEMORY_CACHE_ENABLED=true`) caches LLM, embedding
and decision-model calls in memory and, for a file database, in `<db_path>.llmcache.jsonl`, so an identical call is
answered without reaching the provider. The cache file holds prompts and responses, which include memory text;
delete it together with the database, or call `MemoryCache.clear()`.

## MCP Server

grafeo-memory includes a built-in MCP server so AI agents (Claude Desktop, Cursor, etc.) can use it as a tool.

```bash
uv add grafeo-memory[mcp,openai]    # default model openai:gpt-4o-mini
uv add grafeo-memory[mcp,mistral]   # GRAFEO_MEMORY_MODEL=mistral:...
```

The server needs the provider extra of its model too: OpenAI embeddings by default, Mistral embeddings for a
`mistral:` model.

### Claude Desktop

Add to `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "grafeo-memory": {
      "command": "grafeo-memory-mcp",
      "env": {
        "GRAFEO_MEMORY_MODEL": "openai:gpt-4o-mini",
        "GRAFEO_MEMORY_DB": "./memory.grafeo"
      }
    }
  }
}
```

### Available Tools

| Tool | Description |
| ---- | ----------- |
| `memory_add` | Add a memory by extracting facts from text |
| `memory_add_batch` | Add multiple memories in one batch |
| `memory_search` | Search memories by semantic similarity and graph context |
| `memory_update` | Update an existing memory's text |
| `memory_delete` | Delete a single memory |
| `memory_delete_all` | Delete all memories for a user |
| `memory_list` | List all stored memories |
| `memory_summarize` | Consolidate old memories into topic-grouped summaries |
| `memory_history` | Show change history for a memory |
| `memory_open_decisions` | List questions the memory parked because it was unsure |
| `memory_resolve_decision` | Answer a parked question |
| `memory_manage_roots` | Show or edit the subject roots that scope memories |
| `memory_assemble` | Build a prompt-ready context block with provenance |
| `memory_add_instruction` | Store a standing instruction, certified |

### Environment Variables

| Variable | Default | Description |
| -------- | ------- | ----------- |
| `GRAFEO_MEMORY_MODEL` | `openai:gpt-4o-mini` | pydantic-ai model string |
| `GRAFEO_MEMORY_DB` | *(in-memory)* | Database file path |
| `GRAFEO_MEMORY_USER` | `default` | Default user ID |
| `GRAFEO_MEMORY_YOLO` | *(off)* | Set to `1` for all features |

### Transport

Supports stdio (default), SSE and streamable HTTP:

```bash
grafeo-memory-mcp              # stdio (default)
grafeo-memory-mcp sse          # SSE
grafeo-memory-mcp streamable-http
```

> **Note:** This is different from [grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp), which exposes the raw GrafeoDB database. grafeo-memory-mcp wraps the high-level memory API (extract, reconcile, search, summarize).

## Observability

grafeo-memory supports OpenTelemetry instrumentation via pydantic-ai. When enabled, all LLM calls (extraction, reconciliation, summarization, reranking) are traced automatically.

```python
config = MemoryConfig(instrument=True)  # uses global OTel provider
```

For custom providers:

```python
from grafeo_memory import InstrumentationSettings

config = MemoryConfig(
    instrument=InstrumentationSettings(
        tracer_provider=my_tracer_provider,
        include_content=False,
    )
)
```

## Why grafeo-memory?

| | Traditional stack | grafeo-memory |
|---|---|---|
| Infrastructure | Neo4j + Qdrant (Docker) | **Single .grafeo file** |
| Install size | ~750MB (Docker images) | **~16MB** (uv add) |
| Offline/edge | Requires servers | **Yes** |
| Graph + vector | Separate services | **Unified engine** |
| LLM providers | Varies | **pydantic-ai** (OpenAI, Anthropic, Mistral, Groq, Google) |
| Embeddings | External API required | **Protocol-based** (any provider) |

## API Reference

### `MemoryManager`

- `MemoryManager(model, config=None, *, embedder)`: create memory manager. `model` is a pydantic-ai model string (e.g. `"openai:gpt-4o-mini"`)
- `.add(messages, user_id=None, session_id=None, metadata=None, *, infer=True, importance=1.0, memory_type="semantic")` → `AddResult` (list of `MemoryEvent`)
- `.search(query, user_id=None, k=10, *, filters=None, rerank=True, memory_type=None)` → `SearchResponse` (list of `SearchResult`)
- `.update(memory_id, text)` → `MemoryEvent`: update a memory's text directly. Raises `ValueError` when `memory_id` is not a memory of this manager
- `.get_all(user_id=None, memory_type=None)` → `list[SearchResult]`
- `.delete(memory_id)` → `bool`: delete a memory with its edges (links to entities, session chain, provenance). `False` when `memory_id` is not a memory of this manager; other nodes in a shared database are never touched
- `.delete_all(user_id=None)` → `int` (count deleted, expired memories included)
- `.summarize(user_id=None, *, preserve_recent=5, batch_size=20)` → `AddResult`
- `.history(memory_id)` → `list[HistoryEntry]`: returns change history sorted by timestamp (oldest first), also for deleted memories (see [Memory History](#memory-history))
- `.set_importance(memory_id, importance)` → `bool`
- `.close()`: close the database

Use as a context manager: `with MemoryManager(...) as memory:`. Multiple sessions in the same process are supported.

### `MemoryConfig`

- `db_path`: path to the database, preferably ending in `.grafeo` (None for in-memory, see [Persistence](#persistence))
- `user_id`: default user scope (default `"default"`)
- `session_id`: default session scope
- `agent_id`: default agent scope
- `reconciliation_threshold`: minimum cosine similarity (0.0 to 1.0) for a memory to be considered a reconciliation candidate during `add()`. Lower values find more candidates, higher values require closer matches. At 0.3 (default), loosely related memories are considered. At 0.7+, only very similar memories trigger UPDATE/DELETE decisions
- `search_min_score`: minimum score for search results, 0.0 returns everything (default 0.0)
- `agreement_bonus`: score boost when both vector and graph find the same memory (default 0.1)
- `embedding_dimensions`: vector dimensions (default 1536)
- `enable_importance`: enable composite scoring with recency/frequency/importance (default False)
- `weight_topology`: topology score weight for graph-connected memories (default 0.0, requires `enable_importance`)
- `enable_topology_boost`: re-rank search results by graph connectivity, no LLM call (default False)
- `topology_boost_factor`: strength of topology boost (default 0.2)
- `consolidation_protect_threshold`: protect well-connected memories from summarize (default 0.0, off)
- `instrument`: OpenTelemetry instrumentation, `True` or `InstrumentationSettings` (default False)
- `decision_model`: decision model for non-LLM pipeline stages, a name or a `DecisionModel` (default None)
- `pipeline`: `PipelineConfig`, the engine per stage and the layer toggles (default: LLM everywhere, toggles off)
- `cache`: `True` or `CacheConfig` to cache LLM, embedding and decision calls (default False)
- `history_retention`: grafeo history retention for time travel: `"off"`, `"forever"`, a `timedelta` or seconds (default None, grafeo's default)
- `db_memory_limit_bytes`, `db_spill_limit_bytes`: grafeo memory and spill caps (default None, grafeo's machine default)

### `EmbeddingClient` (Protocol)

- `.embed(texts: list[str]) -> list[list[float]]`: generate embeddings for a batch of texts
- `.dimensions -> int`: return the embedding vector dimensionality

### Return Types

- **`AddResult`**: list subclass of `MemoryEvent`, with `.usage` for LLM token counts
- **`SearchResponse`**: list subclass of `SearchResult`, with `.usage` for LLM token counts
- **`MemoryEvent`**: `.action` (ADD/UPDATE/DELETE/NONE), `.memory_id`, `.text`, `.old_text`
- **`SearchResult`**: `.memory_id`, `.text`, `.score`, `.user_id`, `.metadata`, `.relations`, `.memory_type`
- **`HistoryEntry`**: `.event` (ADD/UPDATE/DELETE), `.old_text`, `.new_text`, `.timestamp` (epoch ms), `.actor_id`, `.role`. Returned sorted by timestamp ascending (oldest first)

### Iteration

```python
# AddResult is iterable:
for event in memory.add("text"):
    print(event.action, event.text)

# SearchResponse is iterable:
for result in memory.search("query"):
    print(result.text, result.score)
```

## Ecosystem

grafeo-memory is part of the GrafeoDB ecosystem:

- **[grafeo](https://github.com/GrafeoDB/grafeo)**: Core graph database engine (Rust)
- **[grafeo-langchain](https://github.com/GrafeoDB/grafeo-langchain)**: LangChain integration
- **[grafeo-llamaindex](https://github.com/GrafeoDB/grafeo-llamaindex)**: LlamaIndex integration
- **[grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp)**: MCP server for raw GrafeoDB access
- **grafeo-memory-mcp** (built-in): MCP server for the memory API (`uv add grafeo-memory[mcp,openai]`)

All packages share the same database file. Build memories with grafeo-memory, query them with grafeo-langchain, expose them via MCP.

## Requirements

- Python 3.12+
- [grafeo](https://pypi.org/project/grafeo/) `>=0.6.2,<0.7` (installed automatically)

## License

Apache-2.0
