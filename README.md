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
uv add grafeo-memory[anthropic]        # + Anthropic embeddings
uv add grafeo-memory[mcp]             # + MCP server for AI agents
uv add grafeo-memory[all]              # all providers
```

Or with pip:

```bash
pip install grafeo-memory[openai]
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
from mistralai import Mistral
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
config = MemoryConfig(db_path="./memory.grafeo")  # single-file database
config = MemoryConfig()                           # in memory, gone when the process ends
```

- A `db_path` ending in `.grafeo` is a single-file database: everything is in that one file (Grafeo adds a
  `memory.grafeo.wal/` directory while it is open, and may keep a `memory.grafeo.spill/` directory next to it).
  This is the format Grafeo recommends from 0.6 on.
- Any other path, such as `./memory.db`, creates a WAL-directory database: a directory instead of a file. It still
  works, but Grafeo 0.6 deprecates it in favor of the single-file format, so use a `.grafeo` path for new databases.
- One process at a time can open a database for writing. Close your app, the CLI or the MCP server before another
  one opens the same path.
- Indexes are (re)created whenever a manager opens a database.

### Upgrading Grafeo (0.5 to 0.6)

grafeo-memory 0.2.3 requires `grafeo>=0.5.44,<0.6`, so it stays on Grafeo 0.5; Grafeo 0.6 support comes with
grafeo-memory 0.3.0. Grafeo 0.6.0 writes a
new file format, and the switch is one way:

1. **Before the first open with 0.6**, stop every process that uses the database (your app, `grafeo-memory`,
   `grafeo-memory-mcp`) and make a backup of the database file if you may want to go back.
2. **The first read-write open with Grafeo 0.6** (any `MemoryManager`, the CLI or the MCP server) migrates a
   `.grafeo` file written by 0.5.x. The old files are kept next to it, unchanged, as `memory.grafeo.pre-0.6` (and
   `memory.grafeo.pre-0.6.wal/`). The migration needs free disk space for a copy of the database.
3. **After that, Grafeo 0.5.x can no longer open the file.** To go back, follow
   [Upgrading from 0.5](https://grafeo.dev/user-guide/persistence/persistent/#upgrading-from-05) in the Grafeo
   docs.
4. Grafeo 0.7 will not read 0.5.x files, so open each database once with 0.6 before upgrading further.

For a WAL-directory database (a `db_path` without `.grafeo`), check the Grafeo 0.6 release notes for how it is
carried over.

Your memories, entities, relations and history are ordinary graph data, so they move with the file: grafeo-memory
needs no migration step of its own.

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
# OpenAI — use OpenAIEmbedder for embeddings
MemoryManager("openai:gpt-4o-mini", config, embedder=OpenAIEmbedder(OpenAI()))

# Anthropic — pair with OpenAI or custom embedder
MemoryManager("anthropic:claude-sonnet-4-5-20250929", config, embedder=embedder)

# Groq — pair with OpenAI or custom embedder
MemoryManager("groq:llama-3.3-70b-versatile", config, embedder=embedder)

# Mistral — use MistralEmbedder for embeddings
MemoryManager("mistral:mistral-small-latest", config, embedder=MistralEmbedder(Mistral()))

# Google — pair with OpenAI or custom embedder
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

## MCP Server

grafeo-memory includes a built-in MCP server so AI agents (Claude Desktop, Cursor, etc.) can use it as a tool.

```bash
uv add grafeo-memory[mcp]
# or: pip install grafeo-memory[mcp]
```

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

config = MemoryConfig(instrument=InstrumentationSettings(
    tracer_provider=my_tracer_provider,
    include_content=False,
))
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
- **grafeo-memory-mcp** (built-in): MCP server for the memory API (`uv add grafeo-memory[mcp]` or `pip install grafeo-memory[mcp]`)

All packages share the same database file. Build memories with grafeo-memory, query them with grafeo-langchain, expose them via MCP.

## Requirements

- Python 3.12+
- [grafeo](https://pypi.org/project/grafeo/) `>=0.5.44,<0.6` (installed automatically)

## License

Apache-2.0
