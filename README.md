[![CI](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml/badge.svg)](https://github.com/GrafeoDB/grafeo-memory/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/GrafeoDB/grafeo-memory/graph/badge.svg)](https://codecov.io/gh/GrafeoDB/grafeo-memory)
[![PyPI](https://img.shields.io/pypi/v/grafeo-memory.svg)](https://pypi.org/project/grafeo-memory/)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENCE)

# grafeo-memory

AI memory layer powered by [GrafeoDB](https://github.com/GrafeoDB/grafeo), an embedded graph database with native vector search.

No servers, no Docker, no Neo4j, no Qdrant. One `.db` file + one LLM.

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
config = MemoryConfig(db_path="./memory.db", user_id="alice")

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
config = MemoryConfig(db_path="./memory.db", user_id="alice")

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
         │  single .db file   │
         └────────────────────┘
```

## Multi-User Isolation

```python
config = MemoryConfig(db_path="./chat_memory.db")

with MemoryManager("openai:gpt-4o-mini", config, embedder=embedder) as memory:
    # Each user's memories are isolated
    memory.add("I love hiking in the mountains", user_id="bob")
    memory.add("I prefer beach vacations", user_id="carol")

    bob_results = memory.search("vacation preferences", user_id="bob")
    # -> hiking, mountains

    carol_results = memory.search("vacation preferences", user_id="carol")
    # -> beach vacations
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

## Why grafeo-memory?

| | Traditional stack | grafeo-memory |
|---|---|---|
| Infrastructure | Neo4j + Qdrant (Docker) | **Single .db file** |
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
- `.update(memory_id, text)` → `MemoryEvent`: update a memory's text directly
- `.get_all(user_id=None, memory_type=None)` → `list[SearchResult]`
- `.delete(memory_id)` → `bool`
- `.delete_all(user_id=None)` → `int` (count deleted)
- `.summarize(user_id=None, *, preserve_recent=5, batch_size=20)` → `AddResult`
- `.history(memory_id)` → `list[HistoryEntry]`
- `.set_importance(memory_id, importance)` → `bool`
- `.close()`: close the database

Use as a context manager: `with MemoryManager(...) as memory:`. Multiple sessions in the same process are supported.

### `MemoryConfig`

- `db_path`: path to database file (None for in-memory)
- `user_id`: default user scope (default `"default"`)
- `session_id`: default session scope
- `agent_id`: default agent scope
- `similarity_threshold`: max embedding distance for reconciliation (default 0.7)
- `embedding_dimensions`: vector dimensions (default 1536)
- `enable_importance`: enable composite scoring with recency/frequency/importance (default False)
- `weight_topology`: topology score weight for graph-connected memories (default 0.0, requires `enable_importance`)
- `enable_topology_boost`: re-rank search results by graph connectivity, no LLM call (default False)
- `topology_boost_factor`: strength of topology boost (default 0.2)

### `EmbeddingClient` (Protocol)

- `.embed(texts: list[str]) -> list[list[float]]`: generate embeddings for a batch of texts
- `.dimensions -> int`: return the embedding vector dimensionality

### Return Types

- **`AddResult`**: list subclass of `MemoryEvent`, with `.usage` for LLM token counts
- **`SearchResponse`**: list subclass of `SearchResult`, with `.usage` for LLM token counts
- **`MemoryEvent`**: `.action` (ADD/UPDATE/DELETE/NONE), `.memory_id`, `.text`, `.old_text`
- **`SearchResult`**: `.memory_id`, `.text`, `.score`, `.user_id`, `.metadata`, `.relations`, `.memory_type`
- **`HistoryEntry`**: `.event`, `.old_text`, `.new_text`, `.timestamp`, `.actor_id`, `.role`

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
- **[grafeo-mcp](https://github.com/GrafeoDB/grafeo-mcp)**: MCP server for AI agents

All packages share the same `.db` file. Build memories with grafeo-memory, query them with grafeo-langchain, expose them via grafeo-mcp.

## Requirements

- Python 3.12+

## License

Apache-2.0
