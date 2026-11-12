# Benchmarks

The harnesses live in the package at `grafeo_memory/bench/` (installed with the
`bench` extra: `uv add 'grafeo-memory[bench]'`). Run them as modules from a checkout:

| Harness | Benchmark | Command |
|---|---|---|
| `grafeo_memory.bench.locomo_eval` | LOCOMO (10 conversations) | `python -m grafeo_memory.bench.locomo_eval all --run <name> --convs all` |
| `grafeo_memory.bench.matrix` | LOCOMO config matrix, rounds 0 to 20 | `python -m grafeo_memory.bench.matrix --round 4 --convs all --extra '--workers 16'` |
| `grafeo_memory.bench.lme_eval` | LongMemEval (500 questions) | `python -m grafeo_memory.bench.lme_eval all --run lme_r0 --cache-path .benchmarks/lme_runs/llm_cache.jsonl` |
| `grafeo_memory.bench.beam_eval` | BEAM 100K (20 conversations) | `python -m grafeo_memory.bench.beam_eval all --run beam_r0` |

Each suite shares one JSONL call cache: no LLM or decision-model call is made
twice, so reruns, retries and config cells replay from the cache at no cost.

Useful flags: `--decision-model jev|fake`, `--memories-engine llm|decision|hybrid`,
`--relations-engine`, `--search-engine`, `--granularity per_fact|pairwise`,
`--escalation confidence|prob_gap|combined`, `--conf-threshold`, `--workers N`, `--cache-path`, and the layer
toggles `--temporal-precheck`, `--ontology`, `--bitemporal`, `--compile`, `--taxonomy`, `--policy`, `--bucket`, plus
`--set KEY=VALUE` for any `MemoryConfig` field by dotted path (`--set retrieval.order=deterministic`). `locomo_eval` also takes
`--judge f1|llm` and `--limit` (smoke subsets); `matrix` takes `--dry-run`, `--report`, `--only <cells>`.

## Paths and isolation

- Code: `grafeo_memory/bench/` (tracked).
- Data and run outputs: `GRAFEO_MEMORY_BENCH_HOME` (default `<repo>/.benchmarks/`),
  resolved by `grafeo_memory.common.paths`. Datasets (`locomo_data/`, `lme_data/`,
  `beam_data/`) and run dirs (`*_runs/`) live there and are git-ignored; a
  redirected bench home still finds the big datasets in the checkout.
- The bench tests set `GRAFEO_MEMORY_BENCH_HOME` to a tmp dir, so test runs never touch the checkout.

## Rounds (matrix)

| Round | Varies |
| --- | --- |
| 0 | Baseline, LLM everywhere |
| 1, 2, 3 | Decision granularity, escalation signal, confidence threshold |
| 4 | Headline on all 10 conversations: `llm`, `decision`, `hybrid` |
| 5, 6, 7 | Temporal pre-check, ontology normalization, both plus a threshold sweep |
| 8, 9 | `compile()`, full-stack headline |
| 10 | Temporal pre-check crossed with the engine |
| 11 | Threshold and granularity on `hybrid` at full size |
| 12 | Retrieval knobs (`k`, diversity, RRF constant, rerank depth, search depth) |
| 13 | Bitemporal storage off and on |

Results aggregate into `<bench_home>/locomo_runs/matrix_round<N>_results.json`;
`--report` re-aggregates without rerunning.

## 0.3.0 against 0.2.2

LOCOMO conversations 0 to 2 (385 questions, categories 1 to 4), Mistral Small 4 for extraction, answers and judge,
the same answer and judge prompts and `k=100`; each version ingests with its own pipeline and the same extraction
instructions.

| Version | Judge score |
| --- | --- |
| 0.2.2 (grafeo 0.5.44) | 0.8727 |
| 0.3.0, default pipeline | 0.9013 |

On all 10 conversations (1,540 questions) 0.3.0's default pipeline scores 0.8753. 0.2.2's locked `mistralai` 1.12.1
is refused by the Mistral API (401), so it ran with mistralai 3.7 and pydantic-ai 2.66. Its search returns a
session id on 17% of results; with the sessions filled in from its database it scores 0.8701, so the gap is not
the missing dates. Three conversations are a small sample: gaps under about 10 questions are noise.

## Results so far

Full LOCOMO (10 conversations, 1,540 questions, categories 1 to 4), Mistral Small 4 for ingest, answers and judge.
Cells in one round differ in one setting; rounds judged by a different model are not comparable. These
rounds ran before the 0.3.0 review changes to the temporal pre-check (stated validity only), the relation
candidate ranking and hybrid escalation, so the rows with those settings are re-measured for 0.3.0.

| Round | Cell | Judge score |
| --- | --- | --- |
| 10 | `llm` | 0.8760 |
| 10 | `llm` + temporal pre-check | 0.8786 |
| 10 | `hybrid` | 0.8805 |
| 10 | `hybrid` + temporal pre-check | 0.8831 |
| 12 | `hybrid`, `k=100` | 0.8929 |
| 12 | `hybrid`, diverse / RRF 30 / rerank 300 / depth 2 | 0.8825 / 0.8810 / 0.8760 / 0.8740 |
| 13 | bitemporal off / on | 0.8903 / 0.8753 |
| 8 | `compile()` off / on | 0.8766 / 0.8753 |

As a pure ablation the earlier temporal pre-check added 0.26 points with 20% fewer LLM calls. `hybrid` uses a 0.75
confidence threshold (0.8831, against 0.8721 at 0.6).

## Datasets

The harnesses read datasets from `<bench_home>/<suite>_data/` (git-ignored); LOCOMO and BEAM are downloaded on
first use, LongMemEval must be placed there. The datasets keep their own licenses:

| Suite | Source | License |
| --- | --- | --- |
| LOCOMO | [snap-research/locomo](https://github.com/snap-research/locomo) | see the repository |
| LongMemEval (cleaned) | [xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned) | MIT |
| BEAM | [Mohammadta/BEAM](https://huggingface.co/datasets/Mohammadta/BEAM) | CC BY-SA 4.0 |

The LOCOMO judge uses Mem0's published answer and judge prompts and LongMemEval its own; both prompt files sit next
to the datasets, outside the package.

## Gates

`tests/test_bench/` runs in CI at no cost: structural checks on the matrix (dry-run cells, report schema), cache
replay and secret-freeness, and the architecture DAG (`tests/test_common/test_architecture.py`). Set
`GRAFEO_MEMORY_BENCH_LIVE=1` (with API keys) to also run the fake-engine smoke cell end to end.

## Targets

LOCOMO judge >= 0.93 (comparable with Mem0's protocol), then LongMemEval (500 questions) and BEAM 100K
(20 conversations) headline runs for the winning configuration.
