# bench/: benchmark harnesses (LOCOMO, LongMemEval, BEAM and the config matrix)

Run as modules, for example `python -m grafeo_memory.bench.matrix --round 4 --convs all`. Requires the `bench`
extra. Data and runs go under `GRAFEO_MEMORY_BENCH_HOME` (default `<repo>/.benchmarks/`). Every LLM and
decision-model call is cached in a shared JSONL file, so reruns cost nothing. No other package module imports
this one (ruff-enforced). See `docs/benchmarks.md`.
