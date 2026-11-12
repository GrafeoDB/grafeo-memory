"""Benchmark harnesses (LOCOMO / LongMemEval / BEAM + matrix rounds).

Lives inside the package but is loaded only by explicit CLI invocation
(`python -m grafeo_memory.bench.matrix ...`); every other layer's ruff.toml
bans importing it. Heavy deps (mistralai, nltk, pyarrow) come from the
`bench` extra. Run/data dirs live under GRAFEO_MEMORY_BENCH_HOME.
"""
