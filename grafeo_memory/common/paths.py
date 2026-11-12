"""Paths: repo root, .env file, and the benchmark home (GRAFEO_MEMORY_BENCH_HOME).

The bench home defaults to <repo>/.benchmarks so existing run dirs and caches
stay valid; the bench tests set the env var to a tmp dir for isolation.
"""

from __future__ import annotations

import os
from pathlib import Path


def repo_root() -> Path:
    """The checkout root when running from a source checkout (the package's parent
    holds ``pyproject.toml``); the current working directory when installed, so a
    wheel never reads ``.env`` from or writes ``.benchmarks/`` into site-packages."""
    checkout = Path(__file__).resolve().parent.parent.parent
    if (checkout / "pyproject.toml").is_file() and (checkout / "grafeo_memory").is_dir():
        return checkout
    return Path.cwd()


def env_file() -> Path:
    """The .env this process reads: GRAFEO_MEMORY_ENV_FILE, else <repo>/.env."""
    custom = os.environ.get("GRAFEO_MEMORY_ENV_FILE")
    return Path(custom) if custom else repo_root() / ".env"


def bench_home() -> Path:
    """Where benchmark harnesses write runs/data: $GRAFEO_MEMORY_BENCH_HOME or <repo>/.benchmarks."""
    custom = os.environ.get("GRAFEO_MEMORY_BENCH_HOME")
    return Path(custom) if custom else repo_root() / ".benchmarks"


def suite_runs(suite: str) -> Path:
    """Run dir for a benchmark suite, created on demand: <bench_home>/<suite>_runs."""
    p = bench_home() / f"{suite}_runs"
    p.mkdir(parents=True, exist_ok=True)
    return p


def data_dir(name: str) -> Path:
    """Dataset dir: <bench_home>/<name> if it exists there, else the checkout's .benchmarks/<name>.

    Big downloaded datasets stay in the checkout; a redirected bench home never
    re-downloads them.
    """
    home = bench_home() / name
    if home.exists():
        return home
    return repo_root() / ".benchmarks" / name


def default_cache(suite: str) -> Path:
    """Default shared JSONL call cache for a suite."""
    return suite_runs(suite) / "llm_cache.jsonl"
