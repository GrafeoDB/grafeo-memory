"""Fixtures for benchmark-gate tests.

Benchmark outputs must never land in the checkout during tests: point
GRAFEO_MEMORY_BENCH_HOME at a tmp dir (grafeo_memory.common.paths honors it;
until then the offline gates below only read code and existing run artifacts).
Harnesses run as modules: python -m grafeo_memory.bench.<name>.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
PY = sys.executable


@pytest.fixture(autouse=True)
def _bench_home_isolated(monkeypatch, tmp_path):
    """Isolate benchmark outputs into a temporary home (deriva's DERIVA_HOME pattern)."""
    monkeypatch.setenv("GRAFEO_MEMORY_BENCH_HOME", str(tmp_path / "bench_home"))
    yield


@pytest.fixture
def repo() -> Path:
    return REPO


@pytest.fixture
def live() -> bool:
    """Live (API-calling) smoke runs only when explicitly enabled."""
    return os.environ.get("GRAFEO_MEMORY_BENCH_LIVE") == "1"
