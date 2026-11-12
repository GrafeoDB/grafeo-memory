"""Architecture gate: enforce the layer DAG on ALL import styles.

Each layer directory ships a ruff.toml whose banned-api table fails lint on
absolute imports of higher layers. This test additionally resolves relative
imports via the AST and applies the same ban tables, read from the ruff.toml
files themselves, so there is a single source of truth.

Layer DAG (docs/architecture.md):
    common/ < providers/ < core/ < services/ < cli/, mcp/, agent_framework.py
    bench/ imports services/core/providers/common; only cli loads bench.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent.parent
PKG = REPO / "grafeo_memory"

# every directory that must carry a ruff.toml with a ban table
LAYER_DIRS = [
    "common",
    "providers",
    "providers/decision",
    "core",
    "core/extraction",
    "core/reconciliation",
    "core/search",
    "core/ontology",
    "services",
    "bench",
    ".",  # package root (__init__, shims, agent_framework)
]

ENTRY_POINTS = {"grafeo_memory.cli", "grafeo_memory.mcp", "grafeo_memory.agent_framework"}


def _banned_for(layer: str) -> set[str]:
    cfg = tomllib.loads((PKG / layer / "ruff.toml").read_text())
    return set(cfg["lint"]["flake8-tidy-imports"]["banned-api"].keys())


def _module_of(path: Path) -> str:
    rel = path.relative_to(REPO).with_suffix("")
    return ".".join(rel.parts)


def _resolve_relative(level: int, module: str | None, importer_module: str) -> str:
    parts = importer_module.split(".")
    pkg = parts[:-1]  # the package containing the module
    base = pkg[: len(pkg) - (level - 1)]
    return ".".join([*base, module]) if module else ".".join(base)


@pytest.mark.parametrize("layer", LAYER_DIRS)
def test_layer_has_ruff_toml(layer: str):
    assert (PKG / layer / "ruff.toml").exists(), f"grafeo_memory/{layer}/ruff.toml missing"
    assert _banned_for(layer), f"grafeo_memory/{layer}/ruff.toml has an empty ban table"


@pytest.mark.parametrize("layer", LAYER_DIRS)
def test_layer_imports_respect_dag(layer: str):
    banned = _banned_for(layer)
    violations: list[str] = []
    for py in sorted((PKG / layer).glob("*.py")):
        if py.name == "ruff.toml":
            continue
        tree = ast.parse(py.read_text())
        importer = _module_of(py)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    target = alias.name
                    if any(target == b or target.startswith(b + ".") for b in banned):
                        violations.append(f"{py.name}: import {target}")
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_relative(node.level, node.module, importer) if node.level > 0 else node.module or ""
                if any(target == b or target.startswith(b + ".") for b in banned):
                    mod = "." * node.level + (node.module or "")
                    violations.append(f"{py.name}: from {mod} import ... -> {target}")
    assert not violations, f"layer {layer} violates the DAG:\n" + "\n".join(violations)


def test_base_layer_bans_every_internal_layer():
    banned = _banned_for("common")
    for top in ("providers", "core", "services", "bench"):
        assert f"grafeo_memory.{top}" in banned
    assert banned >= ENTRY_POINTS


def test_bench_is_banned_everywhere_but_cli():
    for layer in LAYER_DIRS:
        if layer in (".", "bench"):
            continue
        assert "grafeo_memory.bench" in _banned_for(layer), f"{layer} must ban bench"
