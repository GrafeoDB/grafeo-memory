"""Architecture gate: enforce the layer DAG on ALL import styles.

Each layer directory ships a ruff.toml whose banned-api table (TID251) fails
lint on imports of higher layers. This test applies the same ban tables, read
from the ruff.toml files themselves, to every module via the AST: each file is
checked against its nearest ruff.toml, the same lookup ruff uses.

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
    "core/policy",
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


def _is_banned(target: str, banned: set[str]) -> bool:
    return any(target == b or target.startswith(b + ".") for b in banned)


def _layer_of(py: Path) -> str:
    """The directory of the nearest ruff.toml, relative to the package (ruff's own lookup)."""
    for parent in py.parents:
        if (parent / "ruff.toml").exists():
            return parent.relative_to(PKG).as_posix() if parent != PKG else "."
        if parent == PKG:
            break
    raise AssertionError(f"{py} has no ruff.toml above it")


def _modules_of(layer: str) -> list[Path]:
    return [py for py in sorted(PKG.rglob("*.py")) if _layer_of(py) == layer]


def test_every_module_is_in_a_listed_layer():
    unlisted = {_layer_of(py) for py in PKG.rglob("*.py")} - set(LAYER_DIRS)
    assert not unlisted, f"ruff.toml layers missing from LAYER_DIRS: {sorted(unlisted)}"


@pytest.mark.parametrize("layer", LAYER_DIRS)
def test_layer_imports_respect_dag(layer: str):
    banned = _banned_for(layer)
    violations: list[str] = []
    for py in _modules_of(layer):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        importer = _module_of(py)
        name = py.relative_to(PKG).as_posix()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if _is_banned(alias.name, banned):
                        violations.append(f"{name}: import {alias.name}")
            elif isinstance(node, ast.ImportFrom):
                target = _resolve_relative(node.level, node.module, importer) if node.level > 0 else node.module or ""
                mod = "." * node.level + (node.module or "")
                if _is_banned(target, banned):
                    violations.append(f"{name}: from {mod} import ... -> {target}")
                    continue
                # `from grafeo_memory import services` imports a banned subpackage as a member
                for alias in node.names:
                    if _is_banned(f"{target}.{alias.name}", banned):
                        violations.append(f"{name}: from {mod} import {alias.name} -> {target}.{alias.name}")
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
