"""Pre-0.3 module paths still import (as the module at its new home) until 0.7."""

from __future__ import annotations

import importlib
import sys
import warnings

import pytest

import grafeo_memory  # noqa: F401  (installs the aliases)
from grafeo_memory._deprecated import _ALIASES

SUBMODULES = {
    "grafeo_memory.extraction.entities": "grafeo_memory.core.extraction.entities",
    "grafeo_memory.extraction.facts": "grafeo_memory.core.extraction.facts",
    "grafeo_memory.extraction.temporal": "grafeo_memory.core.extraction.temporal",
    "grafeo_memory.reconciliation.memories": "grafeo_memory.core.reconciliation.memories",
    "grafeo_memory.reconciliation.relations": "grafeo_memory.core.reconciliation.relations",
    "grafeo_memory.search.graph": "grafeo_memory.core.search.graph",
    "grafeo_memory.search.vector": "grafeo_memory.core.search.vector",
}


@pytest.mark.parametrize(("old", "new"), sorted({**_ALIASES, **SUBMODULES}.items()))
def test_old_path_is_the_new_module_and_warns(old, new):
    sys.modules.pop(old, None)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        mod = importlib.import_module(old)
    assert mod is importlib.import_module(new)
    assert any(issubclass(w.category, DeprecationWarning) and new in str(w.message) for w in caught)


def test_private_names_and_patching_reach_the_real_module(monkeypatch):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        import grafeo_memory._graph as old_graph
        from grafeo_memory.search.vector import _get_props  # a 0.2.x test import

    from grafeo_memory.core import _graph
    from grafeo_memory.core.search import vector

    assert _get_props is vector._get_props
    monkeypatch.setattr(old_graph, "detach_delete_node", lambda *a, **k: "patched")
    assert _graph.detach_delete_node() == "patched"


def test_unknown_path_still_fails():
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("grafeo_memory.does_not_exist")
