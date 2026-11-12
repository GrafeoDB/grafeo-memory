"""Pre-0.3 import paths, kept working until 0.7.

0.3.0 moved every module into layers (``common/``, ``providers/``, ``core/``, ``services/``). An import of an old
path such as ``grafeo_memory.search.vector`` or ``grafeo_memory.manager`` resolves to the module at its new home
(the same module object, so private names and ``mock.patch`` targets keep working) and emits a
``DeprecationWarning`` naming the new path.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import sys
import warnings

# old path -> new path; submodules of an aliased package follow (``extraction.facts`` -> ``core.extraction.facts``)
_ALIASES: dict[str, str] = {
    "grafeo_memory.manager": "grafeo_memory.services.memory",
    "grafeo_memory.types": "grafeo_memory.common.types",
    "grafeo_memory.schemas": "grafeo_memory.common.schemas",
    "grafeo_memory.settings": "grafeo_memory.common.settings",
    "grafeo_memory.cache": "grafeo_memory.common.cache",
    "grafeo_memory.intervals": "grafeo_memory.common.intervals",
    "grafeo_memory._compat": "grafeo_memory.common.compat",
    "grafeo_memory.embedding": "grafeo_memory.providers.embedding",
    "grafeo_memory._graph": "grafeo_memory.core._graph",
    "grafeo_memory._tracing": "grafeo_memory.core._tracing",
    "grafeo_memory.temporal": "grafeo_memory.core.temporal",
    "grafeo_memory.history": "grafeo_memory.core.history",
    "grafeo_memory.messages": "grafeo_memory.core.messages",
    "grafeo_memory.prompts": "grafeo_memory.core.prompts",
    "grafeo_memory.protocol": "grafeo_memory.core.protocol",
    "grafeo_memory.reranker": "grafeo_memory.core.reranker",
    "grafeo_memory.scoring": "grafeo_memory.core.scoring",
    "grafeo_memory.vision": "grafeo_memory.core.vision",
    "grafeo_memory.communities": "grafeo_memory.core.communities",
    "grafeo_memory.extraction": "grafeo_memory.core.extraction",
    "grafeo_memory.reconciliation": "grafeo_memory.core.reconciliation",
    "grafeo_memory.search": "grafeo_memory.core.search",
}


def _target(fullname: str) -> str | None:
    for old, new in _ALIASES.items():
        if fullname == old or fullname.startswith(old + "."):
            return new + fullname[len(old) :]
    return None


class _AliasLoader(importlib.abc.Loader):
    def __init__(self, target: str) -> None:
        self._target = target

    def create_module(self, spec: importlib.machinery.ModuleSpec):
        warnings.warn(
            f"{spec.name} moved to {self._target}; the old path is removed in grafeo-memory 0.7",
            DeprecationWarning,
            stacklevel=2,
        )
        return importlib.import_module(self._target)

    def exec_module(self, module) -> None:
        """The target module is already executed; nothing to do."""


class _AliasFinder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("grafeo_memory."):
            return None
        new = _target(fullname)
        if new is None:
            return None
        return importlib.machinery.ModuleSpec(fullname, _AliasLoader(new))


def install() -> None:
    """Put the alias finder first, so the path finder never loads a second copy of an aliased submodule."""
    if not any(isinstance(f, _AliasFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _AliasFinder())


install()
