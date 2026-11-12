"""CLI entry point. `grafeo-memory = "grafeo_memory.cli:main"` keeps working, and the
app helpers are re-exported for import. They are copies: patch them in
``grafeo_memory.cli.app``, where ``main`` looks them up."""

from grafeo_memory.cli import app as _app
from grafeo_memory.cli.app import *  # noqa: F403
from grafeo_memory.cli.app import main

globals().update({k: v for k, v in vars(_app).items() if not k.startswith("__")})

__all__ = ["main"]
