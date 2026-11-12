"""Logging for grafeo-memory: a secret-redaction filter for log handlers.

The filter scrubs the values of known secret env vars (API keys) from every log
record a handler emits, so a stray ``log.debug("%s", payload)`` can never print a
key. The library itself adds no handlers; the CLI and the MCP server call
:func:`configure_entry_point_logging`, and applications can call
:func:`install_redaction` on their own handlers.
"""

from __future__ import annotations

import logging
import os

_SECRET_ENV_HINTS = ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "TYPESAFE")


def _secret_values() -> list[str]:
    vals = []
    for name, value in os.environ.items():
        if value and len(value) >= 8 and any(h in name.upper() for h in _SECRET_ENV_HINTS):
            vals.append(value)
    return vals


class SecretRedactingFilter(logging.Filter):
    """Replace secret values in the record message with ***REDACTED***."""

    def filter(self, record: logging.LogRecord) -> bool:
        msg = record.getMessage()
        for v in _secret_values():
            if v in msg:
                msg = msg.replace(v, "***REDACTED***")
        record.msg = msg
        record.args = ()
        return True


def install_redaction(logger: logging.Logger | None = None) -> None:
    """Attach the redaction filter to every handler of ``logger`` (default: the root
    logger). A handler filter sees records from every child logger, which a filter on
    the ``grafeo_memory`` logger itself would not."""
    target = logger or logging.getLogger()
    for handler in target.handlers:
        if not any(isinstance(f, SecretRedactingFilter) for f in handler.filters):
            handler.addFilter(SecretRedactingFilter())


def configure_entry_point_logging() -> None:
    """For the CLI and the MCP server: log to stderr at ``GRAFEO_MEMORY_LOG_LEVEL``
    (default WARNING) with secrets redacted. stdout stays free for command output and
    the MCP stdio transport."""
    level = os.environ.get("GRAFEO_MEMORY_LOG_LEVEL", "WARNING").upper()
    logging.basicConfig(level=level, format="[%(name)s] %(levelname)s %(message)s")
    install_redaction()


def get_logger(name: str = "grafeo_memory") -> logging.Logger:
    """A grafeo-memory logger; redaction applies through :func:`install_redaction`."""
    return logging.getLogger(name)
