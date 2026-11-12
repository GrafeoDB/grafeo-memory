"""Logging for grafeo-memory: a secret-redaction filter for log handlers.

The filter scrubs the values of known secret env vars (API keys), from the process
environment and from the configured ``.env`` file, out of every log record a handler
emits (message and traceback), so a stray ``log.debug("%s", payload)`` can never
print a key. The library itself adds no handlers; the CLI and the MCP server call
:func:`configure_entry_point_logging`, and applications can call
:func:`install_redaction` on their own handlers.
"""

from __future__ import annotations

import logging
import os

from grafeo_memory.common.paths import env_file

_SECRET_ENV_HINTS = ("API_KEY", "SECRET", "TOKEN", "PASSWORD", "TYPESAFE")
# Shorter values are skipped: a name such as ``MAX_TOKENS=1000`` matches a hint, and
# redacting every "1000" would garble logs; real API keys are far longer.
_MIN_SECRET_LEN = 8
_REDACTED = "***REDACTED***"

_dotenv_cache: tuple[tuple[str, int], list[str]] | None = None


def _is_secret(name: str, value: str | None) -> bool:
    return value is not None and len(value) >= _MIN_SECRET_LEN and any(h in name.upper() for h in _SECRET_ENV_HINTS)


def _dotenv_secrets() -> list[str]:
    """Secret values from the .env file settings and get_secret() read, re-read only
    when the file changes (the filter runs on every record)."""
    global _dotenv_cache
    path = env_file()
    try:
        stamp = (str(path), path.stat().st_mtime_ns)
    except OSError:
        return []
    if _dotenv_cache is not None and _dotenv_cache[0] == stamp:
        return _dotenv_cache[1]
    try:
        from dotenv import dotenv_values

        vals = [v for k, v in dotenv_values(path).items() if v is not None and _is_secret(k, v)]
    except Exception:
        vals = []
    _dotenv_cache = (stamp, vals)
    return vals


def _secret_values() -> list[str]:
    vals = [value for name, value in os.environ.items() if _is_secret(name, value)]
    vals.extend(v for v in _dotenv_secrets() if v not in vals)
    return vals


def _scrub(text: str, secrets: list[str]) -> str:
    for v in secrets:
        if v in text:
            text = text.replace(v, _REDACTED)
    return text


class SecretRedactingFilter(logging.Filter):
    """Replace secret values in the record message and traceback with ***REDACTED***."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage()
        except Exception:  # a malformed format call must not raise out of the caller's log statement
            msg = str(record.msg)
        secrets = _secret_values()
        record.msg = _scrub(msg, secrets)
        record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = _scrub(record.exc_text, secrets)
        if record.stack_info:
            record.stack_info = _scrub(record.stack_info, secrets)
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
    (default WARNING; empty means unset) with secrets redacted. stdout stays free for
    command output and the MCP stdio transport."""
    level = (os.environ.get("GRAFEO_MEMORY_LOG_LEVEL") or "WARNING").upper()
    logging.basicConfig(level=level, format="[%(name)s] %(levelname)s %(message)s")
    install_redaction()


def get_logger(name: str = "grafeo_memory") -> logging.Logger:
    """A grafeo-memory logger; redaction applies through :func:`install_redaction`."""
    return logging.getLogger(name)
