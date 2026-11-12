"""Secret handling: resolve API keys from the environment without ever exposing them.

Values are returned as pydantic SecretStr; str()/repr() show '**********'.
Never log or print the underlying value, use the logging filter as backstop.
"""

from __future__ import annotations

import os

from pydantic import SecretStr

from grafeo_memory.common.exceptions import SecretNotFoundError
from grafeo_memory.common.paths import env_file


def _env_value(name: str) -> str | None:
    """Process env wins over the .env file (CI secrets over local file)."""
    v = os.environ.get(name)
    if v:
        return v
    try:
        from dotenv import dotenv_values

        return dotenv_values(env_file()).get(name) or None
    except Exception:
        return None


def get_secret(name: str, *, required: bool = False) -> SecretStr | None:
    """Resolve a secret env var (e.g. TYPESAFE_API_KEY, MISTRAL_API_KEY)."""
    v = _env_value(name)
    if v is None:
        if required:
            raise SecretNotFoundError(f"{name} is not set (env or {env_file()})")
        return None
    return SecretStr(v)
