"""Process-level settings for grafeo-memory.

A single pydantic-settings ``BaseSettings`` with prefix ``GRAFEO_MEMORY_`` and
``.env`` support (the mature step up from ad-hoc ``os.environ`` reads):

- Non-secret tuning: ``GRAFEO_MEMORY_CACHE_ENABLED``, ``GRAFEO_MEMORY_CACHE_MAX_MEMORY_MB``,
  ``GRAFEO_MEMORY_DB_MEMORY_LIMIT_MB``... (see fields below). MB in env, bytes
  in the Python API.
- Secrets stay in ``.env``/env and are read as ``SecretStr``, never logged,
  never serialized into cache keys or traces.

Precedence elsewhere in the codebase stays: CLI flag > env var > .env > default.
"""

from __future__ import annotations

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from grafeo_memory.common.types import CacheConfig, MemoryConfig

MB = 1024 * 1024


class GrafeoMemorySettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="GRAFEO_MEMORY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_ignore_empty=True,  # `GRAFEO_MEMORY_YOLO=` in an MCP env block means unset, as in 0.2.x
    )

    # --- existing process-level knobs (unchanged names) ---
    model: str = "openai:gpt-4o-mini"
    db: str | None = None
    user: str = "default"
    yolo: bool = False

    # --- GRAFEO_MEMORY_CACHE_* ---
    cache_enabled: bool = False
    cache_max_entries: int = 10_000
    cache_max_memory_mb: int = 128
    cache_max_storage_mb: int = 512
    cache_persist_path: str | None = None

    # --- GRAFEO_MEMORY_DB_* engine limits (MB in env; None = grafeo auto default) ---
    db_memory_limit_mb: int | None = None
    db_spill_limit_mb: int | None = None

    # --- decision models (pipeline engines; per-stage config stays Python-side) ---
    decision_model: str | None = None  # registered name, e.g. "jev"; None = LLM-only
    jev_model: str = "jev-latest"

    # --- provider secrets (unprefixed, .env friendly, never logged) ---
    openai_api_key: SecretStr | None = Field(default=None, validation_alias="OPENAI_API_KEY")
    anthropic_api_key: SecretStr | None = Field(default=None, validation_alias="ANTHROPIC_API_KEY")
    google_api_key: SecretStr | None = Field(default=None, validation_alias="GOOGLE_API_KEY")
    mistral_api_key: SecretStr | None = Field(default=None, validation_alias="MISTRAL_API_KEY")
    groq_api_key: SecretStr | None = Field(default=None, validation_alias="GROQ_API_KEY")
    cohere_api_key: SecretStr | None = Field(default=None, validation_alias="CO_API_KEY")
    xai_api_key: SecretStr | None = Field(default=None, validation_alias="XAI_API_KEY")
    deepseek_api_key: SecretStr | None = Field(default=None, validation_alias="DEEPSEEK_API_KEY")
    fireworks_api_key: SecretStr | None = Field(default=None, validation_alias="FIREWORKS_API_KEY")
    together_api_key: SecretStr | None = Field(default=None, validation_alias="TOGETHER_API_KEY")
    cerebras_api_key: SecretStr | None = Field(default=None, validation_alias="CEREBRAS_API_KEY")
    openrouter_api_key: SecretStr | None = Field(default=None, validation_alias="OPENROUTER_API_KEY")
    typesafe_api_key: SecretStr | None = Field(default=None, validation_alias="TYPESAFE_API_KEY")

    def cache_config(self) -> CacheConfig | None:
        """CacheConfig when the cache is enabled, else None."""
        if not self.cache_enabled:
            return None
        return CacheConfig(
            enabled=True,
            max_entries=self.cache_max_entries,
            max_memory_bytes=self.cache_max_memory_mb * MB,
            max_storage_bytes=self.cache_max_storage_mb * MB,
            persist_path=self.cache_persist_path,
        )

    def apply_to(self, config: MemoryConfig) -> MemoryConfig:
        """Copy process-level cache/limit settings into a MemoryConfig (explicit
        fields passed by the caller win)."""
        cc = self.cache_config()
        if cc is not None and config.cache is False:
            config.cache = cc
        if self.db_memory_limit_mb is not None and config.db_memory_limit_bytes is None:
            config.db_memory_limit_bytes = self.db_memory_limit_mb * MB
        if self.db_spill_limit_mb is not None and config.db_spill_limit_bytes is None:
            config.db_spill_limit_bytes = self.db_spill_limit_mb * MB
        if self.decision_model is not None and config.decision_model is None:
            config.decision_model = self.decision_model
        return config
