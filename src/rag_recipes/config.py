"""Application settings loaded from environment and `.env`.

Required fields (`database_url`, `redis_url`, `openai_api_key`) have no
default — instantiating `Settings()` raises a `pydantic.ValidationError`
that names the missing field if they aren't provided. Every other field
has a documented default that mirrors `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from redis.asyncio.connection import parse_url as redis_parse_url


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str
    redis_url: str
    redis_password: str = ""
    openai_api_key: str

    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = False

    personal_api_token: str | None = None
    debug_endpoints_enabled: bool = False

    local_storage_root: str = "./data/storage"

    pdf_window_size_pages: int = 3
    pdf_overlap_pages: int = 1

    search_default_limit: int = 10
    search_keyword_top_k: int = 50
    search_vector_top_k: int = 50
    search_rrf_k: int = 60

    recipe_keyword_boost_title: float = 1.40
    recipe_keyword_boost_ingredients: float = 1.20
    recipe_vector_boost_summary: float = 1.20

    @field_validator("redis_url")
    @classmethod
    def _redis_url_requires_credentials(cls, value: str) -> str:
        # Phase 1.6: reject DSNs without credentials so a half-migrated .env
        # (REDIS_PASSWORD added but REDIS_URL not updated) fails at Settings
        # load instead of silently hitting NOAUTH at runtime. Delegated to
        # redis-py's parser so every scheme the client accepts
        # (redis://, rediss://, unix://) stays valid here too.
        try:
            parsed = redis_parse_url(value)
        except ValueError as exc:
            raise ValueError(f"REDIS_URL is not a valid Redis DSN: {exc}") from exc
        if not parsed.get("password"):
            raise ValueError("REDIS_URL must include credentials, e.g. redis://:pwd@host:port/db")
        return value

    @model_validator(mode="after")
    def _redis_password_matches_url(self) -> Settings:
        # Phase 1.6: catch the mismatch case (REDIS_PASSWORD updated but the
        # password in REDIS_URL drifted, or vice versa) — would otherwise pass
        # Settings load and explode with WRONGPASS at the first arq write.
        # Skipped when REDIS_PASSWORD is unset so envs that authenticate via
        # a fully-credentialed REDIS_URL alone stay supported. parse_url
        # decodes percent-encoded passwords, so the comparison is direct.
        if not self.redis_password:
            return self
        url_password = redis_parse_url(self.redis_url).get("password")
        if url_password != self.redis_password:
            raise ValueError(
                "REDIS_URL password does not match REDIS_PASSWORD; update both in .env"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
