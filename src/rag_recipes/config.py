"""Application settings loaded from environment and `.env`.

Required fields (`database_url`, `redis_url`, `openai_api_key`) have no
default — instantiating `Settings()` raises a `pydantic.ValidationError`
that names the missing field if they aren't provided. Every other field
has a documented default that mirrors `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    database_url: str
    redis_url: str
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


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
