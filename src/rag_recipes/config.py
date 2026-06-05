"""Application settings loaded from environment and `.env`.

Required fields (`database_url`, `redis_url`, `openai_api_key`) have no
default — instantiating `Settings()` raises a `pydantic.ValidationError`
that names the missing field if they aren't provided. Every other field
has a documented default that mirrors `.env.example`.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field, field_validator, model_validator
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
    # Phase 9.5: bound the provider's rate-limit retry loop and per-request
    # timeout. retries=0 disables retries (raise on the first 429); the timeout
    # is a float so it feeds chat.completions.create(timeout=…) without a cast.
    llm_max_rate_limit_retries: int = Field(default=5, ge=0)
    llm_request_timeout_seconds: float = Field(default=60.0, ge=1)
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = Field(default=100, ge=1, le=2048)

    langfuse_host: str = ""
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_enabled: bool = False

    personal_api_token: str | None = None
    debug_endpoints_enabled: bool = False

    local_storage_root: str = "./data/storage"

    pdf_window_size_pages: int = 3
    pdf_overlap_pages: int = 1
    pdf_min_text_chars_for_page: int = 20

    # Identity of the production PDF text extractor (method + build), stamped onto
    # every SourceSpan a run writes so the `auto` reprocess selector (Epic 11.2) can
    # detect whether the extractor changed since a version was produced. Two PyMuPDF
    # builds both stamp extraction_method="embedded_text" but should differ here if
    # the build changed.
    pdf_text_extractor: str = "pymupdf:embedded_text"

    # Phase 9.5: windows per per-batch commit in the extraction loop. A crash
    # rolls back the in-flight batch, so at most batch_size − 1 windows of
    # OpenAI spend are repeated on resume (DECISIONS #4).
    extraction_commit_batch_size: int = Field(default=5, ge=1)

    search_default_limit: int = 10
    search_keyword_top_k: int = 50
    search_vector_top_k: int = 50
    search_rrf_k: int = 60

    recipe_keyword_boost_title: float = 1.40
    recipe_keyword_boost_ingredients: float = 1.20
    recipe_vector_boost_summary: float = 1.20

    worker_max_jobs: int = Field(default=1, ge=1)
    worker_job_timeout_seconds: int = Field(default=600, ge=1)
    worker_keep_result_seconds: int = Field(default=60, ge=0)
    worker_health_check_interval_seconds: int = Field(default=30, ge=1)

    stuck_job_timeout_minutes: int = Field(default=30, ge=1)
    stuck_job_check_interval_minutes: int = Field(default=5, ge=1, le=60)

    # Soft-validation thresholds (Epic 9 Phase 9.3, doc 4 § Soft validation).
    # Review heuristics, not calibrated truth — a candidate below a confidence
    # floor or outside the char band is persisted as needs_review, not dropped.
    extraction_min_overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_boundary_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_normalization_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    extraction_min_recipe_chars: int = Field(default=200, ge=0)
    extraction_max_recipe_chars: int = Field(default=20000, ge=1)

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

    @model_validator(mode="after")
    def _extraction_recipe_char_band_is_ordered(self) -> Settings:
        # The soft-validation "too short" / "too long" rules require a real band:
        # an inverted or collapsed range (max <= min) would make every recipe
        # both too short and too long. Reject at Settings load so the operator
        # fixes the env rather than getting nonsensical needs_review flags.
        if self.extraction_max_recipe_chars <= self.extraction_min_recipe_chars:
            raise ValueError(
                "extraction_max_recipe_chars must be greater than "
                f"extraction_min_recipe_chars; got "
                f"max={self.extraction_max_recipe_chars}, "
                f"min={self.extraction_min_recipe_chars}"
            )
        return self

    @model_validator(mode="after")
    def _stuck_check_interval_divides_60(self) -> Settings:
        # arq's cron(..., minute={...}) encodes "every N minutes" only when N
        # divides 60 — otherwise the schedule skews at the top of each hour
        # (e.g. N=7 maps to {0,7,14,21,28,35,42,49,56} with a 4-minute gap
        # 56→00). Reject non-divisors at Settings load so the operator picks
        # from the documented valid set rather than discovering skew via a
        # missed sweep tick.
        value = self.stuck_job_check_interval_minutes
        if 60 % value != 0:
            raise ValueError(
                "stuck_job_check_interval_minutes must be a divisor of 60 "
                "(valid: 1, 2, 3, 4, 5, 6, 10, 12, 15, 20, 30, 60); "
                f"got {value}"
            )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
