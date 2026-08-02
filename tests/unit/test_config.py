import pytest
from pydantic import ValidationError

from rag_recipes.config import Settings


def test_settings_loads_with_required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.database_url == "postgresql+asyncpg://test/test"
    assert settings.openai_api_key == "sk-test"


def test_pdf_min_text_chars_for_page_defaults_to_20(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.delenv("PDF_MIN_TEXT_CHARS_FOR_PAGE", raising=False)
    settings = Settings(_env_file=None)
    assert settings.pdf_min_text_chars_for_page == 20


def test_search_boost_and_weight_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    s = Settings(_env_file=None)
    # Keyword-side boost table (doc 7 § 7).
    assert s.recipe_keyword_boost_title == 1.40
    assert s.recipe_keyword_boost_ingredients == 1.20
    assert s.recipe_keyword_boost_steps == 1.05
    assert s.recipe_keyword_boost_summary == 1.00
    assert s.recipe_keyword_boost_full == 0.95
    # Vector-side boost table.
    assert s.recipe_vector_boost_summary == 1.20
    assert s.recipe_vector_boost_full == 1.10
    assert s.recipe_vector_boost_steps == 1.00
    assert s.recipe_vector_boost_ingredients == 0.95
    assert s.recipe_vector_boost_title == 0.90
    # Source weights, supporting bonus + cap, and the embedding provider name.
    assert s.keyword_source_weight == 1.0
    assert s.vector_source_weight == 1.0
    assert s.search_supporting_chunk_bonus == 0.05
    assert s.search_supporting_chunk_bonus_cap == 0.15
    assert s.embedding_provider == "openai"


def test_settings_fails_when_database_url_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "database_url" in str(excinfo.value).lower()


def test_settings_fails_when_openai_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "openai_api_key" in str(excinfo.value).lower()


@pytest.mark.parametrize("batch_size", ["0", "3000"])
def test_embedding_batch_size_out_of_range_rejected(
    monkeypatch: pytest.MonkeyPatch, batch_size: str
) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", batch_size)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "embedding_batch_size" in str(excinfo.value).lower()


def test_redis_url_without_credentials_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValidationError, match="must include credentials"):
        Settings(_env_file=None)


def test_redis_url_password_mismatch_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:wrong@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValidationError, match="does not match REDIS_PASSWORD"):
        Settings(_env_file=None)


def test_redis_url_credentialed_without_redis_password_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Envs that authenticate via a fully-credentialed REDIS_URL alone (no
    # separate REDIS_PASSWORD) must keep loading — the mismatch check is
    # skipped when REDIS_PASSWORD is unset.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.delenv("REDIS_PASSWORD", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.redis_password == ""


def test_redis_url_encoded_password_matches_decoded_redis_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The redis-py URL parser decodes percent-encoded passwords; the validator
    # must accept e.g. redis://:p%40ss@... against REDIS_PASSWORD=p@ss.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:p%40ss@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "p@ss")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.redis_password == "p@ss"


def test_redis_url_unix_socket_with_credentials_accepted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # redis-py accepts unix:// DSNs with the password in the query string;
    # the validator delegates to that parser so the same forms work here.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "unix:///tmp/redis.sock?password=redis&db=0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.redis_url.startswith("unix://")


def test_redis_url_invalid_scheme_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "http://localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    with pytest.raises(ValidationError, match="not a valid Redis DSN"):
        Settings(_env_file=None)


def _required_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")


def test_worker_settings_have_expected_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    for var in (
        "WORKER_MAX_JOBS",
        "WORKER_JOB_TIMEOUT_SECONDS",
        "WORKER_KEEP_RESULT_SECONDS",
        "WORKER_HEALTH_CHECK_INTERVAL_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.worker_max_jobs == 1
    assert settings.worker_job_timeout_seconds == 600
    assert settings.worker_keep_result_seconds == 60
    assert settings.worker_health_check_interval_seconds == 30


@pytest.mark.parametrize(
    "var",
    [
        "WORKER_MAX_JOBS",
        "WORKER_JOB_TIMEOUT_SECONDS",
        "WORKER_HEALTH_CHECK_INTERVAL_SECONDS",
    ],
)
def test_worker_positive_int_fields_reject_zero(
    monkeypatch: pytest.MonkeyPatch, var: str
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert var.lower() in str(excinfo.value).lower()


def test_worker_keep_result_seconds_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("WORKER_KEEP_RESULT_SECONDS", "-1")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "worker_keep_result_seconds" in str(excinfo.value).lower()


def test_stuck_job_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    for var in ("STUCK_JOB_TIMEOUT_MINUTES", "STUCK_JOB_CHECK_INTERVAL_MINUTES"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.stuck_job_timeout_minutes == 30
    assert settings.stuck_job_check_interval_minutes == 5


@pytest.mark.parametrize("interval", ["7", "11", "13"])
def test_stuck_job_check_interval_rejects_non_divisors(
    monkeypatch: pytest.MonkeyPatch, interval: str
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("STUCK_JOB_CHECK_INTERVAL_MINUTES", interval)
    with pytest.raises(ValidationError, match="divisor of 60"):
        Settings(_env_file=None)


@pytest.mark.parametrize("interval", ["0", "61"])
def test_stuck_job_check_interval_out_of_bounds_rejected(
    monkeypatch: pytest.MonkeyPatch, interval: str
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("STUCK_JOB_CHECK_INTERVAL_MINUTES", interval)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "stuck_job_check_interval_minutes" in str(excinfo.value).lower()


def test_stuck_job_timeout_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("STUCK_JOB_TIMEOUT_MINUTES", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "stuck_job_timeout_minutes" in str(excinfo.value).lower()


def test_extraction_threshold_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    for var in (
        "EXTRACTION_MIN_OVERALL_CONFIDENCE",
        "EXTRACTION_MIN_BOUNDARY_CONFIDENCE",
        "EXTRACTION_MIN_NORMALIZATION_CONFIDENCE",
        "EXTRACTION_MIN_RECIPE_CHARS",
        "EXTRACTION_MAX_RECIPE_CHARS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.extraction_min_overall_confidence == 0.5
    assert settings.extraction_min_boundary_confidence == 0.5
    assert settings.extraction_min_normalization_confidence == 0.5
    assert settings.extraction_min_recipe_chars == 200
    assert settings.extraction_max_recipe_chars == 20000


@pytest.mark.parametrize(
    "var",
    [
        "EXTRACTION_MIN_OVERALL_CONFIDENCE",
        "EXTRACTION_MIN_BOUNDARY_CONFIDENCE",
        "EXTRACTION_MIN_NORMALIZATION_CONFIDENCE",
    ],
)
@pytest.mark.parametrize("value", ["-0.1", "1.1"])
def test_extraction_confidence_thresholds_out_of_range_rejected(
    monkeypatch: pytest.MonkeyPatch, var: str, value: str
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert var.lower() in str(excinfo.value).lower()


def test_extraction_min_recipe_chars_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_MIN_RECIPE_CHARS", "-1")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "extraction_min_recipe_chars" in str(excinfo.value).lower()


def test_extraction_max_recipe_chars_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_MAX_RECIPE_CHARS", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "extraction_max_recipe_chars" in str(excinfo.value).lower()


def test_extraction_max_must_exceed_min(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_MIN_RECIPE_CHARS", "500")
    monkeypatch.setenv("EXTRACTION_MAX_RECIPE_CHARS", "500")
    with pytest.raises(ValidationError, match="extraction_max_recipe_chars"):
        Settings(_env_file=None)


def test_extraction_batch_and_llm_retry_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    for var in (
        "EXTRACTION_COMMIT_BATCH_SIZE",
        "LLM_MAX_RATE_LIMIT_RETRIES",
        "LLM_REQUEST_TIMEOUT_SECONDS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.extraction_commit_batch_size == 5
    assert settings.llm_max_rate_limit_retries == 5
    assert settings.llm_request_timeout_seconds == 60.0


def test_extraction_commit_batch_size_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("EXTRACTION_COMMIT_BATCH_SIZE", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "extraction_commit_batch_size" in str(excinfo.value).lower()


def test_llm_max_rate_limit_retries_allows_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # 0 is a valid "no retries" configuration — TASK-006 treats it as
    # "raise on the first 429", so the bound is ge=0, not ge=1.
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_MAX_RATE_LIMIT_RETRIES", "0")
    settings = Settings(_env_file=None)
    assert settings.llm_max_rate_limit_retries == 0


def test_llm_max_rate_limit_retries_rejects_negative(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_MAX_RATE_LIMIT_RETRIES", "-1")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "llm_max_rate_limit_retries" in str(excinfo.value).lower()


def test_llm_request_timeout_seconds_rejects_below_one(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_REQUEST_TIMEOUT_SECONDS", "0.5")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "llm_request_timeout_seconds" in str(excinfo.value).lower()


def test_answer_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Epic 17 answer-layer defaults (doc 8 § 3). answer_llm_model defaults to None
    # (resolved to llm_model at the dependency boundary, not here).
    _required_env(monkeypatch)
    for var in (
        "ANSWER_LLM_MODEL",
        "ANSWER_PROMPT_VERSION",
        "ANSWER_SCHEMA_VERSION",
        "ANSWER_CONTEXT_ITEM_LIMIT",
        "ANSWER_MATCHED_CHUNKS_PER_ITEM",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.answer_llm_model is None
    assert settings.answer_prompt_version == "answer-recommendation-v1"
    assert settings.answer_schema_version == "answer.v1"
    assert settings.answer_context_item_limit == 6
    assert settings.answer_matched_chunks_per_item == 3


@pytest.mark.parametrize(
    "var",
    ["ANSWER_CONTEXT_ITEM_LIMIT", "ANSWER_MATCHED_CHUNKS_PER_ITEM"],
)
def test_answer_cap_fields_reject_zero(monkeypatch: pytest.MonkeyPatch, var: str) -> None:
    # The two caps are Field(ge=1): a ≤0 value would silently include nearly all
    # results or force an empty context, so it is rejected at Settings load.
    _required_env(monkeypatch)
    monkeypatch.setenv(var, "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert var.lower() in str(excinfo.value).lower()


def test_search_default_limit_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    # ge=1 so the answers route's effective-limit fallback stays positive (Epic 17.2).
    _required_env(monkeypatch)
    monkeypatch.setenv("SEARCH_DEFAULT_LIMIT", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "search_default_limit" in str(excinfo.value).lower()


def test_reranker_settings_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Epic 18 reranker config; off by default (the rerank step is wired in 18.2).
    _required_env(monkeypatch)
    for var in ("RERANKING_ENABLED", "RERANK_PROVIDER", "RERANK_MODEL", "RERANK_TOP_N"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.reranking_enabled is False
    assert settings.rerank_provider == "openai"
    assert settings.rerank_model == "gpt-4.1"
    assert settings.rerank_top_n == 50


@pytest.mark.parametrize("value", ["0", "-1", "201"])
def test_rerank_top_n_out_of_bounds_rejected(
    monkeypatch: pytest.MonkeyPatch, value: str
) -> None:
    # rerank_top_n is Field(ge=1, le=200): ≤0 would silently disable reranking and a
    # huge value would feed an LLM reranker a costly fan-out (Epic 18.1).
    _required_env(monkeypatch)
    monkeypatch.setenv("RERANK_TOP_N", value)
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "rerank_top_n" in str(excinfo.value).lower()


def test_rerank_hot_path_budget_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    # Epic 18.2 hot-path budget: per-candidate text cap + short reranker timeout.
    _required_env(monkeypatch)
    for var in ("RERANK_MAX_CHARS_PER_CANDIDATE", "RERANK_REQUEST_TIMEOUT_SECONDS"):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.rerank_max_chars_per_candidate == 2000
    assert settings.rerank_request_timeout_seconds == 8.0


def test_rerank_max_chars_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("RERANK_MAX_CHARS_PER_CANDIDATE", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "rerank_max_chars_per_candidate" in str(excinfo.value).lower()


def test_rerank_request_timeout_rejects_non_positive(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("RERANK_REQUEST_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "rerank_request_timeout_seconds" in str(excinfo.value).lower()


def test_rerank_provider_must_be_openai_when_enabled(monkeypatch: pytest.MonkeyPatch) -> None:
    # The data-egress backstop: an unsupported rerank_provider is rejected at load
    # when reranking is enabled (Epic 18.2).
    _required_env(monkeypatch)
    monkeypatch.setenv("RERANKING_ENABLED", "true")
    monkeypatch.setenv("RERANK_PROVIDER", "cohere")
    with pytest.raises(ValidationError, match="rerank_provider must be 'openai'"):
        Settings(_env_file=None)


def test_rerank_provider_unsupported_allowed_when_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A disabled reranker is inert, so any rerank_provider value loads fine.
    _required_env(monkeypatch)
    monkeypatch.setenv("RERANKING_ENABLED", "false")
    monkeypatch.setenv("RERANK_PROVIDER", "cohere")
    settings = Settings(_env_file=None)
    assert settings.rerank_provider == "cohere"


# --- llm_provider switch + Anthropic settings (Epic 19.1) -------------------


def test_llm_provider_defaults_to_openai(monkeypatch: pytest.MonkeyPatch) -> None:
    # Unset config must be byte-for-byte today's behaviour: OpenAI by default,
    # and the Anthropic fields carry their documented defaults.
    _required_env(monkeypatch)
    for var in (
        "LLM_PROVIDER",
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_LLM_MODEL",
        "ANTHROPIC_MAX_TOKENS",
    ):
        monkeypatch.delenv(var, raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "openai"
    assert settings.anthropic_api_key is None
    assert settings.anthropic_llm_model == "claude-sonnet-4-6"
    assert settings.anthropic_max_tokens == 8192


def test_llm_provider_anthropic_without_key_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # The cross-field rule: selecting Anthropic without a key fails at load with a
    # message naming the missing field, not at the first runtime call.
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ValidationError, match="anthropic_api_key"):
        Settings(_env_file=None)


def test_llm_provider_anthropic_with_key_constructs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_LLM_MODEL", raising=False)
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "anthropic"
    assert settings.anthropic_api_key == "sk-ant-test"
    assert settings.anthropic_llm_model == "claude-sonnet-4-6"


def test_llm_provider_invalid_value_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    with pytest.raises(ValidationError, match="llm_provider"):
        Settings(_env_file=None)


def test_structured_output_mode_defaults_to_none(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    # None is the "defer to the provider's registry entry" sentinel. A non-None
    # default could not express that, and would let a provider be paired with a
    # mode its transport cannot serve (Epic 23.4).
    assert Settings(_env_file=None).llm_structured_output_mode is None


def test_structured_output_mode_invalid_value_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_MODE", "nonsense")
    with pytest.raises(ValidationError, match="llm_structured_output_mode"):
        Settings(_env_file=None)


@pytest.mark.parametrize("mode", ["json_schema", "strict_tool", "tool"])
def test_structured_output_mode_accepts_every_supported_value(
    monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("LLM_STRUCTURED_OUTPUT_MODE", mode)
    assert Settings(_env_file=None).llm_structured_output_mode == mode


def test_anthropic_max_tokens_rejects_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    _required_env(monkeypatch)
    monkeypatch.setenv("ANTHROPIC_MAX_TOKENS", "0")
    with pytest.raises(ValidationError) as excinfo:
        Settings(_env_file=None)
    assert "anthropic_max_tokens" in str(excinfo.value).lower()
