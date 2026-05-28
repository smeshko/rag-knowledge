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
