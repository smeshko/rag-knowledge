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
