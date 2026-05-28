"""Unit tests for `rag_recipes.ingestion.queue`."""

from __future__ import annotations

import pytest

from rag_recipes.config import Settings
from rag_recipes.ingestion.queue import _build_redis_settings


def _make_settings(monkeypatch: pytest.MonkeyPatch, redis_url: str, password: str) -> Settings:
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", redis_url)
    monkeypatch.setenv("REDIS_PASSWORD", password)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    return Settings(_env_file=None)


def test_build_redis_settings_parses_dsn_with_password_and_default_db(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch, "redis://:redis@localhost:6379/0", "redis")
    rs = _build_redis_settings(settings)
    assert rs.host == "localhost"
    assert rs.port == 6379
    assert rs.password == "redis"
    assert rs.database == 0
    assert rs.ssl is False


def test_build_redis_settings_round_trips_non_default_db_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch, "redis://:redis@localhost:6379/3", "redis")
    rs = _build_redis_settings(settings)
    assert rs.database == 3


def test_build_redis_settings_decodes_percent_encoded_password(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # arq 0.28's `RedisSettings.from_dsn` uses urlparse().password which
    # silently leaves percent-encoded passwords undecoded. The project's
    # validators already accept the decoded form via redis-py's parser, so
    # _build_redis_settings must agree — otherwise API and worker auth diverge.
    settings = _make_settings(monkeypatch, "redis://:p%40ss@localhost:6379/0", "p@ss")
    rs = _build_redis_settings(settings)
    assert rs.password == "p@ss"


def test_build_redis_settings_sets_ssl_for_rediss_scheme(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = _make_settings(monkeypatch, "rediss://:redis@localhost:6379/0", "redis")
    rs = _build_redis_settings(settings)
    assert rs.ssl is True
