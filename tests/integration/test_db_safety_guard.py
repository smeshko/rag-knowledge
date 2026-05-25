"""Regression tests for the destructive-target guard in conftest.

The integration suite is destructive (it migrates and, in the round-trip test,
runs `alembic downgrade base`). `_assert_safe_test_target` must refuse any
TEST_DATABASE_URL that is not a local, `*_test`-named database — regardless of
the TEST_DATABASE_RESET flag. These exercise pure validation logic and need no
database, so they run even when compose Postgres is unreachable.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import _VALID_DB_NAME, _assert_safe_test_target


def test_allowed_for_local_test_database() -> None:
    _assert_safe_test_target(
        "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes_test",
        "rag_recipes_test",
    )


def test_refused_for_non_local_host() -> None:
    with pytest.raises(RuntimeError, match="non-local"):
        _assert_safe_test_target(
            "postgresql+asyncpg://user:pw@db.prod.internal:5432/rag_recipes_test",
            "rag_recipes_test",
        )


def test_refused_for_non_test_database_name() -> None:
    # The footgun: TEST_DATABASE_URL accidentally pointed at the real dev database.
    with pytest.raises(RuntimeError, match="_test"):
        _assert_safe_test_target(
            "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes",
            "rag_recipes",
        )


@pytest.mark.parametrize(
    "dsn",
    [
        "postgresql+asyncpg://postgres:postgres@/rag_recipes_test?host=db.prod.internal",
        "postgresql+asyncpg://postgres:postgres@localhost/rag_recipes_test?port=6543",
        "postgresql+asyncpg://postgres:postgres@/rag_recipes_test?host=staging-db&port=5432",
    ],
)
def test_refused_for_query_string_host_or_port(dsn: str) -> None:
    # asyncpg honours ?host=/?port=, which would connect off-box even though the
    # URL host is empty/local — the guard must reject these.
    with pytest.raises(RuntimeError, match="query parameters"):
        _assert_safe_test_target(dsn, "rag_recipes_test")


@pytest.mark.parametrize("bad_name", ["rag_recipes_test; DROP", "rag recipes", 'a"b', ""])
def test_refused_for_non_identifier_database_name(bad_name: str) -> None:
    assert _VALID_DB_NAME.match(bad_name) is None
    with pytest.raises(RuntimeError, match="Unsafe test database name"):
        _assert_safe_test_target(
            f"postgresql+asyncpg://postgres:postgres@localhost:5433/{bad_name}",
            bad_name,
        )
