"""Regression tests for the destructive-reset guard in conftest.

These exercise pure validation logic and do not require a database, so they run
even when compose Postgres is unreachable.
"""

from __future__ import annotations

import pytest

from tests.integration.conftest import _VALID_DB_NAME, _assert_resettable


def test_reset_allowed_for_local_test_database() -> None:
    _assert_resettable(
        "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes_test",
        "rag_recipes_test",
    )


def test_reset_refused_for_non_local_host() -> None:
    with pytest.raises(RuntimeError, match="non-local host"):
        _assert_resettable(
            "postgresql+asyncpg://user:pw@db.prod.internal:5432/rag_recipes_test",
            "rag_recipes_test",
        )


def test_reset_refused_for_non_test_database_name() -> None:
    with pytest.raises(RuntimeError, match="_test"):
        _assert_resettable(
            "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes",
            "rag_recipes",
        )


@pytest.mark.parametrize("bad_name", ["rag_recipes_test; DROP", "rag recipes", 'a"b', ""])
def test_valid_db_name_rejects_non_identifiers(bad_name: str) -> None:
    assert _VALID_DB_NAME.match(bad_name) is None
