"""Session-scoped Postgres fixtures for integration tests.

Probes the dev compose Postgres at TEST_DATABASE_URL (default:
postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes_test).
Skips the entire integration test session with a clear remediation message
when compose Postgres is unreachable.
"""

from __future__ import annotations

import asyncio
import os
import re
from collections.abc import AsyncIterator
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import pytest
import pytest_asyncio
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

DEFAULT_TEST_DSN = "postgresql+asyncpg://postgres:postgres@localhost:5433/rag_recipes_test"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
_VALID_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _maintenance_dsn(test_dsn: str) -> str:
    """Replace the database name in a DSN with `postgres` (maintenance DB)."""
    parts = urlparse(test_dsn)
    return urlunparse(parts._replace(path="/postgres"))


def _assert_resettable(dsn: str, db_name: str) -> None:
    """Guard the destructive DROP DATABASE path.

    A misconfigured TEST_DATABASE_URL must never let the reset flag drop a real
    database. Only a local, `*_test`-named database may be dropped.
    """
    host = (urlparse(dsn).hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing TEST_DATABASE_RESET against non-local host {host!r}; "
            "reset only drops databases on localhost."
        )
    if not db_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing TEST_DATABASE_RESET for database {db_name!r}; "
            "only databases whose name ends in '_test' may be dropped."
        )


@pytest.fixture(scope="session")
def postgres_test_dsn() -> str:
    dsn = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DSN)
    maintenance = _maintenance_dsn(dsn)

    async def _check() -> None:
        engine = create_async_engine(maintenance, connect_args={"timeout": 2})
        try:
            async with engine.connect():
                pass
        finally:
            await engine.dispose()

    try:
        asyncio.run(_check())
    except Exception:
        pytest.skip(f"Postgres not reachable at {dsn}; start compose with `just setup`.")
    return dsn


@pytest.fixture(scope="session")
def create_test_database(postgres_test_dsn: str) -> str:
    """Create rag_recipes_test on the dev cluster if absent; recreate when TEST_DATABASE_RESET=1."""
    maintenance = _maintenance_dsn(postgres_test_dsn)
    db_name = urlparse(postgres_test_dsn).path.lstrip("/")
    reset = os.environ.get("TEST_DATABASE_RESET") == "1"

    # db_name is interpolated into CREATE/DROP DATABASE, which cannot take bind
    # parameters for the identifier; reject anything that isn't a plain identifier.
    if not _VALID_DB_NAME.match(db_name):
        raise RuntimeError(f"Unsafe test database name {db_name!r} in TEST_DATABASE_URL.")
    if reset:
        _assert_resettable(postgres_test_dsn, db_name)

    async def _ensure() -> None:
        engine = create_async_engine(maintenance, isolation_level="AUTOCOMMIT")
        try:
            async with engine.connect() as conn:
                exists = await conn.execute(
                    text("SELECT 1 FROM pg_database WHERE datname = :n"),
                    {"n": db_name},
                )
                found = exists.first() is not None
                if found and reset:
                    await conn.execute(
                        text(
                            "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                            "WHERE datname = :n AND pid <> pg_backend_pid()"
                        ),
                        {"n": db_name},
                    )
                    await conn.exec_driver_sql(f'DROP DATABASE "{db_name}"')
                    found = False
                if not found:
                    await conn.exec_driver_sql(f'CREATE DATABASE "{db_name}"')
        finally:
            await engine.dispose()

    asyncio.run(_ensure())
    return postgres_test_dsn


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def test_engine(create_test_database: str) -> AsyncIterator[AsyncEngine]:
    dsn = create_test_database
    cfg = AlembicConfig(str(PROJECT_ROOT / "alembic.ini"))
    cfg.set_main_option("sqlalchemy.url", dsn)
    await asyncio.to_thread(alembic_command.upgrade, cfg, "head")
    # NullPool: each connection is opened on the active (per-test) event loop and
    # closed afterwards, so the session-scoped engine is safe to share across the
    # function-scoped loops that pytest-asyncio creates for individual tests.
    engine = create_async_engine(dsn, poolclass=NullPool)
    try:
        yield engine
    finally:
        await engine.dispose()


@pytest_asyncio.fixture
async def db_session(test_engine: AsyncEngine) -> AsyncIterator[AsyncSession]:
    async with test_engine.connect() as connection:
        trans = await connection.begin()
        session = AsyncSession(bind=connection, expire_on_commit=False)
        nested = await session.begin_nested()
        try:
            yield session
        finally:
            if nested.is_active:
                await nested.rollback()
            await session.close()
            if trans.is_active:
                await trans.rollback()
