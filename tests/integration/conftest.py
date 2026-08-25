"""Session-scoped Postgres fixtures for integration tests.

Probes the dev compose Postgres at TEST_DATABASE_URL (default:
postgresql+asyncpg://postgres:postgres@localhost:5435/rag_recipes_test — the
port docker-compose publishes, and the one .env.example ships).
Skips the entire integration test session with a clear remediation message
when compose Postgres is unreachable.
"""

from __future__ import annotations

import asyncio
import os
import re
import uuid
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from unittest.mock import AsyncMock
from urllib.parse import parse_qs, urlparse, urlunparse

import pytest
import pytest_asyncio
from alembic import command as alembic_command
from alembic.config import Config as AlembicConfig
from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, create_async_engine
from sqlalchemy.pool import NullPool

from rag_recipes.config import get_settings
from rag_recipes.ingestion.queue import _build_redis_settings

# 5435 is the port docker-compose publishes (5432/5433 are other projects'
# databases on this machine — see docker-compose.yml). This default has to
# match it: TEST_DATABASE_URL lives in .env, which pydantic-settings loads for
# Settings but never exports into os.environ, so an unexported shell running
# `just test-integration` lands here — and a wrong default skips the ENTIRE
# integration suite with a "start compose" message on a checkout where compose
# is already running.
DEFAULT_TEST_DSN = "postgresql+asyncpg://postgres:postgres@localhost:5435/rag_recipes_test"
PROJECT_ROOT = Path(__file__).resolve().parents[2]

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1", ""}
_VALID_DB_NAME = re.compile(r"^[A-Za-z0-9_]+$")


def _maintenance_dsn(test_dsn: str) -> str:
    """Replace the database name in a DSN with `postgres` (maintenance DB)."""
    parts = urlparse(test_dsn)
    return urlunparse(parts._replace(path="/postgres"))


def _assert_safe_test_target(dsn: str, db_name: str) -> None:
    """Guard every integration run against a misconfigured TEST_DATABASE_URL.

    The integration suite is destructive by design: it runs migrations, and the
    round-trip test calls `alembic downgrade base`, which drops every table.
    Pointing TEST_DATABASE_URL at a real database (e.g. copied from DATABASE_URL)
    would wipe it. Refuse anything that isn't a local, `*_test`-named database, so
    the destructive paths only ever run against a throwaway test database.
    """
    # db_name is interpolated into CREATE/DROP DATABASE, which cannot take bind
    # parameters for the identifier; reject anything that isn't a plain identifier.
    if not _VALID_DB_NAME.match(db_name):
        raise RuntimeError(f"Unsafe test database name {db_name!r} in TEST_DATABASE_URL.")
    # asyncpg honours host/port given as query parameters, which would override the
    # empty URL host and connect elsewhere — bypassing the local-host check below.
    query = parse_qs(urlparse(dsn).query)
    if "host" in query or "port" in query:
        raise RuntimeError(
            "TEST_DATABASE_URL must not set host/port via query parameters; "
            "asyncpg would connect there, bypassing the local-host guard."
        )
    host = (urlparse(dsn).hostname or "").lower()
    if host not in _LOCAL_HOSTS:
        raise RuntimeError(
            f"Refusing to run the destructive integration suite against non-local "
            f"host {host!r}; TEST_DATABASE_URL must point at a local database."
        )
    if not db_name.endswith("_test"):
        raise RuntimeError(
            f"Refusing to run the destructive integration suite against database "
            f"{db_name!r}; TEST_DATABASE_URL must name a database ending in '_test'."
        )


@pytest.fixture(scope="session")
def postgres_test_dsn() -> str:
    dsn = os.environ.get("TEST_DATABASE_URL", DEFAULT_TEST_DSN)
    # Fail loud before any destructive work if the target looks like a real database.
    _assert_safe_test_target(dsn, urlparse(dsn).path.lstrip("/"))
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
    """Create rag_recipes_test on the dev cluster if absent; recreate when TEST_DATABASE_RESET=1.

    Safety of the target DSN is enforced upstream by `postgres_test_dsn`.
    """
    maintenance = _maintenance_dsn(postgres_test_dsn)
    db_name = urlparse(postgres_test_dsn).path.lstrip("/")
    reset = os.environ.get("TEST_DATABASE_RESET") == "1"

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


# Phase 6.3: documents-route tests must send a bearer token because auth fails
# closed when ``personal_api_token`` is unset.
TEST_API_TOKEN = "test-personal-api-token"
AUTH_HEADERS = {"Authorization": f"Bearer {TEST_API_TOKEN}"}


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return dict(AUTH_HEADERS)


@pytest.fixture(scope="session")
def redis_arq_settings() -> RedisSettings:
    """Build `RedisSettings` against compose Redis; skip if unreachable.

    Mirrors `postgres_test_dsn`'s skip behaviour so integration tests can run
    on a developer machine without compose without failing the suite.
    """
    settings = get_settings()
    rs = _build_redis_settings(settings)

    async def _probe() -> None:
        pool = await create_pool(rs)
        try:
            await pool.ping()
        finally:
            await pool.aclose()

    try:
        asyncio.run(_probe())
    except Exception:
        pytest.skip(
            f"Redis not reachable at {settings.redis_url}; start compose with `just setup`."
        )
    return rs


@pytest_asyncio.fixture
async def arq_pool(redis_arq_settings: RedisSettings) -> AsyncIterator[ArqRedis]:
    pool = await create_pool(redis_arq_settings)
    try:
        yield pool
    finally:
        await pool.aclose()


@pytest.fixture
def arq_queue_name(request: pytest.FixtureRequest) -> str:
    safe = request.node.nodeid
    for ch in ("/", ":", "[", "]"):
        safe = safe.replace(ch, "_")
    return f"arq:test:{safe}-{uuid.uuid4().hex[:8]}"


@pytest_asyncio.fixture
async def arq_queue_cleanup(
    arq_pool: ArqRedis, arq_queue_name: str
) -> AsyncIterator[str]:
    """Yield the queue name; on teardown, delete every arq key for it.

    arq lays out keys as `arq:queue:<name>`, `arq:in-progress:<name>:<job>`,
    `arq:result:<name>:<job>`, `arq:job:<name>:<job>`, etc. A single SCAN
    over `arq:*:<name>*` keeps cleanup exhaustive without enumerating arq's
    internal key schema.
    """
    yield arq_queue_name
    pattern = f"arq:*{arq_queue_name}*"
    cursor = 0
    while True:
        cursor, keys = await arq_pool.scan(cursor=cursor, match=pattern, count=100)
        if keys:
            await arq_pool.delete(*keys)
        if cursor == 0:
            break


@pytest.fixture
def fake_arq_redis() -> AsyncMock:
    """An ArqRedis stand-in whose enqueue_job awaitable always succeeds."""
    m = AsyncMock(spec=ArqRedis)
    m.enqueue_job = AsyncMock(return_value=AsyncMock())  # arq.jobs.Job
    return m


@pytest.fixture(autouse=True)
def _override_arq_redis(fake_arq_redis: AsyncMock) -> Iterator[None]:
    """Override the route's arq dependency so the upload route can enqueue.

    Applies to every integration test (harmless for those that never hit the
    upload route). pytest sets autouse fixtures up before explicitly-requested
    ones, so a test that needs the *real* pool (the end-to-end worker test)
    overrides ``get_arq_redis`` again in its own client fixture and wins.
    """
    from rag_recipes.api.app import app
    from rag_recipes.api.dependencies import get_arq_redis

    app.dependency_overrides[get_arq_redis] = lambda: fake_arq_redis
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_arq_redis, None)


@pytest.fixture
def override_settings_with_token() -> Iterator[None]:
    """Override ``get_settings`` so the auth dependency sees a token configured.

    Documents-route tests opt into this fixture to keep the auth code path
    real (the route still runs ``require_api_token``) while supplying the
    token the test client also sends as a bearer header.
    """
    from rag_recipes.api.app import app
    from rag_recipes.api.dependencies import get_settings
    from rag_recipes.config import Settings

    real = get_settings()
    overridden = real.model_copy(update={"personal_api_token": TEST_API_TOKEN})

    def _override() -> Settings:
        return overridden

    app.dependency_overrides[get_settings] = _override
    try:
        yield
    finally:
        app.dependency_overrides.pop(get_settings, None)
