"""arq worker entry point: `WorkerSettings`, `ping_job`, and shared session helper.

`uv run arq rag_recipes.ingestion.jobs.WorkerSettings` is the documented
process; arq's CLI discovers `WorkerSettings`, builds the connection pool
from `redis_settings`, runs `on_startup` once per process, and dispatches
jobs from `functions`. Phase 7.1 ships only `ping_job` to prove the seam;
Epic 8 adds real ingestion jobs that reuse `langfuse_session_scope`.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from arq.cron import cron

from rag_recipes.config import Settings, get_settings
from rag_recipes.ingestion.cron import sweep_stuck_jobs
from rag_recipes.ingestion.queue import _build_redis_settings
from rag_recipes.providers._observability import (
    ProviderObservability,
    build_provider_observability,
)
from rag_recipes.storage.session import build_engine, build_session_factory


@contextmanager
def langfuse_session_scope(
    observability: ProviderObservability | None,
    session_id: str | None,
) -> Iterator[None]:
    """Open `langfuse.propagate_attributes(session_id=...)` when wired.

    No-ops when `observability` is `None`, when Langfuse is disabled (no
    session scope was wired by `build_provider_observability`), or when
    `session_id` is `None`. Sync CM — `propagate_attributes` is sync.
    """
    if observability is None or session_id is None:
        yield
        return
    # `_session_scope` is the documented seam: `build_provider_observability`
    # wires `langfuse.propagate_attributes` here when Langfuse is enabled,
    # and leaves it `None` otherwise. Accessing it directly keeps the
    # Langfuse import centralised in the factory.
    scope = observability._session_scope  # noqa: SLF001
    if scope is None:
        yield
        return
    with scope(session_id=session_id):
        yield


async def ping_job(
    ctx: dict[str, Any],
    message: str = "ping",
    *,
    _session_id: str | None = None,
) -> str:
    observability = ctx.get("observability")
    with langfuse_session_scope(observability, _session_id):
        return f"pong:{message}"


async def on_startup(ctx: dict[str, Any]) -> None:
    settings: Settings = get_settings()
    ctx["settings"] = settings
    ctx["observability"] = build_provider_observability(settings)
    engine = build_engine(settings)
    ctx["engine"] = engine
    ctx["session_factory"] = build_session_factory(engine)


async def on_shutdown(ctx: dict[str, Any]) -> None:
    observability = ctx.get("observability")
    client = getattr(observability, "_client", None)
    flush = getattr(client, "flush", None)
    if callable(flush):
        # Observability teardown failures must not crash worker shutdown.
        with contextlib.suppress(Exception):
            flush()
    engine = ctx.get("engine")
    if engine is not None:
        await engine.dispose()


_SETTINGS = get_settings()


class WorkerSettings:
    functions = [ping_job]
    cron_jobs = [
        cron(
            sweep_stuck_jobs,
            minute=set(range(0, 60, _SETTINGS.stuck_job_check_interval_minutes)),
            run_at_startup=False,
            unique=True,
            max_tries=1,
            timeout=_SETTINGS.stuck_job_timeout_minutes * 60,
        ),
    ]
    redis_settings = _build_redis_settings(_SETTINGS)
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = _SETTINGS.worker_max_jobs
    job_timeout = _SETTINGS.worker_job_timeout_seconds
    keep_result = _SETTINGS.worker_keep_result_seconds
    health_check_interval = _SETTINGS.worker_health_check_interval_seconds
