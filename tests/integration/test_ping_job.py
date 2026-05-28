"""End-to-end test for the arq worker wiring (Phase 7.1).

Exercises the full enqueue → in-process worker → result loop against the
compose Redis on a per-test queue. Subject under test is the wiring, not
`ping_job` itself (per `tests/AGENTS.md`).
"""

from __future__ import annotations

import asyncio

import pytest
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import JobStatus
from arq.worker import Worker

from rag_recipes.ingestion.jobs import on_shutdown, on_startup, ping_job
from rag_recipes.ingestion.queue import enqueue_job

pytestmark = pytest.mark.asyncio


async def test_ping_job_runs_end_to_end_via_burst_worker(
    arq_pool: ArqRedis,
    redis_arq_settings: RedisSettings,
    arq_queue_cleanup: str,
) -> None:
    queue_name = arq_queue_cleanup
    job = await enqueue_job(
        arq_pool,
        "ping_job",
        "hello",
        session_id="session_abc",
        _queue_name=queue_name,
    )
    assert job is not None

    worker = Worker(
        functions=[ping_job],
        redis_settings=redis_arq_settings,
        burst=True,
        max_jobs=1,
        queue_name=queue_name,
        on_startup=on_startup,
        on_shutdown=on_shutdown,
        poll_delay=0.1,
    )
    try:
        await asyncio.wait_for(worker.async_run(), timeout=10)
    finally:
        await worker.close()

    result = await asyncio.wait_for(job.result(timeout=5), timeout=10)
    assert result == "pong:hello"
    assert await job.status() == JobStatus.complete
