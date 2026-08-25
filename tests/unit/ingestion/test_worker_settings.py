"""Unit tests for the arq WorkerSettings registration."""

from __future__ import annotations

from rag_recipes.config import get_settings
from rag_recipes.ingestion.jobs import WorkerSettings


def _function_named(name: str):
    for fn in WorkerSettings.functions:
        if getattr(fn, "name", None) == name:
            return fn
    raise AssertionError(f"no registered function named {name!r}")


def test_process_document_registered_with_max_tries_3() -> None:
    # Phase 9.5: the extraction loop is resumable, so arq must actually re-drive
    # a killed/timed-out job — max_tries=1 would never trigger the resume path.
    process_document_fn = _function_named("process_document")
    assert process_document_fn.max_tries == 3


def test_process_document_carries_its_own_timeout() -> None:
    # A document is a whole book extracted window-by-window; the worker-wide
    # 600s default cancelled it mid-run and arq retries CancelledError, so a
    # large book died as "max 3 retries exceeded". The per-function timeout must
    # therefore be set AND distinct from the worker default — a None here means
    # arq silently falls back to job_timeout.
    process_document_fn = _function_named("process_document")
    settings = get_settings()
    assert process_document_fn.timeout_s == settings.document_job_timeout_seconds
    assert settings.document_job_timeout_seconds > settings.worker_job_timeout_seconds


def test_index_knowledge_item_uses_the_worker_default_timeout() -> None:
    # The long document timeout must NOT leak onto the single-item index job:
    # that is one chunk+embed round-trip, and arq derives its post-crash
    # in-progress lock TTL from the largest registered timeout.
    index_fn = _function_named("index_knowledge_item")
    assert index_fn.timeout_s is None


def test_index_knowledge_item_registered_with_max_tries_3() -> None:
    # Phase 21.3: the burst-worker integration test hand-wires its own Worker,
    # so this is the only guard on the *production* registration (plan TASK-004).
    index_fn = _function_named("index_knowledge_item")
    assert index_fn.max_tries == 3
