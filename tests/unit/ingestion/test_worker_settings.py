"""Unit tests for the arq WorkerSettings registration."""

from __future__ import annotations

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


def test_index_knowledge_item_registered_with_max_tries_3() -> None:
    # Phase 21.3: the burst-worker integration test hand-wires its own Worker,
    # so this is the only guard on the *production* registration (plan TASK-004).
    index_fn = _function_named("index_knowledge_item")
    assert index_fn.max_tries == 3
