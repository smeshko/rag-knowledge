"""Cron submitter for pending extraction batches (Epic 19.2).

``submit_extraction_batches`` drains ``PENDING`` ``ExtractionBatchItem``s into
Anthropic Message Batches using a **durable claim-before-call protocol** so a
crash/commit-failure after the provider accepts a batch can never re-submit /
double-charge (DECISIONS #7):

1. **Reconcile** stale ``SUBMITTING`` batches (a prior run crashed around the
   provider call) by unconditionally reverting them to ``PENDING`` (the terminal,
   always-terminating fallback). Re-submitting a chunk Anthropic *did* accept is
   wasteful but not corrupting — 19.3 ingests results idempotently keyed on
   ``input_hash``. We do **not** try to "confirm" acceptance by matching a recent
   provider batch on request count: a coincidental match would link items to the
   wrong batch and strand them ``SUBMITTED`` forever (review #2.1); a safe confirm
   needs the per-``custom_id`` results surface 19.3 owns.
2. **Claim** chunks of ``PENDING`` items (``FOR UPDATE SKIP LOCKED`` so concurrent
   crons never grab the same items), chunked by **both** a request count and an
   estimated serialized-bytes cap, committing each chunk to ``SUBMITTING`` (items
   + a ``SUBMITTING`` batch row) *before* the provider call. The items are then
   off the ``PENDING`` gather, so the worst-case crash strands a batch rather than
   re-submitting it.
3. **Submit** each pre-claimed chunk; on success finalize to ``SUBMITTED`` +
   ``provider_batch_id``, on a clean (pre-acceptance) ``LLMTechnicalError`` revert
   the chunk to ``PENDING`` for the next tick.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import func

from rag_recipes.config import Settings
from rag_recipes.ingestion.pipeline.dedup import compute_candidate_score
from rag_recipes.ingestion.pipeline.extraction import (
    SCHEMA_VERSION,
    RecipeExtractionOutput,
)
from rag_recipes.ingestion.pipeline.persist import persist_knowledge_item
from rag_recipes.ingestion.pipeline.windows import Window
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.validation import HardValidationError
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic import map_message_to_structured_output
from rag_recipes.providers.llm.anthropic_batch import (
    AnthropicBatchProvider,
    BatchExtractionRequest,
    BatchResult,
)
from rag_recipes.storage.enums import (
    ExtractionBatchItemStatus,
    ExtractionBatchStatus,
    ExtractionRunStatus,
)
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.source_span import SourceSpan

logger = logging.getLogger(__name__)

# Terminal item status for each non-success result type (used on the duplicate /
# already-ingested path, and at the retry cap).
_TERMINAL_ITEM_STATUS: dict[str, ExtractionBatchItemStatus] = {
    "succeeded": ExtractionBatchItemStatus.SUCCEEDED,
    "errored": ExtractionBatchItemStatus.ERRORED,
    "expired": ExtractionBatchItemStatus.EXPIRED,
    "canceled": ExtractionBatchItemStatus.CANCELED,
}

# Per-request serialization overhead beyond input + schema (custom_id, JSON
# envelope, message wrapper) — a conservative constant for the byte estimate.
_REQUEST_ENVELOPE_BYTES = 512


def _build_batch_provider(settings: Settings) -> AnthropicBatchProvider:
    """Construct the Anthropic batch provider (only the Anthropic path exists)."""
    api_key = settings.anthropic_api_key
    if api_key is None:
        raise ValueError("anthropic_api_key is required for batch submission")
    return AnthropicBatchProvider(
        api_key,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


def _estimate_item_bytes(item: ExtractionBatchItem) -> int:
    return (
        len(item.request_input.encode("utf-8"))
        + len(json.dumps(item.request_schema))
        + _REQUEST_ENVELOPE_BYTES
    )


async def submit_extraction_batches(ctx: dict[str, Any]) -> int:
    """Reconcile, claim, and submit pending extraction batches. Returns items sent."""
    settings: Settings = ctx["settings"]
    session_factory: async_sessionmaker[Any] = ctx["session_factory"]

    if settings.llm_provider != "anthropic":
        logger.info(
            "batch submitter skipped: llm_provider=%s (no batch provider)",
            settings.llm_provider,
        )
        return 0

    provider = _build_batch_provider(settings)
    await _reconcile_submitting_batches(session_factory, settings)

    # Claim phase: pull every available chunk into SUBMITTING *before* any provider
    # call, so the submit phase iterates a fixed set — a per-chunk submit failure
    # reverts only that chunk and never re-claims it this tick.
    claimed: list[tuple[str, list[BatchExtractionRequest]]] = []
    while True:
        chunk = await _claim_chunk(session_factory, settings)
        if chunk is None:
            break
        claimed.append(chunk)

    if not claimed:
        return 0

    submitted = 0
    for batch_id, requests in claimed:
        try:
            result = await provider.submit_batch(requests, idempotency_key=batch_id)
        except LLMTechnicalError as exc:
            # Clean, pre-acceptance failure — revert the chunk so it retries next
            # tick; previously-submitted chunks stay recorded.
            logger.warning(
                "batch %s submit failed pre-acceptance (%s); reverting %d item(s) "
                "to PENDING",
                batch_id,
                exc,
                len(requests),
            )
            await _revert_chunk(session_factory, batch_id)
            continue
        await _finalize_chunk(session_factory, batch_id, result.provider_batch_id)
        submitted += len(requests)

    logger.info(
        "extraction batch submitter sent %d item(s) across %d batch(es)",
        submitted,
        len(claimed),
    )
    return submitted


async def _claim_chunk(
    session_factory: async_sessionmaker[Any], settings: Settings
) -> tuple[str, list[BatchExtractionRequest]] | None:
    """Claim one chunk of PENDING items into a SUBMITTING batch, committed.

    Returns ``(batch_id, requests)`` or ``None`` when no claimable items remain.
    Skips schema-drifted items (``schema_version != SCHEMA_VERSION``); chunks by
    both the request-count and the byte cap (always taking at least one item so a
    single oversized window still ships).
    """
    async with session_factory() as session:
        result = await session.execute(
            select(ExtractionBatchItem)
            .where(
                ExtractionBatchItem.status == ExtractionBatchItemStatus.PENDING,
                ExtractionBatchItem.batch_id.is_(None),
                ExtractionBatchItem.schema_version == SCHEMA_VERSION,
            )
            .order_by(ExtractionBatchItem.created_at)
            .limit(settings.anthropic_batch_max_requests)
            .with_for_update(skip_locked=True)
        )
        candidates = list(result.scalars().all())
        if not candidates:
            return None

        chosen: list[ExtractionBatchItem] = []
        running_bytes = 0
        for item in candidates:
            size = _estimate_item_bytes(item)
            # Always include the first item; otherwise stop before the byte cap.
            if chosen and running_bytes + size > settings.anthropic_batch_max_bytes:
                break
            chosen.append(item)
            running_bytes += size

        batch = ExtractionBatch(
            provider="anthropic",
            provider_batch_id=None,
            model=settings.anthropic_llm_model,
            processing_status=ExtractionBatchStatus.SUBMITTING,
            request_count=len(chosen),
        )
        session.add(batch)
        await session.flush()
        batch_id = batch.id

        requests: list[BatchExtractionRequest] = []
        for item in chosen:
            item.status = ExtractionBatchItemStatus.SUBMITTING
            item.batch_id = batch_id
            requests.append(
                BatchExtractionRequest(
                    custom_id=item.id,
                    input=item.request_input,
                    model=settings.anthropic_llm_model,
                    max_tokens=settings.anthropic_max_tokens,
                    json_schema=item.request_schema,
                )
            )
        await session.commit()
    return batch_id, requests


async def _finalize_chunk(
    session_factory: async_sessionmaker[Any],
    batch_id: str,
    provider_batch_id: str,
) -> None:
    async with session_factory() as session:
        await session.execute(
            update(ExtractionBatch)
            .where(ExtractionBatch.id == batch_id)
            .values(
                processing_status=ExtractionBatchStatus.SUBMITTED,
                provider_batch_id=provider_batch_id,
            )
        )
        await session.execute(
            update(ExtractionBatchItem)
            .where(ExtractionBatchItem.batch_id == batch_id)
            .values(status=ExtractionBatchItemStatus.SUBMITTED)
        )
        await session.commit()


async def _revert_chunk(
    session_factory: async_sessionmaker[Any], batch_id: str
) -> None:
    async with session_factory() as session:
        await session.execute(
            update(ExtractionBatchItem)
            .where(ExtractionBatchItem.batch_id == batch_id)
            .values(status=ExtractionBatchItemStatus.PENDING, batch_id=None)
        )
        await session.execute(
            update(ExtractionBatch)
            .where(ExtractionBatch.id == batch_id)
            .values(processing_status=ExtractionBatchStatus.FAILED)
        )
        await session.commit()


async def _reconcile_submitting_batches(
    session_factory: async_sessionmaker[Any],
    settings: Settings,
) -> None:
    """Revert SUBMITTING batches stranded by a prior crash (DECISIONS #7).

    A SUBMITTING batch past ``anthropic_batch_submitting_timeout_minutes`` means a
    prior run crashed around the provider call. Resolution is the terminal,
    always-terminating fallback: revert it to ``PENDING`` so its items re-submit
    on the next claim. Re-submitting a chunk Anthropic *did* accept is wasteful but
    not corrupting — 19.3 ingests results idempotently keyed on ``input_hash``, so
    duplicate results collapse to one ``ExtractionRun`` (≤2× cost for that chunk in
    the rare crash case).

    We deliberately do **not** "confirm" acceptance by matching a recent provider
    batch on request count (review #2.1): a coincidental same-count match would
    link the items to the *wrong* provider batch, so 19.3 would never deliver their
    results and they'd sit ``SUBMITTED`` (sweep-exempt) forever — strictly worse
    than a dedup-safe re-submit, and the exact "stranded forever" class DECISIONS
    #7 exists to prevent. A safe confirm needs the per-``custom_id`` results surface
    19.3 owns; it can be added there if the re-submit cost ever matters.
    """
    threshold = datetime.now(tz=UTC) - timedelta(
        minutes=settings.anthropic_batch_submitting_timeout_minutes
    )
    async with session_factory() as session:
        stale_ids = list(
            (
                await session.execute(
                    select(ExtractionBatch.id).where(
                        ExtractionBatch.processing_status
                        == ExtractionBatchStatus.SUBMITTING,
                        ExtractionBatch.created_at < threshold,
                    )
                )
            )
            .scalars()
            .all()
        )

    for batch_id in stale_ids:
        logger.warning(
            "reconcile: reverting stale SUBMITTING batch %s to PENDING for "
            "re-submission",
            batch_id,
        )
        await _revert_chunk(session_factory, batch_id)


async def ingest_batch_result(
    session: AsyncSession,
    item: ExtractionBatchItem,
    result: BatchResult,
    *,
    settings: Settings,
) -> None:
    """Turn one batch result into a terminal ``ExtractionRun`` (+ staging candidates).

    Idempotent and atomic (DECISIONS #3): locks the item row, ingests only
    ``SUBMITTED`` items, and skips if a terminal ``ExtractionRun`` already exists
    for ``(document, source_version, input_hash)`` — the cross-phase invariant 19.2
    reconciliation relies on. The run insert + item status flip share the caller's
    transaction (the poller commits per batch). Mirrors the synchronous
    ``run_extraction`` terminal mapping + the staging-persist loop (DECISIONS #1).
    """
    locked = await session.get(ExtractionBatchItem, item.id, with_for_update=True)
    if locked is None or locked.status is not ExtractionBatchItemStatus.SUBMITTED:
        return
    item = locked

    # Idempotency: a run for this window already exists (e.g. a 19.2 reconcile
    # re-submit of an already-accepted chunk) — converge the item terminal, no run.
    existing = await session.scalar(
        select(ExtractionRun.id)
        .where(
            ExtractionRun.document_id == item.document_id,
            ExtractionRun.source_version == item.source_version,
            ExtractionRun.input_hash == item.input_hash,
        )
        .limit(1)
    )
    if existing is not None:
        item.status = _TERMINAL_ITEM_STATUS.get(
            result.result_type, ExtractionBatchItemStatus.REJECTED
        )
        item.result_type = result.result_type
        return

    if result.result_type == "succeeded":
        await _ingest_succeeded(session, item, result, settings=settings)
        return
    if result.result_type == "canceled" or (
        result.result_type == "errored" and not result.retryable
    ):
        await _ingest_terminal_failure(session, item, result, settings=settings)
        return
    # errored(retryable) / expired → bounded re-submit, else REJECTED at the cap.
    await _ingest_retryable_failure(session, item, result, settings=settings)


async def _model_for(session: AsyncSession, item: ExtractionBatchItem, settings: Settings) -> str:
    if item.batch_id is not None:
        model = await session.scalar(
            select(ExtractionBatch.model).where(ExtractionBatch.id == item.batch_id)
        )
        if model:
            return model
    return settings.anthropic_llm_model


def _new_run(
    item: ExtractionBatchItem,
    *,
    model: str,
    status: ExtractionRunStatus,
    output_json: dict[str, Any] | None,
    error_message: str | None,
) -> ExtractionRun:
    return ExtractionRun(
        document_id=item.document_id,
        source_version=item.source_version,
        provider="anthropic",
        model=model,
        prompt_version=item.prompt_version,
        schema_version=item.schema_version,
        input_source_span_ids=item.input_source_span_ids,
        input_hash=item.input_hash,
        status=status,
        output_json=output_json,
        error_message=error_message,
        completed_at=datetime.now(tz=UTC),
    )


async def _ingest_succeeded(
    session: AsyncSession,
    item: ExtractionBatchItem,
    result: BatchResult,
    *,
    settings: Settings,
) -> None:
    model = await _model_for(session, item, settings)
    response = map_message_to_structured_output(
        result.message, provider="anthropic", model=model
    )
    item.result_type = "succeeded"

    if response.output_json is None:
        run = _new_run(
            item,
            model=model,
            status=ExtractionRunStatus.REJECTED,
            output_json=None,
            error_message=response.parse_error,
        )
        session.add(run)
        item.status = ExtractionBatchItemStatus.REJECTED
        item.error_message = response.parse_error
        return

    try:
        parsed = RecipeExtractionOutput.model_validate(response.output_json)
    except ValidationError as exc:
        run = _new_run(
            item,
            model=model,
            status=ExtractionRunStatus.REJECTED,
            output_json=response.output_json,
            error_message=str(exc),
        )
        session.add(run)
        item.status = ExtractionBatchItemStatus.REJECTED
        item.error_message = str(exc)
        return

    run = _new_run(
        item,
        model=model,
        status=ExtractionRunStatus.SUCCESS,
        output_json=response.output_json,
        error_message=None,
    )
    session.add(run)
    await session.flush()  # need run.id for the staging candidates

    window = await _reconstruct_window(session, item)
    for extracted in parsed.items:
        try:
            await persist_knowledge_item(
                session,
                extracted,
                extraction_run_id=run.id,
                document_id=item.document_id,
                source_version=item.source_version,
                window=window,
                staging=True,
                candidate_score=compute_candidate_score(
                    extracted, window_span_ids=item.input_source_span_ids
                ),
            )
        except HardValidationError as exc:
            logger.info(
                "dropping hard-invalid batch candidate for %s: %s",
                item.document_id,
                exc,
            )
    item.status = ExtractionBatchItemStatus.SUCCEEDED


async def _ingest_terminal_failure(
    session: AsyncSession,
    item: ExtractionBatchItem,
    result: BatchResult,
    *,
    settings: Settings,
) -> None:
    model = await _model_for(session, item, settings)
    detail = result.error_type or result.result_type
    error_message = f"batch result {result.result_type}: {detail}"
    session.add(
        _new_run(
            item,
            model=model,
            status=ExtractionRunStatus.REJECTED,
            output_json=None,
            error_message=error_message,
        )
    )
    item.status = (
        ExtractionBatchItemStatus.CANCELED
        if result.result_type == "canceled"
        else ExtractionBatchItemStatus.REJECTED
    )
    item.result_type = result.result_type
    item.error_message = error_message


async def _ingest_retryable_failure(
    session: AsyncSession,
    item: ExtractionBatchItem,
    result: BatchResult,
    *,
    settings: Settings,
) -> None:
    if item.submit_attempts < settings.anthropic_batch_max_submit_attempts:
        # Revert for re-submission by 19.2's submitter. Bump the document heartbeat
        # so the brief PENDING→re-claim window isn't reaped by 19.2's stale-PENDING
        # sweep (DECISIONS #2; cross-phase self-review #1).
        item.status = ExtractionBatchItemStatus.PENDING
        item.batch_id = None
        item.submit_attempts += 1
        item.result_type = result.result_type
        await session.execute(
            update(Document)
            .where(Document.id == item.document_id)
            .values(last_progress_at=func.now())
        )
        return

    # Cap hit — surface as a REJECTED audit run, never retry forever.
    model = await _model_for(session, item, settings)
    error_message = f"{result.result_type} after {item.submit_attempts} attempts"
    session.add(
        _new_run(
            item,
            model=model,
            status=ExtractionRunStatus.REJECTED,
            output_json=None,
            error_message=error_message,
        )
    )
    item.status = _TERMINAL_ITEM_STATUS.get(
        result.result_type, ExtractionBatchItemStatus.REJECTED
    )
    item.result_type = result.result_type
    item.error_message = error_message


async def _reconstruct_window(
    session: AsyncSession, item: ExtractionBatchItem
) -> Window:
    """Rebuild the extraction ``Window`` from the item's recorded span ids.

    ``persist_knowledge_item`` / ``validate_hard`` need a real ``Window``; the
    spans are immutable per ``(document, source_version)`` and the recorded order
    in ``input_source_span_ids`` reproduces the original window exactly.
    """
    rows = (
        await session.execute(
            select(SourceSpan).where(SourceSpan.id.in_(item.input_source_span_ids))
        )
    ).scalars().all()
    by_id = {span.id: span for span in rows}
    ordered = tuple(
        by_id[span_id] for span_id in item.input_source_span_ids if span_id in by_id
    )
    return Window(spans=ordered)


# Local batch statuses still worth polling (have a provider_batch_id, not terminal).
_POLLABLE_BATCH_STATUSES = (
    ExtractionBatchStatus.SUBMITTED,
    ExtractionBatchStatus.IN_PROGRESS,
)


async def poll_extraction_batches(ctx: dict[str, Any]) -> int:
    """Poll in-flight batches; on ``ended``, ingest results + mark the batch done.

    Each batch is processed in its own transaction under a ``FOR UPDATE SKIP
    LOCKED`` row lock so overlapping poller runs never double-process one. A
    provider error on a batch leaves it non-terminal for the next tick (never
    crashes the whole run). Returns the number of results ingested.
    """
    settings: Settings = ctx["settings"]
    session_factory: async_sessionmaker[Any] = ctx["session_factory"]

    if settings.llm_provider != "anthropic":
        logger.info(
            "batch poller skipped: llm_provider=%s (no batch provider)",
            settings.llm_provider,
        )
        return 0

    provider = _build_batch_provider(settings)
    async with session_factory() as session:
        batch_ids = list(
            (
                await session.execute(
                    select(ExtractionBatch.id).where(
                        ExtractionBatch.processing_status.in_(_POLLABLE_BATCH_STATUSES),
                        ExtractionBatch.provider_batch_id.is_not(None),
                    )
                )
            )
            .scalars()
            .all()
        )

    ingested = 0
    touched: set[tuple[str, int]] = set()
    for batch_id in batch_ids:
        count, batch_touched = await _poll_one_batch(
            session_factory, provider, batch_id, settings
        )
        ingested += count
        touched |= batch_touched

    if touched:
        await finalize_completed_documents(session_factory, ctx["redis"], touched)

    logger.info("batch poller ingested %d result(s)", ingested)
    return ingested


async def _poll_one_batch(
    session_factory: async_sessionmaker[Any],
    provider: AnthropicBatchProvider,
    batch_id: str,
    settings: Settings,
) -> tuple[int, set[tuple[str, int]]]:
    async with session_factory() as session:
        batch = (
            await session.execute(
                select(ExtractionBatch)
                .where(ExtractionBatch.id == batch_id)
                .with_for_update(skip_locked=True)
            )
        ).scalar_one_or_none()
        if (
            batch is None
            or batch.provider_batch_id is None
            or batch.processing_status not in _POLLABLE_BATCH_STATUSES
        ):
            # Locked by another poller, or already terminal — skip.
            return 0, set()

        provider_batch_id = batch.provider_batch_id
        try:
            status = await provider.retrieve_batch(provider_batch_id)
            if status.processing_status != "ended":
                # Track the provider's progress locally; re-poll next tick.
                if status.processing_status == "in_progress":
                    batch.processing_status = ExtractionBatchStatus.IN_PROGRESS
                await session.commit()
                return 0, set()

            ingested = 0
            touched: set[tuple[str, int]] = set()
            async for result in provider.iter_results(provider_batch_id):
                item = (
                    await session.execute(
                        select(ExtractionBatchItem).where(
                            ExtractionBatchItem.id == result.custom_id
                        )
                    )
                ).scalar_one_or_none()
                if item is None:
                    continue
                await ingest_batch_result(session, item, result, settings=settings)
                touched.add((item.document_id, item.source_version))
                ingested += 1

            batch.processing_status = ExtractionBatchStatus.ENDED
            batch.completed_at = datetime.now(tz=UTC)
            await session.commit()
            return ingested, touched
        except LLMTechnicalError as exc:
            # Transient provider failure — leave the batch non-terminal for the
            # next tick; nothing is half-ingested (ingest is idempotent per item).
            await session.rollback()
            logger.warning("poll: batch %s provider error (%s); will retry", batch_id, exc)
            return 0, set()


# Non-terminal item statuses for completion detection: a document is ready to
# finalize only when none of its items remain in one of these.
_NON_TERMINAL_ITEM_STATUSES_FOR_COMPLETION = (
    ExtractionBatchItemStatus.PENDING,
    ExtractionBatchItemStatus.SUBMITTING,
    ExtractionBatchItemStatus.SUBMITTED,
)


async def finalize_completed_documents(
    session_factory: async_sessionmaker[Any],
    redis: Any,
    touched: set[tuple[str, int]],
) -> int:
    """Re-drive documents whose every window now has a terminal result.

    Completion is **per-document, not per-batch** (19.2 splits a doc's windows
    across batches; DECISIONS #4): a doc is finalized only when no
    ``ExtractionBatchItem`` for ``(document, source_version)`` is still non-terminal.
    Re-enqueues the normal **resume** path (``batch_mode`` stays ``False``) — the
    existing ``_resume_or_fresh → "resume"`` skips done windows via ``done_hashes``
    and ``_finalize_extraction`` promotes the committed staging candidates. Adds no
    finalize logic; idempotent (a redundant resume no-ops). Returns docs re-driven.
    """
    finalized = 0
    for document_id, source_version in touched:
        async with session_factory() as session:
            remaining = await session.scalar(
                select(func.count())
                .select_from(ExtractionBatchItem)
                .where(
                    ExtractionBatchItem.document_id == document_id,
                    ExtractionBatchItem.source_version == source_version,
                    ExtractionBatchItem.status.in_(
                        _NON_TERMINAL_ITEM_STATUSES_FOR_COMPLETION
                    ),
                )
            )
        if remaining:
            # Windows still in flight (e.g. split across batches) — wait for the
            # last one before re-driving, or finalize would drop un-ingested windows.
            continue
        await enqueue_job(
            redis,
            "process_document",
            document_id,
            source_version=source_version,
            session_id=document_id,
        )
        finalized += 1
    return finalized
