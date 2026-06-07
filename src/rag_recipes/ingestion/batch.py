"""Cron submitter for pending extraction batches (Epic 19.2).

``submit_extraction_batches`` drains ``PENDING`` ``ExtractionBatchItem``s into
Anthropic Message Batches using a **durable claim-before-call protocol** so a
crash/commit-failure after the provider accepts a batch can never re-submit /
double-charge (DECISIONS #7):

1. **Reconcile** stale ``SUBMITTING`` batches (a prior run crashed between the
   provider call and the success commit). Confirm acceptance via a recent-batch
   list-match → finalize to ``SUBMITTED``; otherwise (the terminal, always-
   terminating fallback) revert to ``PENDING`` for re-submission. Re-submitting a
   chunk Anthropic *did* accept is wasteful but not corrupting — 19.3 ingests
   results idempotently keyed on ``input_hash``.
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

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from rag_recipes.config import Settings
from rag_recipes.ingestion.pipeline.extraction import SCHEMA_VERSION
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.anthropic_batch import (
    AnthropicBatchProvider,
    BatchExtractionRequest,
)
from rag_recipes.storage.enums import ExtractionBatchItemStatus, ExtractionBatchStatus
from rag_recipes.storage.models.extraction_batch import ExtractionBatch
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem

logger = logging.getLogger(__name__)

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
