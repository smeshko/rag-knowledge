"""arq worker entry point: `WorkerSettings`, `ping_job`, and shared session helper.

`uv run arq rag_recipes.ingestion.jobs.WorkerSettings` is the documented
process; arq's CLI discovers `WorkerSettings`, builds the connection pool
from `redis_settings`, runs `on_startup` once per process, and dispatches
jobs from `functions`. Phase 7.1 ships only `ping_job` to prove the seam;
Epic 8 adds real ingestion jobs that reuse `langfuse_session_scope`.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from arq.cron import cron
from arq.worker import func as arq_func
from sqlalchemy import Integer, cast, delete, func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from rag_recipes.config import Settings, get_settings
from rag_recipes.ingestion.cron import sweep_stuck_jobs
from rag_recipes.ingestion.pipeline.chunking import persist_chunks_for_ready_items
from rag_recipes.ingestion.pipeline.dedup import (
    CandidateRef,
    compute_candidate_score,
    select_best,
)
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
    run_extraction,
)
from rag_recipes.ingestion.pipeline.pdf_text import (
    EmptyPdfError,
    extract_and_persist_spans,
)
from rag_recipes.ingestion.pipeline.persist import persist_knowledge_item
from rag_recipes.ingestion.pipeline.windows import (
    Window,
    build_windows,
    compute_input_hash,
)
from rag_recipes.ingestion.queue import _build_redis_settings
from rag_recipes.ingestion.status import (
    InvalidTransitionError,
    mark_failed,
    transition_to,
)
from rag_recipes.ingestion.validation import HardValidationError
from rag_recipes.providers._observability import (
    ProviderObservability,
    build_provider_observability,
)
from rag_recipes.providers.errors import (
    FileStorageError,
    LLMTechnicalError,
    PdfExtractionError,
)
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
)
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.knowledge_item import KnowledgeItem
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.session import build_engine, build_session_factory

logger = logging.getLogger(__name__)


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


# Maps a failure's exception type to the structured `reason` recorded on the
# IngestionFailure row. isinstance-checked in insertion order, so list the most
# specific types first (all entries here are disjoint today).
_REASON_FOR: dict[type[Exception], str] = {
    EmptyPdfError: "pdf_empty",
    PdfExtractionError: "pdf_extraction_failed",
    FileStorageError: "file_storage_error",
    LLMTechnicalError: "llm_extraction_failed",
    IntegrityError: "duplicate_span_constraint",
    LookupError: "document_or_asset_not_found",
    InvalidTransitionError: "invalid_status_transition",
}


def _reason_for(exc: BaseException) -> str:
    for cls, reason in _REASON_FOR.items():
        if isinstance(exc, cls):
            return reason
    return "unknown_error"


def _build_llm_provider(
    settings: Settings, observability: ProviderObservability | None
) -> LLMProvider:
    """Construct the production LLM provider from settings.

    Mirrors how 8.1 builds ``PyMuPdfExtractor`` / ``LocalFileStorage`` in-job; the
    seam lets an integration test substitute a ``FakeLLMProvider`` via
    ``ctx["llm_provider"]`` without a real API key.
    """
    return OpenAILLMProvider(
        settings.openai_api_key,
        default_model=settings.llm_model,
        observability=observability,
        max_rate_limit_retries=settings.llm_max_rate_limit_retries,
        request_timeout=settings.llm_request_timeout_seconds,
    )


async def _load_ordered_spans(session: AsyncSession, document_id: str) -> list[SourceSpan]:
    """Load a document's v1 SourceSpans ordered by page_start (build_windows' contract).

    ``page_start`` lives in the JSONB ``locator``; the ``->>`` accessor returns
    text, cast to int so the ordering is numeric (page 10 after page 9, not before).
    """
    result = await session.execute(
        select(SourceSpan)
        .where(
            SourceSpan.document_id == document_id,
            SourceSpan.source_version == 1,
        )
        .order_by(cast(SourceSpan.locator["page_start"].astext, Integer))
    )
    return list(result.scalars().all())


def _chunked(windows: list[Window], size: int) -> Iterator[list[Window]]:
    """Yield ``windows`` in contiguous slices of at most ``size`` (the batch size)."""
    for start in range(0, len(windows), size):
        yield windows[start : start + size]


async def _run_extraction_batches(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    settings: Settings,
    provider: LLMProvider,
    observability: ProviderObservability | None,
) -> None:
    """Extract every window, committing per batch (Phase 9.5, DECISIONS #4).

    Loads the document's ordered spans, builds the overlapping page windows, then
    processes them in chunks of ``settings.extraction_commit_batch_size``. Each
    batch is one transaction: its ``ExtractionRun``s + staging ``KnowledgeItem``s
    (status ``EXTRACTING``, ``candidate_score`` set) + a ``last_progress_at``
    heartbeat commit together. A crash rolls back only the in-flight batch, so at
    most ``batch_size - 1`` windows of OpenAI spend are repeated on resume. The
    heartbeat advances on every batch commit (server ``func.now()``, never via
    ``onupdate``) so the progress-aware stuck-job sweep can tell a slow-but-healthy
    run from a hung one.
    """
    async with session_factory() as session:
        spans = await _load_ordered_spans(session, document_id)
        # Resume skip set (DECISIONS #2): the input_hashes that already have a
        # committed ExtractionRun for this (document_id, source_version). Keyed on
        # the committed audit fact — what survived a prior batch commit — so a
        # re-driven job never re-extracts (and never re-persists a duplicate
        # candidate for) a window it already finished. Robust to run_extraction
        # minting an extra run on its cross-document SUCCESS cache hit.
        done_hashes = set(
            (
                await session.execute(
                    select(ExtractionRun.input_hash).where(
                        ExtractionRun.document_id == document_id,
                        ExtractionRun.source_version == 1,
                    )
                )
            )
            .scalars()
            .all()
        )
    windows = build_windows(
        spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
    )

    for batch in _chunked(windows, settings.extraction_commit_batch_size):
        async with session_factory() as session:
            for window in batch:
                if compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION) in done_hashes:
                    # Already extracted in a prior (interrupted) invocation; its
                    # staging candidates are committed and finalize reads them
                    # from the DB, so skip the provider/DB work entirely.
                    continue
                run = await run_extraction(
                    session,
                    window,
                    source_version=1,
                    document_id=document_id,
                    provider=provider,
                    observability=observability,
                )
                if (
                    run.status is not ExtractionRunStatus.SUCCESS
                    or run.output_json is None
                ):
                    continue
                parsed = RecipeExtractionOutput.model_validate(run.output_json)
                for extracted in parsed.items:
                    try:
                        await persist_knowledge_item(
                            session,
                            extracted,
                            extraction_run_id=run.id,
                            document_id=document_id,
                            source_version=1,
                            window=window,
                            staging=True,
                            candidate_score=compute_candidate_score(
                                extracted, window_span_ids=window.span_ids
                            ),
                        )
                    except HardValidationError as exc:
                        # Per-candidate rejection: persist no row, never fail the
                        # document (9.3 DECISIONS #3). Log and move on.
                        logger.info(
                            "dropping hard-invalid candidate for %s: %s",
                            document_id,
                            exc,
                        )
                        continue
            # Heartbeat the batch commit: only a committed batch counts as
            # progress, so this is set explicitly (not via the updated_at onupdate
            # that bumps on any write).
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(last_progress_at=func.now())
            )
            await session.commit()


async def _finalize_extraction(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
) -> int:
    """Promote staged candidates to final status in one atomic transaction.

    Runs entirely from persisted rows (DECISIONS #3): loads every ``EXTRACTING``
    candidate for ``(document_id, source_version=1)``, rebuilds ``CandidateRef``s
    from the stored ``candidate_score`` / ``normalized_title``, runs
    ``select_best``, deletes the losers, promotes each winner to ``READY`` /
    ``NEEDS_REVIEW`` re-derived from its stored ``structured_data["warnings"]``
    (DECISIONS #1), runs the supersede hook, and advances ``extracting_items →
    validating_items → creating_chunks``. One transaction so no reader ever sees a
    half-finalized document (the atomicity 9.4's Session #4 gave, now scoped to
    finalize). Returns the number of surviving (chosen) items.
    """
    async with session_factory() as session:
        items = list(
            (
                await session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.document_id == document_id,
                        KnowledgeItem.source_version == 1,
                        KnowledgeItem.status == KnowledgeItemStatus.EXTRACTING,
                    )
                )
            )
            .scalars()
            .all()
        )
        candidates = [
            CandidateRef(
                item_id=item.id,
                normalized_title=item.normalized_title,
                # candidate_score is written on every staging row; coerce a stray
                # NULL to 0.0 so a row never silently wins on a missing score.
                candidate_score=item.candidate_score or 0.0,
                extraction_run_id=item.extraction_run_id,
            )
            for item in items
        ]
        chosen, discarded = select_best(candidates)
        logger.info(
            "dedup for %s: %d candidates, %d chosen, %d discarded",
            document_id,
            len(candidates),
            len(chosen),
            len(discarded),
        )
        for ref in discarded:
            logger.info(
                "dedup discard for %s: item=%s title=%r score=%.4f",
                document_id,
                ref.item_id,
                ref.normalized_title,
                ref.candidate_score,
            )
        if discarded:
            await session.execute(
                delete(KnowledgeItem).where(
                    KnowledgeItem.id.in_([ref.item_id for ref in discarded])
                )
            )

        # Promote winners: re-derive final status from the stored warnings
        # (DECISIONS #1) — empty → ready, any warning → needs_review.
        items_by_id = {item.id: item for item in items}
        for ref in chosen:
            item = items_by_id[ref.item_id]
            warnings = item.structured_data.get("warnings") or []
            item.status = (
                KnowledgeItemStatus.NEEDS_REVIEW
                if warnings
                else KnowledgeItemStatus.READY
            )

        # Supersede hook: keep every run from this source_version's pass, so v1's
        # first extraction matches nothing older (no-op). Epic 11 reuses this with
        # a new source-version's runs to retire the prior version (DECISIONS #4).
        run_ids = set(
            (
                await session.execute(
                    select(ExtractionRun.id).where(
                        ExtractionRun.document_id == document_id,
                        ExtractionRun.source_version == 1,
                    )
                )
            )
            .scalars()
            .all()
        )
        await DocumentRepository(session).supersede_prior_items(
            document_id, keep_extraction_run_ids=run_ids
        )

        await transition_to(session, document_id, DocumentStatus.VALIDATING_ITEMS)
        await transition_to(session, document_id, DocumentStatus.CREATING_CHUNKS)
        await session.commit()
        return len(chosen)


async def _create_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
) -> int:
    """Build + persist chunks for the document's ready items (Phase 10.1).

    Runs as its own transaction after ``_finalize_extraction`` has landed the
    document in ``CREATING_CHUNKS``: loads the ``Document`` for its ``category``,
    persists one to five chunks per surviving ``READY`` ``KnowledgeItem``, and
    commits — leaving the document in ``CREATING_CHUNKS`` (the onward
    ``EMBEDDING_CHUNKS`` transition is Phase 10.2). Kept a **separate**
    transaction from finalize on purpose: finalize's bulk discarded-candidate
    ``DELETE`` and a dependent chunk ``INSERT`` cannot be ordered against each
    other in one unit-of-work flush, so chunking reads the committed ready rows
    from a clean session instead. Returns the number of chunks written.
    """
    async with session_factory() as session:
        document = await session.get(Document, document_id)
        if document is None:
            raise LookupError(f"Document not found: {document_id}")
        chunk_count = await persist_chunks_for_ready_items(
            session, document_id=document_id, category=document.category
        )
        await session.commit()
    logger.info("created %d chunks for %s", chunk_count, document_id)
    return chunk_count


async def _resume_or_fresh(session: AsyncSession, document_id: str) -> str:
    """Decide how to (re-)enter ``process_document`` from the document's status.

    Returns ``"fresh"`` for a ``QUEUED`` doc (run text → spans → extraction),
    ``"resume"`` for one already in ``EXTRACTING_ITEMS`` (a re-driven job after a
    kill/timeout — skip straight to the idempotent window loop, which the
    ``input_hash`` skip set makes safe), or ``"skip"`` for any other status (a
    duplicate delivery of an in-flight or completed doc; the caller no-ops rather
    than risk flipping a good row to FAILED). Raises ``LookupError`` when the
    document does not exist.
    """
    status = await session.scalar(
        select(Document.status).where(Document.id == document_id)
    )
    if status is None:
        raise LookupError(f"Document not found: {document_id}")
    if status is DocumentStatus.QUEUED:
        return "fresh"
    if status is DocumentStatus.EXTRACTING_ITEMS:
        return "resume"
    logger.info(
        "process_document skipping %s: status is %s, not resumable",
        document_id,
        status.value,
    )
    return "skip"


async def process_document(
    ctx: dict[str, Any],
    document_id: str,
    *,
    _session_id: str | None = None,
) -> int:
    """Extract per-page text from a Document's PDF and persist SourceSpans.

    Transitions queued → extracting_text → creating_source_spans. On any
    documented exception, calls mark_failed with a structured reason then
    re-raises so arq's result store reflects the failure too. Langfuse session
    defaults to ``document_id`` (the project's session_id == document_id
    convention) when the caller didn't pass ``_session_id``.
    """
    session_factory = ctx["session_factory"]
    settings = ctx["settings"]
    observability = ctx.get("observability")

    with langfuse_session_scope(observability, _session_id or document_id):
        storage = LocalFileStorage(Path(settings.local_storage_root))
        extractor = PyMuPdfExtractor(min_text_chars=settings.pdf_min_text_chars_for_page)
        try:
            # Idempotency / resume guard against duplicate or re-driven delivery.
            # A QUEUED doc runs the full fresh path; one already in
            # EXTRACTING_ITEMS is a re-driven job (kill/timeout) and resumes
            # straight into the idempotent window loop; any other status no-ops
            # (an in-flight or completed doc is the stuck-job cron's concern, not
            # a re-delivery's — flipping a good row to FAILED would be worse).
            async with session_factory() as session:
                entry = await _resume_or_fresh(session, document_id)
                if entry == "skip":
                    return 0
                if entry == "fresh":
                    await transition_to(
                        session, document_id, DocumentStatus.EXTRACTING_TEXT
                    )
                    await session.commit()

            if entry == "fresh":
                async with session_factory() as session:
                    spans_count = await extract_and_persist_spans(
                        session,
                        document_id=document_id,
                        source_version=1,
                        extractor=extractor,
                        storage=storage,
                    )
                    await transition_to(
                        session, document_id, DocumentStatus.CREATING_SOURCE_SPANS
                    )
                    await session.commit()
                logger.debug(
                    "process_document %s: persisted %d spans", document_id, spans_count
                )

            # --- LLM extraction → validate → persist → dedup → chunks signal ---
            # Provider seam (mirrors 8.1's in-job extractor/storage construction):
            # tests inject ctx["llm_provider"]; production builds OpenAI from settings.
            provider: LLMProvider = ctx.get("llm_provider") or _build_llm_provider(
                settings, observability
            )

            if entry == "fresh":
                # Session #3: short-lived transition-only scope. Releases the row
                # lock before the (slow) LLM stage and marks the doc as in
                # extraction so the stuck-job cron sees progress. On resume the doc
                # is already in EXTRACTING_ITEMS, so this transition is skipped.
                async with session_factory() as session:
                    await transition_to(
                        session, document_id, DocumentStatus.EXTRACTING_ITEMS
                    )
                    await session.commit()
            else:
                logger.info(
                    "resuming extraction for %s from extracting_items", document_id
                )

            # Phase 9.5: extract every window committing per batch (durable,
            # heartbeated progress; DECISIONS #4), then promote the staged
            # candidates in one atomic finalize transaction (dedup from persisted
            # rows; DECISIONS #1, #3). Both run inside this method's try/except so a
            # batch-level LLMTechnicalError still routes to mark_failed.
            await _run_extraction_batches(
                session_factory,
                document_id=document_id,
                settings=settings,
                provider=provider,
                observability=observability,
            )
            chosen_count = await _finalize_extraction(
                session_factory, document_id=document_id
            )
            # Phase 10.1: chunk the surviving ready items in their own transaction.
            # Finalize already landed the document in CREATING_CHUNKS, so this stage
            # builds + persists chunks and leaves it there (the onward
            # EMBEDDING_CHUNKS transition is Phase 10.2).
            await _create_chunks(session_factory, document_id=document_id)
            return chosen_count
        except (
            EmptyPdfError,
            PdfExtractionError,
            FileStorageError,
            LLMTechnicalError,
            IntegrityError,
            LookupError,
            InvalidTransitionError,
        ) as exc:
            logger.warning("process_document failed for %s: %s", document_id, exc)
            async with session_factory() as session:
                try:
                    await mark_failed(
                        session,
                        document_id,
                        reason=_reason_for(exc),
                        error_message=str(exc),
                        metadata_json={"exc_type": type(exc).__name__},
                    )
                    await session.commit()
                except InvalidTransitionError:
                    # The doc may already be terminal (e.g. the cron got there
                    # first, or a prior mark_failed succeeded). Swallow — the
                    # original failure is what we re-raise.
                    logger.warning(
                        "mark_failed rejected for %s (already terminal)", document_id
                    )
                    await session.rollback()
            raise


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
    functions = [ping_job, arq_func(process_document, name="process_document", max_tries=3)]
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
