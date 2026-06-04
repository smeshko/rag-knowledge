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
from arq.worker import func
from sqlalchemy import Integer, cast, delete, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.config import Settings, get_settings
from rag_recipes.ingestion.cron import sweep_stuck_jobs
from rag_recipes.ingestion.pipeline.dedup import (
    CandidateRef,
    compute_candidate_score,
    select_best,
)
from rag_recipes.ingestion.pipeline.extraction import (
    RecipeExtractionOutput,
    run_extraction,
)
from rag_recipes.ingestion.pipeline.pdf_text import (
    EmptyPdfError,
    extract_and_persist_spans,
)
from rag_recipes.ingestion.pipeline.persist import persist_knowledge_item
from rag_recipes.ingestion.pipeline.windows import build_windows
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
from rag_recipes.storage.enums import DocumentStatus, ExtractionRunStatus
from rag_recipes.storage.models.document import Document
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
            async with session_factory() as session:
                # Idempotency guard against duplicate / manual re-enqueue. A
                # second delivery for a document that's already past QUEUED
                # would raise InvalidTransitionError on the first transition
                # below, and the except block would then mark_failed — and
                # since CREATING_SOURCE_SPANS -> FAILED is a legal edge, a
                # *successfully processed* document would be silently flipped
                # to FAILED. No-op instead: an in-flight or wedged doc is the
                # stuck-job cron's responsibility, not a re-delivery's.
                status = await session.scalar(
                    select(Document.status).where(Document.id == document_id)
                )
                if status is None:
                    raise LookupError(f"Document not found: {document_id}")
                if status is not DocumentStatus.QUEUED:
                    logger.info(
                        "process_document skipping %s: status is %s, not queued",
                        document_id,
                        status.value,
                    )
                    return 0
                await transition_to(session, document_id, DocumentStatus.EXTRACTING_TEXT)
                await session.commit()
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
            logger.debug("process_document %s: persisted %d spans", document_id, spans_count)

            # --- LLM extraction → validate → persist → dedup → chunks signal ---
            # Provider seam (mirrors 8.1's in-job extractor/storage construction):
            # tests inject ctx["llm_provider"]; production builds OpenAI from settings.
            provider: LLMProvider = ctx.get("llm_provider") or _build_llm_provider(
                settings, observability
            )

            # Session #3: short-lived transition-only scope. Releases the row lock
            # before the (slow) LLM stage and marks the doc as in extraction so the
            # stuck-job cron sees progress (DECISIONS #5).
            async with session_factory() as session:
                await transition_to(session, document_id, DocumentStatus.EXTRACTING_ITEMS)
                await session.commit()

            # Session #4: ONE transaction (DECISIONS #5). Extract every window,
            # persist + score each candidate, select one winner per recipe, prune
            # the losers, run the (v1 no-op) supersede hook, then advance through
            # validating_items to creating_chunks. Atomic so no reader ever sees a
            # half-deduplicated document or a chunks-ready signal with losers still
            # present.
            async with session_factory() as session:
                spans = await _load_ordered_spans(session, document_id)
                windows = build_windows(
                    spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages
                )
                candidates: list[CandidateRef] = []
                run_ids: set[str] = set()
                for window in windows:
                    run = await run_extraction(
                        session,
                        window,
                        source_version=1,
                        document_id=document_id,
                        provider=provider,
                        observability=observability,
                    )
                    run_ids.add(run.id)
                    if (
                        run.status is not ExtractionRunStatus.SUCCESS
                        or run.output_json is None
                    ):
                        continue
                    parsed = RecipeExtractionOutput.model_validate(run.output_json)
                    for extracted in parsed.items:
                        try:
                            item = await persist_knowledge_item(
                                session,
                                extracted,
                                extraction_run_id=run.id,
                                document_id=document_id,
                                source_version=1,
                                window=window,
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
                        candidates.append(
                            CandidateRef(
                                item_id=item.id,
                                normalized_title=item.normalized_title,
                                candidate_score=compute_candidate_score(
                                    extracted, window_span_ids=window.span_ids
                                ),
                                extraction_run_id=run.id,
                            )
                        )

                chosen, discarded = select_best(candidates)
                logger.info(
                    "dedup for %s: %d candidates, %d chosen, %d discarded",
                    document_id,
                    len(candidates),
                    len(chosen),
                    len(discarded),
                )
                # Log each discard before pruning (observability, DECISIONS #6).
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

                # Supersede hook: no-op for first extraction (the keep-set covers
                # every run from this pass, so nothing older matches). Epic 10.3 /
                # Epic 11 are the real consumers (DECISIONS #4).
                await DocumentRepository(session).supersede_prior_items(
                    document_id, keep_extraction_run_ids=run_ids
                )

                await transition_to(
                    session, document_id, DocumentStatus.VALIDATING_ITEMS
                )
                await transition_to(
                    session, document_id, DocumentStatus.CREATING_CHUNKS
                )
                await session.commit()

            return len(chosen)
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
    functions = [ping_job, func(process_document, name="process_document", max_tries=1)]
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
