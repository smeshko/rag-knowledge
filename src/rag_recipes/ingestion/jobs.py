"""arq worker entry point: `WorkerSettings`, `ping_job`, and shared session helper.

`uv run arq rag_recipes.ingestion.jobs.WorkerSettings` is the documented
process; arq's CLI discovers `WorkerSettings`, builds the connection pool
from `redis_settings`, runs `on_startup` once per process, and dispatches
jobs from `functions`. Phase 7.1 ships only `ping_job` to prove the seam;
Epic 8 adds real ingestion jobs that reuse `langfuse_session_scope`.
"""

from __future__ import annotations

import asyncio
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
from rag_recipes.ingestion.batch import (
    poll_extraction_batches,
    submit_extraction_batches,
)
from rag_recipes.ingestion.cron import sweep_stuck_jobs
from rag_recipes.ingestion.pipeline.chunking import (
    build_chunks,
    persist_chunks_for_ready_items,
)
from rag_recipes.ingestion.pipeline.dedup import (
    CandidateRef,
    compute_candidate_score,
    select_best,
)
from rag_recipes.ingestion.pipeline.embedding import embed_chunks
from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    RecipeExtractionOutput,
    WindowExtraction,
    _render_prompt,
    build_recipe_v1_json_schema,
    call_provider_for_window,
    find_cached_extraction,
    record_window_extraction,
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
    format_window_for_llm,
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
    TraceContext,
    build_provider_observability,
)
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.embeddings.openai import OpenAIEmbeddingProvider
from rag_recipes.providers.errors import (
    EmbeddingTechnicalError,
    FileStorageError,
    LLMTechnicalError,
    PdfExtractionError,
)
from rag_recipes.providers.file_storage.local import LocalFileStorage
from rag_recipes.providers.llm.anthropic import _sanitize_schema
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.registry import build_llm_provider
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionBatchItemStatus,
    ExtractionRunStatus,
    KnowledgeItemStatus,
)
from rag_recipes.storage.models.chunk import Chunk
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_batch_item import ExtractionBatchItem
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
    EmbeddingTechnicalError: "embedding_failed",
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
    ``ctx["llm_provider"]`` without a real API key. Construction itself goes
    through the provider registry (Epic 23.4) — this wrapper survives only to keep
    that ``ctx`` seam and the observability injection at the job boundary.
    """
    return build_llm_provider(settings, observability=observability)


def _build_embedding_provider(
    settings: Settings, observability: ProviderObservability | None
) -> EmbeddingProvider:
    """Construct the production embedding provider from settings.

    Same in-job seam as ``_build_llm_provider``: tests inject
    ``ctx["embedding_provider"]`` (a ``FakeEmbeddingProvider``); production builds
    ``OpenAIEmbeddingProvider`` so ``trace_embedding`` batches are attributed to
    the document's Langfuse session.
    """
    return OpenAIEmbeddingProvider(
        settings.openai_api_key,
        model=settings.embedding_model,
        dimensions=settings.embedding_dimensions,
        batch_size=settings.embedding_batch_size,
        observability=observability,
    )


async def _load_ordered_spans(
    session: AsyncSession, document_id: str, source_version: int
) -> list[SourceSpan]:
    """Load a document's SourceSpans for ``source_version``, ordered by page_start.

    Ordering is ``build_windows``' contract. ``page_start`` lives in the JSONB
    ``locator``; the ``->>`` accessor returns text, cast to int so the ordering is
    numeric (page 10 after page 9, not before).
    """
    result = await session.execute(
        select(SourceSpan)
        .where(
            SourceSpan.document_id == document_id,
            SourceSpan.source_version == source_version,
        )
        .order_by(cast(SourceSpan.locator["page_start"].astext, Integer))
    )
    return list(result.scalars().all())


def _chunked(windows: list[Window], size: int) -> Iterator[list[Window]]:
    """Yield ``windows`` in contiguous slices of at most ``size`` (the batch size)."""
    for start in range(0, len(windows), size):
        yield windows[start : start + size]


def _window_key(span_ids: list[str]) -> str:
    """Identity of the page window a set of source spans forms.

    Keys the window on its span set alone — deliberately *not* on the
    ``input_hash``, which also folds in the prompt/schema version. A reuse pass
    re-extracts the same window under a bumped prompt and must be recognised as
    having covered it; hashing the prompt in would make every reuse window look
    untouched. Ordering is normalised so a span-order change never forges a new
    window identity.
    """
    return " ".join(sorted(span_ids))


async def _run_extraction_batches(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    source_version: int,
    settings: Settings,
    provider: LLMProvider,
    observability: ProviderObservability | None,
) -> frozenset[str]:
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

    Returns the ``_window_key`` of every window this pass actually sent to the
    provider — i.e. all windows minus the ones the skip set short-circuited.
    ``_finalize_extraction`` needs it to tell a full re-extraction from a partial
    resume: only the windows in this set have a fresh candidate that may replace
    what an earlier pass already promoted.
    """
    async with session_factory() as session:
        spans = await _load_ordered_spans(session, document_id, source_version)
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
                        ExtractionRun.source_version == source_version,
                    )
                )
            )
            .scalars()
            .all()
        )
    windows = build_windows(spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages)
    reextracted: set[str] = set()

    semaphore = asyncio.Semaphore(settings.extraction_max_concurrent_windows)

    for batch in _chunked(windows, settings.extraction_commit_batch_size):
        async with session_factory() as session:
            pending: list[Window] = []
            cached_runs: dict[int, ExtractionRun] = {}
            for window in batch:
                if compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION) in done_hashes:
                    # Already extracted in a prior (interrupted) invocation; its
                    # staging candidates are committed and finalize reads them
                    # from the DB, so skip the provider/DB work entirely.
                    continue
                # Recorded before the provider call, so a window whose extraction
                # errors still counts as covered: this pass owns it either way, and
                # a reuse must not carry the prior item forward on a failed retry.
                reextracted.add(_window_key(list(window.span_ids)))
                # Cache lookups stay here, ahead of any dispatch: a hit must never
                # cost a provider call just because extraction went concurrent.
                cached_run = await find_cached_extraction(
                    session,
                    window,
                    source_version=source_version,
                    document_id=document_id,
                    provider=provider,
                )
                if cached_run is not None:
                    cached_runs[len(pending)] = cached_run
                pending.append(window)

            # Phase 1 — provider calls only, up to
            # extraction_max_concurrent_windows in flight. No session touches
            # here: an AsyncSession cannot be shared across concurrent tasks, and
            # the LLM call is ~99% of a window's wall time anyway.
            async def _call(window: Window) -> WindowExtraction:
                async with semaphore:
                    return await call_provider_for_window(
                        window,
                        provider=provider,
                        document_id=document_id,
                        observability=observability,
                    )

            to_call = [w for i, w in enumerate(pending) if i not in cached_runs]
            outcomes = iter(await asyncio.gather(*(_call(w) for w in to_call)))

            # Phase 2 — sequential writes in window order, exactly as before.
            # A provider failure still aborts the job, but only *after* the whole
            # batch is recorded and committed: the siblings' SUCCESS rows are
            # paid for, and they only enter the extraction cache once committed.
            # Raising mid-loop would roll them back with the transaction and
            # spend them again on resume — the exact waste gathering was meant
            # to avoid. The first captured error is re-raised after the commit.
            first_error: LLMTechnicalError | None = None
            for index, window in enumerate(pending):
                run = cached_runs.get(index)
                if run is None:
                    outcome = next(outcomes)
                    run = await record_window_extraction(
                        session,
                        outcome,
                        source_version=source_version,
                        document_id=document_id,
                        provider=provider,
                        raise_on_error=False,
                    )
                    if outcome.error is not None and first_error is None:
                        first_error = outcome.error
                if run.status is not ExtractionRunStatus.SUCCESS or run.output_json is None:
                    continue
                parsed = RecipeExtractionOutput.model_validate(run.output_json)
                for extracted in parsed.items:
                    try:
                        await persist_knowledge_item(
                            session,
                            extracted,
                            extraction_run_id=run.id,
                            document_id=document_id,
                            source_version=source_version,
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
            if first_error is not None:
                raise first_error
    return frozenset(reextracted)


# Non-terminal item statuses: a window with an item in one of these is already
# registered / in flight, so registration skips it. The partial-unique index
# scopes idempotency to exactly these (mirrors the migration WHERE clause).
_NON_TERMINAL_BATCH_ITEM_STATUSES = (
    ExtractionBatchItemStatus.PENDING,
    ExtractionBatchItemStatus.SUBMITTING,
    ExtractionBatchItemStatus.SUBMITTED,
)


async def _register_extraction_batch_items(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    source_version: int,
    settings: Settings,
) -> int:
    """Register each window as a ``PENDING`` ``ExtractionBatchItem`` (Epic 19.2).

    Builds the same request ``run_extraction`` would (so the stored
    ``request_input`` / ``input_hash`` match a synchronous run — the key 19.3 uses
    to write ``ExtractionRun``s and re-drive finalize, DECISIONS #5), but instead
    of calling the provider it persists a ``PENDING`` item carrying the rendered
    prompt **and** the sanitized ``recipe.v1`` schema. No synchronous LLM call.

    Idempotent: skips windows already terminal-extracted (``done_hashes``) or
    already-registered (non-terminal items), and treats the partial-unique
    ``IntegrityError`` as a skip under concurrent / re-delivered runs (DECISIONS
    #4). Does **not** transition status — the doc stays in ``EXTRACTING_ITEMS``.
    Returns the number of items newly registered.
    """
    async with session_factory() as session:
        spans = await _load_ordered_spans(session, document_id, source_version)
        done_hashes = set(
            (
                await session.execute(
                    select(ExtractionRun.input_hash).where(
                        ExtractionRun.document_id == document_id,
                        ExtractionRun.source_version == source_version,
                    )
                )
            )
            .scalars()
            .all()
        )
        registered_hashes = set(
            (
                await session.execute(
                    select(ExtractionBatchItem.input_hash).where(
                        ExtractionBatchItem.document_id == document_id,
                        ExtractionBatchItem.source_version == source_version,
                        ExtractionBatchItem.status.in_(_NON_TERMINAL_BATCH_ITEM_STATUSES),
                    )
                )
            )
            .scalars()
            .all()
        )
    windows = build_windows(spans, settings.pdf_window_size_pages, settings.pdf_overlap_pages)
    # The sanitized schema is identical for every window — build it once and store
    # the same dict on each item so submission is a pure transform (DECISIONS #5).
    request_schema = _sanitize_schema(build_recipe_v1_json_schema())

    registered = 0
    for window in windows:
        input_hash = compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION)
        if input_hash in done_hashes or input_hash in registered_hashes:
            continue
        async with session_factory() as session:
            item = ExtractionBatchItem(
                document_id=document_id,
                source_version=source_version,
                input_hash=input_hash,
                input_source_span_ids=window.span_ids,
                request_input=_render_prompt(format_window_for_llm(window)),
                request_schema=request_schema,
                prompt_version=PROMPT_VERSION,
                schema_version=SCHEMA_VERSION,
                status=ExtractionBatchItemStatus.PENDING,
                batch_id=None,
            )
            session.add(item)
            try:
                await session.flush()
            except IntegrityError:
                # The partial-unique index rejected a duplicate non-terminal item
                # (concurrent / re-delivered registration) — idempotent skip.
                await session.rollback()
                registered_hashes.add(input_hash)
                continue
            # Heartbeat so the batch-aware stuck-job sweep sees progress.
            await session.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(last_progress_at=func.now())
            )
            await session.commit()
        registered_hashes.add(input_hash)
        registered += 1
    return registered


async def _finalize_extraction(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    source_version: int,
    reextracted_windows: frozenset[str] | None = None,
) -> int:
    """Promote staged candidates to final status in one atomic transaction.

    Runs entirely from persisted rows (DECISIONS #3): loads every ``EXTRACTING``
    candidate for ``(document_id, source_version)``, rebuilds ``CandidateRef``s
    from the stored ``candidate_score`` / ``normalized_title``, runs
    ``select_best``, deletes the losers, promotes each winner to ``READY`` /
    ``NEEDS_REVIEW`` re-derived from its stored ``structured_data["warnings"]``
    (DECISIONS #1), runs the supersede hook, and advances ``extracting_items →
    validating_items → creating_chunks``. One transaction so no reader ever sees a
    half-finalized document (the atomicity 9.4's Session #4 gave, now scoped to
    finalize). Returns the number of surviving (chosen) items.

    ``reextracted_windows`` is the set of window keys this pass actually called the
    provider for (``_run_extraction_batches``'s return). Items already promoted at
    this ``source_version`` whose producing window is **not** in that set are loaded
    as candidates alongside the staging rows, because this pass never revisited
    their window and therefore holds no replacement for them.

    That distinction is what separates the two same-version passes. A ``reuse``
    re-extracts every window, so nothing is carried, and the keep-set supersede
    below retires the whole prior pass — including recipes whose titles the new
    extraction no longer produces (Epic 11.1's replacement semantics). A ``resume``
    re-extracts only the windows with no committed ``ExtractionRun`` (the
    ``done_hashes`` skip set), so with only that remainder in the dedup the
    supersede would retire every item the earlier passes of the very same ingest
    had already promoted — the whole book, replaced by the last few windows.
    Carrying the untouched windows' items restores the supersede's premise: every
    live item at this version competed, so anything outside the keep-set genuinely
    lost its title.

    ``None`` means "every window was covered" (the legacy full-pass behaviour).
    ``REJECTED`` rows are never carried (terminal audit record, no resurrection),
    nor are ``INDEXING`` rows (an in-flight approval owns them).
    """
    async with session_factory() as session:
        items = list(
            (
                await session.execute(
                    select(KnowledgeItem).where(
                        KnowledgeItem.document_id == document_id,
                        KnowledgeItem.source_version == source_version,
                        KnowledgeItem.status == KnowledgeItemStatus.EXTRACTING,
                    )
                )
            )
            .scalars()
            .all()
        )
        # Losing a dedup group means deletion for a staging row (nothing else
        # references it) but only a status flip for a carried one: it may hold
        # chunks and is part of the document's audit trail.
        staged_ids = {item.id for item in items}
        if reextracted_windows is not None:
            carried = (
                await session.execute(
                    select(KnowledgeItem, ExtractionRun.input_source_span_ids)
                    .join(ExtractionRun, ExtractionRun.id == KnowledgeItem.extraction_run_id)
                    .where(
                        KnowledgeItem.document_id == document_id,
                        KnowledgeItem.source_version == source_version,
                        KnowledgeItem.status.in_(
                            {
                                KnowledgeItemStatus.READY,
                                KnowledgeItemStatus.NEEDS_REVIEW,
                            }
                        ),
                    )
                )
            ).all()
            items.extend(
                item
                for item, span_ids in carried
                if _window_key(span_ids) not in reextracted_windows
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
        discarded_staged = [ref.item_id for ref in discarded if ref.item_id in staged_ids]
        discarded_promoted = [ref.item_id for ref in discarded if ref.item_id not in staged_ids]
        if discarded_staged:
            await session.execute(
                delete(KnowledgeItem).where(KnowledgeItem.id.in_(discarded_staged))
            )
        if discarded_promoted:
            await session.execute(
                update(KnowledgeItem)
                .where(KnowledgeItem.id.in_(discarded_promoted))
                .values(status=KnowledgeItemStatus.SUPERSEDED)
                .execution_options(synchronize_session=False)
            )

        # Promote winners: re-derive final status from the stored warnings
        # (DECISIONS #1) — empty → ready, any warning → needs_review. Only the
        # staging rows are (re)derived: an already-promoted winner keeps the
        # status it holds, so a review decision taken on it is never overwritten.
        items_by_id = {item.id: item for item in items}
        for ref in chosen:
            item = items_by_id[ref.item_id]
            if item.id not in staged_ids:
                continue
            warnings = item.structured_data.get("warnings") or []
            item.status = (
                KnowledgeItemStatus.NEEDS_REVIEW if warnings else KnowledgeItemStatus.READY
            )

        # Whether this version now holds an *accepted* (READY) winner. This — not
        # merely "chosen is non-empty" — is the gate for both superseding the prior
        # set and (re)building chunks (review #2). A pass whose only winners are
        # NEEDS_REVIEW, or a zero-winner pass, must NOT retire the prior active
        # version: the epic's rule is "supersede only after the new run reaches
        # ready" / "a needs_review/failed re-extraction must not auto-supersede"
        # (DECISIONS #5). A surviving already-promoted READY winner satisfies the
        # gate too — the version is searchable either way, and the supersede is now
        # safe because that winner competed in the dedup above.
        has_ready_winner = any(
            items_by_id[ref.item_id].status is KnowledgeItemStatus.READY for ref in chosen
        )

        # Same-version supersede (Epic 11.1): retire the prior pass's items *at this
        # same source_version*, keeping ONLY the runs whose items survived this
        # pass's dedup (the `chosen` set). This is the reuse case — a same-version
        # re-run replaces the prior same-version pass before chunks are (re)built, so
        # persist_chunks_for_ready_items below never re-chunks the retired items.
        # Sound only because the live same-version items were loaded as candidates
        # above: a run outside the keep-set lost its title to a better copy, it was
        # not merely absent from a partial resume's staging set.
        #
        # Crucially this is scoped to `source_version`: it must NOT touch a different
        # (prior-active) version's live items. The cross-version active-version
        # handoff — flipping active_source_version and superseding the prior version
        # — happens later, at the READY gate in _index_and_finalize, so a
        # post-finalize embedding/index failure can never retire the prior version
        # while leaving the new one unsearchable (review #1).
        #
        # Gate on an accepted READY replacement: a pass with no ready winner must NOT
        # supersede — that would retire the prior live ready set with nothing
        # searchable to replace it. Skip the call so the prior set survives until a
        # pass actually produces a ready replacement.
        if has_ready_winner:
            keep_run_ids = {ref.extraction_run_id for ref in chosen}
            await DocumentRepository(session).supersede_prior_items(
                document_id,
                keep_extraction_run_ids=keep_run_ids,
                source_version=source_version,
            )

        await transition_to(session, document_id, DocumentStatus.VALIDATING_ITEMS)
        doc = await transition_to(session, document_id, DocumentStatus.CREATING_CHUNKS)
        # Phase 10.1: build + persist the chunks for the surviving ready items in
        # the SAME transaction that lands the document in CREATING_CHUNKS, so the
        # status flip and the chunks commit atomically. This makes the stage
        # crash-safe: a failure before commit rolls finalize back to EXTRACTING_ITEMS
        # (the resumable entry point), and a committed CREATING_CHUNKS document
        # always has its chunks — never the partial state a separate post-finalize
        # transaction would leave. The onward CREATING_CHUNKS -> EMBEDDING_CHUNKS
        # transition is Phase 10.2. `doc` is the row transition_to just locked, so
        # its `category` is read without a second query. The chunk INSERTs do not
        # collide with the discarded-candidate DELETE above: that DELETE already
        # executed (removing the losers) before any chunk is built for a winner.
        #
        # Build chunks only when this pass produced a ready replacement. Without the
        # gate a zero-/needs_review-winner reuse (which preserves the prior ready
        # set above) would re-enter persist_chunks_for_ready_items and append a
        # SECOND set of chunks for the already-chunked prior items (review #2). When
        # there is no ready winner the prior items keep their existing chunks
        # untouched; on a fresh run with no ready winner there were no chunks to
        # build anyway, so the gate is a no-op there.
        if has_ready_winner:
            chunk_count = await persist_chunks_for_ready_items(
                session,
                document_id=document_id,
                source_version=source_version,
                category=doc.category,
            )
        else:
            chunk_count = 0
        # Advance the stuck-job heartbeat: the post-extraction stages
        # (chunking/embedding/indexing) are no longer sweep-exempt, so each must
        # report progress or a long-but-healthy run would be reaped on the stale
        # last_progress_at frozen at the final extraction batch (review #1).
        await session.execute(
            update(Document).where(Document.id == document_id).values(last_progress_at=func.now())
        )
        logger.info("created %d chunks for %s", chunk_count, document_id)
        await session.commit()
        return len(chosen)


async def _embed_document_chunks(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    source_version: int,
    provider: EmbeddingProvider,
    batch_size: int,
) -> int:
    """Embed the document's chunks (Phase 10.2): CREATING_CHUNKS → EMBEDDING_CHUNKS.

    Transitions the document into ``EMBEDDING_CHUNKS`` and upserts one
    ``ChunkEmbedding`` per chunk **in one transaction**, so the status flip and
    the embedding rows commit atomically. On an ``EmbeddingTechnicalError`` mid-run
    the whole transaction rolls back — the document stays in ``CREATING_CHUNKS``
    with no embedding rows — and ``process_document``'s handler then marks it
    ``FAILED`` (like every other technical failure). A committed ``EMBEDDING_CHUNKS``
    document therefore always carries its embeddings; there is no exempt
    resting state left half-embedded for a crash to expose. Leaves the document in
    ``EMBEDDING_CHUNKS`` (the onward ``INDEXING``/terminal transition is Phase 10.3).
    Returns the number of embeddings written. ``embed_chunks`` upserts on
    ``(chunk_id, provider, model)``, so a replay is idempotent.
    """
    async with session_factory() as session:
        await transition_to(session, document_id, DocumentStatus.EMBEDDING_CHUNKS)
        # Embed only chunks whose parent KnowledgeItem is READY *and at this run's
        # source_version*. The READY filter excludes a reuse run's superseded prior
        # items (their chunks are retained for audit, not re-embedded). The
        # source_version filter isolates a new_source_version run to its own chunks:
        # during that run the prior active version is still READY (its supersede +
        # the active flip happen later, at the READY gate), so without it a v(new)
        # run would re-embed the prior version's chunks — wasted spend, and a stray
        # EmbeddingTechnicalError on those old chunks would falsely fail this run
        # (review #2). On a fresh run every chunk is READY-parented at this version.
        result = await session.execute(
            select(Chunk)
            .join(KnowledgeItem, Chunk.parent_id == KnowledgeItem.id)
            .where(
                Chunk.document_id == document_id,
                KnowledgeItem.source_version == source_version,
                KnowledgeItem.status == KnowledgeItemStatus.READY,
            )
        )
        chunks = list(result.scalars().all())
        embeddings = await embed_chunks(
            session,
            chunks,
            provider=provider,
            batch_size=batch_size,
            trace_context=TraceContext(session_id=document_id),
        )
        # Advance the stuck-job heartbeat (see _finalize_extraction): EMBEDDING_CHUNKS
        # is no longer sweep-exempt, so a freshly-embedded document must carry a fresh
        # last_progress_at or the sweep would reap it on the stale extraction heartbeat.
        await session.execute(
            update(Document).where(Document.id == document_id).values(last_progress_at=func.now())
        )
        await session.commit()
    logger.info("embedded %d chunks for %s", len(embeddings), document_id)
    return len(embeddings)


async def _index_and_finalize(
    session_factory: async_sessionmaker[AsyncSession],
    *,
    document_id: str,
    source_version: int,
) -> DocumentStatus:
    """Mark indexed and land the terminal status (Phase 10.3).

    The FTS GIN and HNSW indexes are DB-managed (auto-updated on insert by the
    10.3 migration), so there is no per-document indexing work — ``INDEXING`` is a
    transient marker. Transitions ``EMBEDDING_CHUNKS → INDEXING`` then to the
    terminal status (``READY`` when the document has at least one chunk, else
    ``NEEDS_REVIEW`` — chunks only ever come from ready items per 10.1, so "≥ 1
    chunk" is the searchability signal, DECISIONS #3) **in one transaction**, so a
    pre-commit crash rolls back to the resumable ``EMBEDDING_CHUNKS`` and the
    terminal status is reached atomically. Returns the terminal status.

    This is also the active-version handoff gate (Epic 11.2): reaching ``READY``
    is the genuine "this version is live and searchable" point. The flip of
    ``active_source_version`` to ``source_version`` and the supersede of every
    *other* version's items happen here, atomically, only on ``READY`` — so a
    failure anywhere before this (extraction OR embedding) never retires the prior
    version (review #1). On ``NEEDS_REVIEW`` neither happens, so the prior active
    version stays live.
    """
    async with session_factory() as session:
        await transition_to(session, document_id, DocumentStatus.INDEXING)
        # Count only chunks whose parent KnowledgeItem is READY *and at this run's
        # source_version*. Two reasons: (1) a reuse run (Epic 11.1) can supersede the
        # prior pass's items while their chunks are retained for audit — those must
        # not make the document look searchable. (2) A new_source_version run's
        # searchability is THIS version's chunks: if its winners are all NEEDS_REVIEW
        # (no v(new) chunks) the document must land NEEDS_REVIEW even though the prior
        # version's chunks still exist, so the active-version handoff below does not
        # fire and the prior version stays live (review #1). On a fresh run every
        # chunk is READY-parented at this version, so the filter is a no-op there.
        chunk_count = await session.scalar(
            select(func.count())
            .select_from(Chunk)
            .join(KnowledgeItem, Chunk.parent_id == KnowledgeItem.id)
            .where(
                Chunk.document_id == document_id,
                KnowledgeItem.source_version == source_version,
                KnowledgeItem.status == KnowledgeItemStatus.READY,
            )
        )
        terminal = (
            DocumentStatus.READY
            if chunk_count and chunk_count >= 1
            else DocumentStatus.NEEDS_REVIEW
        )
        doc = await transition_to(session, document_id, terminal)
        if terminal is DocumentStatus.READY:
            # Active-version handoff: only now that this version is searchable do we
            # flip the active version to it and retire every *other* version's items.
            # The keep-set is every run at THIS source_version, so a new_source_version
            # run supersedes the prior active version's items while a fresh/reuse run
            # (whose only items are at this version) retires nothing extra. Co-located
            # with the flip in this one transaction so the two never diverge. (The
            # same-version reuse supersede already ran in _finalize_extraction; this
            # is idempotent over it.)
            keep_run_ids = set(
                (
                    await session.execute(
                        select(ExtractionRun.id).where(
                            ExtractionRun.document_id == document_id,
                            ExtractionRun.source_version == source_version,
                        )
                    )
                )
                .scalars()
                .all()
            )
            await DocumentRepository(session).supersede_prior_items(
                document_id, keep_extraction_run_ids=keep_run_ids
            )
            doc.active_source_version = source_version
            await session.flush()
        await session.commit()
    logger.info("finalized %s -> %s (%d chunks)", document_id, terminal.value, chunk_count or 0)
    return terminal


async def _resume_or_fresh(
    session: AsyncSession, document_id: str, *, reuse_source_spans: bool = False
) -> str:
    """Decide how to (re-)enter ``process_document`` from the document's status.

    Returns ``"fresh"`` for a ``QUEUED`` doc (run text → spans → extraction),
    ``"reuse"`` for a ``QUEUED`` doc dispatched with ``reuse_source_spans=True``
    (a reuse reprocess — skip the PDF text stage and re-run item extraction over
    the existing spans, Epic 11.1), ``"resume"`` for one already in
    ``EXTRACTING_ITEMS`` (a re-driven job after a kill/timeout — skip straight to
    the idempotent window loop, which the ``input_hash`` skip set makes safe),
    ``"embed"`` for one in ``CREATING_CHUNKS`` (a re-driven job after a crash
    between the atomic finalize commit and the embedding commit — its chunks are
    persisted, so resume straight into the idempotent embedding stage), or
    ``"skip"`` for any other status (a duplicate delivery of an in-flight or
    completed doc; the caller no-ops rather than risk flipping a good row to
    FAILED). Raises ``LookupError`` when the document does not exist.
    """
    status = await session.scalar(select(Document.status).where(Document.id == document_id))
    if status is None:
        raise LookupError(f"Document not found: {document_id}")
    if status is DocumentStatus.QUEUED:
        return "reuse" if reuse_source_spans else "fresh"
    if status is DocumentStatus.EXTRACTING_ITEMS:
        return "resume"
    if status is DocumentStatus.CREATING_CHUNKS:
        return "embed"
    if status is DocumentStatus.EMBEDDING_CHUNKS:
        return "index"
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
    source_version: int = 1,
    reuse_source_spans: bool = False,
    batch_mode: bool = False,
    _session_id: str | None = None,
) -> int:
    """Run a Document's full ingestion lifecycle to its terminal status.

    Drives queued → extracting_text → creating_source_spans → extracting_items →
    validating_items → creating_chunks → embedding_chunks → indexing → ready
    (or needs_review when zero chunks were produced). Each stage commits its own
    transaction; the staged flow has resumable entry points (``_resume_or_fresh``)
    so a re-driven job after a crash continues from the last committed status
    rather than re-running completed work. On any documented exception, calls
    mark_failed with a structured reason then re-raises so arq's result store
    reflects the failure too. Langfuse session defaults to ``document_id`` (the
    project's session_id == document_id convention) when the caller didn't pass
    ``_session_id``.

    When ``reuse_source_spans`` is true (Epic 11.1 reuse reprocess), the PDF text
    stage is skipped entirely: the document transitions ``QUEUED →
    EXTRACTING_ITEMS`` directly (it never enters ``EXTRACTING_TEXT`` /
    ``CREATING_SOURCE_SPANS`` and ``extract_and_persist_spans`` is not called),
    and item extraction re-runs over the already-persisted ``SourceSpan`` rows for
    ``source_version``. Finalize supersedes the prior pass's items (see
    ``_finalize_extraction``); no new spans are written.
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
                entry = await _resume_or_fresh(
                    session, document_id, reuse_source_spans=reuse_source_spans
                )
                if entry == "skip":
                    return 0
                if entry == "fresh":
                    await transition_to(session, document_id, DocumentStatus.EXTRACTING_TEXT)
                    await session.commit()
                elif entry == "reuse":
                    # Reuse reprocess (Epic 11.1): skip the PDF text stage. Jump
                    # straight to item extraction over the existing spans via the
                    # QUEUED -> EXTRACTING_ITEMS edge — the doc never enters
                    # EXTRACTING_TEXT / CREATING_SOURCE_SPANS.
                    await transition_to(session, document_id, DocumentStatus.EXTRACTING_ITEMS)
                    await session.commit()

            # Staged flow with resumable entry points: the entry selects the
            # starting stage and all paths converge on the terminal index stage.
            # fresh/resume -> extraction(+finalize) -> embed -> index; embed (resume
            # from creating_chunks) -> embed -> index; index (resume from
            # embedding_chunks) -> index. Each stage commits atomically, so a crash
            # rolls back to the resumable status it started from.
            chosen_count = 0

            if entry in ("fresh", "resume", "reuse"):
                if entry == "fresh":
                    async with session_factory() as session:
                        spans_count = await extract_and_persist_spans(
                            session,
                            document_id=document_id,
                            source_version=source_version,
                            extractor=extractor,
                            storage=storage,
                            extractor_identity=settings.pdf_text_extractor,
                        )
                        await transition_to(
                            session, document_id, DocumentStatus.CREATING_SOURCE_SPANS
                        )
                        await session.commit()
                    logger.debug(
                        "process_document %s: persisted %d spans",
                        document_id,
                        spans_count,
                    )

                if entry == "fresh":
                    # Session #3: short-lived transition-only scope. Releases the row
                    # lock before the (slow) LLM stage and marks the doc as in
                    # extraction so the stuck-job cron sees progress. On resume the
                    # doc is already in EXTRACTING_ITEMS, so this is skipped.
                    async with session_factory() as session:
                        await transition_to(session, document_id, DocumentStatus.EXTRACTING_ITEMS)
                        await session.commit()
                elif entry == "reuse":
                    # The reuse entry already landed the doc in EXTRACTING_ITEMS
                    # (skipping the text stage), so just run the item loop over the
                    # existing v=source_version spans.
                    logger.info(
                        "reuse reprocess for %s: re-extracting items over existing "
                        "v%d spans (text stage skipped)",
                        document_id,
                        source_version,
                    )
                else:
                    logger.info("resuming extraction for %s from extracting_items", document_id)

                # Epic 19.2 batch path: register each window as a PENDING
                # ExtractionBatchItem (no synchronous LLM call) and return, leaving
                # the doc parked in EXTRACTING_ITEMS for the cron submitter + 19.3
                # poller to drive. No finalize/embed/index here.
                if batch_mode:
                    registered = await _register_extraction_batch_items(
                        session_factory,
                        document_id=document_id,
                        source_version=source_version,
                        settings=settings,
                    )
                    logger.info(
                        "registered %d batch item(s) for %s; parked in "
                        "EXTRACTING_ITEMS for batch submission",
                        registered,
                        document_id,
                    )
                    return registered

                # LLM extraction -> validate -> persist -> dedup -> chunks. Provider
                # seam (mirrors 8.1's in-job extractor/storage construction): tests
                # inject ctx["llm_provider"]; production builds the provider from
                # settings (OpenAI or Anthropic per llm_provider).
                provider: LLMProvider = ctx.get("llm_provider") or _build_llm_provider(
                    settings, observability
                )

                # Phase 9.5: extract every window committing per batch (durable,
                # heartbeated progress; DECISIONS #4), then promote the staged
                # candidates in one atomic finalize transaction (dedup from persisted
                # rows; DECISIONS #1, #3). Both run inside this try/except so a
                # batch-level LLMTechnicalError still routes to mark_failed.
                reextracted_windows = await _run_extraction_batches(
                    session_factory,
                    document_id=document_id,
                    source_version=source_version,
                    settings=settings,
                    provider=provider,
                    observability=observability,
                )
                chosen_count = await _finalize_extraction(
                    session_factory,
                    document_id=document_id,
                    source_version=source_version,
                    reextracted_windows=reextracted_windows,
                )

            if entry in ("fresh", "resume", "reuse", "embed"):
                # Phase 10.2: embed the persisted chunks (CREATING_CHUNKS ->
                # EMBEDDING_CHUNKS). The "embed" entry resumes here after a crash
                # between the finalize commit and the embedding commit (chunks
                # persisted; embed_chunks upserts, so the replay is idempotent).
                # Provider seam mirrors the LLM one.
                if entry == "embed":
                    logger.info("resuming embedding for %s from creating_chunks", document_id)
                embedding_provider: EmbeddingProvider = ctx.get(
                    "embedding_provider"
                ) or _build_embedding_provider(settings, observability)
                await _embed_document_chunks(
                    session_factory,
                    document_id=document_id,
                    source_version=source_version,
                    provider=embedding_provider,
                    batch_size=settings.embedding_batch_size,
                )

            if entry == "index":
                # Resume after a crash between the embedding commit and the terminal
                # commit: the embeddings are persisted at EMBEDDING_CHUNKS, so run
                # only the idempotent index + terminal stage.
                logger.info("resuming indexing for %s from embedding_chunks", document_id)

            # Phase 10.3: mark indexed and land the terminal status (READY with
            # >= 1 chunk, else NEEDS_REVIEW) atomically, completing the lifecycle.
            # Also the Epic 11.2 active-version handoff gate (flip + cross-version
            # supersede on READY only).
            await _index_and_finalize(
                session_factory, document_id=document_id, source_version=source_version
            )
            return chosen_count
        except (
            EmptyPdfError,
            PdfExtractionError,
            FileStorageError,
            LLMTechnicalError,
            EmbeddingTechnicalError,
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
                    logger.warning("mark_failed rejected for %s (already terminal)", document_id)
                    await session.rollback()
            raise


async def index_knowledge_item(
    ctx: dict[str, Any],
    item_id: str,
    *,
    _session_id: str | None = None,
) -> int:
    """Index one approved knowledge item: ``indexing → ready`` + chunk + embed
    + active-version handoff, in a single transaction (Epic 21.3, plan D1).

    The approve-side worker for ``POST /knowledge-items/{item_id}/review``.
    Everything commits atomically, so a provider failure rolls back to
    ``indexing`` and arq retries (``max_tries=3``); if every retry is
    exhausted, the item-level pass in ``sweep_stuck_jobs`` returns the row to
    ``needs_review`` (plan D1b) — never an unrecoverable state.

    Lock order is **Document first, item second** — the same order
    ``_index_and_finalize`` uses (``transition_to`` locks the Document row,
    then ``supersede_prior_items`` bulk-updates items); the inverse order
    could deadlock against a concurrent reprocess READY gate.

    Idempotency: the ``status is INDEXING`` guard makes a re-delivered job for
    a completed (``ready``), reverted (``needs_review``) or superseded item a
    no-op — which is also what makes the D1b sweep and the route's
    compensating revert race-safe against a late-arriving original job.

    Staleness (plan D9 rule 4): a reprocess can land between the route's 409
    guard and this job running, so staleness is re-checked here *before any
    chunking*; a stale item is reverted to ``needs_review`` — never flipped to
    READY, which would commit a chunked-but-permanently-unsearchable row (the
    forward-only handoff refuses to move the active version backwards while
    search requires version equality, ``retrieval/_sql.py``).

    Handoff (plan D1a, mirrors ``_index_and_finalize``'s READY gate,
    forward-only per D9): when ``active_source_version`` is NULL (a doc that
    landed terminal NEEDS_REVIEW never reached the READY gate) or older than
    the item's version, flip it to the item's version and supersede every
    other version's items (rejected/indexing rows are spared by the
    repository guard). Equal → no-op; **older item → no-op** — a symmetric
    ``!=`` would supersede the newer live generation.

    Langfuse scope opens on ``_session_id`` (the route passes the document id
    per the session==document convention) with ``item_id`` as the fallback —
    the document id is unknown until the item is loaded inside the
    transaction, so the scope cannot be opened on it directly.

    Returns the number of chunk rows carried by the item on success, 0 on any
    no-op path.
    """
    session_factory = ctx["session_factory"]
    settings: Settings = ctx["settings"]
    observability = ctx.get("observability")

    with langfuse_session_scope(observability, _session_id or item_id):
        async with session_factory() as session:
            # Lock-free resolve of the parent document id, so the Document can
            # be locked FIRST (see docstring).
            document_id = await session.scalar(
                select(KnowledgeItem.document_id).where(KnowledgeItem.id == item_id)
            )
            if document_id is None:
                logger.warning("index_knowledge_item: item %s not found", item_id)
                return 0
            doc = await session.get(Document, document_id, with_for_update=True)
            if doc is None:
                logger.warning(
                    "index_knowledge_item: document %s vanished for item %s",
                    document_id,
                    item_id,
                )
                return 0
            item = await session.get(KnowledgeItem, item_id, with_for_update=True)
            if item is None or item.status is not KnowledgeItemStatus.INDEXING:
                logger.info(
                    "index_knowledge_item: no-op for %s (status %s)",
                    item_id,
                    "missing" if item is None else item.status.value,
                )
                return 0

            # Run-time staleness re-check (D9 rule 4) — BEFORE any chunking.
            current_version = await session.scalar(
                select(func.max(KnowledgeItem.source_version)).where(
                    KnowledgeItem.document_id == document_id
                )
            )
            if current_version is not None and item.source_version < current_version:
                item.status = KnowledgeItemStatus.NEEDS_REVIEW
                await session.commit()
                logger.warning(
                    "index_knowledge_item: item %s went stale (v%d < v%d); "
                    "reverted to needs_review without chunking",
                    item_id,
                    item.source_version,
                    current_version,
                )
                return 0

            # Flip before building: build_chunks returns [] for non-READY.
            item.status = KnowledgeItemStatus.READY
            # Defensive idempotency: needs_review-born items never have chunks;
            # the guard covers hand-seeded/degraded rows.
            existing_chunk = await session.scalar(
                select(Chunk.id).where(Chunk.parent_id == item_id).limit(1)
            )
            if existing_chunk is None:
                session.add_all(build_chunks(item, category=doc.category))
                await session.flush()
            chunks = list(
                (await session.execute(select(Chunk).where(Chunk.parent_id == item_id)))
                .scalars()
                .all()
            )
            provider = ctx.get("embedding_provider") or _build_embedding_provider(
                settings, observability
            )
            # Upserts on (chunk_id, provider, model) — replay-idempotent.
            await embed_chunks(
                session,
                chunks,
                provider=provider,
                batch_size=settings.embedding_batch_size,
                trace_context=TraceContext(session_id=document_id),
            )

            # Forward-only active-version handoff (D1a/D9).
            if doc.active_source_version is None or item.source_version > doc.active_source_version:
                keep_run_ids = set(
                    (
                        await session.execute(
                            select(ExtractionRun.id).where(
                                ExtractionRun.document_id == document_id,
                                ExtractionRun.source_version == item.source_version,
                            )
                        )
                    )
                    .scalars()
                    .all()
                )
                await DocumentRepository(session).supersede_prior_items(
                    document_id, keep_extraction_run_ids=keep_run_ids
                )
                doc.active_source_version = item.source_version
                await session.flush()
            await session.commit()
    logger.info(
        "index_knowledge_item: indexed %s (%d chunks) for document %s",
        item_id,
        len(chunks),
        document_id,
    )
    return len(chunks)


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
    functions = [
        ping_job,
        arq_func(
            process_document,
            name="process_document",
            max_tries=3,
            # Its own timeout, not the worker-wide `job_timeout` below: a full
            # book's sequential window loop outlives the default several times
            # over (see Settings.document_job_timeout_seconds).
            timeout=_SETTINGS.document_job_timeout_seconds,
        ),
        arq_func(index_knowledge_item, name="index_knowledge_item", max_tries=3),
    ]
    cron_jobs = [
        cron(
            sweep_stuck_jobs,
            minute=set(range(0, 60, _SETTINGS.stuck_job_check_interval_minutes)),
            run_at_startup=False,
            unique=True,
            max_tries=1,
            timeout=_SETTINGS.stuck_job_timeout_minutes * 60,
        ),
        cron(
            submit_extraction_batches,
            minute=set(range(0, 60, _SETTINGS.anthropic_batch_submit_interval_minutes)),
            run_at_startup=False,
            unique=True,
            max_tries=1,
            timeout=_SETTINGS.worker_job_timeout_seconds,
        ),
        cron(
            poll_extraction_batches,
            minute=set(range(0, 60, _SETTINGS.anthropic_batch_poll_interval_minutes)),
            run_at_startup=False,
            unique=True,
            max_tries=1,
            timeout=_SETTINGS.worker_job_timeout_seconds,
        ),
    ]
    redis_settings = _build_redis_settings(_SETTINGS)
    on_startup = on_startup
    on_shutdown = on_shutdown
    max_jobs = _SETTINGS.worker_max_jobs
    job_timeout = _SETTINGS.worker_job_timeout_seconds
    keep_result = _SETTINGS.worker_keep_result_seconds
    health_check_interval = _SETTINGS.worker_health_check_interval_seconds
