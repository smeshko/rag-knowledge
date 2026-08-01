"""POST /api/v1/documents — upload a PDF, register a Document + SourceAsset.

Three failure classes (see PLAN Decisions):

* **Pre-storage** (duplicate-lookup DB error or ``put_object`` raising) —
  nothing durably stored, return ``internal_error`` with no deletion.
* **Post-storage pre-commit** (``add_*``/``flush`` error before
  ``session.commit()``). The orphan is safe to remove: ``rollback`` +
  ``delete_object(key)``. ``IntegrityError`` here is the duplicate race;
  recover by re-fetching the winner.
* **Commit-ambiguous** (``session.commit()`` itself raises). The DB may
  have committed despite the error; deleting the file could strand a
  committed row pointing at a missing PDF. Keep the file, ``rollback``
  best-effort, return ``internal_error``.
"""

from __future__ import annotations

import hashlib
import logging
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

from arq.connections import ArqRedis
from fastapi import APIRouter, Depends, File, Form, UploadFile
from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import (
    get_arq_redis,
    get_file_storage,
    get_session,
    get_settings,
)
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.documents import (
    BatchUploadErrorCode,
    BatchUploadItemResult,
    BatchUploadItemStatus,
    BatchUploadResponse,
    DocumentCounts,
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    IngestionFailureInfo,
    IngestionProgress,
    IngestionStatusResponse,
    ReprocessRequest,
    ReprocessResponse,
    UploadIngestion,
    UploadResponse,
)
from rag_recipes.config import Settings
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.status import is_terminal
from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.storage.enums import DocumentStatus, ReprocessMode, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository
from rag_recipes.storage.repositories.failures import FailuresRepository

router = APIRouter(tags=["documents"])

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_DEFAULT_FILENAME = "upload.pdf"

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0

# doc 6 §5: a document is terminal once it lands in one of these states.
_TERMINAL_DOCUMENT_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED}
)

def _parse_enum[E: StrEnum](
    enum_cls: type[E], raw: str | None, *, field: str
) -> E | None:
    """Coerce an optional string to a `StrEnum` member or raise the
    doc-6 ``invalid_request`` envelope.

    Filter params are typed ``str | None`` rather than enums so FastAPI's raw
    422 never fires before the handler runs — every validation error stays
    inside the ``ApiError`` envelope.
    """
    if raw is None or raw == "":
        return None
    try:
        return enum_cls(raw)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message=f"Invalid value for {field!r}.",
            details={"field": field, "value": raw},
        ) from exc


def _parse_int(
    raw: str | None,
    *,
    field: str,
    default: int,
    minimum: int,
    maximum: int | None = None,
) -> int:
    """Parse an optional integer query param with explicit bounds; raise
    the doc-6 ``invalid_request`` envelope on any failure."""
    if raw is None or raw == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message=f"{field!r} must be an integer.",
            details={"field": field, "value": raw},
        ) from exc
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f">= {minimum}" if maximum is None else f"in [{minimum}, {maximum}]"
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message=f"{field!r} must be {bounds}.",
            details={"field": field, "value": raw},
        )
    return value


async def _best_effort_rollback(session: AsyncSession) -> None:
    try:
        await session.rollback()
    except Exception:
        logger.exception("rollback failed during upload compensation")


async def _best_effort_delete(storage: FileStorageProvider, key: str) -> None:
    try:
        await storage.delete_object(key)
    except Exception:
        # Orphan leak: log the key so it can be reconciled. Do not surface
        # the cleanup error; the caller's primary path (e.g. duplicate-race
        # winner recovery, or returning a 500) must still run.
        logger.exception("failed to delete orphan upload key %s", key)


def _is_pdf(data: bytes) -> bool:
    return data.startswith(_PDF_MAGIC)


def _derive_title(filename: str) -> str:
    stem = Path(filename).stem
    return stem if stem else filename


def _safe_filename(filename: str | None) -> str:
    return filename if filename else _DEFAULT_FILENAME


@router.post("/documents", status_code=201)
async def upload_document(
    file: Annotated[UploadFile | None, File()] = None,
    category: Annotated[str, Form()] = "recipes",
    subcategory: Annotated[str | None, Form()] = None,
    title: Annotated[str | None, Form()] = None,
    author: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    storage: FileStorageProvider = Depends(get_file_storage),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
) -> Any:
    return await _handle_upload(
        file=file,
        category=category,
        subcategory=subcategory,
        title=title,
        author=author,
        language=language,
        session=session,
        storage=storage,
        arq_redis=arq_redis,
    )


async def _handle_upload(
    *,
    file: UploadFile | None,
    category: str,
    subcategory: str | None,
    title: str | None,
    author: str | None,
    language: str | None,
    session: AsyncSession,
    storage: FileStorageProvider,
    arq_redis: ArqRedis,
) -> UploadResponse:
    document, is_duplicate = await _create_document_from_upload(
        file=file,
        category=category,
        subcategory=subcategory,
        title=title,
        author=author,
        language=language,
        session=session,
        storage=storage,
    )
    # Fresh-insert path only — kick off synchronous ingestion. Duplicate / race
    # recoveries must not re-enqueue (they returned an already-processing doc).
    if not is_duplicate:
        try:
            await enqueue_job(
                arq_redis, "process_document", document.id, session_id=document.id
            )
        except Exception:
            # The document row is committed and queued; the 7.2 stuck-job cron
            # will mark it failed if it never gets picked up. Surface the error
            # in logs so an operator can manually re-enqueue. Do NOT fail the 201.
            logger.exception(
                "Failed to enqueue process_document for %s; stuck-job cron will catch it",
                document.id,
            )

    return UploadResponse(
        document=DocumentResponse.model_validate(document),
        ingestion=UploadIngestion(status=document.status.value),
    )


async def _create_document_from_upload(
    *,
    file: UploadFile | None,
    category: str,
    subcategory: str | None,
    title: str | None,
    author: str | None,
    language: str | None,
    session: AsyncSession,
    storage: FileStorageProvider,
) -> tuple[Document, bool]:
    """Validate + dedup + store + create a ``Document`` from an uploaded PDF.

    Returns ``(document, is_duplicate)``: ``is_duplicate`` is ``True`` for a
    content-hash hit or a duplicate-race winner (an already-existing document is
    returned), ``False`` for a fresh insert (committed + refreshed). Raises the
    documented ``ApiError`` envelopes on each failure class (module docstring).
    **Enqueueing is the caller's job** — this helper never starts ingestion, so
    the synchronous (``POST /documents``) and batch (``POST /documents/batch``)
    endpoints can route the created document differently.
    """
    if file is None:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="A 'file' multipart field is required.",
            details={"field": "file"},
        )
    data = await file.read()
    if not data:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="A 'file' multipart field is required.",
            details={"field": "file"},
        )

    if not _is_pdf(data):
        raise ApiError(
            status_code=415,
            code=ErrorCode.UNSUPPORTED_FILE_TYPE,
            message="Only PDF uploads are supported.",
            details={"expected": "application/pdf"},
        )

    content_hash = hashlib.sha256(data).hexdigest()
    repo = DocumentRepository(session)

    # Pre-storage: duplicate lookup. A DB error here means nothing was
    # stored; map to internal_error, no deletion.
    try:
        existing_asset = await repo.get_source_asset_by_content_hash(content_hash)
    except Exception as exc:
        raise ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to check for duplicate uploads.",
        ) from exc

    if existing_asset is not None:
        existing_document = await repo.get_document_by_asset_id(existing_asset.id)
        if existing_document is None:
            raise ApiError(
                status_code=500,
                code=ErrorCode.INTERNAL_ERROR,
                message="Existing SourceAsset has no Document; inconsistent state.",
            )
        return existing_document, True

    asset_id = new_id(SourceAsset.ID_PREFIX)
    key = f"source-assets/{asset_id}/original.pdf"
    original_filename = _safe_filename(file.filename)
    final_title = title if title is not None else _derive_title(original_filename)
    final_author = author if author is not None else ""

    # Pre-storage: storage write. ``LocalFileStorage`` writes atomically,
    # so a failure here leaves nothing durably stored — no deletion.
    try:
        stored = await storage.put_object(key, data, "application/pdf")
    except FileStorageError as exc:
        raise ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to store the uploaded file.",
        ) from exc

    # Post-storage pre-commit work — ``add_*``/``flush``. Errors here are
    # safe to clean up: ``rollback`` + ``delete_object``. The IntegrityError
    # branch recovers the winner of a content-hash race.
    try:
        await repo.add_source_asset(
            id=asset_id,
            source_type=SourceType.PDF,
            original_filename=original_filename,
            storage_provider=stored.storage_provider,
            storage_key=stored.storage_key,
            content_hash=content_hash,
            upload_status=UploadStatus.UPLOADED,
        )
        document = await repo.add_document(
            asset_id=asset_id,
            category=category,
            subcategory=subcategory,
            title=final_title,
            author=final_author,
            source_type=SourceType.PDF,
            language=language,
            active_source_version=None,
            status=DocumentStatus.QUEUED,
        )
    except IntegrityError:
        # Rollback and orphan cleanup are best-effort: a degraded
        # filesystem or session must not abort the winner re-fetch, which
        # is the whole point of the duplicate-race branch. Failures are
        # logged so the orphan key can be reconciled later.
        await _best_effort_rollback(session)
        await _best_effort_delete(storage, key)
        winner = await repo.get_source_asset_by_content_hash(content_hash)
        if winner is None:
            raise ApiError(
                status_code=500,
                code=ErrorCode.INTERNAL_ERROR,
                message="Duplicate race recovery failed: winning SourceAsset not found.",
            ) from None
        winner_document = await repo.get_document_by_asset_id(winner.id)
        if winner_document is None:
            raise ApiError(
                status_code=500,
                code=ErrorCode.INTERNAL_ERROR,
                message="Duplicate race recovery failed: winning Document not found.",
            ) from None
        return winner_document, True
    except Exception as exc:
        await _best_effort_rollback(session)
        await _best_effort_delete(storage, key)
        raise ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to persist the uploaded document.",
        ) from exc

    # Commit-ambiguous: a commit() exception does not prove the txn
    # aborted, so do NOT delete the file. Best-effort rollback, then 500.
    try:
        await session.commit()
    except Exception as exc:
        await _best_effort_rollback(session)
        raise ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Failed to commit the uploaded document.",
        ) from exc

    await session.refresh(document)
    return document, False


def _anthropic_batch_enabled(settings: Settings) -> bool:
    return settings.llm_provider == "anthropic" and bool(settings.anthropic_api_key)


def _batch_error_code(code: ErrorCode) -> BatchUploadErrorCode:
    """Map an ``ApiError.code`` to the batch-item error code (D2).

    By value, with an ``internal_error`` fallback so a future raise site with
    a code outside the reachable set cannot break item construction.
    """
    try:
        return BatchUploadErrorCode(code.value)
    except ValueError:
        return BatchUploadErrorCode.INTERNAL_ERROR


@router.post("/documents/batch", status_code=201)
async def upload_documents_batch(
    files: Annotated[list[UploadFile], File()],
    category: Annotated[str, Form()] = "recipes",
    subcategory: Annotated[str | None, Form()] = None,
    language: Annotated[str | None, Form()] = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    storage: FileStorageProvider = Depends(get_file_storage),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any:
    """Upload a cohort of PDFs and defer extraction into the Anthropic batch path.

    Each file is created exactly as ``POST /documents`` does (shared helper), but
    enqueued with ``batch_mode=True`` so ``process_document`` registers windows
    instead of running synchronous extraction. Per-file failures (non-PDF, etc.)
    yield an ``error`` item without aborting the cohort.

    **Refuses creation unless the Anthropic batch path is enabled** (``llm_provider
    == "anthropic"`` + a key): otherwise the created docs would register windows
    that no submitter could ever drain, stranding them in ``EXTRACTING_ITEMS``
    (DECISIONS #3; Codex round-1 #2).
    """
    if not _anthropic_batch_enabled(settings):
        raise ApiError(
            status_code=409,
            code=ErrorCode.INVALID_REQUEST,
            message=(
                "Batch upload requires the Anthropic batch path "
                "(LLM_PROVIDER=anthropic + ANTHROPIC_API_KEY)."
            ),
        )

    results: list[BatchUploadItemResult] = []
    for file in files:
        filename = _safe_filename(file.filename)
        try:
            document, is_duplicate = await _create_document_from_upload(
                file=file,
                category=category,
                subcategory=subcategory,
                title=None,
                author=None,
                language=language,
                session=session,
                storage=storage,
            )
        except ApiError as exc:
            # Per-file failure (non-PDF, storage/commit error) — record and keep
            # processing the rest of the cohort.
            results.append(
                BatchUploadItemResult(
                    filename=filename,
                    status=BatchUploadItemStatus.ERROR,
                    error=exc.message,
                    error_code=_batch_error_code(exc.code),
                )
            )
            continue

        if is_duplicate:
            results.append(
                BatchUploadItemResult(
                    filename=filename,
                    status=BatchUploadItemStatus.DUPLICATE,
                    document_id=document.id,
                )
            )
            continue

        try:
            await enqueue_job(
                arq_redis,
                "process_document",
                document.id,
                batch_mode=True,
                session_id=document.id,
            )
        except Exception:
            logger.exception(
                "Failed to enqueue batch process_document for %s; stuck-job cron "
                "will catch it",
                document.id,
            )
        results.append(
            BatchUploadItemResult(
                filename=filename,
                status=BatchUploadItemStatus.CREATED,
                document_id=document.id,
            )
        )

    return BatchUploadResponse(
        items=results,
        total=len(results),
        created=sum(1 for r in results if r.status is BatchUploadItemStatus.CREATED),
        duplicates=sum(
            1 for r in results if r.status is BatchUploadItemStatus.DUPLICATE
        ),
        errors=sum(1 for r in results if r.status is BatchUploadItemStatus.ERROR),
    )


@router.get("/documents")
async def list_documents(
    category: str | None = None,
    status: str | None = None,
    source_type: str | None = None,
    limit: str | None = None,
    offset: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    status_enum = _parse_enum(DocumentStatus, status, field="status")
    source_type_enum = _parse_enum(SourceType, source_type, field="source_type")
    limit_int = _parse_int(
        limit,
        field="limit",
        default=_LIST_LIMIT_DEFAULT,
        minimum=1,
        maximum=_LIST_LIMIT_MAX,
    )
    offset_int = _parse_int(
        offset,
        field="offset",
        default=_LIST_OFFSET_DEFAULT,
        minimum=0,
    )
    # `category` is free text per the plan — no enum validation.
    category_value = category if category else None
    repo = DocumentRepository(session)
    documents = await repo.list_documents(
        category=category_value,
        status=status_enum,
        source_type=source_type_enum,
        limit=limit_int,
        offset=offset_int,
    )
    return DocumentListResponse(
        documents=[DocumentListItem.model_validate(doc) for doc in documents],
    )


async def _require_document(repo: DocumentRepository, document_id: str) -> Any:
    document = await repo.get_document(document_id)
    if document is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=f"Document {document_id!r} not found.",
            details={"document_id": document_id},
        )
    return document


@router.get("/documents/{document_id}")
async def get_document(
    document_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    repo = DocumentRepository(session)
    document = await _require_document(repo, document_id)
    item_counts = await repo.count_knowledge_items(document_id)
    counts = DocumentCounts(
        source_spans=await repo.count_source_spans(document_id),
        knowledge_items=item_counts.total,
        ready_items=item_counts.ready,
        needs_review_items=item_counts.needs_review,
        chunks=await repo.count_chunks(document_id),
    )
    return DocumentDetailResponse(
        document=DocumentResponse.model_validate(document),
        counts=counts,
    )


@router.get("/documents/{document_id}/status")
async def get_document_status(
    document_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    """Report ingestion progress for a document.

    For a non-terminal doc the in-flight version is the highest existing span
    version (Epic 11.2, DECISIONS #5) — ``max + 1`` work for a new_source_version
    run already shows up as v(new) spans, while initial ingestion / a reuse run
    stays on the current version; it defaults to ``1`` before any span exists.
    Progress is suppressed only when the in-flight version equals the *already
    active* version — a reuse reprocess re-runs over existing spans without
    creating new ones, so the surviving spans are not this run's progress. A
    new_source_version run (in-flight > active) and initial ingestion (no active
    version) both report the in-flight version's spans. Terminal docs report their
    active version's final counts (a failed doc with active=null gets (0, None)).
    """
    repo = DocumentRepository(session)
    document = await _require_document(repo, document_id)
    is_doc_terminal = is_terminal(document.status)
    if is_doc_terminal:
        current_source_version = None
        version_for_progress = document.active_source_version
    else:
        in_flight = await repo.max_source_version(document_id) or 1
        current_source_version = in_flight
        if (
            document.active_source_version is not None
            and in_flight == document.active_source_version
        ):
            # Reprocess of the active version (reuse) in flight: the surviving
            # active-version spans are NOT this run's progress yet.
            version_for_progress = None
        else:
            version_for_progress = in_flight
    if version_for_progress is None:
        pages_processed, pages_total = 0, None
    else:
        pages_processed, pages_total = await repo.get_pages_progress(
            document_id, version_for_progress
        )
    # Latest failure (D1): fetched only on the FAILED branch; a failed doc with
    # no failure row (legacy) serializes `failure: null` rather than erroring.
    failure: IngestionFailureInfo | None = None
    if document.status is DocumentStatus.FAILED:
        latest = await FailuresRepository(session).latest_failure(document_id)
        if latest is not None:
            failure = IngestionFailureInfo(
                reason=latest.reason,
                stage=latest.last_status.value,
                failed_at=latest.failed_at,
            )
    return IngestionStatusResponse(
        document_id=document.id,
        status=document.status.value,
        active_source_version=document.active_source_version,
        current_source_version=current_source_version,
        progress=IngestionProgress(
            stage=document.status.value,
            message=None,
            pages_total=pages_total,
            pages_processed=pages_processed,
        ),
        terminal=is_doc_terminal,
        failure=failure,
    )


def _resolve_auto_mode(
    *,
    latest_run: ExtractionRun | None,
    active_identity: str | None,
    settings: Settings,
) -> ReprocessMode:
    """Resolve `mode="auto"` to a concrete reuse/new-version mode (DECISIONS #4).

    Reuse the existing spans (re-run only the downstream extraction) when the
    document is fully current: the latest run's prompt/schema match today's
    constants AND the same extractor produced the active version. Otherwise —
    a changed or unknown (legacy, None) extractor identity, a drifted
    prompt/schema, or no prior run — re-extract into a new source version. The
    new-version default is the safe side: a needless re-extraction is merely
    costly, whereas a missed one would skip required work (the dangerous miss).
    """
    if (
        latest_run is not None
        and latest_run.prompt_version == PROMPT_VERSION
        and latest_run.schema_version == SCHEMA_VERSION
        and active_identity is not None
        and active_identity == settings.pdf_text_extractor
    ):
        return ReprocessMode.REUSE_SOURCE_SPANS
    return ReprocessMode.NEW_SOURCE_VERSION


async def _enqueue_reprocess(
    arq_redis: ArqRedis,
    document_id: str,
    *,
    source_version: int,
    reuse_source_spans: bool,
) -> None:
    """Best-effort enqueue of a reprocess run (mirrors upload_document).

    The row is already committed and QUEUED, so a failed enqueue is logged — not
    fatal — and the stuck-job cron is the backstop for the un-picked-up doc. Called
    AFTER the commit so a rolled-back guard (the 409 path) never leaves a job queued.
    """
    try:
        await enqueue_job(
            arq_redis,
            "process_document",
            document_id,
            session_id=document_id,
            source_version=source_version,
            reuse_source_spans=reuse_source_spans,
        )
    except Exception:
        logger.exception(
            "Failed to enqueue process_document for %s; stuck-job cron will catch it",
            document_id,
        )


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    body: ReprocessRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
    settings: Settings = Depends(get_settings),  # noqa: B008
) -> Any:
    try:
        mode = ReprocessMode(body.mode)
    except ValueError as exc:
        raise ApiError(
            status_code=400,
            code=ErrorCode.INVALID_REQUEST,
            message="Invalid value for 'mode'.",
            details={"field": "mode", "value": body.mode},
        ) from exc

    repo = DocumentRepository(session)
    document = await repo.get_document(document_id)
    if document is None:
        raise ApiError(
            status_code=404,
            code=ErrorCode.DOCUMENT_NOT_FOUND,
            message=f"Document {document_id!r} not found.",
            details={"document_id": document_id},
        )
    previous = document.active_source_version

    # Atomic guarded transition: the WHERE-clause status filter closes the
    # check-then-write race. Two concurrent POSTs cannot both pass because
    # Postgres serializes the row UPDATEs; the loser sees status='queued'
    # (no longer in the terminal set) and matches 0 rows.
    result = await session.execute(
        update(Document)
        .where(
            Document.id == document_id,
            Document.status.in_(_TERMINAL_DOCUMENT_STATUSES),
        )
        .values(
            status=DocumentStatus.QUEUED,
            last_reprocess_mode=mode.value,
            last_reprocess_reason=body.reason,
            # Reset the extraction heartbeat (review #3): sweep_stuck_jobs reaps on
            # coalesce(last_progress_at, updated_at). A terminal doc carries a
            # last_progress_at from its *original* run, possibly days old; leaving it
            # would let the cron mark this freshly-requeued reprocess FAILED before
            # the worker even starts. NULL falls back to the just-bumped updated_at,
            # matching a fresh upload's pre-progress state.
            last_progress_at=None,
        )
        .execution_options(synchronize_session=False)
    )
    if result.rowcount == 0:  # type: ignore[attr-defined]
        raise ApiError(
            status_code=409,
            code=ErrorCode.INGESTION_ALREADY_RUNNING,
            message="Document is not in a terminal state.",
            details={"document_id": document_id},
        )
    await session.commit()

    # Resolve `auto` to a concrete mode (DECISIONS #4). `last_reprocess_mode` keeps
    # the *requested* mode ("auto", recorded in the UPDATE above) — the audit stays
    # honest about what the user asked; the resolved decision goes to the log.
    effective_mode = mode
    if mode is ReprocessMode.AUTO:
        latest_run = await repo.get_latest_extraction_run(document_id)
        active_identity = (
            await repo.get_version_extractor_identity(document_id, previous)
            if previous is not None
            else None
        )
        effective_mode = _resolve_auto_mode(
            latest_run=latest_run,
            active_identity=active_identity,
            settings=settings,
        )
        logger.info(
            "auto reprocess for %s resolved to %s", document_id, effective_mode.value
        )

    # Dispatch the resolved run after the commit. `current_source_version` reflects
    # the version the run targets: the active version for reuse (unchanged), or the
    # new version for a new-source-version run.
    current_source_version = previous
    if effective_mode is ReprocessMode.REUSE_SOURCE_SPANS:
        # Reuse the existing version: the active version if the doc has one, else
        # the highest existing span version. Do NOT fabricate source_version=1 — a
        # terminal doc with no spans (active and max both None, e.g. an early FAILED
        # whose first extraction never persisted spans) has no source text to reuse,
        # so skip the enqueue (the row still sits QUEUED for the cron backstop).
        resolved_version = (
            previous
            if previous is not None
            else await repo.max_source_version(document_id)
        )
        if resolved_version is not None:
            await _enqueue_reprocess(
                arq_redis,
                document_id,
                source_version=resolved_version,
                reuse_source_spans=True,
            )
    elif effective_mode is ReprocessMode.NEW_SOURCE_VERSION:
        # Re-extract the PDF text into a fresh version = max(existing) + 1 (v1 when
        # the doc has no spans). `active_source_version` stays on the OLD version
        # during the run; the worker flips it to the new version on success (Epic
        # 11.2 TASK-003). Concurrency: the terminal->QUEUED guarded UPDATE above
        # already serialises reprocess starts, so two near-simultaneous requests
        # can't both compute the same max+1 (the loser 409s before reaching here);
        # the SourceSpan (document_id, source_version, locator_hash) unique key is
        # the backstop.
        new_version = (await repo.max_source_version(document_id) or 0) + 1
        current_source_version = new_version
        await _enqueue_reprocess(
            arq_redis,
            document_id,
            source_version=new_version,
            reuse_source_spans=False,
        )

    return ReprocessResponse(
        document_id=document_id,
        status=DocumentStatus.QUEUED.value,
        previous_active_source_version=previous,
        current_source_version=current_source_version,
    )
