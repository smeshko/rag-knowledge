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

from rag_recipes.api.dependencies import get_arq_redis, get_file_storage, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.documents import (
    DocumentCounts,
    DocumentDetailResponse,
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    IngestionProgress,
    IngestionStatusResponse,
    ReprocessRequest,
    ReprocessResponse,
    UploadIngestion,
    UploadResponse,
)
from rag_recipes.ingestion.queue import enqueue_job
from rag_recipes.ingestion.status import is_terminal
from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.storage.enums import DocumentStatus, ReprocessMode, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository

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
        return UploadResponse(
            document=DocumentResponse.model_validate(existing_document),
            ingestion=UploadIngestion(status=existing_document.status.value),
        )

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
        return UploadResponse(
            document=DocumentResponse.model_validate(winner_document),
            ingestion=UploadIngestion(status=winner_document.status.value),
        )
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

    # Fresh-insert path only — kick off ingestion. The duplicate-recovery
    # paths above return before reaching here and must not re-enqueue.
    try:
        await enqueue_job(arq_redis, "process_document", document.id, session_id=document.id)
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

    8.2 covers initial ingestion only: ``current_source_version`` is ``1``
    for any non-terminal status and ``None`` for terminal ones. Epic 11 must
    revisit this logic for reprocess scenarios where ``active`` != ``current``
    (a ready doc at v1 with v2 in flight).
    """
    repo = DocumentRepository(session)
    document = await _require_document(repo, document_id)
    is_doc_terminal = is_terminal(document.status)
    current_source_version = None if is_doc_terminal else 1
    if is_doc_terminal:
        # Report final counts for terminal docs that have an active version (a
        # ready doc at v1 shows its spans); failed docs with active=null
        # legitimately get (0, None).
        version_for_progress = document.active_source_version
    elif document.active_source_version is not None:
        # Reprocess in flight: the doc is non-terminal but already has an active
        # version from a prior run, so the new run's spans don't exist yet.
        # Suppress progress (the surviving old-version spans are NOT this run's
        # work). Epic 11 will compute the real in-flight version here.
        version_for_progress = None
    else:
        # Initial ingestion: in-flight version is 1.
        version_for_progress = 1
    if version_for_progress is None:
        pages_processed, pages_total = 0, None
    else:
        pages_processed, pages_total = await repo.get_pages_progress(
            document_id, version_for_progress
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
    )


@router.post("/documents/{document_id}/reprocess")
async def reprocess_document(
    document_id: str,
    body: ReprocessRequest,
    session: AsyncSession = Depends(get_session),  # noqa: B008
    arq_redis: ArqRedis = Depends(get_arq_redis),  # noqa: B008
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

    # Epic 11.1: only `reuse_source_spans` actually starts work here. `auto` /
    # `new_source_version` are accepted (row flipped to QUEUED, mode/reason
    # recorded above) but enqueue NO job — dispatching the reuse pipeline for them
    # would run the wrong work (reuse skips PDF re-extraction and would supersede
    # live items against stale source text while last_reprocess_mode records a
    # different mode). The stuck-job cron is the existing backstop for any
    # un-picked-up QUEUED doc.
    # TODO(11.2): add the auto selector + new_source_version re-extraction enqueue.
    if mode is ReprocessMode.REUSE_SOURCE_SPANS:
        # Resolve the version to reuse: the active version if the doc has one,
        # else the highest existing span version. Do NOT fabricate
        # source_version=1 — a terminal doc with no spans (active and max both
        # None, e.g. an early FAILED whose first extraction never persisted spans)
        # has no source text to reuse, so a reuse job would be a queued no-op.
        # Skip the enqueue entirely in that case.
        resolved_version = (
            previous
            if previous is not None
            else await repo.max_source_version(document_id)
        )
        if resolved_version is not None:
            # Best-effort, mirroring upload_document: the row is committed and
            # QUEUED, so a failed enqueue is logged (not fatal) and the stuck-job
            # cron will catch the un-picked-up doc. Enqueue AFTER the commit so a
            # rolled-back guard (the 409 path above) never leaves a job queued.
            try:
                await enqueue_job(
                    arq_redis,
                    "process_document",
                    document_id,
                    session_id=document_id,
                    source_version=resolved_version,
                    reuse_source_spans=True,
                )
            except Exception:
                logger.exception(
                    "Failed to enqueue reuse process_document for %s; "
                    "stuck-job cron will catch it",
                    document_id,
                )

    # `current_source_version` mirrors `active_source_version`: a reuse reprocess
    # re-runs the existing version, so the active version is unchanged.
    return ReprocessResponse(
        document_id=document_id,
        status=DocumentStatus.QUEUED.value,
        previous_active_source_version=previous,
        current_source_version=previous,
    )
