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

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_file_storage, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.documents import (
    DocumentListItem,
    DocumentListResponse,
    DocumentResponse,
    UploadIngestion,
    UploadResponse,
)
from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository

router = APIRouter(tags=["documents"])

logger = logging.getLogger(__name__)

_PDF_MAGIC = b"%PDF-"
_DEFAULT_FILENAME = "upload.pdf"

_LIST_LIMIT_DEFAULT = 50
_LIST_LIMIT_MAX = 200
_LIST_OFFSET_DEFAULT = 0

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
) -> Any:
    try:
        return await _handle_upload(
            file=file,
            category=category,
            subcategory=subcategory,
            title=title,
            author=author,
            language=language,
            session=session,
            storage=storage,
        )
    except ApiError as err:
        return JSONResponse(status_code=err.status_code, content=err.to_body())
    except Exception:
        # Final safety net: no failure escapes the doc-6 envelope shape.
        # No deletion here — earlier handlers already managed any orphan.
        fallback = ApiError(
            status_code=500,
            code=ErrorCode.INTERNAL_ERROR,
            message="Unexpected server error.",
        )
        return JSONResponse(status_code=fallback.status_code, content=fallback.to_body())


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
    try:
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
    except ApiError as err:
        return JSONResponse(status_code=err.status_code, content=err.to_body())
