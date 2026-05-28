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

import contextlib
import hashlib
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_file_storage, get_session
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.schemas.documents import DocumentResponse, UploadIngestion, UploadResponse
from rag_recipes.providers.errors import FileStorageError
from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.repositories.documents import DocumentRepository

router = APIRouter(tags=["documents"])

_PDF_MAGIC = b"%PDF-"
_DEFAULT_FILENAME = "upload.pdf"


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
        await session.rollback()
        await storage.delete_object(key)
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
        await session.rollback()
        await storage.delete_object(key)
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
        with contextlib.suppress(Exception):
            await session.rollback()
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
