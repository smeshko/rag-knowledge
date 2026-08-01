"""Unit tests for the typed batch-upload item schema (Epic 21.1, D2).

``BatchUploadItemResult.status`` is a closed ``StrEnum`` — unknown statuses
are unrepresentable — and error items carry a ``BatchUploadErrorCode`` mapped
from ``ApiError.code`` with an ``internal_error`` fallback.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_recipes.api.errors import ErrorCode
from rag_recipes.api.routes.documents import _batch_error_code
from rag_recipes.api.schemas.documents import (
    BatchUploadErrorCode,
    BatchUploadItemResult,
    BatchUploadItemStatus,
)


@pytest.mark.parametrize("status", ["created", "duplicate", "error"])
def test_item_accepts_the_three_statuses(status: str) -> None:
    item = BatchUploadItemResult(filename="a.pdf", status=BatchUploadItemStatus(status))
    assert item.status == status
    assert item.model_dump()["status"] == status


def test_item_rejects_unknown_status() -> None:
    with pytest.raises(ValidationError):
        BatchUploadItemResult.model_validate({"filename": "a.pdf", "status": "bogus"})


def test_error_code_defaults_to_none() -> None:
    item = BatchUploadItemResult(
        filename="a.pdf", status=BatchUploadItemStatus.CREATED
    )
    assert item.error_code is None


@pytest.mark.parametrize(
    ("api_code", "expected"),
    [
        (ErrorCode.INVALID_REQUEST, BatchUploadErrorCode.INVALID_REQUEST),
        (ErrorCode.UNSUPPORTED_FILE_TYPE, BatchUploadErrorCode.UNSUPPORTED_FILE_TYPE),
        (ErrorCode.INTERNAL_ERROR, BatchUploadErrorCode.INTERNAL_ERROR),
    ],
)
def test_batch_error_code_maps_reachable_codes(
    api_code: ErrorCode, expected: BatchUploadErrorCode
) -> None:
    assert _batch_error_code(api_code) is expected


@pytest.mark.parametrize(
    "api_code",
    [
        ErrorCode.DUPLICATE_SOURCE_ASSET,
        ErrorCode.DOCUMENT_NOT_FOUND,
        ErrorCode.UNAUTHORIZED,
        ErrorCode.INGESTION_ALREADY_RUNNING,
        ErrorCode.KNOWLEDGE_ITEM_NOT_FOUND,
    ],
)
def test_batch_error_code_falls_back_to_internal_error(api_code: ErrorCode) -> None:
    assert _batch_error_code(api_code) is BatchUploadErrorCode.INTERNAL_ERROR
