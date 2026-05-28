"""Shared API error envelope.

Defines the doc-6 ``{"error": {"code", "message", "details"}}`` body shape
and the ``ApiError`` exception that routes raise to opt into it. Phase 6.1
renders the envelope from the route directly; Phase 6.3 will register a
centralised FastAPI exception handler that converts any uncaught
``ApiError`` into the same response — this module is the seam it plugs into.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_REQUEST = "invalid_request"
    UNSUPPORTED_FILE_TYPE = "unsupported_file_type"
    DUPLICATE_SOURCE_ASSET = "duplicate_source_asset"
    DOCUMENT_NOT_FOUND = "document_not_found"
    INTERNAL_ERROR = "internal_error"


def error_body(
    *,
    code: ErrorCode,
    message: str,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "error": {
            "code": code.value,
            "message": message,
            "details": details if details is not None else {},
        }
    }


class ApiError(Exception):
    def __init__(
        self,
        status_code: int,
        code: ErrorCode,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details if details is not None else {}

    def to_body(self) -> dict[str, Any]:
        return error_body(code=self.code, message=self.message, details=self.details)
