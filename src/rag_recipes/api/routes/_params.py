"""Shared query-param parsing for API routes (doc 6 error envelope).

Extracted from ``documents.py`` in Phase 21.3 so the review-items listing and
the documents/debug routes share one parser pair. Filter params are typed
``str | None`` on the routes rather than enums/ints so FastAPI's raw 422 never
fires before the handler runs — every validation error stays inside the
``ApiError`` envelope.
"""

from __future__ import annotations

from enum import StrEnum

from rag_recipes.api.errors import ApiError, ErrorCode


def parse_enum[E: StrEnum](
    enum_cls: type[E], raw: str | None, *, field: str
) -> E | None:
    """Coerce an optional string to a `StrEnum` member or raise the
    doc-6 ``invalid_request`` envelope."""
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


def parse_int(
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
