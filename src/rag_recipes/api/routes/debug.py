"""Dev-only debug endpoints (doc 6 § 9).

Three read-only audit views — extraction-runs list/detail and source-spans list.
The whole router is gated by ``require_debug_enabled`` (404 with FastAPI's default
body when ``debug_endpoints_enabled`` is false, so production neither serves nor
acknowledges them) and registered ``include_in_schema=False`` so it never appears in
the production OpenAPI. The personal API token is still enforced (global app gate).
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.dependencies import get_session, require_debug_enabled
from rag_recipes.api.errors import ApiError, ErrorCode
from rag_recipes.api.routes.documents import _parse_enum, _parse_int
from rag_recipes.api.schemas.debug import (
    ExtractionRunDetail,
    ExtractionRunListResponse,
    ExtractionRunSummary,
    SourceSpanListResponse,
    SourceSpanSummary,
)
from rag_recipes.storage.enums import ExtractionRunStatus
from rag_recipes.storage.repositories.debug import DebugRepository

router = APIRouter(
    tags=["debug"],
    dependencies=[Depends(require_debug_enabled)],
    include_in_schema=False,
)


def _optional_version(raw: str | None) -> int | None:
    """Parse an optional ``source_version`` query param (>= 1) or None when absent."""
    if not raw:
        return None
    return _parse_int(raw, field="source_version", default=1, minimum=1)


@router.get(
    "/documents/{document_id}/extraction-runs",
    response_model=ExtractionRunListResponse,
)
async def list_extraction_runs(
    document_id: str,
    source_version: str | None = None,
    status: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    version = _optional_version(source_version)
    status_enum = _parse_enum(ExtractionRunStatus, status, field="status")
    runs = await DebugRepository(session).list_extraction_runs(
        document_id, source_version=version, status=status_enum
    )
    return ExtractionRunListResponse(
        extraction_runs=[ExtractionRunSummary.model_validate(run) for run in runs]
    )


@router.get("/extraction-runs/{run_id}", response_model=ExtractionRunDetail)
async def get_extraction_run(
    run_id: str,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    run = await DebugRepository(session).get_extraction_run(run_id)
    if run is None:
        # Within the (already-open) dev gate, a normal project 404 envelope is fine —
        # the non-acknowledgement constraint only applies to the dev-OFF gate.
        raise ApiError(
            status_code=404,
            code=ErrorCode.INVALID_REQUEST,
            message=f"Extraction run {run_id!r} not found.",
            details={"run_id": run_id},
        )
    return ExtractionRunDetail.model_validate(run)


@router.get(
    "/documents/{document_id}/source-spans",
    response_model=SourceSpanListResponse,
    summary="List a document's source spans (debug-only)",
    description=(
        "Returns full source-span text for a document. WARNING: source span text "
        "can contain copyrighted source content — this is a dev-only debug endpoint "
        "and is unavailable (404) in production."
    ),
)
async def list_source_spans(
    document_id: str,
    source_version: str | None = None,
    page_start: str | None = None,
    page_end: str | None = None,
    session: AsyncSession = Depends(get_session),  # noqa: B008
) -> Any:
    version = _optional_version(source_version)
    start = (
        _parse_int(page_start, field="page_start", default=1, minimum=1)
        if page_start
        else None
    )
    end = (
        _parse_int(page_end, field="page_end", default=1, minimum=1)
        if page_end
        else None
    )
    spans = await DebugRepository(session).list_source_spans(
        document_id, source_version=version, page_start=start, page_end=end
    )
    return SourceSpanListResponse(
        source_spans=[SourceSpanSummary.model_validate(span) for span in spans]
    )
