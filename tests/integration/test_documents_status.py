"""Integration tests for GET /api/v1/documents/{document_id}/status."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Iterator

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session
from rag_recipes.storage.enums import DocumentStatus, SourceType, UploadStatus
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.document import Document
from rag_recipes.storage.models.source_asset import SourceAsset
from rag_recipes.storage.models.source_span import SourceSpan

# Per Decision #4 in the plan: terminal = {ready, needs_review, failed}. Phase
# 8.2 replaced 6.2's `current_source_version == active_source_version` mirror
# with the real rule: `1` for any non-terminal status (initial-ingestion
# scope) and `None` for terminal statuses, matching doc 6 §5's in-progress
# example (`active: null, current: 1`).
TERMINAL_STATUSES: frozenset[DocumentStatus] = frozenset(
    {DocumentStatus.READY, DocumentStatus.NEEDS_REVIEW, DocumentStatus.FAILED}
)


async def _seed_document(
    session: AsyncSession,
    *,
    content_hash: str,
    status: DocumentStatus = DocumentStatus.QUEUED,
    active_source_version: int | None = None,
) -> Document:
    asset_id = new_id(SourceAsset.ID_PREFIX)
    session.add(
        SourceAsset(
            id=asset_id,
            source_type=SourceType.PDF,
            original_filename="example.pdf",
            storage_provider="local",
            storage_key=f"source-assets/{asset_id}/original.pdf",
            content_hash=content_hash,
            upload_status=UploadStatus.UPLOADED,
        )
    )
    await session.flush()
    document = Document(
        asset_id=asset_id,
        category="recipes",
        subcategory=None,
        title="Example",
        author="",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=active_source_version,
        status=status,
    )
    session.add(document)
    await session.flush()
    await session.refresh(document)
    return document


async def _seed_spans(
    session: AsyncSession,
    *,
    document: Document,
    source_version: int,
    pages: int,
) -> None:
    for page in range(1, pages + 1):
        marker = f"{document.id}-{source_version}-{page}"
        session.add(
            SourceSpan(
                document_id=document.id,
                source_version=source_version,
                source_type=SourceType.PDF,
                locator={
                    "type": "pdf_page_range",
                    "page_start": page,
                    "page_end": page,
                },
                locator_hash=hashlib.sha256(marker.encode()).hexdigest(),
                text=f"text-{marker}",
                text_hash=hashlib.sha256(f"text-{marker}".encode()).hexdigest(),
            )
        )
    await session.flush()


@pytest.fixture
def client(
    db_session: AsyncSession,
    override_settings_with_token: None,
    auth_headers: dict[str, str],
) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
            headers=auth_headers,
        )
    finally:
        app.dependency_overrides.pop(get_session, None)


@pytest.mark.asyncio
async def test_status_returns_doc_section_5_shape(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(db_session, content_hash="hash-status-1")
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "document_id",
        "status",
        "active_source_version",
        "current_source_version",
        "progress",
        "terminal",
    }
    assert body["document_id"] == document.id
    assert body["status"] == "queued"
    assert body["active_source_version"] is None
    # 8.2 rule: non-terminal → current_source_version=1 (doc 6 §5 in-progress
    # example), diverging from active_source_version (null mid-ingestion).
    assert body["current_source_version"] == 1
    assert body["terminal"] is False
    progress = body["progress"]
    assert set(progress.keys()) == {"stage", "message", "pages_total", "pages_processed"}
    assert progress["stage"] == "queued"
    assert progress["message"] is None
    assert progress["pages_total"] is None
    assert progress["pages_processed"] == 0


@pytest.mark.parametrize("status", list(DocumentStatus))
@pytest.mark.asyncio
async def test_terminal_flag_matches_documented_set_for_every_status(
    client: httpx.AsyncClient,
    db_session: AsyncSession,
    status: DocumentStatus,
) -> None:
    """`terminal` is true iff status is in {ready, needs_review, failed}.

    Parametrized over all 11 ``DocumentStatus`` members so an implementation
    that special-cases only ``queued`` as non-terminal cannot pass.
    """
    document = await _seed_document(
        db_session, content_hash=f"hash-term-{status.value}", status=status
    )
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == status.value
    assert body["progress"]["stage"] == status.value
    is_doc_terminal = status in TERMINAL_STATUSES
    assert body["terminal"] is is_doc_terminal
    # 8.2: current_source_version is 1 for any non-terminal status, None for
    # terminal — verified across all 11 statuses.
    assert body["current_source_version"] == (None if is_doc_terminal else 1)


@pytest.mark.asyncio
async def test_status_queued_document_reports_current_version_1_no_pages(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session, content_hash="hash-queued-prog", status=DocumentStatus.QUEUED
    )
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["current_source_version"] == 1
    assert body["active_source_version"] is None
    assert body["progress"]["pages_processed"] == 0
    assert body["progress"]["pages_total"] is None
    assert body["terminal"] is False


@pytest.mark.asyncio
async def test_status_extracting_text_reports_current_version_1_no_pages(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session,
        content_hash="hash-extracting-prog",
        status=DocumentStatus.EXTRACTING_TEXT,
    )
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "extracting_text"
    assert body["current_source_version"] == 1
    assert body["active_source_version"] is None
    assert body["progress"]["pages_processed"] == 0
    assert body["progress"]["pages_total"] is None
    assert body["terminal"] is False


@pytest.mark.asyncio
async def test_status_creating_source_spans_with_three_spans_reports_3_3(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session,
        content_hash="hash-spans-prog",
        status=DocumentStatus.CREATING_SOURCE_SPANS,
    )
    await _seed_spans(db_session, document=document, source_version=1, pages=3)
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["current_source_version"] == 1
    assert body["active_source_version"] is None
    assert body["progress"]["pages_processed"] == 3
    assert body["progress"]["pages_total"] == 3
    assert body["terminal"] is False


@pytest.mark.asyncio
async def test_status_ready_document_with_three_spans_reports_final_counts(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session,
        content_hash="hash-ready-prog",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    await _seed_spans(db_session, document=document, source_version=1, pages=3)
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["current_source_version"] is None
    assert body["active_source_version"] == 1
    assert body["progress"]["pages_processed"] == 3
    assert body["progress"]["pages_total"] == 3
    assert body["terminal"] is True


@pytest.mark.asyncio
async def test_status_failed_document_no_spans_no_active_version_reports_zero_none(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session, content_hash="hash-failed-prog", status=DocumentStatus.FAILED
    )
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["current_source_version"] is None
    assert body["active_source_version"] is None
    assert body["progress"]["pages_processed"] == 0
    assert body["progress"]["pages_total"] is None
    assert body["terminal"] is True


@pytest.mark.asyncio
async def test_status_needs_review_reports_final_counts_via_active_version(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    document = await _seed_document(
        db_session,
        content_hash="hash-needs-review-prog",
        status=DocumentStatus.NEEDS_REVIEW,
        active_source_version=1,
    )
    await _seed_spans(db_session, document=document, source_version=1, pages=2)
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["current_source_version"] is None
    assert body["active_source_version"] == 1
    assert body["progress"]["pages_processed"] == 2
    assert body["progress"]["pages_total"] == 2
    assert body["terminal"] is True


@pytest.mark.asyncio
async def test_status_reprocess_in_flight_does_not_report_stale_spans(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    """A reprocess-queued doc must not report its old run's spans as progress.

    ``POST /reprocess`` moves a terminal doc back to ``queued`` while keeping
    ``active_source_version`` set and leaving the old-version spans in place.
    The new run's spans don't exist yet, so progress must be ``(0, None)`` — not
    the surviving v1 counts. Regression for review round-1 #1: without the
    reprocess-in-flight guard the route would count the stale v1 spans and
    report ``pages_processed == pages_total`` for a doc that hasn't started.
    Epic 11 owns the real in-flight ``current_source_version`` arithmetic.
    """
    document = await _seed_document(
        db_session,
        content_hash="hash-reprocess-in-flight",
        status=DocumentStatus.QUEUED,
        active_source_version=1,
    )
    # Old run's spans survive the reprocess (the route only updates Document).
    await _seed_spans(db_session, document=document, source_version=1, pages=3)
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "queued"
    assert body["active_source_version"] == 1
    assert body["current_source_version"] == 1
    assert body["terminal"] is False
    # Stale v1 spans must NOT leak into progress for the not-yet-started run.
    assert body["progress"]["pages_processed"] == 0
    assert body["progress"]["pages_total"] is None


@pytest.mark.asyncio
async def test_status_unknown_id_returns_404_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents/doc_does_not_exist/status")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "document_not_found"
