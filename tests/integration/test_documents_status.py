"""Integration tests for GET /api/v1/documents/{document_id}/status."""

from __future__ import annotations

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

# Per Decision #4 in the plan: terminal = {ready, needs_review, failed}. The
# `current_source_version == active_source_version` mirror below is a
# *temporary Epic-8 placeholder* and diverges from doc 6 §5's documented
# in-progress example (`active: null, current: 1`); Epic 8 Phase 8.2 must
# replace this mirror with the real in-progress version once versioned spans
# exist. Do not freeze this assertion as a permanent contract.
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


@pytest.fixture
def client(db_session: AsyncSession) -> Iterator[httpx.AsyncClient]:
    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    transport = httpx.ASGITransport(app=app)
    try:
        yield httpx.AsyncClient(transport=transport, base_url="http://testserver")
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
    # TEMPORARY Epic-8 placeholder: mirrors active_source_version. Doc 6 §5
    # expects these to diverge mid-ingestion (active: null, current: 1) once
    # versioned spans land. Epic 8 Phase 8.2 must replace this mirror.
    assert body["current_source_version"] == body["active_source_version"]
    assert body["terminal"] is False
    progress = body["progress"]
    assert set(progress.keys()) == {"stage", "message", "pages_total", "pages_processed"}
    assert progress["stage"] == "queued"
    assert progress["message"] is None
    assert progress["pages_total"] is None
    assert progress["pages_processed"] is None


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
    assert body["terminal"] is (status in TERMINAL_STATUSES)


@pytest.mark.asyncio
async def test_current_source_version_mirrors_active_when_set(
    client: httpx.AsyncClient, db_session: AsyncSession
) -> None:
    # TEMPORARY Epic-8 placeholder — see TERMINAL_STATUSES comment above.
    document = await _seed_document(
        db_session,
        content_hash="hash-mirror-1",
        status=DocumentStatus.READY,
        active_source_version=1,
    )
    async with client:
        response = await client.get(f"/api/v1/documents/{document.id}/status")
    assert response.status_code == 200
    body = response.json()
    assert body["active_source_version"] == 1
    assert body["current_source_version"] == 1
    assert body["progress"]["pages_total"] is None
    assert body["progress"]["pages_processed"] is None


@pytest.mark.asyncio
async def test_status_unknown_id_returns_404_envelope(client: httpx.AsyncClient) -> None:
    async with client:
        response = await client.get("/api/v1/documents/doc_does_not_exist/status")
    assert response.status_code == 404
    body = response.json()
    assert body["error"]["code"] == "document_not_found"
