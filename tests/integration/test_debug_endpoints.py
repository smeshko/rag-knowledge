"""Integration tests for the dev-only debug endpoints (doc 6 § 9)."""

from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.api.app import app
from rag_recipes.api.dependencies import get_session, get_settings
from rag_recipes.ingestion.pipeline.extraction import PROMPT_VERSION, SCHEMA_VERSION
from rag_recipes.storage.enums import (
    DocumentStatus,
    ExtractionRunStatus,
    SourceType,
    UploadStatus,
)
from rag_recipes.storage.ids import new_id
from rag_recipes.storage.models.extraction_run import ExtractionRun
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository
from tests.integration.conftest import AUTH_HEADERS, TEST_API_TOKEN

pytestmark = pytest.mark.asyncio


@asynccontextmanager
async def _client(
    db_session: AsyncSession, *, debug_enabled: bool, with_auth: bool = True
) -> AsyncIterator[httpx.AsyncClient]:
    settings = get_settings().model_copy(
        update={
            "personal_api_token": TEST_API_TOKEN,
            "debug_endpoints_enabled": debug_enabled,
        }
    )

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield db_session

    app.dependency_overrides[get_session] = _override_session
    app.dependency_overrides[get_settings] = lambda: settings
    transport = httpx.ASGITransport(app=app)
    headers = AUTH_HEADERS if with_auth else {}
    try:
        async with httpx.AsyncClient(
            transport=transport, base_url="http://testserver", headers=headers
        ) as client:
            yield client
    finally:
        for dep in (get_session, get_settings):
            app.dependency_overrides.pop(dep, None)


async def _seed(db_session: AsyncSession) -> dict[str, str]:
    repo = DocumentRepository(db_session)
    aid = new_id("asset")
    asset = await repo.add_source_asset(
        id=aid,
        source_type=SourceType.PDF,
        original_filename="cookbook.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{aid}/original.pdf",
        content_hash=hashlib.sha256(new_id("h").encode()).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    doc = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="Cookbook",
        author="Chef",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=1,
        status=DocumentStatus.READY,
    )

    def _run(
        source_version: int, status: ExtractionRunStatus, **extra: object
    ) -> ExtractionRun:
        run = ExtractionRun(
            document_id=doc.id,
            source_version=source_version,
            provider="openai",
            model="gpt-x",
            prompt_version=PROMPT_VERSION,
            schema_version=SCHEMA_VERSION,
            input_source_span_ids=["span_1"],
            input_hash=new_id("hash"),
            status=status,
            **extra,
        )
        db_session.add(run)
        return run

    success = _run(1, ExtractionRunStatus.SUCCESS, output_json={"items": [{"t": 1}]})
    failed = _run(1, ExtractionRunStatus.FAILED, error_message="boom")
    _run(2, ExtractionRunStatus.SUCCESS)
    await db_session.flush()

    for page in (1, 2, 3):
        locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
        db_session.add(
            SourceSpan(
                id=new_id("span"),
                document_id=doc.id,
                source_version=1,
                source_type=SourceType.PDF,
                locator=locator,
                locator_hash=hashlib.sha256(f"{doc.id}-{page}".encode()).hexdigest(),
                text=f"copyrighted recipe text page {page}",
                text_hash=hashlib.sha256(f"t{page}".encode()).hexdigest(),
            )
        )
    await db_session.flush()
    return {"document_id": doc.id, "success_run": success.id, "failed_run": failed.id}


async def test_extraction_runs_list_and_filters(db_session: AsyncSession) -> None:
    seeded = await _seed(db_session)
    doc = seeded["document_id"]
    async with _client(db_session, debug_enabled=True) as client:
        all_runs = (await client.get(f"/api/v1/documents/{doc}/extraction-runs")).json()
        v1 = (
            await client.get(
                f"/api/v1/documents/{doc}/extraction-runs?source_version=1"
            )
        ).json()
        failed = (
            await client.get(
                f"/api/v1/documents/{doc}/extraction-runs?status=failed"
            )
        ).json()
        bad = await client.get(
            f"/api/v1/documents/{doc}/extraction-runs?status=bogus"
        )
    assert len(all_runs["extraction_runs"]) == 3
    # The list item omits the heavy output_json.
    assert "output_json" not in all_runs["extraction_runs"][0]
    assert len(v1["extraction_runs"]) == 2
    assert len(failed["extraction_runs"]) == 1
    assert failed["extraction_runs"][0]["status"] == "failed"
    assert bad.status_code == 400
    assert bad.json()["error"]["code"] == "invalid_request"


async def test_extraction_run_detail_includes_output_and_error(
    db_session: AsyncSession,
) -> None:
    seeded = await _seed(db_session)
    async with _client(db_session, debug_enabled=True) as client:
        success = (
            await client.get(f"/api/v1/extraction-runs/{seeded['success_run']}")
        ).json()
        failed = (
            await client.get(f"/api/v1/extraction-runs/{seeded['failed_run']}")
        ).json()
        missing = await client.get("/api/v1/extraction-runs/run_nope")
    assert success["output_json"] == {"items": [{"t": 1}]}
    assert success["input_source_span_ids"] == ["span_1"]
    assert "input_hash" in success
    assert failed["error_message"] == "boom"
    assert failed["output_json"] is None
    # Within the open dev gate, an unknown run uses the project 404 envelope.
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "invalid_request"


async def test_source_spans_full_text_and_page_filter(db_session: AsyncSession) -> None:
    seeded = await _seed(db_session)
    doc = seeded["document_id"]
    async with _client(db_session, debug_enabled=True) as client:
        all_spans = (await client.get(f"/api/v1/documents/{doc}/source-spans")).json()
        window = (
            await client.get(
                f"/api/v1/documents/{doc}/source-spans?page_start=2&page_end=3"
            )
        ).json()
    assert len(all_spans["source_spans"]) == 3
    # Full (copyright-sensitive) text is returned.
    assert all_spans["source_spans"][0]["text"].startswith("copyrighted recipe text")
    pages = sorted(s["locator"]["page_start"] for s in window["source_spans"])
    assert pages == [2, 3]


async def test_disabled_returns_404_identical_to_unknown_route(
    db_session: AsyncSession,
) -> None:
    seeded = await _seed(db_session)
    doc = seeded["document_id"]
    async with _client(db_session, debug_enabled=False) as client:
        unknown = await client.get("/api/v1/__no_such_route__")
        runs = await client.get(f"/api/v1/documents/{doc}/extraction-runs")
        run = await client.get(f"/api/v1/extraction-runs/{seeded['success_run']}")
        spans = await client.get(f"/api/v1/documents/{doc}/source-spans")
    assert unknown.status_code == 404
    for resp in (runs, run, spans):
        # Byte-for-byte identical to an unknown route: production never acknowledges.
        assert resp.status_code == 404
        assert resp.json() == unknown.json() == {"detail": "Not Found"}


async def test_missing_token_returns_401_even_when_dev_on(
    db_session: AsyncSession,
) -> None:
    seeded = await _seed(db_session)
    doc = seeded["document_id"]
    async with _client(db_session, debug_enabled=True, with_auth=False) as client:
        resp = await client.get(f"/api/v1/documents/{doc}/extraction-runs")
    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "unauthorized"


async def test_debug_routes_hidden_from_openapi_with_copyright_warning() -> None:
    schema = app.openapi()
    for path in (
        "/api/v1/documents/{document_id}/extraction-runs",
        "/api/v1/extraction-runs/{run_id}",
        "/api/v1/documents/{document_id}/source-spans",
    ):
        assert path not in schema["paths"]
    # The copyright warning lives on the route's description (hidden from OpenAPI).
    span_routes = [
        r
        for r in app.routes
        if getattr(r, "path", "") == "/api/v1/documents/{document_id}/source-spans"
    ]
    assert span_routes
    assert "copyright" in (span_routes[0].description or "").lower()
