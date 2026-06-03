"""Integration tests for ingestion.pipeline.extraction.run_extraction.

Real Postgres (``test_engine`` / ``db_session``) with an injected
``FakeLLMProvider``. Exercises the four call/parse outcomes 9.2 records —
success, technical failure, provider parse-rejection, and Pydantic-rejection —
asserting the persisted ``ExtractionRun`` row in each case. The cache-hit path
lands in TASK-004.
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.ingestion.pipeline.extraction import (
    PROMPT_VERSION,
    SCHEMA_VERSION,
    run_extraction,
)
from rag_recipes.ingestion.pipeline.windows import Window, compute_input_hash
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
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

SOURCE_VERSION = 1


def _canned_success_payload() -> dict[str, Any]:
    """A valid doc-4 § Strict Output Shape payload (one recipe)."""
    return {
        "items": [
            {
                "item_type": "recipe",
                "title": "Tomato and White Bean Soup",
                "summary": "A simple soup with pantry ingredients.",
                "body_text": "Tomato and White Bean Soup\nServes 4\n...",
                "source_span_ids": ["span_a", "span_b"],
                "structured_data": {
                    "schema": "recipe.v1",
                    "yield": "Serves 4",
                    "prep_time": None,
                    "cook_time": "35 minutes",
                    "total_time": None,
                    "ingredients_text": "2 tbsp olive oil\n1 onion, diced...",
                    "ingredients": [
                        {
                            "position": 1,
                            "raw_text": "2 tbsp olive oil",
                            "quantity_text": "2",
                            "quantity_value": 2,
                            "unit_raw": "tbsp",
                            "unit_normalized": "tablespoon",
                            "item_text": "olive oil",
                            "item_normalized": "olive oil",
                            "preparation": None,
                            "notes": None,
                            "confidence": {
                                "overall": 0.95,
                                "quantity": 0.98,
                                "unit": 0.96,
                                "item": 0.97,
                                "normalization": 0.9,
                            },
                        }
                    ],
                    "steps_text": "Heat the oil in a large pot...",
                    "steps": [
                        {
                            "step_number": 1,
                            "text": "Heat the oil in a large pot.",
                            "source_span_ids": ["span_b"],
                            "confidence": {"overall": 0.92, "ordering": 0.9},
                        }
                    ],
                },
                "confidence": {
                    "overall": 0.88,
                    "boundary": 0.82,
                    "fields": {
                        "title": 0.96,
                        "summary": 0.84,
                        "yield": 0.9,
                        "ingredients": 0.91,
                        "steps": 0.87,
                    },
                },
                "warnings": [],
            }
        ]
    }


def _make_span(document_id: str, page: int, text: str) -> SourceSpan:
    locator = {"type": "pdf_page_range", "page_start": page, "page_end": page}
    return SourceSpan(
        document_id=document_id,
        source_version=SOURCE_VERSION,
        source_type=SourceType.PDF,
        locator=locator,
        locator_hash=hashlib.sha256(str(locator).encode()).hexdigest(),
        text=text,
        text_hash=hashlib.sha256(text.encode()).hexdigest(),
    )


async def _make_document_and_window(session: AsyncSession) -> tuple[str, Window]:
    """Insert a Document + two per-page SourceSpans; return (id, Window)."""
    repo = DocumentRepository(session)
    pdf_bytes = b"%PDF-1.4 run-extraction fixture"
    asset = await repo.add_source_asset(
        id=new_id("asset"),
        source_type=SourceType.PDF,
        original_filename="recipe.pdf",
        storage_provider="fake",
        storage_key=f"source-assets/{new_id('asset')}/original.pdf",
        content_hash=hashlib.sha256(pdf_bytes).hexdigest(),
        upload_status=UploadStatus.UPLOADED,
    )
    document = await repo.add_document(
        asset_id=asset.id,
        category="recipes",
        subcategory=None,
        title="My Recipe",
        author="Alice",
        source_type=SourceType.PDF,
        language=None,
        active_source_version=None,
        status=DocumentStatus.QUEUED,
    )
    spans = [
        _make_span(document.id, 1, "Tomato and White Bean Soup\nServes 4"),
        _make_span(document.id, 2, "Heat the oil in a large pot."),
    ]
    session.add_all(spans)
    await session.flush()
    return document.id, Window(spans=tuple(spans))


async def _runs_for(session: AsyncSession, document_id: str) -> list[ExtractionRun]:
    result = await session.execute(
        select(ExtractionRun).where(ExtractionRun.document_id == document_id)
    )
    return list(result.scalars().all())


@pytest.mark.asyncio
async def test_success_records_success_run(db_session: AsyncSession) -> None:
    document_id, window = await _make_document_and_window(db_session)
    provider = FakeLLMProvider(default_output=_canned_success_payload())

    run = await run_extraction(
        db_session,
        window,
        source_version=SOURCE_VERSION,
        document_id=document_id,
        provider=provider,
    )

    assert run.status == ExtractionRunStatus.SUCCESS
    assert run.output_json == _canned_success_payload()
    assert run.completed_at is not None
    assert run.error_message is None
    assert run.provider == "fake"
    assert run.model == "fake-model"
    assert run.prompt_version == PROMPT_VERSION
    assert run.schema_version == SCHEMA_VERSION
    assert run.input_source_span_ids == window.span_ids
    assert run.input_hash == compute_input_hash(window, PROMPT_VERSION, SCHEMA_VERSION)
    assert len(provider.calls) == 1
    # The provider received the trace-bearing request our prompt produced.
    assert provider.calls[0].prompt_version == PROMPT_VERSION

    rows = await _runs_for(db_session, document_id)
    assert len(rows) == 1


@pytest.mark.asyncio
async def test_technical_failure_records_failed_and_reraises(db_session: AsyncSession) -> None:
    document_id, window = await _make_document_and_window(db_session)
    provider = FakeLLMProvider(fail_technically=True)

    with pytest.raises(LLMTechnicalError):
        await run_extraction(
            db_session,
            window,
            source_version=SOURCE_VERSION,
            document_id=document_id,
            provider=provider,
        )

    rows = await _runs_for(db_session, document_id)
    assert len(rows) == 1
    run = rows[0]
    assert run.status == ExtractionRunStatus.FAILED
    assert run.error_message
    assert run.output_json is None
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_provider_parse_rejection_records_rejected(db_session: AsyncSession) -> None:
    document_id, window = await _make_document_and_window(db_session)
    canned = StructuredOutputResponse(
        output_json=None,
        parse_error="model refused to answer",
        raw_text="I cannot help with that.",
        usage=TokenUsage(input_tokens=10, output_tokens=0),
        provider="fake",
        model="fake-model",
    )
    provider = FakeLLMProvider(default_output=canned)

    run = await run_extraction(
        db_session,
        window,
        source_version=SOURCE_VERSION,
        document_id=document_id,
        provider=provider,
    )

    assert run.status == ExtractionRunStatus.REJECTED
    assert run.error_message == "model refused to answer"
    assert run.output_json is None
    assert run.completed_at is not None


@pytest.mark.asyncio
async def test_pydantic_rejection_retains_raw_output(db_session: AsyncSession) -> None:
    document_id, window = await _make_document_and_window(db_session)
    invalid = _canned_success_payload()
    del invalid["items"][0]["title"]  # violates recipe.v1 (title required)
    provider = FakeLLMProvider(default_output=invalid)

    run = await run_extraction(
        db_session,
        window,
        source_version=SOURCE_VERSION,
        document_id=document_id,
        provider=provider,
    )

    assert run.status == ExtractionRunStatus.REJECTED
    assert run.error_message  # carries the ValidationError text
    assert run.output_json == invalid  # raw object retained for audit
    assert run.completed_at is not None
