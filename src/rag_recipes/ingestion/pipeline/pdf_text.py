"""PDF text extraction pipeline stage (doc 3 § 3-4).

Pure stage: read a Document's PDF bytes, extract per-page text, and persist one
``SourceSpan`` per page at the given ``source_version``. No status transitions
and no arq/job concerns — those live in ``ingestion.jobs.process_document``.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.providers.file_storage.base import FileStorageProvider
from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.storage.enums import SourceType
from rag_recipes.storage.models.source_span import SourceSpan
from rag_recipes.storage.repositories.documents import DocumentRepository

__all__ = ["EmptyPdfError", "extract_and_persist_spans"]


class EmptyPdfError(ValueError):
    """Raised when the PDF has no pages."""


def _sha256_json(d: dict[str, Any]) -> str:
    """SHA-256 of a dict serialised to canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


async def extract_and_persist_spans(
    session: AsyncSession,
    *,
    document_id: str,
    source_version: int,
    extractor: PdfTextExtractor,
    storage: FileStorageProvider,
    extractor_identity: str | None = None,
) -> int:
    """Extract per-page text for a document and persist one SourceSpan per page.

    Returns the number of spans written. Raises ``LookupError`` if the document
    or its asset is missing, ``EmptyPdfError`` if the PDF has no pages, and
    ``sqlalchemy.exc.IntegrityError`` if spans for this
    ``(document_id, source_version)`` already exist (the uniqueness contract).

    ``extractor_identity`` (e.g. ``"pymupdf:embedded_text"``) is stamped into each
    span's ``locator["meta"]`` so the Epic 11.2 ``auto`` reprocess selector can tell
    whether the extractor changed since this version was produced. It is excluded
    from ``locator_hash`` (meta is), so it never affects span uniqueness.
    """
    repo = DocumentRepository(session)
    document = await repo.get_document(document_id)
    if document is None:
        raise LookupError(f"Document {document_id} not found")
    asset = await repo.get_source_asset_for_document(document)
    if asset is None:
        raise LookupError(f"SourceAsset for document {document_id} not found")

    pdf_bytes = await storage.get_object(asset.storage_key)
    pages = await extractor.extract_pages(pdf_bytes)
    if not pages:
        raise EmptyPdfError("PDF contains no pages")

    spans: list[SourceSpan] = []
    for page in pages:
        # locator_core is the uniqueness key input; meta is carried in the
        # persisted locator but excluded from locator_hash so the hash stays
        # stable across extraction-method/confidence changes (DECISIONS #3).
        locator_core = {
            "type": "pdf_page_range",
            "page_start": page.page_number,
            "page_end": page.page_number,
        }
        locator_with_meta = {
            **locator_core,
            "meta": {
                "confidence": page.confidence,
                "extraction_method": page.extraction_method,
                "suspicious": page.confidence == 0.0,
                "extractor_identity": extractor_identity,
            },
        }
        spans.append(
            SourceSpan(
                document_id=document_id,
                source_version=source_version,
                source_type=SourceType.PDF,
                locator=locator_with_meta,
                locator_hash=_sha256_json(locator_core),
                text=page.text,
                text_hash=_sha256_text(page.text),
            )
        )

    session.add_all(spans)
    await session.flush()
    return len(spans)
