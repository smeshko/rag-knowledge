"""In-memory FakePdfTextExtractor returning canned pages keyed on content hash.

Production code (doc 13 § 9): callers register page fixtures by the sha256 of
the PDF bytes; unknown bytes yield a placeholder page or raise, configurably.
"""

from __future__ import annotations

import hashlib

from rag_recipes.providers.errors import PdfExtractionError
from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

__all__ = ["FakePdfTextExtractor"]


class FakePdfTextExtractor(PdfTextExtractor):
    """PDF extractor returning canned ``PdfPageText`` lists keyed on content hash."""

    def __init__(
        self,
        pages_by_hash: dict[str, list[PdfPageText]] | None = None,
        *,
        fallback_to_placeholder: bool = True,
    ) -> None:
        self._pages_by_hash: dict[str, list[PdfPageText]] = {
            key: [page.model_copy(deep=True) for page in pages]
            for key, pages in (pages_by_hash or {}).items()
        }
        self._fallback_to_placeholder = fallback_to_placeholder

    @staticmethod
    def content_hash(file: bytes) -> str:
        return hashlib.sha256(file).hexdigest()

    async def extract_pages(self, file: bytes) -> list[PdfPageText]:
        pages = self._pages_by_hash.get(self.content_hash(file))
        if pages is not None:
            return [page.model_copy(deep=True) for page in pages]
        if self._fallback_to_placeholder:
            return [
                PdfPageText(
                    page_number=1,
                    text="",
                    extraction_method="fake",
                    confidence=None,
                )
            ]
        raise PdfExtractionError("no canned pages registered for the given bytes")
