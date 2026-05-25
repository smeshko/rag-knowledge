"""PdfTextExtractor interface (doc 11 § 2)."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_recipes.providers.pdf_extractor.types import PdfPageText

__all__ = ["PdfTextExtractor"]


class PdfTextExtractor(ABC):
    """Abstract extractor turning a PDF's bytes into per-page text.

    Async for call-site uniformity; the synchronous PyMuPDF real implementation
    (Epic 4) runs its CPU-bound extraction in a threadpool behind this signature.
    Technical failures raise ``PdfExtractionError`` (``providers.errors``).
    """

    @abstractmethod
    async def extract_pages(self, file: bytes) -> list[PdfPageText]:
        """Extract text from each page of the PDF ``file``."""
