"""Pydantic payload types for the PDF text extractor (doc 11 § 2)."""

from __future__ import annotations

from pydantic import BaseModel

__all__ = ["PdfPageText"]


class PdfPageText(BaseModel):
    """Extracted text for a single PDF page.

    ``extraction_method`` is a free string (e.g. ``"embedded_text"``); OCR and
    layout methods are anticipated later. ``confidence`` is nullable for methods
    that do not report one.
    """

    page_number: int
    text: str
    extraction_method: str
    confidence: float | None = None
