"""Reusable contract suite for PdfTextExtractor implementations (doc 12 § 2).

Subclasses override ``provider``, ``known_input`` (bytes the provider can
extract) and ``expected_pages`` (the page list it should return for that
input). Named ``…Contract`` so pytest does not collect the abstract base.
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

__all__ = ["PdfExtractorContract"]


class PdfExtractorContract:
    """Interface guarantees every PdfTextExtractor must satisfy."""

    @pytest.fixture
    def provider(self) -> PdfTextExtractor:
        raise NotImplementedError("subclasses must override the `provider` fixture")

    @pytest.fixture
    def known_input(self) -> bytes:
        raise NotImplementedError("subclasses must override the `known_input` fixture")

    @pytest.fixture
    def expected_pages(self) -> list[PdfPageText]:
        raise NotImplementedError("subclasses must override the `expected_pages` fixture")

    async def test_known_input_returns_expected_pages(
        self,
        provider: PdfTextExtractor,
        known_input: bytes,
        expected_pages: list[PdfPageText],
    ) -> None:
        assert await provider.extract_pages(known_input) == expected_pages

    async def test_returned_pages_are_pdfpagetext(
        self, provider: PdfTextExtractor, known_input: bytes
    ) -> None:
        pages = await provider.extract_pages(known_input)
        assert pages
        for page in pages:
            assert isinstance(page, PdfPageText)
            assert isinstance(page.page_number, int)
            assert isinstance(page.text, str)
            assert isinstance(page.extraction_method, str)
            assert page.confidence is None or isinstance(page.confidence, float)
