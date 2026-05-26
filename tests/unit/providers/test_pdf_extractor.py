"""Bind FakePdfTextExtractor and the real PyMuPdfExtractor to the contract suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_recipes.providers.pdf_extractor.fake import FakePdfTextExtractor
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText
from tests.contracts.pdf_extractor import PdfExtractorContract

_KNOWN = b"%PDF-1.4 known recipe"
_EXPECTED = [
    PdfPageText(page_number=1, text="page one", extraction_method="fake", confidence=0.9),
    PdfPageText(page_number=2, text="page two", extraction_method="fake"),
]

_FIXTURE = Path("data/fixtures/pdfs/sample_recipe.pdf")
_FIXTURE_PAGES = [
    PdfPageText(
        page_number=1,
        text="Classic Pancakes\nIngredients\n2 cups flour\n2 eggs\n1 cup milk\n1 tbsp sugar\n",
        extraction_method="embedded_text",
        confidence=None,
    ),
    PdfPageText(
        page_number=2,
        text=(
            "Instructions\nMix the dry ingredients.\nWhisk in the eggs and milk.\n"
            "Cook on a hot griddle until golden.\n"
        ),
        extraction_method="embedded_text",
        confidence=None,
    ),
    PdfPageText(
        page_number=3,
        text="",
        extraction_method="embedded_text",
        confidence=0.0,
    ),
]


class TestFakePdfExtractor(PdfExtractorContract):
    @pytest.fixture
    def provider(self) -> FakePdfTextExtractor:
        return FakePdfTextExtractor(
            {FakePdfTextExtractor.content_hash(_KNOWN): _EXPECTED}
        )

    @pytest.fixture
    def known_input(self) -> bytes:
        return _KNOWN

    @pytest.fixture
    def expected_pages(self) -> list[PdfPageText]:
        return _EXPECTED


class TestPyMuPdfExtractor(PdfExtractorContract):
    @pytest.fixture
    def provider(self) -> PyMuPdfExtractor:
        return PyMuPdfExtractor(min_text_chars=20)

    @pytest.fixture
    def known_input(self) -> bytes:
        return _FIXTURE.read_bytes()

    @pytest.fixture
    def expected_pages(self) -> list[PdfPageText]:
        return _FIXTURE_PAGES
