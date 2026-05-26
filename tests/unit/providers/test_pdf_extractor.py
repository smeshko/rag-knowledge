"""Bind FakePdfTextExtractor to the shared PdfExtractor contract suite."""

from __future__ import annotations

import pytest

from rag_recipes.providers.pdf_extractor.fake import FakePdfTextExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText
from tests.contracts.pdf_extractor import PdfExtractorContract

_KNOWN = b"%PDF-1.4 known recipe"
_EXPECTED = [
    PdfPageText(page_number=1, text="page one", extraction_method="fake", confidence=0.9),
    PdfPageText(page_number=2, text="page two", extraction_method="fake"),
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
