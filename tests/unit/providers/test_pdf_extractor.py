"""Bind FakePdfTextExtractor to the shared PdfExtractor contract suite."""

from __future__ import annotations

import pytest

from rag_recipes.providers.errors import PdfExtractionError
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

    async def test_unknown_input_returns_placeholder(
        self, provider: FakePdfTextExtractor
    ) -> None:
        pages = await provider.extract_pages(b"unregistered bytes")
        assert len(pages) == 1
        assert pages[0].page_number == 1
        assert pages[0].extraction_method == "fake"
        assert pages[0].confidence is None

    async def test_unknown_input_raises_when_fallback_disabled(self) -> None:
        provider = FakePdfTextExtractor(fallback_to_placeholder=False)
        with pytest.raises(PdfExtractionError):
            await provider.extract_pages(b"unregistered bytes")

    async def test_returned_pages_are_isolated_from_fixture(
        self, provider: FakePdfTextExtractor
    ) -> None:
        pages = await provider.extract_pages(_KNOWN)
        pages.append(
            PdfPageText(page_number=99, text="injected", extraction_method="fake")
        )
        pages[0].text = "mutated"

        again = await provider.extract_pages(_KNOWN)
        assert again == _EXPECTED
        assert len(again) == 2
        assert again[0].text == "page one"
