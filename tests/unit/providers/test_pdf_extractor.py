"""Bind FakePdfTextExtractor and the real PyMuPdfExtractor to the contract suite."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pymupdf
import pytest

from rag_recipes.providers.errors import PdfExtractionError
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


class _CloseSpyDoc:
    """Wraps a real PyMuPDF document, recording how often it is closed.

    The extractor uses ``with pymupdf.open(...) as doc`` and iterates the value
    returned by ``__enter__``; ``__exit__`` here closes the wrapped document so
    we can assert closure on both the success and post-open-failure paths.
    """

    def __init__(self, doc: Any) -> None:
        self._doc = doc
        self.close_calls = 0

    def __enter__(self) -> Any:
        return self._doc

    def __exit__(self, *exc_info: object) -> bool:
        self._doc.close()
        self.close_calls += 1
        return False


def _spy_open(monkeypatch: pytest.MonkeyPatch) -> list[_CloseSpyDoc]:
    """Patch ``pymupdf.open`` to wrap the real document in a close-spy."""
    original_open = pymupdf.open
    spies: list[_CloseSpyDoc] = []

    def fake_open(*args: object, **kwargs: object) -> _CloseSpyDoc:
        spy = _CloseSpyDoc(original_open(*args, **kwargs))
        spies.append(spy)
        return spy

    monkeypatch.setattr(pymupdf, "open", fake_open)
    return spies


async def test_extract_pages_is_deterministic() -> None:
    extractor = PyMuPdfExtractor(min_text_chars=20)
    data = _FIXTURE.read_bytes()
    assert await extractor.extract_pages(data) == await extractor.extract_pages(data)


async def test_pages_are_one_indexed_in_document_order() -> None:
    extractor = PyMuPdfExtractor(min_text_chars=20)
    pages = await extractor.extract_pages(_FIXTURE.read_bytes())
    assert [page.page_number for page in pages] == [1, 2, 3]


async def test_sparse_page_flagged_and_method_constant() -> None:
    extractor = PyMuPdfExtractor(min_text_chars=20)
    pages = await extractor.extract_pages(_FIXTURE.read_bytes())
    assert all(page.extraction_method == "embedded_text" for page in pages)
    assert pages[-1].confidence == 0.0
    assert all(page.confidence is None for page in pages[:-1])


async def test_non_pdf_bytes_raise_pdf_extraction_error() -> None:
    # pymupdf.open(stream=...) raises during open, before any document exists,
    # so there is nothing to close on this path — only the error is asserted.
    extractor = PyMuPdfExtractor(min_text_chars=20)
    with pytest.raises(PdfExtractionError):
        await extractor.extract_pages(b"not a pdf")


async def test_document_closed_on_success_path(monkeypatch: pytest.MonkeyPatch) -> None:
    spies = _spy_open(monkeypatch)
    extractor = PyMuPdfExtractor(min_text_chars=20)
    await extractor.extract_pages(_FIXTURE.read_bytes())
    assert [spy.close_calls for spy in spies] == [1]


async def test_document_closed_on_post_open_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    spies = _spy_open(monkeypatch)

    def boom(*args: object, **kwargs: object) -> str:
        raise RuntimeError("extraction blew up after open")

    monkeypatch.setattr(pymupdf.Page, "get_text", boom)
    extractor = PyMuPdfExtractor(min_text_chars=20)
    with pytest.raises(PdfExtractionError):
        await extractor.extract_pages(_FIXTURE.read_bytes())
    assert [spy.close_calls for spy in spies] == [1]


async def test_concurrent_extractions_against_synthetic_fixture() -> None:
    """Regression test for the Phase 4.2 PyMuPDF concurrency footgun.

    PyMuPDF runs MuPDF in single-threaded mode (reinit_singlethreaded() at
    import); concurrent extract calls from multiple executor threads risk
    native crashes or silent corruption. The provider now dispatches via a
    module-level single-worker executor, so N concurrent calls must all
    return identical, correct results.
    """
    extractor = PyMuPdfExtractor(min_text_chars=20)
    pdf_bytes = _FIXTURE.read_bytes()
    results = await asyncio.gather(
        *[extractor.extract_pages(pdf_bytes) for _ in range(8)]
    )
    for pages in results:
        assert [p.model_dump() for p in pages] == [p.model_dump() for p in _FIXTURE_PAGES]
