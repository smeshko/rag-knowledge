"""Unit tests for _citation_label (doc 7 § 10).

Locator keys (``page_start`` / ``page_end``) match what the Epic 8 span writer
(``ingestion/pipeline/pdf_text.py``) stamps into ``SourceSpan.locator``.
"""

from __future__ import annotations

from rag_recipes.retrieval.search import _citation_label


def test_single_page() -> None:
    assert _citation_label({"page_start": 42, "page_end": 42}) == "page 42"


def test_page_range_uses_en_dash() -> None:
    assert _citation_label({"page_start": 42, "page_end": 43}) == "pages 42–43"


def test_missing_end_reads_single_page() -> None:
    assert _citation_label({"page_start": 7}) == "page 7"


def test_missing_start_yields_empty_label() -> None:
    assert _citation_label({}) == ""
