"""Unit tests for the pure page-windowing layer in ingestion.pipeline.windows.

No DB, no session: ``SourceSpan`` instances are constructed detached and only
their ``id`` / ``text`` / ``text_hash`` / ``locator`` attributes are read.
"""

from __future__ import annotations

import hashlib

import pytest

from rag_recipes.ingestion.pipeline.windows import (
    Window,
    build_windows,
    format_window_for_llm,
)
from rag_recipes.storage.models.source_span import SourceSpan


def _make_span(page: int, text: str | None = None) -> SourceSpan:
    """Build a detached per-page SourceSpan for windowing tests."""
    body = text if text is not None else f"text of page {page}"
    return SourceSpan(
        id=f"span_{page:03d}",
        text=body,
        text_hash=hashlib.sha256(body.encode("utf-8")).hexdigest(),
        locator={"type": "pdf_page_range", "page_start": page, "page_end": page},
    )


def _ranges(windows: list[Window]) -> list[tuple[int, int]]:
    return [w.page_range for w in windows]


@pytest.mark.parametrize(
    ("n_pages", "expected"),
    [
        (1, [(1, 1)]),
        (3, [(1, 3)]),
        (5, [(1, 3), (3, 5)]),
        (7, [(1, 3), (3, 5), (5, 7)]),
        (9, [(1, 3), (3, 5), (5, 7), (7, 9)]),
    ],
)
def test_build_windows_size3_overlap1(n_pages: int, expected: list[tuple[int, int]]) -> None:
    spans = [_make_span(p) for p in range(1, n_pages + 1)]
    windows = build_windows(spans, window_size=3, overlap=1)
    assert _ranges(windows) == expected


def test_build_windows_overlap0_is_non_overlapping() -> None:
    spans = [_make_span(p) for p in range(1, 10)]
    windows = build_windows(spans, window_size=3, overlap=0)
    assert _ranges(windows) == [(1, 3), (4, 6), (7, 9)]


def test_build_windows_fewer_spans_than_window_size() -> None:
    spans = [_make_span(1), _make_span(2)]
    windows = build_windows(spans, window_size=3, overlap=1)
    assert len(windows) == 1
    assert windows[0].span_ids == ["span_001", "span_002"]


def test_build_windows_single_span() -> None:
    windows = build_windows([_make_span(1)], window_size=3, overlap=1)
    assert len(windows) == 1
    assert windows[0].page_range == (1, 1)


def test_build_windows_empty() -> None:
    assert build_windows([], window_size=3, overlap=1) == []


@pytest.mark.parametrize(
    ("window_size", "overlap"),
    [
        (0, 0),
        (3, -1),
        (3, 3),
        (3, 4),
    ],
)
def test_build_windows_invalid_args_raise(window_size: int, overlap: int) -> None:
    spans = [_make_span(p) for p in range(1, 5)]
    with pytest.raises(ValueError):
        build_windows(spans, window_size=window_size, overlap=overlap)


def test_window_is_frozen() -> None:
    window = Window(spans=(_make_span(1),))
    with pytest.raises((AttributeError, TypeError)):
        window.spans = ()  # type: ignore[misc]


def test_window_span_ids_and_page_range() -> None:
    spans = (_make_span(2), _make_span(3), _make_span(4))
    window = Window(spans=spans)
    assert window.span_ids == ["span_002", "span_003", "span_004"]
    assert window.page_range == (2, 4)


def test_format_window_for_llm_two_spans() -> None:
    window = Window(
        spans=(
            _make_span(42, "text of span 42"),
            _make_span(43, "text of span 43"),
        )
    )
    expected = (
        "[SOURCE_SPAN span_042 | PDF page 42]\n"
        "text of span 42\n"
        "\n"
        "[SOURCE_SPAN span_043 | PDF page 43]\n"
        "text of span 43"
    )
    assert format_window_for_llm(window) == expected


def test_format_window_for_llm_single_span_has_no_trailing_separator() -> None:
    window = Window(spans=(_make_span(7, "lone page"),))
    assert format_window_for_llm(window) == "[SOURCE_SPAN span_007 | PDF page 7]\nlone page"
