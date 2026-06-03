"""Unit tests for the pure page-windowing layer in ingestion.pipeline.windows.

No DB, no session: ``SourceSpan`` instances are constructed detached and only
their ``id`` / ``text`` / ``text_hash`` / ``locator`` attributes are read.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from rag_recipes.ingestion.pipeline.windows import (
    Window,
    build_windows,
    compute_input_hash,
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


def test_compute_input_hash_is_deterministic() -> None:
    window = Window(spans=(_make_span(1), _make_span(2)))
    first = compute_input_hash(window, prompt_version="p1", schema_version="s1")
    second = compute_input_hash(window, prompt_version="p1", schema_version="s1")
    assert first == second


def test_compute_input_hash_matches_canonical_json() -> None:
    span = _make_span(1, "page one")
    window = Window(spans=(span,))
    expected_doc = {
        "prompt_version": "p1",
        "schema_version": "s1",
        "spans": [{"id": span.id, "text_hash": span.text_hash}],
    }
    canonical = json.dumps(expected_doc, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    assert compute_input_hash(window, prompt_version="p1", schema_version="s1") == expected


def test_compute_input_hash_changes_with_prompt_version() -> None:
    window = Window(spans=(_make_span(1), _make_span(2)))
    base = compute_input_hash(window, prompt_version="p1", schema_version="s1")
    assert compute_input_hash(window, prompt_version="p2", schema_version="s1") != base


def test_compute_input_hash_changes_with_schema_version() -> None:
    window = Window(spans=(_make_span(1), _make_span(2)))
    base = compute_input_hash(window, prompt_version="p1", schema_version="s1")
    assert compute_input_hash(window, prompt_version="p1", schema_version="s2") != base


def test_compute_input_hash_changes_with_span_order() -> None:
    s1, s2 = _make_span(1), _make_span(2)
    base = compute_input_hash(Window(spans=(s1, s2)), prompt_version="p1", schema_version="s1")
    reordered = compute_input_hash(
        Window(spans=(s2, s1)), prompt_version="p1", schema_version="s1"
    )
    assert reordered != base


def test_compute_input_hash_changes_with_text_hash() -> None:
    base = compute_input_hash(
        Window(spans=(_make_span(1, "original"),)),
        prompt_version="p1",
        schema_version="s1",
    )
    changed = compute_input_hash(
        Window(spans=(_make_span(1, "different"),)),
        prompt_version="p1",
        schema_version="s1",
    )
    assert changed != base


def test_compute_input_hash_changes_with_span_id() -> None:
    span = _make_span(1, "page one")
    other = _make_span(1, "page one")
    object.__setattr__(other, "id", "span_999")
    base = compute_input_hash(Window(spans=(span,)), prompt_version="p1", schema_version="s1")
    changed = compute_input_hash(Window(spans=(other,)), prompt_version="p1", schema_version="s1")
    assert changed != base
