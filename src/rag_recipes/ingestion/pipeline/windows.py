"""Page-windowing layer for LLM recipe extraction (doc 3 § 5, doc 4).

Pure functions only: no DB session, no provider, no storage, no ``Settings``.
The caller loads ordered ``SourceSpan`` rows for the active ``source_version``
and injects the window size / overlap (from ``Settings.pdf_window_size_pages``
and ``Settings.pdf_overlap_pages``); this module never touches I/O.

Windowing rule: ``build_windows`` emits ``spans[start:start+window_size]`` for
``start`` stepping by ``window_size - overlap`` and stops after the first window
that includes the last span. This reproduces the doc-3 § 5 sequence exactly
(``1-3, 3-5, 5-7`` for 7 pages; ``1-3, 3-5, 5-7, 7-9`` for 9 pages) and avoids
trailing sub-windows fully contained in their predecessor (DECISIONS #1).
"""

from __future__ import annotations

from dataclasses import dataclass

from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["Window", "build_windows"]


@dataclass(frozen=True)
class Window:
    """An immutable group of per-page ``SourceSpan`` rows for one extraction call."""

    spans: tuple[SourceSpan, ...]

    @property
    def span_ids(self) -> list[str]:
        """Ordered span ids carried by this window."""
        return [span.id for span in self.spans]

    @property
    def page_range(self) -> tuple[int, int]:
        """``(first span's page_start, last span's page_end)``."""
        return (self.spans[0].locator["page_start"], self.spans[-1].locator["page_end"])


def build_windows(
    spans: list[SourceSpan],
    window_size: int,
    overlap: int,
) -> list[Window]:
    """Group pre-ordered per-page spans into overlapping page windows.

    ``spans`` is assumed pre-sorted by ``locator["page_start"]`` (the caller's
    responsibility, mirroring ``test_process_document_job.py``); this function
    does not re-sort. Empty ``spans`` yields ``[]``.

    Raises ``ValueError`` if ``window_size < 1`` or ``overlap`` is outside
    ``[0, window_size)`` — an ``overlap >= window_size`` would make ``step <= 0``
    and loop forever.
    """
    if window_size < 1:
        raise ValueError(f"window_size must be >= 1, got {window_size}")
    if not 0 <= overlap < window_size:
        raise ValueError(
            f"overlap must satisfy 0 <= overlap < window_size, "
            f"got overlap={overlap}, window_size={window_size}"
        )
    if not spans:
        return []

    step = window_size - overlap
    windows: list[Window] = []
    start = 0
    while True:
        windows.append(Window(spans=tuple(spans[start : start + window_size])))
        if start + window_size >= len(spans):
            break
        start += step
    return windows
