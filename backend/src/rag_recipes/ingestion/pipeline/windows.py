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

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rag_recipes.storage.models.source_span import SourceSpan

__all__ = ["Window", "build_windows", "compute_input_hash", "format_window_for_llm"]


def _sha256_json(d: dict[str, Any]) -> str:
    """SHA-256 of a dict serialised to canonical JSON (sorted keys, no whitespace).

    Reimplements the canonical form used in ``pipeline/pdf_text._sha256_json``
    locally — that symbol is module-private, so it is not imported across modules.
    """
    canonical = json.dumps(d, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


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


def format_window_for_llm(window: Window) -> str:
    """Render a window as the doc-4 ``[SOURCE_SPAN <id> | PDF page N]`` input text.

    One block per span — ``f"[SOURCE_SPAN {span.id} | PDF page {page}]\\n{text}"``
    where ``page`` is the span's ``locator["page_start"]`` — joined by a blank
    line. The span ids are load-bearing: the LLM must echo them in its output.

    This shape is part of the prompt contract. Any observable change to it must
    bump ``prompt_version`` (which feeds ``compute_input_hash``), otherwise a
    stale cached extraction could be reused against the new format.
    """
    blocks = [
        f"[SOURCE_SPAN {span.id} | PDF page {span.locator['page_start']}]\n{span.text}"
        for span in window.spans
    ]
    return "\n\n".join(blocks)


def compute_input_hash(
    window: Window,
    prompt_version: str,
    schema_version: str,
) -> str:
    """Deterministic SHA-256 hex used as the extraction cache / dedup key.

    Hashes canonical JSON of
    ``{"prompt_version": …, "schema_version": …, "spans": [{"id", "text_hash"}, …]}``.
    Hashing ``(id, text_hash)`` pairs — not raw text or the formatted string —
    keys the cache to immutable source-text identity while staying stable against
    cosmetic format changes (DECISIONS #2). ``prompt_version`` is inside the hash,
    so a meaningful format change (which must bump it) correctly invalidates the
    cache.
    """
    payload = {
        "prompt_version": prompt_version,
        "schema_version": schema_version,
        "spans": [{"id": span.id, "text_hash": span.text_hash} for span in window.spans],
    }
    return _sha256_json(payload)
