"""PyMuPdfExtractor — embedded-text PDF extraction via PyMuPDF (doc 11 § 2).

PyMuPDF is AGPL-3.0 (doc 13 § 3). That is acceptable for this project's
personal / self-hosted scope; if the project is ever distributed publicly, swap
to ``pdfplumber`` (or another permissively-licensed library) behind the
``PdfTextExtractor`` interface rather than shipping under AGPL.

Imports ``pymupdf`` (not the legacy ``fitz`` alias): the ``fitz`` package ships
no ``py.typed`` and fails strict mypy, whereas ``pymupdf`` is typed. PyMuPDF's
own functions remain unannotated, so the few API calls carry targeted
``# type: ignore[no-untyped-call]`` (kept honest by ``warn_unused_ignores``).
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from threading import Lock

import pymupdf

from rag_recipes.providers.errors import PdfExtractionError
from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

__all__ = ["PyMuPdfExtractor"]

# PyMuPDF runs MuPDF in single-threaded mode (it calls reinit_singlethreaded()
# at import); dispatching concurrent extractions onto multiple threads risks
# native crashes or silent corruption (Phase 4.2 REVIEW.md round-1 #1 /
# round-2 #1). Funnel every extraction through one dedicated worker thread so
# MuPDF only ever sees a single caller, regardless of in-flight concurrency.
# If a future PyMuPDF release drops the single-threaded constraint, this
# executor can be removed and extract_pages can revert to asyncio.to_thread.
_PDF_EXECUTOR: ThreadPoolExecutor | None = None
_PDF_EXECUTOR_LOCK = Lock()


def _get_pdf_executor() -> ThreadPoolExecutor:
    """Lazily build the process-global single-worker PyMuPDF executor.

    Double-checked locking keeps initialisation thread-safe while avoiding the
    import-time cost of spawning the worker thread (test collection imports this
    module without necessarily extracting anything).
    """
    global _PDF_EXECUTOR
    if _PDF_EXECUTOR is None:
        with _PDF_EXECUTOR_LOCK:
            if _PDF_EXECUTOR is None:
                _PDF_EXECUTOR = ThreadPoolExecutor(
                    max_workers=1, thread_name_prefix="pymupdf"
                )
    return _PDF_EXECUTOR


class PyMuPdfExtractor(PdfTextExtractor):
    """Extract embedded text from a PDF's bytes, flagging sparse pages.

    Pages whose stripped text is shorter than ``min_text_chars`` carry
    ``confidence=0.0`` (a "revisit with OCR" marker); normal pages carry
    ``None``. ``extraction_method`` is ``"embedded_text"`` for every page.
    """

    def __init__(self, min_text_chars: int) -> None:
        self._min_text_chars = min_text_chars

    async def extract_pages(self, file: bytes) -> list[PdfPageText]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(_get_pdf_executor(), self._extract, file)

    def _extract(self, file: bytes) -> list[PdfPageText]:
        try:
            with pymupdf.open(stream=file, filetype="pdf") as doc:  # type: ignore[no-untyped-call]
                pages: list[PdfPageText] = []
                for index, page in enumerate(doc):
                    text = page.get_text("text")
                    confidence = 0.0 if len(text.strip()) < self._min_text_chars else None
                    pages.append(
                        PdfPageText(
                            page_number=index + 1,
                            text=text,
                            extraction_method="embedded_text",
                            confidence=confidence,
                        )
                    )
                return pages
        except Exception as exc:
            raise PdfExtractionError(f"PyMuPDF extraction failed: {exc}") from exc
