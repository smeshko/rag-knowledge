"""Manual smoke test for PyMuPdfExtractor.

Runs the real extractor against a PDF and prints per-page output so you can
eyeball extraction quality and sparse-page flagging.

    uv run python scripts/smoke_pdf_extract.py [PDF_PATH]

PDF_PATH defaults to the committed fixture data/fixtures/pdfs/sample_recipe.pdf.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from rag_recipes.config import get_settings
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor

_DEFAULT = Path("data/fixtures/pdfs/sample_recipe.pdf")


async def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT
    settings = get_settings()
    extractor = PyMuPdfExtractor(min_text_chars=settings.pdf_min_text_chars_for_page)
    pages = await extractor.extract_pages(path.read_bytes())

    print(f"{path} -> {len(pages)} page(s)")
    for page in pages:
        preview = page.text.strip().replace("\n", " ")[:60]
        print(f"  page {page.page_number}: confidence={page.confidence!r} preview={preview!r}")


if __name__ == "__main__":
    asyncio.run(main())
