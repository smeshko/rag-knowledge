"""Deterministically (re)generate ``sample_recipe.pdf`` for the PDF extractor tests.

Run with ``uv run python data/fixtures/pdfs/generate.py``. The output is
byte-stable: PDF metadata dates are pinned and the document ``/ID`` is fixed, so
re-running produces an identical file (the determinism acceptance criterion).

The fixture has three pages of recipe-like text plus one deliberately empty page
(page 3) that falls below ``PDF_MIN_TEXT_CHARS_FOR_PAGE`` so the extractor's
sparse-page flagging is exercised.
"""

from __future__ import annotations

from pathlib import Path

import pymupdf

OUTPUT = Path(__file__).with_name("sample_recipe.pdf")

# Pinned so reruns are byte-identical; PyMuPDF would otherwise stamp "now".
_FIXED_DATE = "D:20260101000000Z"
_FIXED_ID = b"\x00" * 16

_PAGES: list[str] = [
    "Classic Pancakes\nIngredients\n2 cups flour\n2 eggs\n1 cup milk\n1 tbsp sugar",
    "Instructions\nMix the dry ingredients.\nWhisk in the eggs and milk.\n"
    "Cook on a hot griddle until golden.",
    "",  # intentionally empty -> sub-threshold page
]


def build() -> bytes:
    doc = pymupdf.open()
    try:
        for body in _PAGES:
            page = doc.new_page()
            if body:
                page.insert_text((72, 72), body, fontsize=12, fontname="helv")
        doc.set_metadata(
            {
                "title": "Classic Pancakes",
                "author": "rag-recipes fixtures",
                "subject": "PDF extractor test fixture",
                "keywords": "",
                "creator": "data/fixtures/pdfs/generate.py",
                "producer": "rag-recipes",
                "creationDate": _FIXED_DATE,
                "modDate": _FIXED_DATE,
            }
        )
        doc.xref_set_key(-1, "ID", f"[<{_FIXED_ID.hex()}><{_FIXED_ID.hex()}>]")
        return doc.tobytes(garbage=4, deflate=True)
    finally:
        doc.close()


def main() -> None:
    OUTPUT.write_bytes(build())
    print(f"wrote {OUTPUT} ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
