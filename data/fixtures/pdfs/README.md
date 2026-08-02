# PDF test fixtures

## `sample_recipe.pdf`

A small, deterministic four-page PDF used by the `PyMuPdfExtractor` tests
(`tests/unit/providers/test_pdf_extractor.py`) and the fixture cutter tests
(`tests/unit/evals/test_fixture_cutter.py`):

- **Page 1** — recipe title + ingredients (well above the sparse threshold).
- **Page 2** — instructions (well above the threshold).
- **Page 3** — intentionally empty, so its extracted text is below
  `PDF_MIN_TEXT_CHARS_FOR_PAGE` (default 20) and the extractor flags it with
  `confidence=0.0`.
- **Page 4** — `Plate 4`: also below the threshold and also flagged, but **not
  empty**. It stands in for a real cookbook's captioned image plate and exists
  so tests can prove a flagged page keeps its *text*, not merely its slot —
  with only an empty sparse page, an implementation that blanked flagged pages
  would pass.

The file is a committed binary so tests exercise the real bytes-in path.

### Regenerating

```bash
uv run python data/fixtures/pdfs/generate.py
```

`generate.py` pins the PDF metadata dates and document `/ID`, so re-running
produces a **byte-identical** file — `git status` shows no diff afterwards. If
you change the page text, regenerate and update the expected pages in
`TestPyMuPdfExtractor`.
