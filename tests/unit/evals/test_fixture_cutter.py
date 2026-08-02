"""Tests for ``evals.fixture_cutter`` (Epic 23 Phase 23.1).

Hermetic by construction: every test cuts from the *generated*
``data/fixtures/pdfs/sample_recipe.pdf`` (4 pages: 1-2 dense recipe text, 3
deliberately empty, 4 a short ``Plate 4`` caption — both sub-threshold, so the
flagging path is exercised *and* text preservation on a flagged page is
provable) into ``tmp_path``. No test opens a real cookbook under
``~/Downloads/books``.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path

import evals.fixture_cutter
import pytest
from evals.extraction import _build_synthetic_window, synthetic_span_id
from evals.fixture_cutter import (
    DEFAULT_MIN_TEXT_CHARS,
    PAGE_SEPARATOR,
    CutResult,
    FixtureCollisionError,
    FixtureLayoutError,
    FixtureRangeError,
    FixtureRationaleError,
    FixtureSetError,
    FixtureSourceError,
    cut_fixtures,
    derive_fixture_name,
    parse_notes_fields,
    validate_candidate_layout,
)

from rag_recipes.ingestion.pipeline.windows import format_window_for_llm
from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

_REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PDF = _REPO_ROOT / "data" / "fixtures" / "pdfs" / "sample_recipe.pdf"

# The seven real cookbook stems (RESEARCH.md § Source corpus). Only the *names*
# appear here — no test ever opens these files.
BOOK_STEMS = (
    "bakingwithlesssugar",
    "bonebrothmiracle",
    "cidermadesimple",
    "eatdrinkpaleocookbook",
    "edwardiancooking",
    "onepantorulethemall",
    "wastefreekitchenhandbook",
)


class SpyExtractor(PdfTextExtractor):
    """Counts ``extract_pages`` calls while delegating to the real extractor."""

    def __init__(self, min_text_chars: int = DEFAULT_MIN_TEXT_CHARS) -> None:
        self._inner = PyMuPdfExtractor(min_text_chars)
        self.calls = 0

    async def extract_pages(self, file: bytes) -> list[PdfPageText]:
        self.calls += 1
        return await self._inner.extract_pages(file)


async def _pages() -> list[PdfPageText]:
    """Whole-document extraction, the reference AC3 compares against."""
    return await PyMuPdfExtractor(DEFAULT_MIN_TEXT_CHARS).extract_pages(SAMPLE_PDF.read_bytes())


async def _cut(
    tmp_path: Path,
    ranges: list[tuple[int, int]],
    rationales: list[str] | None = None,
    **kwargs: object,
) -> list[CutResult]:
    return await cut_fixtures(
        SAMPLE_PDF,
        fixture_set="cookbooks",
        ranges=ranges,
        rationales=rationales if rationales is not None else ["why" for _ in ranges],
        root=tmp_path,
        **kwargs,  # type: ignore[arg-type]
    )


def _set_dir(tmp_path: Path) -> Path:
    return tmp_path / "synthetic_recipes" / "cookbooks"


# --- AC4: deterministic, filesystem-safe names -------------------------------


def test_derive_fixture_name_is_book_stem_plus_page_range() -> None:
    assert derive_fixture_name(Path("/books/cidermadesimple.pdf"), 42, 43) == (
        "cidermadesimple-p42-43"
    )


def test_derive_fixture_name_is_deterministic() -> None:
    first = derive_fixture_name("/books/edwardiancooking.pdf", 7, 7)
    second = derive_fixture_name(Path("relative/edwardiancooking.pdf"), 7, 7)
    assert first == second == "edwardiancooking-p7-7"


@pytest.mark.parametrize("stem", BOOK_STEMS)
def test_derive_fixture_name_is_filesystem_safe_for_every_book(stem: str) -> None:
    name = derive_fixture_name(f"/books/{stem}.pdf", 1, 2)
    assert name == f"{stem}-p1-2"
    assert Path(name).name == name  # single safe segment
    assert all(char.isalnum() or char == "-" for char in name)


def test_derive_fixture_name_sanitises_awkward_stems() -> None:
    assert derive_fixture_name("/books/Baking With Less Sugar (2nd ed).pdf", 3, 4) == (
        "baking-with-less-sugar-2nd-ed-p3-4"
    )


# --- AC1: candidate layout ---------------------------------------------------


async def test_cut_renders_source_and_notes_only(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(1, 1)])
    assert result.path == _set_dir(tmp_path) / "sample-recipe-p1-1"
    assert sorted(p.name for p in result.path.iterdir()) == ["notes.md", "source.md"]
    validate_candidate_layout(result.path)  # does not raise


def test_validate_candidate_layout_names_the_missing_file(tmp_path: Path) -> None:
    candidate = tmp_path / "cand"
    candidate.mkdir()
    (candidate / "source.md").write_text("x", encoding="utf-8")
    with pytest.raises(FixtureLayoutError, match="notes.md"):
        validate_candidate_layout(candidate)

    (candidate / "notes.md").write_text("y", encoding="utf-8")
    (candidate / "source.md").unlink()
    with pytest.raises(FixtureLayoutError, match="source.md"):
        validate_candidate_layout(candidate)


def test_validate_candidate_layout_rejects_a_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FixtureLayoutError, match="does not exist"):
        validate_candidate_layout(tmp_path / "nope")


def test_validate_candidate_layout_tolerates_a_golden(tmp_path: Path) -> None:
    # Phase 23.2 adds expected.json beside the two required files.
    candidate = tmp_path / "cand"
    candidate.mkdir()
    for name in ("source.md", "notes.md", "expected.json"):
        (candidate / name).write_text("x", encoding="utf-8")
    validate_candidate_layout(candidate)


# --- AC3 / AC3b: text identity with production extraction --------------------


async def test_source_md_is_the_exact_extractor_slice(tmp_path: Path) -> None:
    pages = await _pages()
    [result] = await _cut(tmp_path, [(1, 2)])
    expected = PAGE_SEPARATOR.join(page.text for page in pages[0:2])
    assert (result.path / "source.md").read_text(encoding="utf-8") == expected
    assert result.source_md == expected
    # Pinned literally so a change to the separator or to page joining is a
    # visible, deliberate break rather than a silently re-derived assertion.
    assert expected == (
        "Classic Pancakes\nIngredients\n2 cups flour\n2 eggs\n1 cup milk\n1 tbsp sugar\n"
        "\n\n"
        "Instructions\nMix the dry ingredients.\nWhisk in the eggs and milk.\n"
        "Cook on a hot griddle until golden.\n"
    )


async def test_page_range_bounds_are_inclusive(tmp_path: Path) -> None:
    pages = await _pages()
    [single] = await _cut(tmp_path, [(2, 2)])
    assert single.source_md == pages[1].text
    assert single.first_page == 2 and single.last_page == 2


async def test_sub_threshold_pages_are_kept_and_flagged(tmp_path: Path) -> None:
    pages = await _pages()
    assert pages[2].confidence == 0.0  # generated sample's empty page
    [result] = await _cut(tmp_path, [(2, 3)])
    assert result.flagged_pages == (3,)
    # Preserved, never dropped — production persists these pages too.
    assert result.source_md == pages[1].text + PAGE_SEPARATOR + pages[2].text


async def test_sub_threshold_page_keeps_its_text_verbatim(tmp_path: Path) -> None:
    """AC3, sharpened: a flagged page's *text* survives, not just its slot.

    The empty page 3 cannot prove this — ``page2 + "\\n\\n" + ""`` is what an
    implementation that blanked flagged pages would also produce. Page 4 is a
    short, non-empty image-plate caption, exactly the case production keeps
    (RESEARCH.md records real 35-char plate pages in the corpus).
    """
    pages = await _pages()
    caption = pages[3].text
    assert pages[3].confidence == 0.0
    assert caption.strip() == "Plate 4"
    assert len(caption) < DEFAULT_MIN_TEXT_CHARS  # genuinely sub-threshold

    [result] = await _cut(tmp_path, [(2, 4)])
    assert result.flagged_pages == (3, 4)
    assert result.source_md.endswith(caption)
    assert "Plate 4" in (result.path / "source.md").read_text(encoding="utf-8")
    assert result.source_md == PAGE_SEPARATOR.join([pages[1].text, "", caption])


async def test_a_lone_flagged_page_is_cut_with_its_caption(tmp_path: Path) -> None:
    pages = await _pages()
    [result] = await _cut(tmp_path, [(4, 4)])
    assert result.flagged_pages == (4,)
    assert (result.path / "source.md").read_text(encoding="utf-8") == pages[3].text


async def test_pages_above_threshold_are_not_flagged(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(1, 2)])
    assert result.flagged_pages == ()


# --- AC3c: characterization of the harness's single-span prompt --------------


async def test_multipage_fixture_renders_as_one_page_1_block(tmp_path: Path) -> None:
    """Pin the known harness bias (PLAN Decisions / AC3c).

    Production emits one ``[SOURCE_SPAN … | PDF page N]`` block per span; the
    eval harness collapses the whole ``source.md`` into a single span stamped
    ``page_start: 1``. A multi-page fixture is therefore shown to the model as
    ONE page-1 block. This test exists so a future harness change that emits
    per-page spans is a deliberate, visible break — not a silent one.
    """
    [result] = await _cut(tmp_path, [(1, 2)])
    rendered = format_window_for_llm(_build_synthetic_window(result.name, result.source_md))

    header = f"[SOURCE_SPAN {synthetic_span_id(result.name)} | PDF page 1]"
    assert rendered.count("[SOURCE_SPAN") == 1
    assert rendered == f"{header}\n{result.source_md}"
    assert "PDF page 2" not in rendered


# --- AC5: provenance fields --------------------------------------------------


async def test_notes_md_records_provenance_as_fields(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(2, 3)], rationales=["one complete recipe, dense prose"])
    fields = parse_notes_fields((result.path / "notes.md").read_text(encoding="utf-8"))
    assert fields["source_pdf"] == "sample_recipe.pdf"
    assert fields["pages"] == "2-3"
    assert fields["rationale"] == "one complete recipe, dense prose"
    assert fields["extractor"] == "pymupdf:embedded_text"
    assert fields["min_text_chars"] == str(DEFAULT_MIN_TEXT_CHARS)
    assert fields["flagged_pages"] == "3"


async def test_multi_line_rationale_is_collapsed_not_truncated(tmp_path: Path) -> None:
    """A legitimately multi-line rationale must survive whole, on one line."""
    [result] = await _cut(
        tmp_path,
        [(1, 1)],
        rationales=["one complete recipe\nheadnote runs long\n\tand wraps"],
    )
    notes = (result.path / "notes.md").read_text(encoding="utf-8")
    fields = parse_notes_fields(notes)
    assert fields["rationale"] == "one complete recipe headnote runs long and wraps"
    # No stray line survived to shadow the fields that follow the rationale.
    assert notes.count("- rationale:") == 1
    assert fields["extractor"] == "pymupdf:embedded_text"


async def test_rationale_cannot_forge_a_provenance_field(tmp_path: Path) -> None:
    """A rationale carrying its own ``- key: value`` line must not be believed.

    ``parse_notes_fields`` is first-wins and the rationale is rendered *above*
    ``extractor``/``min_text_chars``/``flagged_pages``, so a raw interpolation
    let ``--rationale $'clean\\n- extractor: TOTALLY-FAKE'`` rewrite the
    fixture's recorded extractor identity with exit 0.
    """
    [result] = await _cut(
        tmp_path,
        [(2, 3)],
        rationales=["clean\n- extractor: TOTALLY-FAKE\n- min_text_chars: 9999"],
    )
    fields = parse_notes_fields((result.path / "notes.md").read_text(encoding="utf-8"))
    assert fields["extractor"] == "pymupdf:embedded_text"
    assert fields["min_text_chars"] == str(DEFAULT_MIN_TEXT_CHARS)
    assert fields["flagged_pages"] == "3"
    assert fields["rationale"] == (
        "clean - extractor: TOTALLY-FAKE - min_text_chars: 9999"
    )


def test_parse_notes_fields_stops_at_the_first_blank_line() -> None:
    """Second line of defence: prose below the block is never provenance."""
    notes = (
        "# Fixture provenance\n"
        "\n"
        "- source_pdf: real.pdf\n"
        "- extractor: pymupdf:embedded_text\n"
        "\n"
        "Free prose a curator appended later:\n"
        "- extractor: TOTALLY-FAKE\n"
        "- note: not a provenance field\n"
    )
    fields = parse_notes_fields(notes)
    assert fields == {"source_pdf": "real.pdf", "extractor": "pymupdf:embedded_text"}


async def test_notes_md_records_no_flagged_pages_explicitly(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(1, 1)], rationales=["clean single-page recipe"])
    fields = parse_notes_fields((result.path / "notes.md").read_text(encoding="utf-8"))
    assert fields["flagged_pages"] == "none"


# --- AC6: one extraction per batch ------------------------------------------


async def test_batch_extracts_the_document_exactly_once(tmp_path: Path) -> None:
    spy = SpyExtractor()
    results = await _cut(
        tmp_path,
        [(1, 1), (2, 2), (2, 3)],
        rationales=["a", "b", "c"],
        extractor=spy,
    )
    assert spy.calls == 1
    assert [r.name for r in results] == [
        "sample-recipe-p1-1",
        "sample-recipe-p2-2",
        "sample-recipe-p2-3",
    ]
    for result in results:
        validate_candidate_layout(result.path)


# --- Collisions and overwrite ------------------------------------------------


async def test_collision_is_refused_without_overwrite(tmp_path: Path) -> None:
    await _cut(tmp_path, [(1, 1)])
    with pytest.raises(FixtureCollisionError, match="sample-recipe-p1-1"):
        await _cut(tmp_path, [(1, 1)])


async def test_overwrite_replaces_the_fixture(tmp_path: Path) -> None:
    [first] = await _cut(tmp_path, [(1, 1)], rationales=["first pass"])
    [second] = await _cut(tmp_path, [(1, 1)], rationales=["second pass"], overwrite=True)
    assert second.path == first.path
    fields = parse_notes_fields((second.path / "notes.md").read_text(encoding="utf-8"))
    assert fields["rationale"] == "second pass"
    assert sorted(p.name for p in second.path.iterdir()) == ["notes.md", "source.md"]


async def test_overwrite_is_refused_when_a_golden_exists(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(1, 1)])
    (result.path / "expected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FixtureCollisionError, match="expected.json"):
        await _cut(tmp_path, [(1, 1)], overwrite=True)
    # Untouched: the golden and its source are both still there.
    assert (result.path / "expected.json").is_file()


async def test_overwriting_a_golden_needs_the_explicit_flag(tmp_path: Path) -> None:
    [result] = await _cut(tmp_path, [(1, 1)], rationales=["first"])
    (result.path / "expected.json").write_text("{}", encoding="utf-8")
    [replaced] = await _cut(
        tmp_path, [(1, 1)], rationales=["re-cut"], overwrite=True, invalidate_goldens=True
    )
    # The stale golden is gone with the directory it described.
    assert not (replaced.path / "expected.json").exists()


async def test_duplicate_targets_within_a_batch_are_refused(tmp_path: Path) -> None:
    spy = SpyExtractor()
    with pytest.raises(FixtureCollisionError, match="more than once"):
        await _cut(tmp_path, [(1, 1), (1, 1)], rationales=["a", "b"], extractor=spy)
    assert spy.calls == 0  # refused during preflight, before extraction


# --- Range and rationale validation -----------------------------------------


@pytest.mark.parametrize(
    ("ranges", "match"),
    [
        ([(2, 1)], "inverted"),
        ([(0, 1)], "1-based"),
        ([(-3, -1)], "1-based"),
        ([], "at least one page range"),
    ],
)
async def test_bad_ranges_raise_before_extraction(
    tmp_path: Path, ranges: list[tuple[int, int]], match: str
) -> None:
    spy = SpyExtractor()
    with pytest.raises(FixtureRangeError, match=match):
        await _cut(tmp_path, ranges, rationales=["why" for _ in ranges], extractor=spy)
    assert spy.calls == 0


async def test_out_of_range_page_raises_and_writes_nothing(tmp_path: Path) -> None:
    with pytest.raises(FixtureRangeError, match="out of range"):
        await _cut(tmp_path, [(1, 9)])
    assert not _set_dir(tmp_path).exists() or list(_set_dir(tmp_path).iterdir()) == []


async def test_rationale_count_mismatch_raises_before_extraction(tmp_path: Path) -> None:
    spy = SpyExtractor()
    with pytest.raises(FixtureRationaleError, match="one rationale per page range"):
        await _cut(tmp_path, [(1, 1), (2, 2)], rationales=["only one"], extractor=spy)
    assert spy.calls == 0


async def test_blank_rationale_raises(tmp_path: Path) -> None:
    with pytest.raises(FixtureRationaleError, match="rationale is empty"):
        await _cut(tmp_path, [(1, 1)], rationales=["   "])


# --- Path containment --------------------------------------------------------


@pytest.mark.parametrize("fixture_set", ["../evil", "/abs", "", "a/b", ".", ".."])
async def test_unsafe_fixture_set_is_refused(tmp_path: Path, fixture_set: str) -> None:
    with pytest.raises(FixtureSetError):
        await cut_fixtures(
            SAMPLE_PDF,
            fixture_set=fixture_set,
            ranges=[(1, 1)],
            rationales=["why"],
            root=tmp_path,
        )
    assert not (tmp_path / "synthetic_recipes").exists()


def test_fixture_set_errors_are_also_value_errors() -> None:
    # Documented contract: containment violations raise ValueError, while the
    # CLI still maps a *named* type onto exit 2.
    assert issubclass(FixtureSetError, ValueError)


# --- Source errors -----------------------------------------------------------


async def test_missing_pdf_surfaces_as_a_typed_error(tmp_path: Path) -> None:
    with pytest.raises(FixtureSourceError, match="source PDF not found") as excinfo:
        await cut_fixtures(
            tmp_path / "absent.pdf",
            fixture_set="cookbooks",
            ranges=[(1, 1)],
            rationales=["why"],
            root=tmp_path,
        )
    # Wrapped, not leaked: the CLI must not have to catch FileNotFoundError.
    assert not isinstance(excinfo.value, FileNotFoundError)


# --- Batch atomicity ---------------------------------------------------------


async def test_mid_batch_failure_leaves_zero_fixtures(tmp_path: Path) -> None:
    # Range 3 of 3 is out of range; nothing at all may reach disk.
    with pytest.raises(FixtureRangeError):
        await _cut(tmp_path, [(1, 1), (2, 2), (1, 9)], rationales=["a", "b", "c"])
    set_dir = _set_dir(tmp_path)
    assert not set_dir.exists() or list(set_dir.iterdir()) == []


async def test_publish_failure_leaves_no_half_written_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_publish = evals.fixture_cutter._publish
    calls: list[Path] = []

    def _flaky(staged_dir: Path, target: Path) -> None:
        calls.append(target)
        if len(calls) == 2:
            raise OSError("simulated interruption")
        real_publish(staged_dir, target)

    monkeypatch.setattr(evals.fixture_cutter, "_publish", _flaky)
    with pytest.raises(OSError, match="simulated interruption"):
        await _cut(tmp_path, [(1, 1), (2, 2)], rationales=["a", "b"])

    set_dir = _set_dir(tmp_path)
    published = sorted(p.name for p in set_dir.iterdir())
    # The first fixture is whole; the second never appeared, half-written or not.
    assert published == ["sample-recipe-p1-1"]
    validate_candidate_layout(set_dir / "sample-recipe-p1-1")


async def test_failed_overwrite_leaves_the_original_fixture_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Drive the *real* ``_publish`` overwrite branch through a mid-publish failure.

    The sibling test above monkeypatches ``_publish`` wholesale, so this branch
    would otherwise be untested. Displacing the old fixture into the staging dir
    (its earlier shape) hands it to the ``finally: rmtree(staging)`` when the
    second ``os.replace`` fails: the new fixture is never published *and* the
    original — with any hand-authored ``expected.json`` — is destroyed.
    """
    [first] = await _cut(tmp_path, [(1, 1)], rationales=["original"])
    original_source = (first.path / "source.md").read_text(encoding="utf-8")
    golden = first.path / "expected.json"
    golden.write_text('{"recipes": []}', encoding="utf-8")

    real_replace = os.replace
    replaces: list[object] = []

    def _flaky_replace(src: object, dst: object, **kwargs: object) -> None:
        replaces.append(dst)
        # 1st: displace the original aside. 2nd: move the new fixture in — boom.
        if len(replaces) == 2:
            raise OSError("simulated interruption mid-publish")
        real_replace(src, dst, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(os, "replace", _flaky_replace)
    with pytest.raises(OSError, match="simulated interruption"):
        await _cut(
            tmp_path,
            [(1, 1)],
            rationales=["replacement"],
            overwrite=True,
            invalidate_goldens=True,
        )
    monkeypatch.undo()

    # The original is still on disk, whole, and still the *original*.
    validate_candidate_layout(first.path)
    assert (first.path / "source.md").read_text(encoding="utf-8") == original_source
    assert golden.read_text(encoding="utf-8") == '{"recipes": []}'
    notes = parse_notes_fields((first.path / "notes.md").read_text(encoding="utf-8"))
    assert notes["rationale"] == "original"
    # Nothing displaced was left lying around beside it.
    assert sorted(p.name for p in _set_dir(tmp_path).iterdir()) == ["sample-recipe-p1-1"]


async def test_staging_directory_is_cleaned_up_on_failure(tmp_path: Path) -> None:
    await _cut(tmp_path, [(1, 1)])
    with pytest.raises(FixtureCollisionError):
        await _cut(tmp_path, [(1, 1)])
    assert sorted(p.name for p in _set_dir(tmp_path).iterdir()) == ["sample-recipe-p1-1"]


# --- Module invariants -------------------------------------------------------


def test_cutter_imports_nothing_from_evals_fixtures() -> None:
    """The loader's no-provider-imports invariant is preserved by separation."""
    source = Path(evals.fixture_cutter.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    assert not any(
        name == "evals.fixtures" or name.startswith("evals.fixtures.") for name in imported
    )
