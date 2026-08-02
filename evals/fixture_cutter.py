"""Cut real cookbook PDF page ranges into recipe *fixture candidates* (Epic 23 Phase 23.1).

Given a PDF **path** and one or more inclusive, 1-based page ranges, this module
renders ``data/fixtures/synthetic_recipes/<set>/<name>/{source.md, notes.md}``
using the **production** ``PyMuPdfExtractor``.

Three things a future reader must not "fix":

1. **Why this is not in ``evals/fixtures.py``.** That module's docstring pins an
   invariant — "Loaders are pure filesystem I/O — no ``Settings``, provider, or
   DB imports" — and the cutter needs ``PyMuPdfExtractor``, a provider. So the
   cutter lives here and imports nothing from ``evals.fixtures``.

2. **Phase 23.1 deliberately ships candidates without ``expected.json``.**
   Goldens are Phase 23.2's job. Until then
   ``evals.fixtures.load_recipe_fixtures("<set>")`` raises ``FileNotFoundError``
   for a set cut by this module, because it reads ``expected.json`` unguarded
   (``evals/fixtures.py:85-86``). That is expected, not a bug: the phase's
   acceptance gate is :func:`validate_candidate_layout`, and making the loader
   silently skip incomplete fixtures would hide a typo'd fixture name in 23.2.

3. **A fixture reproduces production's *text*, not production's *span
   structure*.** Production renders one ``[SOURCE_SPAN <id> | PDF page N]``
   block **per span** (``ingestion/pipeline/windows.py``), whereas the eval
   harness's ``_build_synthetic_window`` (``evals/extraction.py``) collapses the
   whole ``source.md`` into a **single** span stamped ``page_start: 1``. A
   multi-page fixture is therefore presented to the model as one page-1 block.
   That known harness bias is pinned by a characterization test; prefer
   single-page ranges where a recipe permits.

**Page separator.** ``source.md`` is exactly
``PAGE_SEPARATOR.join(page.text for page in selected_pages)`` — no header, no
trailing newline added, no normalisation. The pages come from **one**
whole-document ``extract_pages`` call that is then sliced, so the text is
byte-identical to what ingestion sees for the same pages. Sub-threshold pages
(``confidence == 0.0``) are **kept with their text**, exactly as production
keeps them (``providers/pdf_extractor/pymupdf.py:77`` flags, never drops); they
are merely reported in :attr:`CutResult.flagged_pages` so the curator can re-cut.
A range in which *every* page is sub-threshold is the one exception: it would
render an empty ``source.md``, and therefore an empty prompt in 23.2, so it is
refused rather than written.
"""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from rag_recipes.providers.errors import PdfExtractionError
from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

__all__ = [
    "DEFAULT_MIN_TEXT_CHARS",
    "FIXTURES_ROOT",
    "PAGE_SEPARATOR",
    "CutResult",
    "FixtureCollisionError",
    "FixtureCutterError",
    "FixtureLayoutError",
    "FixtureRangeError",
    "FixtureRationaleError",
    "FixtureSetError",
    "FixtureSourceError",
    "cut_fixtures",
    "derive_fixture_name",
    "parse_notes_fields",
    "validate_candidate_layout",
]

FIXTURES_ROOT = Path(__file__).resolve().parents[1] / "data" / "fixtures"

#: Separator joining consecutive pages' text inside ``source.md``. Part of the
#: fixture contract — changing it changes every fixture's bytes and therefore
#: every golden authored against them.
PAGE_SEPARATOR = "\n\n"

#: Mirrors ``Settings.pdf_min_text_chars_for_page`` (``config.py:84``). Held as a
#: constant rather than read from ``Settings`` so cutting a fixture never needs
#: operator credentials.
DEFAULT_MIN_TEXT_CHARS = 20

_UNSAFE_NAME_CHARS = re.compile(r"[^a-z0-9]+")


class FixtureCutterError(ValueError):
    """Base for every caller error this module raises.

    Subclasses ``ValueError`` so the documented "path containment violations
    raise ``ValueError``" contract holds, while still giving the CLI named
    types to map onto exit 2 without catching bare ``ValueError``.
    """


class FixtureRangeError(FixtureCutterError):
    """A page range is malformed, inverted, outside the document, or yields no text."""


class FixtureRationaleError(FixtureCutterError):
    """Rationales are missing, blank, or do not match the range count."""


class FixtureCollisionError(FixtureCutterError):
    """A target fixture already exists, or a batch names the same target twice."""


class FixtureSourceError(FixtureCutterError):
    """The source PDF is missing, unreadable, or failed to extract."""


class FixtureSetError(FixtureCutterError):
    """``fixture_set`` is not a single safe path segment inside the fixtures root."""


class FixtureLayoutError(FixtureCutterError):
    """A candidate directory is missing a required file."""


@dataclass(frozen=True)
class CutResult:
    """One rendered fixture candidate.

    ``flagged_pages`` holds the 1-based page numbers the extractor marked
    ``confidence=0.0`` (below ``min_text_chars``). Their text is still present in
    ``source_md`` — production keeps such pages too — but a cut that is mostly
    image plates is almost certainly a bad cut, so the CLI warns on them.
    """

    name: str
    path: Path
    first_page: int
    last_page: int
    source_md: str
    flagged_pages: tuple[int, ...]


def derive_fixture_name(pdf_path: Path | str, first_page: int, last_page: int) -> str:
    """Return the deterministic fixture name ``<book-stem>-p<first>-<last>``.

    Pure and filesystem-safe: the stem is lower-cased and every run of
    non-``[a-z0-9]`` characters collapses to a single ``-``. Names are derived
    rather than free-form because the name becomes
    ``evals.extraction.synthetic_span_id(name)``, the span id Phase 23.2's
    goldens cite — a free-form name risks a silent overwrite that would
    invalidate goldens with no error surface.
    """
    stem = _UNSAFE_NAME_CHARS.sub("-", Path(pdf_path).stem.lower()).strip("-")
    if not stem:
        raise FixtureSourceError(f"PDF filename yields an empty fixture stem: {pdf_path}")
    return f"{stem}-p{first_page}-{last_page}"


def validate_candidate_layout(directory: Path) -> None:
    """Assert a candidate dir carries a non-empty ``source.md`` and a ``notes.md``.

    Phase 23.1's acceptance gate, standing in for ``load_recipe_fixtures``,
    which reads ``expected.json`` unguarded and so raises for every goldenless
    candidate. Extra files are tolerated — Phase 23.2 adds ``expected.json``
    beside these two.

    ``source.md`` must hold something other than whitespace: an all-image-plate
    window produces a structurally valid but *empty* candidate, which would pass
    a presence-only gate and then become an empty prompt in 23.2.
    """
    if not directory.is_dir():
        raise FixtureLayoutError(f"candidate directory does not exist: {directory}")
    missing = [name for name in ("source.md", "notes.md") if not (directory / name).is_file()]
    if missing:
        raise FixtureLayoutError(
            f"candidate {directory} is missing required file(s): {', '.join(missing)}"
        )
    if not (directory / "source.md").read_text(encoding="utf-8").strip():
        raise FixtureLayoutError(
            f"candidate {directory} has an empty source.md: a fixture with no text "
            f"would be scored as an empty prompt"
        )


def parse_notes_fields(notes_md: str) -> dict[str, str]:
    """Parse ``notes.md``'s ``- key: value`` provenance block into a mapping.

    Provenance is recorded as discrete fields rather than free prose so a stale
    or mis-cut fixture is diagnosable — and assertable — later.

    Only the **first** ``- key: value`` block is parsed: the first blank line
    after it ends the scan. Together with the whitespace collapsing applied to
    the rationale at cut time, that stops prose further down the file from
    passing itself off as provenance.
    """
    fields: dict[str, str] = {}
    started = False
    for line in notes_md.splitlines():
        stripped = line.strip()
        if not stripped:
            if started:
                break  # header block ended; everything below is prose
            continue
        if not stripped.startswith("- ") or ":" not in stripped:
            continue
        key, _, value = stripped[2:].partition(":")
        key = key.strip()
        if key and key not in fields:
            fields[key] = value.strip()
            started = True
    return fields


def render_source_md(pages: Sequence[PdfPageText]) -> str:
    """Join selected pages' text with :data:`PAGE_SEPARATOR`, verbatim."""
    return PAGE_SEPARATOR.join(page.text for page in pages)


def _render_notes_md(
    *,
    source_pdf: str,
    first_page: int,
    last_page: int,
    rationale: str,
    min_text_chars: int,
    flagged_pages: Sequence[int],
) -> str:
    flagged = ", ".join(str(page) for page in flagged_pages) if flagged_pages else "none"
    # Collapsed to a single line: notes.md's provenance block is line-oriented
    # and first-wins, so a raw multi-line rationale would either truncate at its
    # first newline or — worse — inject a `- extractor: ...` line that shadows
    # the real one and falsifies the fixture's provenance.
    rationale = " ".join(rationale.split())
    return (
        "# Fixture provenance\n"
        "\n"
        f"- source_pdf: {source_pdf}\n"
        f"- pages: {first_page}-{last_page}\n"
        f"- rationale: {rationale}\n"
        "- extractor: pymupdf:embedded_text\n"
        f"- min_text_chars: {min_text_chars}\n"
        f"- page_separator: {PAGE_SEPARATOR!r}\n"
        f"- flagged_pages: {flagged}\n"
        "\n"
        "Cut by `rag-evals fixtures cut` (Epic 23 Phase 23.1). No `expected.json` yet — "
        "goldens are authored in Phase 23.2. Flagged pages are below `min_text_chars` "
        "(image plates); their text is preserved, matching production.\n"
    )


def _resolve_set_dir(fixture_set: str, root: Path | None) -> Path:
    """Resolve ``<root>/synthetic_recipes/<set>``, refusing anything outside it."""
    if not fixture_set or fixture_set != Path(fixture_set).name or fixture_set in (".", ".."):
        raise FixtureSetError(
            f"fixture_set must be a single safe path segment, got {fixture_set!r}"
        )
    recipes_root = ((FIXTURES_ROOT if root is None else root) / "synthetic_recipes").resolve()
    set_dir = (recipes_root / fixture_set).resolve()
    if set_dir == recipes_root or not set_dir.is_relative_to(recipes_root):
        raise FixtureSetError(
            f"fixture_set {fixture_set!r} escapes the fixtures root {recipes_root}"
        )
    return set_dir


def _validate_ranges(ranges: Sequence[tuple[int, int]]) -> None:
    if not ranges:
        raise FixtureRangeError("at least one page range is required")
    for first_page, last_page in ranges:
        if first_page < 1:
            raise FixtureRangeError(
                f"page range {first_page}-{last_page}: pages are 1-based, first page must be >= 1"
            )
        if last_page < first_page:
            raise FixtureRangeError(
                f"page range {first_page}-{last_page} is inverted: last page precedes first"
            )


def _validate_rationales(ranges: Sequence[tuple[int, int]], rationales: Sequence[str]) -> None:
    if len(rationales) != len(ranges):
        raise FixtureRationaleError(
            f"expected one rationale per page range: got {len(rationales)} "
            f"rationale(s) for {len(ranges)} range(s)"
        )
    for (first_page, last_page), rationale in zip(ranges, rationales, strict=True):
        if not rationale.strip():
            raise FixtureRationaleError(f"page range {first_page}-{last_page}: rationale is empty")


def _preflight_targets(
    *,
    pdf_path: Path,
    set_dir: Path,
    ranges: Sequence[tuple[int, int]],
    overwrite: bool,
    invalidate_goldens: bool,
) -> list[tuple[str, Path]]:
    """Derive every target, refusing duplicates and collisions before any write."""
    targets: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for first_page, last_page in ranges:
        name = derive_fixture_name(pdf_path, first_page, last_page)
        if name in seen:
            raise FixtureCollisionError(
                f"batch names fixture {name!r} more than once: duplicate page range "
                f"{first_page}-{last_page}"
            )
        seen.add(name)
        target = set_dir / name
        if target.exists():
            if not overwrite:
                raise FixtureCollisionError(
                    f"fixture already exists: {target} (pass overwrite=True to replace it)"
                )
            if (target / "expected.json").is_file() and not invalidate_goldens:
                raise FixtureCollisionError(
                    f"refusing to overwrite {target}: it carries expected.json, and replacing "
                    f"source.md would leave the golden silently describing different text "
                    f"(pass invalidate_goldens=True to accept that)"
                )
        targets.append((name, target))
    return targets


async def cut_fixtures(
    pdf_path: Path | str,
    *,
    fixture_set: str,
    ranges: Sequence[tuple[int, int]],
    rationales: Sequence[str],
    min_text_chars: int = DEFAULT_MIN_TEXT_CHARS,
    root: Path | None = None,
    overwrite: bool = False,
    invalidate_goldens: bool = False,
    extractor: PdfTextExtractor | None = None,
) -> list[CutResult]:
    """Cut one or more page ranges of ``pdf_path`` into fixture candidates.

    The whole batch is preflighted — ranges, rationale cardinality, path
    containment, name collisions, duplicate targets — **before the first write**,
    so a failure on range 5 of 8 leaves zero fixtures on disk. Each fixture is
    then rendered into a staging directory and ``os.replace``-d into place, so an
    interrupted run never publishes a ``source.md`` without its ``notes.md``.

    ``extract_pages`` is called **exactly once** for the whole batch: it takes
    whole-file bytes, so a per-range invocation would re-parse the entire book
    (``eatdrinkpaleocookbook.pdf`` is 190 MB).
    """
    pdf = Path(pdf_path)
    set_dir = _resolve_set_dir(fixture_set, root)
    _validate_ranges(ranges)
    _validate_rationales(ranges, rationales)
    if not pdf.is_file():
        raise FixtureSourceError(f"source PDF not found: {pdf}")
    targets = _preflight_targets(
        pdf_path=pdf,
        set_dir=set_dir,
        ranges=ranges,
        overwrite=overwrite,
        invalidate_goldens=invalidate_goldens,
    )

    try:
        file_bytes = pdf.read_bytes()
    except OSError as exc:
        raise FixtureSourceError(f"could not read source PDF {pdf}: {exc}") from exc
    reader = PyMuPdfExtractor(min_text_chars) if extractor is None else extractor
    try:
        pages = await reader.extract_pages(file_bytes)
    except PdfExtractionError as exc:
        raise FixtureSourceError(f"could not extract {pdf}: {exc}") from exc

    for first_page, last_page in ranges:
        if last_page > len(pages):
            raise FixtureRangeError(
                f"page range {first_page}-{last_page} is out of range: "
                f"{pdf.name} has {len(pages)} page(s)"
            )

    set_dir.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".cut-", dir=set_dir))
    try:
        staged: list[tuple[Path, CutResult]] = []
        for (name, target), (first_page, last_page), rationale in zip(
            targets, ranges, rationales, strict=True
        ):
            selected = pages[first_page - 1 : last_page]
            source_md = render_source_md(selected)
            flagged = tuple(page.page_number for page in selected if page.confidence == 0.0)
            if not source_md.strip():
                # Caught here rather than only by the layout gate so the operator
                # is told which range is unusable, not which staging path is.
                raise FixtureRangeError(
                    f"page range {first_page}-{last_page} of {pdf.name} yields no text: "
                    f"every selected page is below {min_text_chars} characters "
                    f"(image plates?) — pick a different range"
                )
            staged_dir = staging / name
            staged_dir.mkdir()
            (staged_dir / "source.md").write_text(source_md, encoding="utf-8")
            (staged_dir / "notes.md").write_text(
                _render_notes_md(
                    source_pdf=pdf.name,
                    first_page=first_page,
                    last_page=last_page,
                    rationale=rationale.strip(),
                    min_text_chars=min_text_chars,
                    flagged_pages=flagged,
                ),
                encoding="utf-8",
            )
            validate_candidate_layout(staged_dir)
            staged.append(
                (
                    staged_dir,
                    CutResult(
                        name=name,
                        path=target,
                        first_page=first_page,
                        last_page=last_page,
                        source_md=source_md,
                        flagged_pages=flagged,
                    ),
                )
            )

        results: list[CutResult] = []
        for staged_dir, result in staged:
            _publish(staged_dir, result.path)
            results.append(result)
        return results
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _publish(staged_dir: Path, target: Path) -> None:
    """Move a fully-rendered staging dir onto ``target`` atomically.

    On overwrite the existing fixture is displaced to a sibling **inside the
    set directory**, never into the staging dir: staging is ``rmtree``-d by
    :func:`cut_fixtures`'s ``finally``, so a failure between the two
    ``os.replace`` calls would take the original fixture — and any hand-authored
    ``expected.json`` it carries — down with it. Displaced-aside stays on disk
    until the new fixture is in place, and is restored if publishing fails.
    """
    if not target.exists():
        os.replace(staged_dir, target)
        return
    # os.replace refuses a non-empty destination directory, so the existing
    # fixture is renamed aside (atomic) and only deleted once the new one lands.
    displaced = target.parent / f".replaced-{target.name}"
    shutil.rmtree(displaced, ignore_errors=True)  # stale leftover from a crash
    os.replace(target, displaced)
    try:
        os.replace(staged_dir, target)
    except BaseException:
        os.replace(displaced, target)
        raise
    shutil.rmtree(displaced, ignore_errors=True)
