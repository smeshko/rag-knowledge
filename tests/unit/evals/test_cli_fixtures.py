"""Tests for ``rag-evals fixtures cut`` (Epic 23 Phase 23.1).

Hermetic: cuts come from the generated ``data/fixtures/pdfs/sample_recipe.pdf``
into ``tmp_path`` via a monkeypatched ``FIXTURES_ROOT``; no test opens a real
cookbook, and no test writes into the repo's ``data/fixtures``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import evals.cli
import evals.fixture_cutter
import pytest
from evals.cli import app
from evals.fixture_cutter import DEFAULT_MIN_TEXT_CHARS, parse_notes_fields
from typer.testing import CliRunner

from rag_recipes.providers.pdf_extractor.base import PdfTextExtractor
from rag_recipes.providers.pdf_extractor.pymupdf import PyMuPdfExtractor
from rag_recipes.providers.pdf_extractor.types import PdfPageText

runner = CliRunner()

_REPO_ROOT = Path(__file__).resolve().parents[3]
SAMPLE_PDF = _REPO_ROOT / "data" / "fixtures" / "pdfs" / "sample_recipe.pdf"

#: Module that must stay unimported while ``--help`` renders.
PROVIDER_MODULE = "rag_recipes.providers.pdf_extractor.pymupdf"


@pytest.fixture
def fixtures_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setattr(evals.fixture_cutter, "FIXTURES_ROOT", tmp_path)
    return tmp_path


def _set_dir(root: Path) -> Path:
    return root / "synthetic_recipes" / "cookbooks"


# --- AC2: --help never imports the PDF provider ------------------------------


_HELP_SENTINEL = """
import sys

from typer.testing import CliRunner

from evals.cli import app

assert "{module}" not in sys.modules, "importing evals.cli already pulled in the provider"
assert "evals.fixture_cutter" not in sys.modules, "importing evals.cli pulled in the cutter"

result = CliRunner().invoke(app, {argv})
assert result.exit_code == 0, result.output

assert "{module}" not in sys.modules, "--help imported the PyMuPDF provider"
assert "evals.fixture_cutter" not in sys.modules, "--help imported the cutter"
print("OK")
"""


@pytest.mark.parametrize(
    "argv", [["fixtures", "--help"], ["fixtures", "cut", "--help"], ["--help"]]
)
def test_help_leaves_the_pdf_provider_unimported(argv: list[str]) -> None:
    """A ``sys.modules`` sentinel, run in a clean interpreter.

    Stronger than "no provider was constructed": the in-process pytest session
    has already imported PyMuPDF for other tests, so the check is only
    meaningful in a subprocess that imports nothing but ``evals.cli``.
    """
    script = textwrap.dedent(_HELP_SENTINEL).format(module=PROVIDER_MODULE, argv=argv)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        cwd=_REPO_ROOT,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "OK" in completed.stdout


def test_root_help_lists_the_fixtures_sub_app() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "fixtures" in result.output


def test_fixtures_help_lists_cut() -> None:
    result = runner.invoke(app, ["fixtures", "--help"])
    assert result.exit_code == 0
    assert "cut" in result.output


def test_cut_min_text_chars_default_tracks_the_cutter_constant() -> None:
    # The CLI duplicates the constant so --help stays provider-free; this pins
    # the duplication so the two cannot drift.
    assert evals.cli._CUT_MIN_TEXT_CHARS == DEFAULT_MIN_TEXT_CHARS


# --- Happy paths -------------------------------------------------------------


def test_single_range_writes_the_fixture_and_prints_name_and_path(fixtures_root: Path) -> None:
    result = runner.invoke(
        app,
        [
            "fixtures",
            "cut",
            "--pdf",
            str(SAMPLE_PDF),
            "--set",
            "cookbooks",
            "--pages",
            "1-2",
            "--rationale",
            "one complete pancake recipe",
        ],
    )
    assert result.exit_code == 0, result.output
    target = _set_dir(fixtures_root) / "sample-recipe-p1-2"
    assert sorted(p.name for p in target.iterdir()) == ["notes.md", "source.md"]
    assert "sample-recipe-p1-2" in result.output
    assert str(target) in result.output
    fields = parse_notes_fields((target / "notes.md").read_text(encoding="utf-8"))
    assert fields["rationale"] == "one complete pancake recipe"


def test_single_page_shorthand_is_accepted(fixtures_root: Path) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", "2", "--rationale", "method only"],
    )
    assert result.exit_code == 0, result.output
    assert (_set_dir(fixtures_root) / "sample-recipe-p2-2").is_dir()


def test_multi_range_emits_n_fixtures_from_one_extraction(
    fixtures_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """AC6 — the book is parsed once for the whole batch."""
    calls: list[int] = []

    class _Spy(PdfTextExtractor):
        def __init__(self, min_text_chars: int) -> None:
            self._inner = PyMuPdfExtractor(min_text_chars)

        async def extract_pages(self, file: bytes) -> list[PdfPageText]:
            calls.append(1)
            return await self._inner.extract_pages(file)

    monkeypatch.setattr(evals.fixture_cutter, "PyMuPdfExtractor", _Spy)

    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", "1-1", "--rationale", "a",
         "--pages", "2-2", "--rationale", "b",
         "--pages", "1-2", "--rationale", "c"],
    )
    assert result.exit_code == 0, result.output
    assert len(calls) == 1
    names = sorted(p.name for p in _set_dir(fixtures_root).iterdir())
    assert names == ["sample-recipe-p1-1", "sample-recipe-p1-2", "sample-recipe-p2-2"]


def test_sub_threshold_pages_are_reported_as_a_warning(fixtures_root: Path) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", "2-3", "--rationale", "trailing plate"],
    )
    assert result.exit_code == 0, result.output
    # CliRunner folds stderr into output by default.
    assert "warning" in result.output
    assert "page(s) 3" in result.output


def test_overwrite_replaces_an_existing_fixture(fixtures_root: Path) -> None:
    base = ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks", "--pages", "1-1"]
    assert runner.invoke(app, [*base, "--rationale", "first"]).exit_code == 0
    result = runner.invoke(app, [*base, "--rationale", "second", "--overwrite"])
    assert result.exit_code == 0, result.output
    notes = (_set_dir(fixtures_root) / "sample-recipe-p1-1" / "notes.md").read_text(
        encoding="utf-8"
    )
    assert parse_notes_fields(notes)["rationale"] == "second"


# --- Caller errors all exit 2 ------------------------------------------------


def test_collision_without_overwrite_exits_2_naming_the_path(fixtures_root: Path) -> None:
    args = ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
            "--pages", "1-1", "--rationale", "why"]
    assert runner.invoke(app, args).exit_code == 0
    result = runner.invoke(app, args)
    assert result.exit_code == 2
    assert "sample-recipe-p1-1" in result.output
    assert "--overwrite" in result.output or "overwrite" in result.output


def test_overwriting_a_golden_bearing_fixture_exits_2(fixtures_root: Path) -> None:
    args = ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
            "--pages", "1-1", "--rationale", "why"]
    assert runner.invoke(app, args).exit_code == 0
    golden = _set_dir(fixtures_root) / "sample-recipe-p1-1" / "expected.json"
    golden.write_text("{}", encoding="utf-8")
    result = runner.invoke(app, [*args, "--overwrite"])
    assert result.exit_code == 2
    assert "expected.json" in result.output
    assert golden.is_file()


@pytest.mark.parametrize(
    ("pages", "needle"),
    [
        ("3-1", "inverted"),
        ("0-1", "1-based"),
        ("1-99", "out of range"),
        ("banana", "invalid --pages"),
        ("1-", "invalid --pages"),
        ("", "invalid --pages"),
    ],
)
def test_bad_pages_exit_2_with_the_problem_named(
    fixtures_root: Path, pages: str, needle: str
) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", pages, "--rationale", "why"],
    )
    assert result.exit_code == 2, result.output
    assert needle in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)


def test_an_all_plate_range_exits_2(fixtures_root: Path) -> None:
    """Page 3 of the sample is empty; a fixture of it alone would be an empty prompt."""
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", "3", "--rationale", "nothing but a plate"],
    )
    assert result.exit_code == 2, result.output
    assert "yields no text" in result.output
    assert not _set_dir(fixtures_root).exists() or list(_set_dir(fixtures_root).iterdir()) == []


def test_rationale_help_documents_the_positional_pairing() -> None:
    """--pages/--rationale pair positionally and only counts are checked, so the
    strict-interleave requirement has to be stated where the operator will read it."""
    result = runner.invoke(app, ["fixtures", "cut", "--help"])
    assert result.exit_code == 0
    assert "positionally" in result.output
    assert "interleave" in result.output


def test_readme_documents_the_positional_pairing() -> None:
    readme = (_REPO_ROOT / "evals" / "README.md").read_text(encoding="utf-8")
    assert "positionally" in readme
    assert "strictly interleaved" in readme


def test_missing_pdf_exits_2_without_a_traceback(fixtures_root: Path, tmp_path: Path) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(tmp_path / "absent.pdf"), "--set", "cookbooks",
         "--pages", "1-1", "--rationale", "why"],
    )
    assert result.exit_code == 2
    assert "source PDF not found" in result.output
    assert not isinstance(result.exception, FileNotFoundError)


def test_rationale_count_mismatch_exits_2(fixtures_root: Path) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", "cookbooks",
         "--pages", "1-1", "--pages", "2-2", "--rationale", "only one"],
    )
    assert result.exit_code == 2
    assert "one rationale per page range" in result.output
    assert not _set_dir(fixtures_root).exists()


@pytest.mark.parametrize("fixture_set", ["../evil", "/abs", "a/b"])
def test_unsafe_set_exits_2(fixtures_root: Path, fixture_set: str) -> None:
    result = runner.invoke(
        app,
        ["fixtures", "cut", "--pdf", str(SAMPLE_PDF), "--set", fixture_set,
         "--pages", "1-1", "--rationale", "why"],
    )
    assert result.exit_code == 2
    assert "fixture_set" in result.output
    assert not (fixtures_root / "synthetic_recipes").exists()


def test_missing_required_options_exit_2() -> None:
    assert runner.invoke(app, ["fixtures", "cut"]).exit_code == 2
    assert (
        runner.invoke(app, ["fixtures", "cut", "--pdf", str(SAMPLE_PDF)]).exit_code == 2
    )
