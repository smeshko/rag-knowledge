"""CI conformance gate over the real fixture tree (Epic 23.2 TASK-001).

This is the only test in the suite that reads `data/fixtures/` as data rather
than building its own. That coupling is deliberate: a golden is repo *content*,
and content defects (a typo'd key, an unparseable duration) are invisible to
every behavioural test because the scorers degrade silently rather than raising.

Two failure modes this file is shaped to avoid:

* **Passing vacuously.** `load_recipe_fixtures` returns `[]` for a set that does
  not exist, so a test that merely iterates it is green over zero fixtures. Every
  enrolled set therefore asserts a minimum count.
* **Escaping by omission.** A new set that nobody enrolls would never be gated.
  So non-enrolment of an on-disk directory is itself a failure.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from evals.golden_schema import parse_verification_status, validate_expected_json

FIXTURE_ROOT = Path("data/fixtures/synthetic_recipes")


@dataclass(frozen=True)
class GoldenSetPolicy:
    """How strictly one fixture set is gated."""

    #: Fail below this many fixtures. Guards against a half-committed set
    #: passing because every fixture present happens to be well formed.
    minimum: int
    #: Require a parseable `Verification:` field in each fixture's notes.md.
    require_verification: bool
    #: The set is gitignored, so CI never sees it. Absence skips with a reason;
    #: PRESENCE enforces every rule in full. There is no partial mode — a
    #: locally broken set must fail on the machine that has it, which is the
    #: only machine that can fix it.
    local_only: bool = False


_GOLDEN_SETS: dict[str, GoldenSetPolicy] = {
    # Committed, CI-visible, and deliberately untouched by Epic 23.2.
    "synthetic": GoldenSetPolicy(minimum=2, require_verification=False),
    # Gitignored: real cookbook excerpts cannot enter a public repo (PR #61).
    "cookbooks": GoldenSetPolicy(minimum=40, require_verification=True, local_only=True),
}


def _fixture_dirs(set_name: str) -> list[Path]:
    root = FIXTURE_ROOT / set_name
    return sorted(d for d in root.iterdir() if d.is_dir())


def test_every_set_on_disk_is_enrolled() -> None:
    """A set nobody enrolled is a set nobody gates.

    Rejected alternative: auto-discovery with a skip for sets lacking goldens.
    A silent skip is exactly how a typo'd directory name or a half-committed set
    passes CI — the same pattern 23.1 already rejected in the loader.
    """
    on_disk = {d.name for d in FIXTURE_ROOT.iterdir() if d.is_dir()}
    assert on_disk <= set(_GOLDEN_SETS), (
        f"unenrolled fixture set(s) {sorted(on_disk - set(_GOLDEN_SETS))} — "
        "add them to _GOLDEN_SETS so they are gated, or delete them"
    )


@pytest.mark.parametrize("set_name", sorted(_GOLDEN_SETS))
def test_the_set_is_present_and_large_enough(set_name: str) -> None:
    policy = _GOLDEN_SETS[set_name]
    root = FIXTURE_ROOT / set_name
    if not root.is_dir():
        if policy.local_only:
            pytest.skip(
                f"{set_name} is gitignored (see .gitignore, PR #61) and absent here — "
                "expected in CI. Regenerate locally from data/fixtures/cookbooks_ranges.json."
            )
        pytest.fail(f"committed fixture set {set_name!r} is missing from {FIXTURE_ROOT}")

    fixtures = _fixture_dirs(set_name)
    assert len(fixtures) >= policy.minimum, (
        f"{set_name}: {len(fixtures)} fixtures, expected at least {policy.minimum} — "
        "a partially present set would otherwise pass by being uniformly well formed"
    )


@pytest.mark.parametrize("set_name", sorted(_GOLDEN_SETS))
def test_every_golden_in_the_set_conforms(set_name: str) -> None:
    policy = _GOLDEN_SETS[set_name]
    root = FIXTURE_ROOT / set_name
    if not root.is_dir() and policy.local_only:
        pytest.skip(f"{set_name} is gitignored and absent here")

    failures: list[str] = []
    for fixture in _fixture_dirs(set_name):
        path = fixture / "expected.json"
        if not path.is_file():
            failures.append(f"{fixture.name}: no expected.json")
            continue
        try:
            expected = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            failures.append(f"{fixture.name}: expected.json is not valid JSON ({exc})")
            continue
        errors = validate_expected_json(fixture.name, expected)
        failures += [f"{fixture.name}: {error}" for error in errors]

    assert not failures, "\n".join(failures)


@pytest.mark.parametrize(
    "set_name", sorted(s for s, p in _GOLDEN_SETS.items() if p.require_verification)
)
def test_every_fixture_declares_its_verification_status(set_name: str) -> None:
    """Presence of the status, not its value.

    The strict "no fixture is still `draft`" assertion lands with the human
    verification pass (TASK-005), not here — otherwise every agent-completable
    task would be red for the entire phase, and a red suite that is *expected*
    to be red teaches everyone to ignore it.
    """
    root = FIXTURE_ROOT / set_name
    if not root.is_dir() and _GOLDEN_SETS[set_name].local_only:
        pytest.skip(f"{set_name} is gitignored and absent here")

    missing = [
        fixture.name
        for fixture in _fixture_dirs(set_name)
        if parse_verification_status(
            (fixture / "notes.md").read_text(encoding="utf-8")
            if (fixture / "notes.md").is_file()
            else None
        )
        is None
    ]
    assert not missing, (
        f"{set_name}: fixture(s) with no parseable 'Verification:' field: {missing}. "
        "Absent is a failure, never a default — a forgotten fixture must not look finished."
    )
