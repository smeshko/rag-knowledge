"""Golden-replay run over the real `cookbooks` set (Epic 23.2 TASK-004).

Feeds each fixture's own golden back through the real driver as a canned
provider response, so the loader, hard validation, soft validation, and every
objective scorer traverse the whole 42-fixture set with no live call. This is
the executable form of the epic's `rag-evals extraction --fixtures cookbooks`
criterion — the CLI itself has no offline path by design.

**A green run here proves reachability and self-consistency, never
correctness.** A perfect score says the golden is well formed, cites the right
span, and is fully reachable by an ideal extractor. Whether the golden matches
the recipe is exactly and only the human verification pass (TASK-005).

Like the conformance gate, this skips when the gitignored set is absent (CI)
and enforces in full when it is present (the machine that has the fixtures).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from evals.extraction import run_extraction_eval
from evals.fixtures import load_recipe_fixtures

from rag_recipes.providers.llm.fake import FakeLLMProvider
from tests.unit.evals.eval_utils import (
    SettingsStandIn,
    golden_to_recipe_output,
    request_hash,
)

FIXTURES_ROOT = Path("data/fixtures")
SET_DIR = FIXTURES_ROOT / "synthetic_recipes" / "cookbooks"

#: AC12 — per-field golden coverage floors. A corpus where nearly every golden
#: writes `null` for a field would leave that field's accuracy resting on a
#: handful of fixtures without anyone noticing. Measured on the drafted set
#: (2026-08-05): yield 42/42; at-least-one-time 7/42 — only
#: `onepantorulethemall` prints a stated time ("TOTAL TIME:"), the other five
#: books state none, and inventing one would score correct extractions wrong.
#: The time floor is therefore 6 (plan asked 8; the corpus cannot honestly meet
#: it), set below the measured 7 so one dropped fixture does not flip the suite.
YIELD_COVERAGE_FLOOR = 8
TIME_COVERAGE_FLOOR = 6

pytestmark = pytest.mark.skipif(
    not SET_DIR.is_dir(),
    reason="cookbooks is gitignored (PR #61) and absent here — expected in CI",
)


@pytest.fixture(scope="module")
def replay_run(tmp_path_factory: pytest.TempPathFactory) -> dict:
    """One driver run over the full set, shared by every assertion below."""
    fixtures = load_recipe_fixtures("cookbooks", root=FIXTURES_ROOT)
    assert len(fixtures) >= 40, "partially present set — regenerate before asserting on it"

    provider = FakeLLMProvider(
        {
            request_hash(fixture.name, fixture.source_md): golden_to_recipe_output(
                fixture.name, fixture.source_md, fixture.expected
            )
            for fixture in fixtures
        }
    )
    import asyncio

    run = asyncio.run(
        run_extraction_eval(
            "cookbooks",
            "golden-replay",
            llm_provider=provider,
            fixtures_root=FIXTURES_ROOT,
            reports_root=tmp_path_factory.mktemp("reports"),
            settings=SettingsStandIn(),
        )
    )
    return json.loads((run.path / "results.json").read_text(encoding="utf-8"))


def test_the_whole_set_is_reachable(replay_run: dict) -> None:
    """AC11 — zero extraction_failed, zero hard_validation_failed.

    A fixture that fails hard validation does not score low — it vanishes from
    the report. This is the assertion that no golden is silently unreachable.
    """
    assert replay_run["status"] == "completed"
    per_fixture = replay_run["results"]["per_fixture"]  # a list, one entry per fixture
    not_scored = {
        entry["name"]: entry["status"] for entry in per_fixture if entry["status"] != "scored"
    }
    assert not_scored == {}, f"unreachable fixtures: {not_scored}"
    assert len(per_fixture) >= 40
    assert replay_run["results"]["aggregate"]["extraction_success_rate"] == 1.0


def test_an_ideal_extraction_scores_perfectly(replay_run: dict) -> None:
    """AC11 — the goldens are self-consistent under every objective scorer.

    Anything below 1.0 here means a golden disagrees with itself — e.g. a
    raw_text the aligner cannot match — since the "extraction" IS the golden.
    """
    accuracy = replay_run["results"]["aggregate"]["field_accuracy"]
    for field in (
        "title_exact",
        "title_normalized",
        "ingredient_count",
        "step_count",
        "ingredients_detail_f1",
        "source_span_ids_f1",
    ):
        assert accuracy[field] == 1.0, f"{field}: {accuracy[field]}"


def test_golden_coverage_meets_the_floors(replay_run: dict) -> None:
    """AC12 — enough goldens actually carry the optional fields.

    Null is the honest value for an unstated time, but a set that is null
    nearly everywhere measures nothing on that field. Counted from the same
    loaded fixtures the run scored — the `replay_run` dependency pins that.
    """
    assert replay_run["status"] == "completed"
    fixtures = load_recipe_fixtures("cookbooks", root=FIXTURES_ROOT)
    with_yield = sum(
        1 for fixture in fixtures if fixture.expected["structured_data"]["yield"] is not None
    )
    with_a_time = sum(
        1
        for fixture in fixtures
        if any(
            fixture.expected["structured_data"][field] is not None
            for field in ("prep_time", "cook_time", "total_time")
        )
    )
    assert with_yield >= YIELD_COVERAGE_FLOOR, f"yield coverage: {with_yield}"
    assert with_a_time >= TIME_COVERAGE_FLOOR, f"time coverage: {with_a_time}"
