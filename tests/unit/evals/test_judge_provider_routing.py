"""The judge runs where it is told to (Epic 23.3 TASK-002).

Two properties, and both matter:

* **Unset** — the judge uses the *same provider object* as extraction. Identity,
  not equality: an equivalent-but-distinct provider would pass an equality check
  while doubling construction, and would mean the "no behaviour change" claim was
  approximately rather than exactly true.
* **Set** — each provider receives only its own calls. Asserted on the *content*
  of the recorded requests, never on call counts alone: a count-only assertion
  passes even if both providers served both roles in the right proportion, which
  is precisely the confound being removed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.extraction import PROMPT_VERSION, run_extraction_eval
from evals.judges import JUDGE_SCHEMA_VERSION

from rag_recipes.providers.llm.fake import FakeLLMProvider
from tests.unit.evals.eval_utils import (
    SettingsStandIn,
    write_judge_prompt,
    write_smoke_set,
)

_JUDGE_OK = {"rating": "pass", "critique": "Reads fine."}


def _judge_only_provider(label: str) -> FakeLLMProvider:
    """A fake that answers judge requests and is labelled so calls are traceable."""
    provider = FakeLLMProvider(default_output=_JUDGE_OK, default_model=f"{label}-model")
    provider.provider = label
    return provider


async def _run(tmp_path: Path, **kwargs: Any):  # noqa: ANN202
    fixtures_root = tmp_path / "fixtures"
    extraction_provider = write_smoke_set(fixtures_root, judge_output=_JUDGE_OK)
    write_judge_prompt(fixtures_root)
    run = await run_extraction_eval(
        "smoke",
        "judge-routing",
        llm_provider=extraction_provider,
        judge="summary_quality",
        fixtures_root=fixtures_root,
        reports_root=tmp_path / "reports",
        judge_cache_root=tmp_path / "judge-cache",
        settings=SettingsStandIn(),
        **kwargs,
    )
    return run, extraction_provider


def _schema_versions(provider: FakeLLMProvider) -> list[str]:
    return [call.schema_version for call in provider.calls]


async def test_unset_judge_provider_reuses_the_same_object(tmp_path: Path) -> None:
    """AC1 — identity, not equality."""
    run, extraction_provider = await _run(tmp_path)

    # The one provider served both roles: its recorded calls carry BOTH the
    # extraction schema version and the judge's.
    versions = set(_schema_versions(extraction_provider))
    assert JUDGE_SCHEMA_VERSION in versions
    assert len(versions) > 1, "the single provider should have served both roles"

    doc = json.loads((run.path / "results.json").read_text(encoding="utf-8"))
    assert doc["results"]["judge"]["provider"] == "fake"


async def test_a_separate_judge_provider_receives_only_judge_calls(
    tmp_path: Path,
) -> None:
    """AC2 — routing asserted on request content, not on call counts."""
    judge_provider = _judge_only_provider("judge-vendor")
    _run_dir, extraction_provider = await _run(tmp_path, judge_provider=judge_provider)

    judge_versions = _schema_versions(judge_provider)
    extraction_versions = _schema_versions(extraction_provider)

    assert judge_versions, "the judge provider was never called"
    # Every judge-provider call is a judge call...
    assert set(judge_versions) == {JUDGE_SCHEMA_VERSION}
    # ...and the extraction provider saw no judge call at all.
    assert JUDGE_SCHEMA_VERSION not in extraction_versions
    assert extraction_versions, "the extraction provider was never called"

    # The prompt versions confirm it from the other side: extraction requests
    # carry the extraction prompt version, judge requests the judge prompt's.
    assert all(call.prompt_version == PROMPT_VERSION for call in extraction_provider.calls)
    assert all(call.prompt_version != PROMPT_VERSION for call in judge_provider.calls)


async def test_judge_provenance_names_the_judge_provider_not_the_extractor(
    tmp_path: Path,
) -> None:
    """AC3 — a committed baseline must say who graded it."""
    judge_provider = _judge_only_provider("judge-vendor")
    run, extraction_provider = await _run(tmp_path, judge_provider=judge_provider)

    judge_payload = json.loads((run.path / "results.json").read_text(encoding="utf-8"))[
        "results"
    ]["judge"]

    assert judge_payload["provider"] == "judge-vendor"
    assert judge_payload["model"] == "judge-vendor-model"
    # The extraction provider's identity must NOT leak into judge provenance.
    assert judge_payload["provider"] != extraction_provider.provider
