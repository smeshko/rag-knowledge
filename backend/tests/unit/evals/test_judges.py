"""Offline tests for the LLM-as-judge layer (Epic 15 Phase 15.2).

Every judge call goes through ``FakeLLMProvider`` — no real API, no key, no
``live`` marker.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from evals.judges import (
    JUDGE_SCHEMA_VERSION,
    Judge,
    JudgeError,
    JudgeRating,
    _build_verdict_json_schema,
    load_judge,
)
from evals.models import JudgePrompt

from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage

_PROMPT_TEXT = (
    "# Summary quality judge\n"
    "# version: v1\n\n"
    "Rate the summary.\n\n"
    "## Extracted output\n\n{extracted_output}\n\n"
    "## Expected (golden) values\n\n{expected_output}\n\n"
    "## Source text\n\n{source_text}\n"
)


def _judge(provider: FakeLLMProvider) -> Judge:
    prompt = JudgePrompt(name="summary_quality", version="v1", text=_PROMPT_TEXT)
    return Judge(name="summary_quality", version="v1", llm_provider=provider, prompt=prompt)


async def test_judge_returns_typed_rating_with_identity_and_metadata() -> None:
    provider = FakeLLMProvider(
        default_output={"rating": "pass", "critique": "Faithfully captures the dish."}
    )
    rating = await _judge(provider).judge("EXTRACTED", "GOLDEN", "SOURCE")
    assert isinstance(rating, JudgeRating)
    assert rating.judge_name == "summary_quality"
    assert rating.judge_version == "v1"
    assert rating.rating == "pass"
    assert rating.critique == "Faithfully captures the dish."
    assert rating.metadata == {
        "provider": "fake",
        "model": "fake-model",
        "prompt_version": "summary_quality-v1",
        "schema_version": JUDGE_SCHEMA_VERSION,
    }


async def test_prompt_placeholders_are_substituted_and_versions_are_judge_specific() -> None:
    provider = FakeLLMProvider(default_output={"rating": "fail", "critique": "Too generic."})
    await _judge(provider).judge("EXTRACTED-XX", "GOLDEN-YY", "SOURCE-ZZ")
    (request,) = provider.calls
    assert "EXTRACTED-XX" in request.input
    assert "GOLDEN-YY" in request.input
    assert "SOURCE-ZZ" in request.input
    for token in ("{extracted_output}", "{expected_output}", "{source_text}"):
        assert token not in request.input
    # Distinct from extraction's recipe-extraction-v1 / recipe.v1, so fake
    # request hashes (and canned responses) can never collide across the two.
    assert request.prompt_version == "summary_quality-v1"
    assert request.schema_version == JUDGE_SCHEMA_VERSION


def _all_keys(node: object) -> set[str]:
    keys: set[str] = set()
    if isinstance(node, dict):
        keys.update(node)
        for value in node.values():
            keys.update(_all_keys(value))
    elif isinstance(node, list):
        for item in node:
            keys.update(_all_keys(item))
    return keys


def test_verdict_schema_is_strict_and_carries_no_constraint_keywords() -> None:
    schema = _build_verdict_json_schema()
    assert schema["additionalProperties"] is False
    assert schema["required"] == ["rating", "critique"]
    assert set(schema["properties"]) == {"rating", "critique"}
    # Anthropic's sanitizer strips these, so their presence would be a false
    # promise of enforcement.
    assert not _all_keys(schema) & {
        "minLength",
        "maxLength",
        "pattern",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "uniqueItems",
        "multipleOf",
    }


async def test_provider_rejection_raises_judge_error() -> None:
    provider = FakeLLMProvider(
        default_output=StructuredOutputResponse(
            output_json=None,
            parse_error="output truncated by provider (stop_reason=max_tokens)",
            raw_text="{\"rating\":",
            usage=TokenUsage(input_tokens=5, output_tokens=5),
            provider="fake",
            model="fake-model",
        )
    )
    with pytest.raises(JudgeError, match="no output"):
        await _judge(provider).judge("EXTRACTED", "GOLDEN", "SOURCE")


async def test_malformed_verdict_raises_judge_error() -> None:
    # Reachable in production: the Anthropic path is non-strict forced tool
    # use, so the schema is a request, not a guarantee.
    provider = FakeLLMProvider(default_output={"rating": "maybe", "critique": "Hmm."})
    with pytest.raises(JudgeError, match="verdict"):
        await _judge(provider).judge("EXTRACTED", "GOLDEN", "SOURCE")


def _write_prompt(root: Path, name: str, text: str) -> None:
    prompts = root / "judge_prompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (prompts / f"{name}.md").write_text(text, encoding="utf-8")


def test_load_judge_reads_prompt_and_version(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "summary_quality", _PROMPT_TEXT)
    judge = load_judge("summary_quality", FakeLLMProvider(), root=tmp_path)
    assert judge.name == "summary_quality"
    assert judge.version == "v1"
    assert judge.prompt_version == "summary_quality-v1"
    assert judge.model == "fake-model"


def test_load_judge_without_version_raises(tmp_path: Path) -> None:
    _write_prompt(tmp_path, "summary_quality", "# Summary quality judge\n\nRate it.\n")
    with pytest.raises(JudgeError, match="version"):
        load_judge("summary_quality", FakeLLMProvider(), root=tmp_path)


def test_load_judge_missing_prompt_surfaces_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_judge("nonexistent", FakeLLMProvider(), root=tmp_path)


def test_committed_prompts_load_with_version_v1() -> None:
    # The three shipped prompts must always carry a parseable version.
    for name in ("summary_quality", "boundary_correctness", "step_text_quality"):
        judge = load_judge(name, FakeLLMProvider())
        assert judge.version == "v1"
