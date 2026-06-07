"""Opt-in real-API smoke test for AnthropicLLMProvider (marked ``live``).

Deselected by default (``addopts = "-m 'not live'"``); run via ``just test-live``
or ``uv run pytest -m live tests/live/test_anthropic_llm.py``. Beyond a tiny
wiring smoke, this sends the **real** sanitized ``recipe.v1`` / ``answer.v1``
production schemas through one Sonnet 4.6 structured call each — the authoritative
check that those schemas clear Claude's structured-output limits (union/optional
/grammar-complexity), which the static keyword guard in the unit tests cannot
prove (round-1 #3, round-3 #1).
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.answers.schema import build_answer_v1_json_schema
from rag_recipes.ingestion.pipeline.extraction import build_recipe_v1_json_schema
from rag_recipes.providers.llm.anthropic import AnthropicLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from tests.live.conftest import LiveCredentials

_SMOKE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


def _provider(credentials: LiveCredentials) -> AnthropicLLMProvider:
    assert credentials.anthropic_api_key is not None  # guaranteed by the fixture skip
    return AnthropicLLMProvider(
        api_key=credentials.anthropic_api_key,
        default_model=credentials.anthropic_llm_model,
    )


def _request(
    credentials: LiveCredentials,
    *,
    prompt: str,
    schema: dict[str, Any],
    schema_version: str,
) -> StructuredOutputRequest:
    return StructuredOutputRequest(
        provider="anthropic",
        model=credentials.anthropic_llm_model,
        prompt_version="live-smoke",
        schema_version=schema_version,
        input=prompt,
        json_schema=schema,
    )


@pytest.mark.live
async def test_anthropic_llm_smoke(
    live_anthropic_credentials: LiveCredentials,
) -> None:
    provider = _provider(live_anthropic_credentials)
    request = _request(
        live_anthropic_credentials,
        prompt="Return a JSON object with ok set to true.",
        schema=_SMOKE_SCHEMA,
        schema_version="live.v1",
    )

    response = await provider.generate_structured_output(request)

    assert isinstance(response.output_json, dict)
    assert response.raw_text
    assert response.usage.input_tokens > 0
    assert response.provider == "anthropic"


@pytest.mark.live
async def test_anthropic_llm_accepts_real_recipe_v1_schema(
    live_anthropic_credentials: LiveCredentials,
) -> None:
    # Proves the production recipe.v1 schema clears Claude's structured-output
    # limits. An empty-extraction prompt keeps the call cheap and deterministic.
    provider = _provider(live_anthropic_credentials)
    request = _request(
        live_anthropic_credentials,
        prompt=(
            "The following page contains no recipes. Extract an empty recipe "
            "set: return items as an empty list."
        ),
        schema=build_recipe_v1_json_schema(),
        schema_version="recipe.v1",
    )

    response = await provider.generate_structured_output(request)

    assert response.parse_error is None
    assert isinstance(response.output_json, dict)
    assert isinstance(response.output_json["items"], list)


@pytest.mark.live
async def test_anthropic_llm_accepts_real_answer_v1_schema(
    live_anthropic_credentials: LiveCredentials,
) -> None:
    # Proves the production answer.v1 schema (with its nested ``answer`` object)
    # clears Claude's limits. Assert the NESTED shape — a degenerate
    # ``{"answer": "x", ...}`` must not pass (round-3 #1; answers/schema.py).
    provider = _provider(live_anthropic_credentials)
    request = _request(
        live_anthropic_credentials,
        prompt=(
            "Produce a minimal answer object. The 'answer' field is an object "
            "with a short 'style' string, a short 'text' string, and an empty "
            "'citations' list. Return empty 'recommendations' and empty top-level "
            "'citations' lists."
        ),
        schema=build_answer_v1_json_schema(),
        schema_version="answer.v1",
    )

    response = await provider.generate_structured_output(request)

    assert response.parse_error is None
    assert isinstance(response.output_json, dict)
    answer = response.output_json["answer"]
    assert isinstance(answer, dict)
    assert isinstance(answer["style"], str)
    assert isinstance(answer["text"], str)
    assert isinstance(answer["citations"], list)
    assert isinstance(response.output_json["recommendations"], list)
    assert isinstance(response.output_json["citations"], list)
