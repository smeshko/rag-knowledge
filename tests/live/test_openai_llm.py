"""Opt-in real-API smoke test for OpenAILLMProvider (marked ``live``).

Deselected by default (``addopts = "-m 'not live'"``); run via ``just test-live``.
Keeps the prompt and schema tiny — this is a wiring smoke check, not an
extraction-quality test.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.providers.llm.openai import OpenAILLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from tests.live.conftest import LiveCredentials

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"ok": {"type": "boolean"}},
    "required": ["ok"],
    "additionalProperties": False,
}


@pytest.mark.live
async def test_openai_llm_generates_structured_output(
    live_credentials: LiveCredentials,
) -> None:
    provider = OpenAILLMProvider(
        api_key=live_credentials.openai_api_key,
        default_model=live_credentials.llm_model,
    )
    request = StructuredOutputRequest(
        provider="openai",
        model=live_credentials.llm_model,
        prompt_version="live-smoke",
        schema_version="live.v1",
        input="Return a JSON object with ok set to true.",
        json_schema=_SCHEMA,
    )

    response = await provider.generate_structured_output(request)

    assert isinstance(response.output_json, dict)
    assert response.raw_text
    assert response.usage.input_tokens > 0
    assert response.provider == "openai"
