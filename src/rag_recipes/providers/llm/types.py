"""Pydantic payload types for the LLM provider (doc 11 § 3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, model_validator

__all__ = ["StructuredOutputRequest", "StructuredOutputResponse", "TokenUsage"]


class StructuredOutputRequest(BaseModel):
    """Request for a structured (JSON-Schema-constrained) LLM generation.

    ``provider`` / ``model`` are free strings (multi-provider by design), matching
    the ``ExtractionRun`` columns of the same name.
    """

    provider: str
    model: str
    prompt_version: str
    schema_version: str
    input: str
    json_schema: dict[str, Any]


class TokenUsage(BaseModel):
    """Token counts reported by the provider for a single generation."""

    input_tokens: int
    output_tokens: int


class StructuredOutputResponse(BaseModel):
    """Result of a structured LLM generation.

    A *technical* failure (transport, timeout, provider error) is raised as
    ``LLMTechnicalError`` and never reaches this model. But un-parseable or
    schema-non-conforming model output is **not** a technical failure — it is a
    rejection the extraction layer decides on (doc 11 § 3). The response carries
    that case explicitly: ``output_json`` is ``None`` when the model output could
    not be parsed into a JSON object, and ``parse_error`` describes why. On a
    clean parse, ``output_json`` holds the object and ``parse_error`` is ``None``.

    ``raw_text`` is always retained so the model's raw output survives a parse
    failure for debugging (doc 11 § 3).
    """

    output_json: dict[str, Any] | None
    parse_error: str | None = None
    raw_text: str
    usage: TokenUsage
    provider: str
    model: str

    @model_validator(mode="after")
    def _exactly_one_state(self) -> StructuredOutputResponse:
        if self.output_json is None:
            if not self.parse_error:
                raise ValueError("rejected output requires a non-empty parse_error")
        elif self.parse_error is not None:
            raise ValueError("parsed output_json must not carry a parse_error")
        return self
