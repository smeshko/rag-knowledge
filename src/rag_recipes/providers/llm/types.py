"""Pydantic payload types for the LLM provider (doc 11 § 3)."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel

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

    ``raw_text`` is retained deliberately so the model's raw output survives a
    structured-parse failure for debugging (doc 11 § 3).
    """

    output_json: dict[str, Any]
    raw_text: str
    usage: TokenUsage
    provider: str
    model: str
