"""The ``rerank.v1`` JSON schema for the LLM listwise reranker (Epic 18.2).

Minimal by design: the model returns only an ordered ``ranking`` of ``chunk_id`` +
``relevance_score`` (most relevant first). The retrieval facade maps the order back
onto the existing ``MergedChunk`` objects (it derives rank from list position, not a
trusted rank field), so the schema carries no rank. Authored for OpenAI strict mode:
every object lists all properties in ``required`` and sets ``additionalProperties:
false``.
"""

from __future__ import annotations

from typing import Any

__all__ = ["RERANK_SCHEMA_VERSION", "RERANK_V1_SCHEMA"]

RERANK_SCHEMA_VERSION = "rerank.v1"

RERANK_V1_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["ranking"],
    "properties": {
        "ranking": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["chunk_id", "relevance_score"],
                "properties": {
                    "chunk_id": {"type": "string"},
                    "relevance_score": {"type": "number"},
                },
            },
        },
    },
}
