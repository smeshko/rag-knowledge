"""Pydantic fixture models for the evaluation harness (doc 12 § 4, doc 13 topic 12).

Scaffold shipped by Epic 14 Phase 14.1. These models describe the four fixture
kinds under ``data/fixtures/`` — synthetic recipes, queries + qrels, judge
prompts, and judge-alignment records. Scoring and metric logic that consumes
them lands in Epics 15/16.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

__all__ = [
    "JudgeAlignmentRecord",
    "JudgePrompt",
    "Qrel",
    "QueryFixture",
    "QueryFixtureSet",
    "RecipeFixture",
]


class RecipeFixture(BaseModel):
    """One synthetic recipe fixture: raw source plus its golden extraction.

    ``expected`` is the parsed ``expected.json`` kept as an opaque dict — it is
    deliberately **not** re-validated against the live ``RecipeExtractionOutput``
    model (DECISIONS #3); golden-vs-live comparison is Epic 15's concern.
    """

    name: str
    source_md: str
    expected: dict[str, Any]
    notes: str | None = None


class QueryFixture(BaseModel):
    """One row of ``queries.tsv``: a golden retrieval query."""

    query_id: str
    query_text: str


class Qrel(BaseModel):
    """One row of ``qrels.tsv``: a relevance judgement for (query, item).

    ``relevance`` defaults to ``1`` (binary relevance today), leaving room for
    graded ``0/1/2/3`` judgements later (doc 13 topic 11).
    """

    query_id: str
    knowledge_item_id: str
    relevance: int = 1


class QueryFixtureSet(BaseModel):
    """A BEIR-style pair of golden queries and their relevance judgements."""

    name: str
    queries: list[QueryFixture]
    qrels: list[Qrel]


class JudgePrompt(BaseModel):
    """A versioned LLM-judge prompt loaded from ``judge_prompts/<name>.md``."""

    name: str
    version: str | None = None
    text: str


class JudgeAlignmentRecord(BaseModel):
    """Human-vs-judge alignment data for one fixture (doc 13 topic 12).

    All rating fields are optional so a partially-annotated record loads.
    """

    fixture_id: str
    human_rating: str | None = None
    judge_rating: str | None = None
    agreement_status: str | None = None
    run_metadata: dict[str, Any] = Field(default_factory=dict)
