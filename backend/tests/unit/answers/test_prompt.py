"""Unit tests for the answer prompt and context-pack serialization."""

from __future__ import annotations

import pytest

from rag_recipes.answers.context_pack import (
    ContextChunk,
    ContextCitation,
    ContextDocument,
    ContextItem,
    ContextPack,
)
from rag_recipes.answers.prompt import (
    ANSWER_PROMPT_VERSION,
    GROUNDING_RULES,
    PROMPTS_BY_VERSION,
    RECOMMENDATION_PROMPT,
    render_answer_input,
)
from rag_recipes.config import Settings


def _pack() -> ContextPack:
    return ContextPack(
        query="cozy white bean soups",
        items=[
            ContextItem(
                context_item_id="ctx_1",
                knowledge_item_id="item_123",
                title="Tomato and White Bean Soup",
                summary="A simple soup.",
                document=ContextDocument(
                    document_id="doc_1", title="Simple Food", author="Author"
                ),
                matched_chunks=[
                    ContextChunk(
                        chunk_id="chunk_1",
                        chunk_type="recipe_ingredients",
                        text="2 cans white beans",
                        citation_id="cite_1",
                    )
                ],
                citations=[
                    ContextCitation(
                        citation_id="cite_1",
                        source_span_id="span_42",
                        label="Simple Food, page 42",
                    )
                ],
            )
        ],
    )


def test_grounding_rules_present_in_recommendation_prompt() -> None:
    # The doc 8 § 5 rules must be encoded in the prompt.
    assert GROUNDING_RULES in RECOMMENDATION_PROMPT
    for fragment in (
        "Use only the provided context",
        "Do not invent",
        "Cite every recommended item",
        "Do not cite sources that were not provided",
        "concise",
    ):
        assert fragment in RECOMMENDATION_PROMPT


def test_prompt_registered_under_its_version() -> None:
    assert PROMPTS_BY_VERSION[ANSWER_PROMPT_VERSION] is RECOMMENDATION_PROMPT


def test_render_answer_input_is_deterministic() -> None:
    pack = _pack()
    assert render_answer_input("cozy white bean soups", pack) == render_answer_input(
        "cozy white bean soups", pack
    )


def test_render_answer_input_surfaces_allowed_ids_and_text() -> None:
    out = render_answer_input("cozy white bean soups", _pack())
    assert "cozy white bean soups" in out
    assert "Allowed citation IDs: cite_1" in out
    assert "Allowed knowledge_item_ids: item_123" in out
    # The chunk text and citation id are serialized into the pack JSON.
    assert "2 cans white beans" in out
    assert "span_42" in out


def test_render_answer_input_empty_pack_reports_none() -> None:
    out = render_answer_input("q", ContextPack(query="q", items=[]))
    assert "Allowed citation IDs: (none)" in out
    assert "Allowed knowledge_item_ids: (none)" in out


def test_prompt_version_matches_settings_default(monkeypatch: pytest.MonkeyPatch) -> None:
    # Drift guard between the module constant and the Settings default.
    monkeypatch.setenv("DATABASE_URL", "postgresql+asyncpg://test/test")
    monkeypatch.setenv("REDIS_URL", "redis://:redis@localhost:6379/0")
    monkeypatch.setenv("REDIS_PASSWORD", "redis")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    settings = Settings(_env_file=None)
    assert settings.answer_prompt_version == ANSWER_PROMPT_VERSION
