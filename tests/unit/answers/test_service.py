"""Unit tests for the answer service: citation validation + safe-fallback assembly.

``generate_answer``'s DB seams (``search``, structured-data / chunk fetches, result
projection) are monkeypatched so the orchestration, validation, and fallback logic
run fully in memory against a ``FakeLLMProvider``. The pure helpers
(``validate_citations`` / ``build_response_citations``) are tested directly.
"""

from __future__ import annotations

from typing import Any

import pytest

from rag_recipes.answers import service as service_module
from rag_recipes.answers.context_pack import (
    ChunkInput,
    ContextChunk,
    ContextCitation,
    ContextDocument,
    ContextItem,
    ContextPack,
)
from rag_recipes.answers.service import (
    FALLBACK_WARNING,
    NO_RESULTS_WARNING,
    build_response_citations,
    generate_answer,
    validate_citations,
)
from rag_recipes.config import Settings
from rag_recipes.providers.llm.fake import FakeLLMProvider
from rag_recipes.providers.llm.types import StructuredOutputResponse, TokenUsage
from rag_recipes.retrieval.types import (
    KnowledgeItemResult,
    MatchedChunkRef,
    ResultDocument,
    ResultItem,
    SearchDebug,
    SearchRequest,
    SearchResult,
    SourceCitation,
)
from rag_recipes.storage.enums import ChunkType

# --- pure-helper fixtures -------------------------------------------------------


def _pack_one_item() -> ContextPack:
    return ContextPack(
        query="q",
        items=[
            ContextItem(
                context_item_id="ctx_1",
                knowledge_item_id="item_1",
                title="Soup",
                summary=None,
                document=ContextDocument(document_id="d1", title="Book", author="A"),
                matched_chunks=[
                    ContextChunk(
                        chunk_id="c1",
                        chunk_type="recipe_summary",
                        text="beans",
                        citation_id="cite_1",
                    )
                ],
                citations=[
                    ContextCitation(
                        citation_id="cite_1", source_span_id="span_1", label="Book p1"
                    )
                ],
            ),
            ContextItem(
                context_item_id="ctx_2",
                knowledge_item_id="item_2",
                title="Stew",
                summary=None,
                document=ContextDocument(document_id="d2", title="Book2", author="B"),
                matched_chunks=[
                    ContextChunk(
                        chunk_id="c2",
                        chunk_type="recipe_summary",
                        text="lentils",
                        citation_id="cite_2",
                    )
                ],
                citations=[
                    ContextCitation(
                        citation_id="cite_2", source_span_id="span_2", label="Book2 p2"
                    )
                ],
            ),
        ],
    )


def _valid_answer_json() -> dict[str, Any]:
    return {
        "answer": {"style": "recommendation", "text": "Try the soup.", "citations": ["cite_1"]},
        "recommendations": [
            {
                "knowledge_item_id": "item_1",
                "reason": "uses beans",
                "citation_ids": ["cite_1"],
            }
        ],
        "citations": [],
    }


def test_validate_citations_accepts_valid() -> None:
    assert validate_citations(_valid_answer_json(), _pack_one_item()) == []


def test_validate_citations_rejects_unknown_answer_cite() -> None:
    bad = _valid_answer_json()
    bad["answer"]["citations"] = ["cite_99"]
    errors = validate_citations(bad, _pack_one_item())
    assert any("cite_99" in e for e in errors)


def test_validate_citations_rejects_recommendation_without_citation() -> None:
    bad = _valid_answer_json()
    bad["recommendations"][0]["citation_ids"] = []
    errors = validate_citations(bad, _pack_one_item())
    assert any("no citations" in e for e in errors)


def test_validate_citations_rejects_unknown_item() -> None:
    bad = _valid_answer_json()
    bad["recommendations"][0]["knowledge_item_id"] = "item_unknown"
    errors = validate_citations(bad, _pack_one_item())
    assert any("item_unknown" in e for e in errors)


def test_validate_citations_rejects_cross_item_binding() -> None:
    # item_1 recommendation cites cite_2, which belongs to item_2 — misattribution.
    bad = _valid_answer_json()
    bad["recommendations"][0]["citation_ids"] = ["cite_2"]
    errors = validate_citations(bad, _pack_one_item())
    assert any("belongs to a different item" in e for e in errors)


def test_validate_citations_flags_missing_answer_block() -> None:
    # A parseable object with no "answer" key is structurally malformed → error,
    # not a crash (review #1, finding 1).
    bad = {"recommendations": [], "citations": []}
    errors = validate_citations(bad, _pack_one_item())
    assert any("answer block is missing" in e for e in errors)


def test_validate_citations_flags_null_answer_block() -> None:
    bad: dict[str, Any] = {"answer": None, "recommendations": [], "citations": []}
    errors = validate_citations(bad, _pack_one_item())
    assert any("answer block is missing or not an object" in e for e in errors)


def test_validate_citations_flags_non_list_recommendations() -> None:
    bad: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "x", "citations": []},
        "recommendations": "oops",
        "citations": [],
    }
    errors = validate_citations(bad, _pack_one_item())
    assert any("recommendations is missing or not a list" in e for e in errors)


def test_validate_citations_tolerates_null_citations_without_crashing() -> None:
    # answer.citations is null (wrong type) → coerced to [], no membership error.
    ok: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "x", "citations": None},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_1"]}
        ],
        "citations": [],
    }
    assert validate_citations(ok, _pack_one_item()) == []


def test_validate_citations_tolerates_unhashable_citation_ids() -> None:
    # A nested list/dict among citation ids must not raise `TypeError: unhashable`
    # in the membership lookup — it's reported as an invalid citation (review #3).
    bad: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "x", "citations": [["nested"]]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": [{"k": "v"}]}
        ],
        "citations": [],
    }
    errors = validate_citations(bad, _pack_one_item())
    assert errors  # reported, not crashed
    assert any("unknown citation_id" in e for e in errors)


def test_validate_citations_tolerates_unhashable_knowledge_item_id() -> None:
    bad: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "x", "citations": []},
        "recommendations": [
            {"knowledge_item_id": ["item_1"], "reason": "r", "citation_ids": ["cite_1"]}
        ],
        "citations": [],
    }
    errors = validate_citations(bad, _pack_one_item())
    assert any("unknown knowledge_item_id" in e for e in errors)


def test_validate_citations_allows_empty_recommendations_with_answer_citation() -> None:
    # summary/direct_answer may omit recommendations, but the answer must still cite.
    ok: dict[str, Any] = {
        "answer": {"style": "summary", "text": "x", "citations": ["cite_1"]},
        "recommendations": [],
        "citations": [],
    }
    assert validate_citations(ok, _pack_one_item()) == []


def test_validate_citations_rejects_zero_citation_generated_answer() -> None:
    # An answer with no recommendations AND no answer.citations is ungrounded.
    bad: dict[str, Any] = {
        "answer": {"style": "summary", "text": "x", "citations": []},
        "recommendations": [],
        "citations": [],
    }
    errors = validate_citations(bad, _pack_one_item())
    assert any("no valid citations" in e for e in errors)


def test_build_response_citations_reconstructs_from_pack() -> None:
    cites = build_response_citations(["cite_1"], _pack_one_item())
    assert len(cites) == 1
    assert cites[0].citation_id == "cite_1"
    assert cites[0].knowledge_item_id == "item_1"
    assert cites[0].source_span_id == "span_1"
    assert cites[0].label == "Book p1"


def test_build_response_citations_skips_unknown_ids() -> None:
    assert build_response_citations(["cite_unknown"], _pack_one_item()) == []


# --- generate_answer orchestration (DB seams monkeypatched) ---------------------


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        database_url="postgresql+asyncpg://t/t",
        redis_url="redis://:r@localhost:6379/0",
        redis_password="r",
        openai_api_key="sk-test",
    )


def _make_search_result(*, items: list[KnowledgeItemResult]) -> SearchResult:
    return SearchResult(
        items=items,
        debug=SearchDebug(
            mode="hybrid",
            normalized_query="q",
            keyword_candidates=0,
            vector_candidates=0,
            merged_chunks=0,
            grouped_items=len(items),
        ),
    )


def _retrieval_item(*, item_id: str, chunk_id: str, span_id: str) -> KnowledgeItemResult:
    return KnowledgeItemResult(
        item=ResultItem(
            knowledge_item_id=item_id,
            item_type="recipe",
            title=f"Title {item_id}",
            summary="s",
            status="ready",
        ),
        document=ResultDocument(document_id=f"d-{item_id}", title="Book", author="A"),
        item_score=1.0,
        matched_chunks=[
            MatchedChunkRef(chunk_id=chunk_id, chunk_type=ChunkType.RECIPE_SUMMARY, score=1.0)
        ],
        source_citations=[SourceCitation(source_span_id=span_id, label="Book p1")],
    )


def _patch_db_seams(
    monkeypatch: pytest.MonkeyPatch,
    *,
    result: SearchResult,
    chunk_inputs: dict[str, ChunkInput],
) -> None:
    async def fake_search(
        session: Any, request: Any, *, provider: Any, settings: Any, reranker: Any = None
    ) -> Any:
        return result

    async def fake_structured(session: Any, res: Any) -> dict[str, dict[str, Any]]:
        return {item.item.knowledge_item_id: {"structured_data": {}} for item in res.items}

    async def fake_chunk_inputs(
        session: Any, res: Any, *, item_limit: int, chunks_per_item: int
    ) -> dict[str, ChunkInput]:
        return chunk_inputs

    async def fake_project(session: Any, res: Any, *, structured: Any = None) -> list[Any]:
        # Sentinel projection — a non-empty marker list so "results present" is checkable.
        return ["RESULT"] * len(res.items)

    monkeypatch.setattr(service_module, "search", fake_search)
    monkeypatch.setattr(service_module, "fetch_item_structured_data", fake_structured)
    monkeypatch.setattr(service_module, "_fetch_chunk_inputs", fake_chunk_inputs)
    monkeypatch.setattr(service_module, "project_results", fake_project)


def _request() -> SearchRequest:
    return SearchRequest(query="cozy soups", limit=10)


@pytest.mark.asyncio
async def test_generate_answer_success_reconstructs_from_pack(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    # The model emits cite_1 (the pack id) and bogus citation detail; the service
    # must reconstruct citations[] from the pack, ignoring the model's objects.
    canned = {
        "answer": {"style": "recommendation", "text": "Try it.", "citations": ["cite_1"]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "beans", "citation_ids": ["cite_1"]}
        ],
        "citations": [{"citation_id": "cite_1", "knowledge_item_id": "WRONG"}],
    }
    llm = FakeLLMProvider(default_output=canned)

    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=llm,
        embedding_provider=object(),
        settings=_settings(),
    )

    assert out.is_fallback is False
    assert out.answer.text == "Try it."
    assert out.recommendations[0].title == "Title item_1"  # reconstructed from pack
    assert out.recommendations[0].knowledge_item_id == "item_1"
    assert len(out.citations) == 1
    assert out.citations[0].knowledge_item_id == "item_1"  # pack, not the model's "WRONG"
    assert out.citations[0].source_span_id == "span_1"
    assert out.results == ["RESULT"]
    assert out.warnings == []


@pytest.mark.asyncio
async def test_generate_answer_success_omits_results_when_not_requested(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned = {
        "answer": {"style": "recommendation", "text": "x", "citations": ["cite_1"]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_1"]}
        ],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=False,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is False
    assert out.results == []  # dropped on the success path


@pytest.mark.asyncio
async def test_generate_answer_empty_results_falls_back_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_db_seams(monkeypatch, result=_make_search_result(items=[]), chunk_inputs={})
    llm = FakeLLMProvider(default_output={"answer": {}})
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=False,
        llm_provider=llm,
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [NO_RESULTS_WARNING]
    assert out.results == []
    assert llm.calls == ()  # no LLM call for empty retrieval


@pytest.mark.asyncio
async def test_generate_answer_empty_pack_falls_back_without_llm_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    # Chunk references a span NOT in the item's source_citations → uncitable → empty pack.
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="x", source_span_ids=["span_NOPE"])},
    )
    llm = FakeLLMProvider(default_output={"answer": {}})
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=llm,
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]
    assert out.results == ["RESULT"]  # fallback always carries results
    assert llm.calls == ()


@pytest.mark.asyncio
async def test_generate_answer_unknown_cite_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned = {
        "answer": {"style": "recommendation", "text": "x", "citations": ["cite_99"]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_99"]}
        ],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=False,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]
    assert out.results == ["RESULT"]  # fallback keeps results despite include_results=False


@pytest.mark.asyncio
async def test_generate_answer_parse_error_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    rejected = StructuredOutputResponse(
        output_json=None,
        parse_error="model output is not valid JSON",
        raw_text="{not json",
        usage=TokenUsage(input_tokens=1, output_tokens=0),
        provider="fake",
        model="fake-model",
    )
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=rejected),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]


@pytest.mark.asyncio
async def test_generate_answer_technical_failure_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=FakeLLMProvider(fail_technically=True),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True  # LLMTechnicalError → safe fallback, not a 502
    assert out.warnings == [FALLBACK_WARNING]


@pytest.mark.asyncio
async def test_generate_answer_malformed_output_falls_back_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A parseable object with no "answer" key (parse_error is None, but not schema-
    # conforming) must funnel to the safe fallback, never crash with a 500
    # (review #1, finding 1).
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    malformed = {"recommendations": [], "citations": []}  # no "answer" key
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=malformed),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]
    assert out.results == ["RESULT"]


@pytest.mark.asyncio
async def test_generate_answer_unhashable_citation_ids_falls_back_no_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Nested-list citation ids would crash the (pre-guard) validate_citations
    # membership lookup with TypeError → must funnel to the safe fallback (review #3).
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "x", "citations": [["nested"]]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": [["nested"]]}
        ],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]
    assert out.results == ["RESULT"]
    # The malformed-output fallback must still carry debug like every other path
    # (review #1, 17.3) — the route's T/T gate depends on it being non-None.
    assert out.debug is not None
    assert out.debug.prompt_version == "answer-recommendation-v2"


@pytest.mark.asyncio
async def test_generate_answer_null_citations_succeeds_without_crash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # answer.citations is null (wrong type) — coerced, no crash, success path.
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned = {
        "answer": {"style": "recommendation", "text": "ok", "citations": None},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_1"]}
        ],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=False,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is False
    assert out.answer.citations == []  # null coerced to empty, no crash
    assert out.citations[0].citation_id == "cite_1"  # picked up from the recommendation


@pytest.mark.asyncio
async def test_generate_answer_summary_empty_recommendations_succeeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # summary may return empty recommendations as long as the answer cites ≥1 id;
    # the resolved per-style prompt_version flows into the debug payload.
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned: dict[str, Any] = {
        "answer": {"style": "summary", "text": "A summary.", "citations": ["cite_1"]},
        "recommendations": [],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="summary",
        include_results=False,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is False
    assert out.answer.style == "summary"
    assert out.recommendations == []
    assert out.citations[0].citation_id == "cite_1"
    assert out.debug is not None
    assert out.debug.prompt_version == "answer-summary-v2"


@pytest.mark.asyncio
async def test_generate_answer_summary_zero_citations_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A generated summary with no recommendations AND no answer.citations is
    # ungrounded → safe fallback, not a citation-free answer.
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned: dict[str, Any] = {
        "answer": {"style": "summary", "text": "ungrounded", "citations": []},
        "recommendations": [],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="summary",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]


@pytest.mark.asyncio
async def test_generate_answer_summary_cross_item_binding_falls_back(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Binding is NOT relaxed for the new styles: a present recommendation citing a
    # cite from another item still fails → fallback.
    result = _make_search_result(
        items=[
            _retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1"),
            _retrieval_item(item_id="item_2", chunk_id="c2", span_id="span_2"),
        ]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={
            "c1": ChunkInput(text="a", source_span_ids=["span_1"]),
            "c2": ChunkInput(text="b", source_span_ids=["span_2"]),
        },
    )
    canned: dict[str, Any] = {
        "answer": {"style": "summary", "text": "x", "citations": ["cite_1"]},
        # item_1 recommendation citing cite_2 (belongs to item_2) — misattribution.
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_2"]}
        ],
        "citations": [],
    }
    out = await generate_answer(
        None,
        _request(),
        style="summary",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda c: c["answer"].__setitem__("text", 42), id="text-int"),
        pytest.param(lambda c: c["answer"].__setitem__("text", {"n": "x"}), id="text-object"),
        pytest.param(
            lambda c: c["recommendations"][0].__setitem__("reason", 7), id="reason-int"
        ),
        pytest.param(
            lambda c: c["recommendations"][0].__setitem__("reason", ["a"]), id="reason-list"
        ),
    ],
)
async def test_generate_answer_truthy_nonstring_scalar_falls_back(
    monkeypatch: pytest.MonkeyPatch, mutate: Any
) -> None:
    # A truthy non-string text/reason passes citation validation but would trip
    # Pydantic model construction — it must degrade to the safe fallback, not a 500
    # (review #1, finding 1 / round 2).
    result = _make_search_result(
        items=[_retrieval_item(item_id="item_1", chunk_id="c1", span_id="span_1")]
    )
    _patch_db_seams(
        monkeypatch,
        result=result,
        chunk_inputs={"c1": ChunkInput(text="beans", source_span_ids=["span_1"])},
    )
    canned: dict[str, Any] = {
        "answer": {"style": "recommendation", "text": "ok", "citations": ["cite_1"]},
        "recommendations": [
            {"knowledge_item_id": "item_1", "reason": "r", "citation_ids": ["cite_1"]}
        ],
        "citations": [],
    }
    mutate(canned)
    out = await generate_answer(
        None,
        _request(),
        style="recommendation",
        include_results=True,
        llm_provider=FakeLLMProvider(default_output=canned),
        embedding_provider=object(),
        settings=_settings(),
    )
    assert out.is_fallback is True
    assert out.warnings == [FALLBACK_WARNING]
    assert out.results == ["RESULT"]
