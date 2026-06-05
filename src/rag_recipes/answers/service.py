"""Answer service: retrieval reuse → context pack → LLM → citation validation (doc 8).

``generate_answer`` orchestrates the query-time answer flow and returns an internal
``AnswerResult`` the route projects to ``AnswerResponse``. The grounding guarantee
is enforced here, not trusted to the model:

* Citations are validated for **membership and binding** — every cited ``cite_N``
  exists in the context pack, every recommendation has ≥1 citation, every
  recommended ``knowledge_item_id`` is a pack item, and each of a recommendation's
  ``citation_ids`` belongs to *that* recommendation's item (a cross-item citation
  is a misattribution and fails).
* The response ``citations[]`` detail objects and each ``recommendations[].title``
  are **reconstructed from the pack**, never trusted from the LLM (the model emits
  only ``cite_N`` references).
* Parse failure, invalid/misattributed citations, **or** ``LLMTechnicalError`` all
  take the same safe-fallback path: return the retrieved results plus a warning,
  never an ungrounded or misattributed answer (one unified failure path, no 502).
* A fallback always carries the retrieved ``results`` (even when
  ``include_results=False``); the success-path drop applies only to a real answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from rag_recipes.answers.context_pack import ChunkInput, ContextPack, build_context_pack
from rag_recipes.answers.prompt import render_answer_input
from rag_recipes.answers.schema import build_answer_v1_json_schema
from rag_recipes.api.schemas.answers import (
    AnswerBody,
    AnswerCitation,
    Recommendation,
)
from rag_recipes.api.schemas.search import KnowledgeItemResult
from rag_recipes.api.search_projection import (
    build_structured_preview,
    fetch_item_structured_data,
    project_results,
)
from rag_recipes.config import Settings
from rag_recipes.providers.embeddings.base import EmbeddingProvider
from rag_recipes.providers.errors import LLMTechnicalError
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest
from rag_recipes.retrieval.search import search
from rag_recipes.retrieval.types import SearchRequest
from rag_recipes.storage.models.chunk import Chunk

# doc 8 § 7 — the safe-fallback message shown when no citation-safe answer is possible.
FALLBACK_WARNING = (
    "I found relevant results, but could not generate a citation-safe answer. "
    "Here are the retrieved items instead."
)
# Shown when retrieval returns nothing to ground an answer on.
NO_RESULTS_WARNING = "No relevant results were found for this query."


@dataclass
class AnswerResult:
    """Internal result the answers route projects into ``AnswerResponse``.

    ``is_fallback`` lets the route honor the "fallback always keeps results, success
    drops them when ``include_results=False``" rule. ``results`` is always populated
    when there are retrieved items (the route applies the success-path drop).
    """

    query: str
    answer: AnswerBody
    recommendations: list[Recommendation] = field(default_factory=list)
    citations: list[AnswerCitation] = field(default_factory=list)
    results: list[KnowledgeItemResult] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_fallback: bool = False


async def generate_answer(
    session: AsyncSession,
    request: SearchRequest,
    *,
    style: str,
    include_results: bool,
    llm_provider: LLMProvider,
    embedding_provider: EmbeddingProvider,
    settings: Settings,
) -> AnswerResult:
    """Run retrieval, synthesize a grounded answer, and validate its citations.

    ``request.limit`` is the route-normalized positive ``effective_limit``, so the
    context-pack ``item_limit`` is always ≥ 1.
    """
    result = await search(session, request, provider=embedding_provider, settings=settings)

    if not result.items:
        # Nothing to ground an answer on — fallback without an LLM call.
        return _fallback(request.query, style, results=[], warning=NO_RESULTS_WARNING)

    item_limit = min(request.limit, settings.answer_context_item_limit)
    chunks_per_item = settings.answer_matched_chunks_per_item

    structured = await fetch_item_structured_data(session, result)
    preview_by_item = {
        item_id: build_structured_preview(extra.get("structured_data", {})).model_dump(
            by_alias=True
        )
        for item_id, extra in structured.items()
    }
    chunk_input_by_id = await _fetch_chunk_inputs(
        session, result, item_limit=item_limit, chunks_per_item=chunks_per_item
    )

    pack = build_context_pack(
        request.query,
        result.items,
        chunk_input_by_id,
        item_limit=item_limit,
        chunks_per_item=chunks_per_item,
        structured_preview_by_item_id=preview_by_item,
    )

    # The pack drops uncitable chunks and items with no citable emitted chunk — if
    # nothing citable survives, there's no grounded answer to make.
    if not pack.items:
        results = await project_results(session, result, structured=structured)
        return _fallback(request.query, style, results=results, warning=FALLBACK_WARNING)

    try:
        response = await llm_provider.generate_structured_output(
            StructuredOutputRequest(
                provider=llm_provider.provider,
                model=llm_provider.default_model,
                prompt_version=settings.answer_prompt_version,
                schema_version=settings.answer_schema_version,
                input=render_answer_input(request.query, pack),
                json_schema=build_answer_v1_json_schema(),
            )
        )
    except LLMTechnicalError:
        # Unified safe-fallback path: a technical LLM failure is treated like a
        # parse/citation failure — never a 502, always show what we found.
        results = await project_results(session, result, structured=structured)
        return _fallback(request.query, style, results=results, warning=FALLBACK_WARNING)

    answer_json = response.output_json
    if response.parse_error is not None or answer_json is None:
        results = await project_results(session, result, structured=structured)
        return _fallback(request.query, style, results=results, warning=FALLBACK_WARNING)

    errors = validate_citations(answer_json, pack)
    if errors:
        results = await project_results(session, result, structured=structured)
        return _fallback(request.query, style, results=results, warning=FALLBACK_WARNING)

    # Success: reconstruct the answer body / recommendations / citations from the
    # pack. validate_citations checks citation membership/binding and the structural
    # shape, but not every scalar type (a truthy non-string `text`/`reason` would
    # trip Pydantic) — so guard model construction and degrade any residual surprise
    # to the safe fallback, never a 500 (review #1, finding 1 / round 2). The DB
    # projection stays OUTSIDE the guard so a genuine DB failure still surfaces.
    try:
        answer_body, recommendations, response_citations = _build_success_payload(
            answer_json, pack, style=style
        )
    except (ValidationError, TypeError):
        results = await project_results(session, result, structured=structured)
        return _fallback(request.query, style, results=results, warning=FALLBACK_WARNING)

    results = (
        await project_results(session, result, structured=structured)
        if include_results
        else []
    )
    return AnswerResult(
        query=request.query,
        answer=answer_body,
        recommendations=recommendations,
        citations=response_citations,
        results=results,
        warnings=[],
        is_fallback=False,
    )


def _build_success_payload(
    answer_json: dict[str, Any], pack: ContextPack, *, style: str
) -> tuple[AnswerBody, list[Recommendation], list[AnswerCitation]]:
    """Reconstruct the answer body, recommendations, and response citations from the pack.

    Pure (no I/O). Accessors stay defensive, but a residual non-conforming scalar
    (e.g. a truthy non-string ``text``/``reason``) is allowed to raise
    ``ValidationError``/``TypeError`` during model construction so the caller routes
    to the safe fallback rather than letting it escape as a 500.
    """
    raw_answer = answer_json.get("answer")
    answer_block = raw_answer if isinstance(raw_answer, dict) else {}
    recommendations_json = [
        rec for rec in _as_list(answer_json.get("recommendations")) if isinstance(rec, dict)
    ]
    title_by_item = {item.knowledge_item_id: item.title for item in pack.items}

    answer_citation_ids = [
        c for c in _as_list(answer_block.get("citations")) if isinstance(c, str)
    ]
    used_cite_ids: list[str] = list(answer_citation_ids)
    for rec in recommendations_json:
        for cid in _as_list(rec.get("citation_ids")):
            if isinstance(cid, str) and cid not in used_cite_ids:
                used_cite_ids.append(cid)

    recommendations: list[Recommendation] = []
    for rec in recommendations_json:
        item_id = rec.get("knowledge_item_id", "")
        recommendations.append(
            Recommendation(
                knowledge_item_id=item_id,
                title=title_by_item.get(item_id, "") if isinstance(item_id, str) else "",
                reason=rec.get("reason") or "",
                citation_ids=[
                    c for c in _as_list(rec.get("citation_ids")) if isinstance(c, str)
                ],
            )
        )
    answer_body = AnswerBody(
        style=style,
        text=answer_block.get("text") or "",
        citations=answer_citation_ids,
    )
    return answer_body, recommendations, build_response_citations(used_cite_ids, pack)


def _as_list(value: Any) -> list[Any]:
    """Coerce a model-supplied field to a list (``[]`` for anything non-list).

    ``parse_error is None`` only guarantees a JSON *object*, not a schema-conforming
    one — the provider does no post-parse JSON-Schema validation. So a wrong-typed
    field (``null``, a string, …) must degrade to a validation failure, never an
    ``AttributeError``/``TypeError`` that would escape ``generate_answer`` as a 500.
    """
    return value if isinstance(value, list) else []


def validate_citations(answer_json: dict[str, Any], pack: ContextPack) -> list[str]:
    """Return citation-validation errors (empty ⇒ valid). Membership **and** binding.

    Rules (doc 8 § 7, plus binding): (a) every cited ``citation_id`` exists in the
    pack; (b) every recommendation has ≥1 citation; (c) every recommended
    ``knowledge_item_id`` is a pack item; (d) each of a recommendation's
    ``citation_ids`` belongs to *that* recommendation's item.

    Structurally malformed (but parseable) output — a missing/non-object ``answer``,
    a non-list ``recommendations``, a non-object recommendation — is itself a
    validation error, so a non-conforming model response funnels to the safe
    fallback rather than crashing (review #1, finding 1).
    """
    cite_owner: dict[str, str] = {}
    for item in pack.items:
        for citation in item.citations:
            cite_owner[citation.citation_id] = item.knowledge_item_id
    pack_item_ids = {item.knowledge_item_id for item in pack.items}

    errors: list[str] = []

    answer_block = answer_json.get("answer")
    if not isinstance(answer_block, dict):
        errors.append("answer block is missing or not an object")
        answer_block = {}
    # `isinstance(cid, str)` is checked *before* the membership lookup so an
    # unhashable nested value (e.g. a list) can never reach `cid in cite_owner` and
    # raise `TypeError` — it is reported as an invalid citation instead (review #3).
    for cid in _as_list(answer_block.get("citations")):
        if not isinstance(cid, str) or cid not in cite_owner:
            errors.append(f"answer cites unknown citation_id {cid!r}")

    recommendations = answer_json.get("recommendations")
    if not isinstance(recommendations, list):
        errors.append("recommendations is missing or not a list")
        recommendations = []
    for index, rec in enumerate(recommendations):
        if not isinstance(rec, dict):
            errors.append(f"recommendation[{index}] is not an object")
            continue
        rec_item = rec.get("knowledge_item_id")
        citation_ids = _as_list(rec.get("citation_ids"))
        if not isinstance(rec_item, str) or rec_item not in pack_item_ids:
            errors.append(
                f"recommendation[{index}] cites unknown knowledge_item_id {rec_item!r}"
            )
        if not citation_ids:
            errors.append(f"recommendation[{index}] has no citations")
        for cid in citation_ids:
            if not isinstance(cid, str) or cid not in cite_owner:
                errors.append(
                    f"recommendation[{index}] cites unknown citation_id {cid!r}"
                )
            elif cite_owner[cid] != rec_item:
                errors.append(
                    f"recommendation[{index}] cites {cid!r} which belongs to a "
                    f"different item ({cite_owner[cid]!r}, not {rec_item!r})"
                )

    return errors


def build_response_citations(
    used_cite_ids: list[str], pack: ContextPack
) -> list[AnswerCitation]:
    """Reconstruct response ``citations[]`` from the pack (never from the LLM)."""
    detail: dict[str, AnswerCitation] = {}
    for item in pack.items:
        for citation in item.citations:
            detail[citation.citation_id] = AnswerCitation(
                citation_id=citation.citation_id,
                knowledge_item_id=item.knowledge_item_id,
                source_span_id=citation.source_span_id,
                label=citation.label,
            )
    return [detail[cid] for cid in used_cite_ids if cid in detail]


async def _fetch_chunk_inputs(
    session: AsyncSession,
    result: Any,
    *,
    item_limit: int,
    chunks_per_item: int,
) -> dict[str, ChunkInput]:
    """Batch-fetch ``text`` + ``source_span_ids`` for the chunks the pack will consider.

    Only the chunks within the builder's cap window (top ``item_limit`` items, top
    ``chunks_per_item`` matched chunks each) are fetched — the rest are never read.
    """
    chunk_ids = [
        mc.chunk_id
        for item in result.items[:item_limit]
        for mc in item.matched_chunks[:chunks_per_item]
    ]
    if not chunk_ids:
        return {}
    rows = (
        await session.execute(
            select(Chunk.id, Chunk.text, Chunk.source_span_ids).where(
                Chunk.id.in_(chunk_ids)
            )
        )
    ).all()
    return {
        row.id: ChunkInput(text=row.text, source_span_ids=list(row.source_span_ids or []))
        for row in rows
    }


def _fallback(
    query: str,
    style: str,
    *,
    results: list[KnowledgeItemResult],
    warning: str,
) -> AnswerResult:
    return AnswerResult(
        query=query,
        answer=AnswerBody(style=style, text=warning, citations=[]),
        recommendations=[],
        citations=[],
        results=results,
        warnings=[warning],
        is_fallback=True,
    )
