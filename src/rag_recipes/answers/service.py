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

    # Success: reconstruct citations + recommendation titles from the pack.
    answer_block = answer_json["answer"]
    recommendations_json = answer_json.get("recommendations", [])
    title_by_item = {item.knowledge_item_id: item.title for item in pack.items}

    used_cite_ids: list[str] = list(answer_block.get("citations", []))
    for rec in recommendations_json:
        for cid in rec.get("citation_ids", []):
            if cid not in used_cite_ids:
                used_cite_ids.append(cid)

    recommendations = [
        Recommendation(
            knowledge_item_id=rec["knowledge_item_id"],
            title=title_by_item.get(rec["knowledge_item_id"], ""),
            reason=rec.get("reason", ""),
            citation_ids=list(rec.get("citation_ids", [])),
        )
        for rec in recommendations_json
    ]

    results = (
        await project_results(session, result, structured=structured)
        if include_results
        else []
    )
    return AnswerResult(
        query=request.query,
        answer=AnswerBody(
            style=style,
            text=answer_block.get("text", ""),
            citations=list(answer_block.get("citations", [])),
        ),
        recommendations=recommendations,
        citations=build_response_citations(used_cite_ids, pack),
        results=results,
        warnings=[],
        is_fallback=False,
    )


def validate_citations(answer_json: dict[str, Any], pack: ContextPack) -> list[str]:
    """Return citation-validation errors (empty ⇒ valid). Membership **and** binding.

    Rules (doc 8 § 7, plus binding): (a) every cited ``citation_id`` exists in the
    pack; (b) every recommendation has ≥1 citation; (c) every recommended
    ``knowledge_item_id`` is a pack item; (d) each of a recommendation's
    ``citation_ids`` belongs to *that* recommendation's item.
    """
    cite_owner: dict[str, str] = {}
    for item in pack.items:
        for citation in item.citations:
            cite_owner[citation.citation_id] = item.knowledge_item_id
    pack_item_ids = {item.knowledge_item_id for item in pack.items}

    errors: list[str] = []

    answer_block = answer_json.get("answer", {})
    for cid in answer_block.get("citations", []):
        if cid not in cite_owner:
            errors.append(f"answer cites unknown citation_id {cid!r}")

    for index, rec in enumerate(answer_json.get("recommendations", [])):
        rec_item = rec.get("knowledge_item_id")
        citation_ids = rec.get("citation_ids", [])
        if rec_item not in pack_item_ids:
            errors.append(
                f"recommendation[{index}] cites unknown knowledge_item_id {rec_item!r}"
            )
        if not citation_ids:
            errors.append(f"recommendation[{index}] has no citations")
        for cid in citation_ids:
            if cid not in cite_owner:
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
