"""Retrieval evaluation runner (Epic 16, doc 12 § 7 / § 10).

``run_retrieval_eval`` loads a golden query set (Epic 14 fixtures), drives
``POST /api/v1/search`` once per query at the selected mode, folds each
query's ranked ``KnowledgeItem.id`` list into a pytrec_eval run, computes the
retrieval metrics (``evals.metrics.retrieval``), and writes ``results.json`` +
``summary.md`` through Epic 14's :class:`~evals.reports.ReportRun`.

Injection seams (all keyword-only): ``search`` (the async search caller),
``report_factory`` (binds ``ReportRun`` to a custom root/settings), and
``settings``. Unit tests inject all three; only a real, credentialed live run
uses the defaults — see :func:`_default_search`.
"""

from __future__ import annotations

from contextlib import AsyncExitStack, asynccontextmanager
from typing import TYPE_CHECKING, Any, Protocol

from evals.fixtures import load_query_fixtures
from evals.metrics.retrieval import build_run_dict, compute_retrieval_metrics

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Callable
    from pathlib import Path

    from evals.reports import ReportRun

__all__ = ["VALID_MODES", "RetrievalSettingsLike", "SearchCaller", "run_retrieval_eval"]

VALID_MODES = frozenset({"hybrid", "keyword", "vector"})


class RetrievalSettingsLike(Protocol):
    """The settings attributes the runner reads (and nothing more)."""

    @property
    def search_default_limit(self) -> int: ...

    @property
    def embedding_provider(self) -> str: ...

    @property
    def embedding_model(self) -> str: ...

    @property
    def reranking_enabled(self) -> bool: ...

    @property
    def rerank_provider(self) -> str: ...

    @property
    def rerank_model(self) -> str: ...

    @property
    def rerank_top_n(self) -> int: ...


class SearchCaller(Protocol):
    """Async search seam: ``(query_text, *, mode, limit) -> response envelope``."""

    async def __call__(
        self, query_text: str, *, mode: str, limit: int
    ) -> dict[str, Any]: ...


@asynccontextmanager
async def _default_search(settings: Any) -> AsyncIterator[SearchCaller]:
    """The in-process **live** search caller — real app, real providers.

    LIVE PATH — never exercised by automated tests. The unoverridden app's
    ``get_embedding_provider`` constructs a real ``OpenAIEmbeddingProvider``
    (and ``get_reranker_provider`` a real ``OpenAIRerankerProvider`` when
    ``reranking_enabled``), so any hybrid/vector search through this caller is
    a real network call requiring operator-supplied credentials. Tests must
    inject a fake ``search`` instead.

    A context manager because the app's **lifespan must run**: ``get_session``
    reads ``request.app.state.session_factory``, which only the lifespan sets
    (``api/app.py``) — ``httpx.ASGITransport`` does not run lifespan events, so
    driving the app without it made every query 500 on ``AttributeError``.
    Integration tests never hit this because they override ``get_session``.
    Entering the lifespan also means a reachable Redis (the lifespan builds the
    arq pool), same as serving the app under uvicorn.

    Client and lifespan are entered **once per run**, not per query.
    """
    import httpx

    from rag_recipes.api.app import app

    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://rag-evals",
            headers={"Authorization": f"Bearer {settings.personal_api_token}"},
        ) as client,
    ):

        async def _search(query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
            response = await client.post(
                "/api/v1/search",
                json={
                    "query": query_text,
                    "mode": mode,
                    "limit": limit,
                    "filters": {"exclude_needs_review": True},
                    # `SearchRequestBody.include_debug` defaults to False, so
                    # without this the per-query breakdown's debug section is
                    # unreachable on the live path. The route ANDs it with
                    # `Settings.debug_endpoints_enabled` and pops the key when
                    # gated off, which is exactly the "included when dev mode
                    # is enabled" contract (Phase 16.2 acceptance).
                    "include_debug": True,
                },
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body

        yield _search


def _fold_qrels(qrels: list[Any]) -> dict[str, dict[str, int]]:
    """Group list-shaped ``Qrel`` rows into pytrec_eval's ``{qid: {item: rel}}``.

    Later duplicate ``(query_id, knowledge_item_id)`` rows overwrite earlier
    ones (plan Scope).
    """
    folded: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        folded.setdefault(qrel.query_id, {})[qrel.knowledge_item_id] = int(qrel.relevance)
    return folded


def _check_query_ids_align(
    query_set: str, queries: list[Any], qrels: dict[str, dict[str, int]]
) -> None:
    """Reject a fixture set whose ``queries.tsv`` and ``qrels.tsv`` disagree.

    Metrics aggregate over the **qrels** query ids (DECISIONS #5), so the two
    sides diverge silently and in opposite directions:

    - a query with no qrels row is searched, then dropped from ``per_query``
      and from the aggregate denominator entirely — a hard, unjudged query
      vanishes while the headline stays falsely perfect;
    - a qrels row naming a query id that ``queries.tsv`` never lists is never
      searched, so it scores a permanent ``0.0`` and drags the headline down.

    Both are fixture bugs rather than retrieval signal, and neither is visible
    in the report, so fail loudly before spending any search call. Two empty
    sides stay the legitimate empty-dataset case ``load_query_fixtures``
    allows.
    """
    query_ids = {query.query_id for query in queries}
    unjudged = sorted(query_ids - set(qrels))
    orphaned = sorted(set(qrels) - query_ids)
    if not unjudged and not orphaned:
        return
    problems = []
    if unjudged:
        problems.append(f"queries with no qrels row: {', '.join(unjudged)}")
    if orphaned:
        problems.append(f"qrels rows for unknown query ids: {', '.join(orphaned)}")
    raise ValueError(
        f"inconsistent query fixture set {query_set!r} — {'; '.join(problems)}"
    )


#: The merged ``RetrievalDebugInfo`` keys (``api/schemas/search.py``) — the
#: only keys the per-query debug section may render. ``filters_applied`` and
#: ``chunk_type_boosts`` do not exist on the wire schema.
_DEBUG_KEYS = (
    "retrieval_mode",
    "normalized_query",
    "embedding_model",
    "keyword_top_k",
    "vector_top_k",
    "keyword_candidates",
    "vector_candidates",
    "merged_candidates",
    "grouped_items",
    "rerank_applied",
)


def _chunk_types(result: dict[str, Any]) -> str:
    """Comma-join the matched chunk types, first-occurrence order, de-duplicated."""
    seen: dict[str, None] = {}
    for chunk in result.get("matched_chunks", []):
        seen.setdefault(chunk["chunk_type"], None)
    return ", ".join(seen) or "—"


def _result_score(result: dict[str, Any]) -> str:
    """``max(matched_chunks[].score)`` — the envelope has no item-level score."""
    scores = [chunk["score"] for chunk in result.get("matched_chunks", [])]
    return f"{max(scores):.4f}" if scores else "—"


def _citations(result: dict[str, Any]) -> str:
    labels = [citation["label"] for citation in result.get("source_citations", [])]
    return "; ".join(labels) or "—"


def _render_per_query_md(
    *,
    run_block: dict[str, Any],
    query_records: list[dict[str, Any]],
    qrels: dict[str, dict[str, int]],
    k: int,
) -> str:
    """Per-query breakdown: query text, expected items, top-k table, debug.

    The ``score`` column is ``max(matched_chunks[].score)`` — merged Epic 13
    projects no item-level score into the API envelope. The ``debug`` section
    is rendered only when the envelope carried the key (``envelope.get`` —
    merged ``routes/search.py`` pops it entirely when gating is off) and only
    with keys that exist on the merged ``RetrievalDebugInfo`` schema.
    """
    rerank_state = "on" if run_block["reranking_enabled"] else "off"
    lines = [
        "# Per-query retrieval breakdown",
        "",
        f"Query set: `{run_block['query_set']}` · mode: `{run_block['mode']}` · "
        f"k={k} · reranking: {rerank_state}",
        "",
        "`score` is `max(matched_chunks[].score)` — the search envelope exposes no "
        "item-level fused score.",
    ]
    for record in query_records:
        expected = _relevant_items(qrels.get(record["query_id"], {}))
        expected_ids = ", ".join(f"`{item_id}`" for item_id in expected) or "—"
        lines += [
            "",
            f"## {record['query_id']} — \"{record['query_text']}\"",
            "",
            f"Expected items (qrels): {expected_ids}",
            "",
            "| rank | item_id | score | matched chunk types | citations |",
            "|---|---|---|---|---|",
        ]
        for rank, result in enumerate(record["results"][:k], start=1):
            lines.append(
                f"| {rank} | {result['item']['id']} | {_result_score(result)} "
                f"| {_chunk_types(result)} | {_citations(result)} |"
            )
        debug = record["debug"]
        if debug is not None:
            lines += ["", "### Debug", ""]
            lines += [
                f"- {key}: {debug[key]}" for key in _DEBUG_KEYS if key in debug
            ]
    return "\n".join(lines) + "\n"


def _relevant_items(judged: dict[str, int]) -> list[str]:
    """The judged items a query is actually *expected* to retrieve.

    A qrels row with relevance ``0`` is an explicit "judged, not relevant"
    verdict — pytrec_eval scores it as such, and the graded ``0/1/2/3`` scale
    the metrics wrapper documents makes it a first-class value. Feeding it
    into the expected-item set would label a non-relevant item "expected" in
    ``per_query.md`` and, worse, make 16.2's diff flag it *leaving* the top-k
    as a regression when that is an improvement.
    """
    return [item_id for item_id, relevance in judged.items() if relevance > 0]


def _expected_item_ranks(
    expected: dict[str, int], retrieved_ids: list[str]
) -> dict[str, int | None]:
    """Map each relevant item to its 1-based retrieved rank, or ``None`` if absent."""
    positions = {item_id: rank for rank, item_id in enumerate(retrieved_ids, start=1)}
    return {item_id: positions.get(item_id) for item_id in _relevant_items(expected)}


def _run_block(
    *, query_set: str, mode: str, k: int, limit: int, settings: RetrievalSettingsLike
) -> dict[str, Any]:
    """The run-config record; two runs differing here are not comparable."""
    block: dict[str, Any] = {
        "query_set": query_set,
        "mode": mode,
        "k": k,
        "limit": limit,
        "embedding_provider": settings.embedding_provider,
        "embedding_model": settings.embedding_model,
        "reranking_enabled": settings.reranking_enabled,
    }
    if settings.reranking_enabled:
        block["rerank_provider"] = settings.rerank_provider
        block["rerank_model"] = settings.rerank_model
        block["rerank_top_n"] = settings.rerank_top_n
    return block


def _render_summary(
    *,
    run_block: dict[str, Any],
    aggregate: dict[str, float],
    per_query: dict[str, dict[str, float]],
    sample_size: int = 3,
) -> str:
    """Aggregate headline + secondaries and a best/worst-by-NDCG@10 sample."""
    rerank_state = "on" if run_block["reranking_enabled"] else "off"
    by_ndcg = sorted(per_query.items(), key=lambda entry: entry[1]["ndcg_cut_10"])
    worst = by_ndcg[:sample_size]
    best = list(reversed(by_ndcg[-sample_size:]))
    lines = [
        "# Retrieval evaluation summary",
        "",
        f"Query set: `{run_block['query_set']}` · mode: `{run_block['mode']}` · "
        f"k={run_block['k']} · limit={run_block['limit']} · reranking: {rerank_state}",
        "",
        "## Aggregate",
        "",
        f"- **NDCG@10 (headline): {aggregate['ndcg_cut_10']:.4f}**",
        f"- Recall@5: {aggregate['recall_5']:.4f}",
        f"- Recall@10: {aggregate['recall_10']:.4f}",
        f"- MRR: {aggregate['recip_rank']:.4f}",
        "",
        "## Best queries by NDCG@10",
        "",
    ]
    lines += [f"- `{qid}`: {measures['ndcg_cut_10']:.4f}" for qid, measures in best]
    lines += ["", "## Worst queries by NDCG@10", ""]
    lines += [f"- `{qid}`: {measures['ndcg_cut_10']:.4f}" for qid, measures in worst]
    return "\n".join(lines) + "\n"


async def run_retrieval_eval(
    query_set: str,
    k: int,
    label: str,
    mode: str = "hybrid",
    *,
    search: SearchCaller | None = None,
    report_factory: Callable[[str], ReportRun] | None = None,
    settings: RetrievalSettingsLike | None = None,
    fixtures_root: Path | None = None,
) -> ReportRun:
    """Run the retrieval eval for ``query_set`` and write the report.

    Validates ``mode``, ``k``, and the fixture set's query-id consistency
    before any search call, folds the list-shaped ``QueryFixtureSet`` into
    pytrec_eval qrels, drives one search per query at
    ``limit = max(k, settings.search_default_limit)``, and writes the payload
    ``{"report_type": "retrieval", "run": ..., "aggregate": ..., "per_query":
    ...}`` (nested by ``ReportRun.write_results`` under the ``"results"`` key).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}: expected one of {sorted(VALID_MODES)}")
    if k < 1:
        # k reaches a bare slice (`results[:k]`) and the diff's top-k cutoff:
        # k=0 renders an empty breakdown and makes every expected item look
        # dropped-from-top-k, and a negative k silently trims the *tail* of
        # each result list instead of erroring.
        raise ValueError(f"invalid k {k!r}: expected a positive integer")
    if settings is None:
        from rag_recipes.config import get_settings

        settings = get_settings()
    if report_factory is None:
        from evals.reports import ReportRun as _ReportRun

        report_factory = lambda run_label: _ReportRun(run_label)  # noqa: E731

    fixture_set = load_query_fixtures(query_set, root=fixtures_root)
    qrels = _fold_qrels(fixture_set.qrels)
    _check_query_ids_align(query_set, fixture_set.queries, qrels)
    limit = max(k, settings.search_default_limit)

    # Collect the *full* result objects per query (DECISIONS #4, Phase 16.2):
    # the per-query breakdown and the run dict share this one pass.
    query_records: list[dict[str, Any]] = []
    async with AsyncExitStack() as stack:
        if search is None:
            # Live default: app lifespan + one HTTP client for the whole run.
            # An injected `search` leaves the stack empty — nothing is entered.
            search = await stack.enter_async_context(_default_search(settings))
        for query in fixture_set.queries:
            envelope = await search(query.query_text, mode=mode, limit=limit)
            query_records.append(
                {
                    "query_id": query.query_id,
                    "query_text": query.query_text,
                    "results": envelope["results"],
                    # .get(): merged routes/search.py pops the key when gating is off.
                    "debug": envelope.get("debug"),
                }
            )
    retrieved_ids = {
        record["query_id"]: [result["item"]["id"] for result in record["results"]]
        for record in query_records
    }

    run = {query_id: build_run_dict(ids) for query_id, ids in retrieved_ids.items()}
    metrics = compute_retrieval_metrics(qrels, run)

    run_block = _run_block(query_set=query_set, mode=mode, k=k, limit=limit, settings=settings)
    per_query = {
        query_id: {
            "metrics": measures,
            "expected_item_ranks": _expected_item_ranks(
                qrels.get(query_id, {}), retrieved_ids.get(query_id, [])
            ),
            "retrieved_ids": retrieved_ids.get(query_id, []),
        }
        for query_id, measures in metrics.per_query.items()
    }
    payload = {
        "report_type": "retrieval",
        "run": run_block,
        "aggregate": metrics.aggregate,
        "per_query": per_query,
    }

    with report_factory(label) as report:
        report.write_results(payload)
        report.write_summary(
            _render_summary(
                run_block=run_block,
                aggregate=metrics.aggregate,
                per_query=metrics.per_query,
            )
        )
        # ReportRun has no per_query.md writer (its writers are summary.md /
        # results.json / per_item_breakdowns.md), so write via run.path.
        (report.path / "per_query.md").write_text(
            _render_per_query_md(
                run_block=run_block, query_records=query_records, qrels=qrels, k=k
            ),
            encoding="utf-8",
        )
    return report
