"""Retrieval evaluation runner (Epic 16, doc 12 § 7 / § 10).

``run_retrieval_eval`` loads a golden query set (Epic 14 fixtures), drives
``POST /api/v1/search`` once per query at the selected mode, folds each
query's ranked ``KnowledgeItem.id`` list into a pytrec_eval run, computes the
retrieval metrics (``evals.metrics.retrieval``), and writes ``results.json`` +
``summary.md`` through Epic 14's :class:`~evals.reports.ReportRun`.

Injection seams (all keyword-only): ``search`` (the async search caller),
``report_factory`` (binds ``ReportRun`` to a custom root/settings), and
``settings``. Unit tests inject all three; only a real, credentialed live run
uses the defaults — see :func:`_build_default_search`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from evals.fixtures import load_query_fixtures
from evals.metrics.retrieval import build_run_dict, compute_retrieval_metrics

if TYPE_CHECKING:
    from collections.abc import Callable

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


def _build_default_search(settings: Any) -> SearchCaller:
    """Build the in-process **live** search caller — real app, real providers.

    LIVE PATH — never exercised by automated tests. The unoverridden app's
    ``get_embedding_provider`` constructs a real ``OpenAIEmbeddingProvider``
    (and ``get_reranker_provider`` a real ``OpenAIRerankerProvider`` when
    ``reranking_enabled``), so any hybrid/vector search through this caller is
    a real network call requiring operator-supplied credentials. Tests must
    inject a fake ``search`` instead.
    """
    import httpx

    from rag_recipes.api.app import app

    async def _search(query_text: str, *, mode: str, limit: int) -> dict[str, Any]:
        transport = httpx.ASGITransport(app=app)
        headers = {"Authorization": f"Bearer {settings.personal_api_token}"}
        async with httpx.AsyncClient(
            transport=transport, base_url="http://rag-evals", headers=headers
        ) as client:
            response = await client.post(
                "/api/v1/search",
                json={
                    "query": query_text,
                    "mode": mode,
                    "limit": limit,
                    "filters": {"exclude_needs_review": True},
                },
            )
            response.raise_for_status()
            body: dict[str, Any] = response.json()
            return body

    return _search


def _fold_qrels(qrels: list[Any]) -> dict[str, dict[str, int]]:
    """Group list-shaped ``Qrel`` rows into pytrec_eval's ``{qid: {item: rel}}``.

    Later duplicate ``(query_id, knowledge_item_id)`` rows overwrite earlier
    ones (plan Scope).
    """
    folded: dict[str, dict[str, int]] = {}
    for qrel in qrels:
        folded.setdefault(qrel.query_id, {})[qrel.knowledge_item_id] = int(qrel.relevance)
    return folded


def _expected_item_ranks(
    expected: dict[str, int], retrieved_ids: list[str]
) -> dict[str, int | None]:
    """Map each expected item to its 1-based retrieved rank, or ``None`` if absent."""
    positions = {item_id: rank for rank, item_id in enumerate(retrieved_ids, start=1)}
    return {item_id: positions.get(item_id) for item_id in expected}


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
) -> ReportRun:
    """Run the retrieval eval for ``query_set`` and write the report.

    Validates ``mode`` before any search call, folds the list-shaped
    ``QueryFixtureSet`` into pytrec_eval qrels, drives one search per query at
    ``limit = max(k, settings.search_default_limit)``, and writes the payload
    ``{"report_type": "retrieval", "run": ..., "aggregate": ..., "per_query":
    ...}`` (nested by ``ReportRun.write_results`` under the ``"results"`` key).
    """
    if mode not in VALID_MODES:
        raise ValueError(f"invalid mode {mode!r}: expected one of {sorted(VALID_MODES)}")
    if settings is None:
        from rag_recipes.config import get_settings

        settings = get_settings()
    if search is None:
        search = _build_default_search(settings)
    if report_factory is None:
        from evals.reports import ReportRun as _ReportRun

        report_factory = lambda run_label: _ReportRun(run_label)  # noqa: E731

    fixture_set = load_query_fixtures(query_set)
    qrels = _fold_qrels(fixture_set.qrels)
    limit = max(k, settings.search_default_limit)

    retrieved_ids: dict[str, list[str]] = {}
    for query in fixture_set.queries:
        envelope = await search(query.query_text, mode=mode, limit=limit)
        retrieved_ids[query.query_id] = [
            result["item"]["id"] for result in envelope["results"]
        ]

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
    return report
