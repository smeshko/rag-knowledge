"""The query-time answer prompts and context-pack serialization (doc 8 §§ 5, 9).

The grounding rules (doc 8 § 5) keep the answer derived, not canonical: the model
may explain and compare retrieved items but must cite using only the ``cite_N`` ids
present in the context pack, and must never invent recipes, ingredients, times, or
sources. Each answer style (doc 8 § 9 — ``recommendation``/``summary``/``comparison``/
``direct_answer``) has its own **versioned** prompt variant sharing those rules; only
the task instruction and the ``answer.style`` value differ. ``render_answer_input``
prepends the resolved per-style prompt to the serialized pack so the model receives
the grounding rules together with the exact ``citation_id`` space its citations are
later validated against (Phase 17.2, doc 8 § 7).

``ANSWER_PROMPT_VERSION`` mirrors the ``Settings.answer_prompt_version`` default (a
drift test guards the pair); ``resolve_prompt_version`` returns the per-style version
that flows into both ``StructuredOutputRequest`` and the debug payload.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from rag_recipes.answers.context_pack import ContextPack

__all__ = [
    "ANSWER_PROMPT_VERSION",
    "GROUNDING_RULES",
    "PROMPT_BY_STYLE",
    "PROMPTS_BY_VERSION",
    "RECOMMENDATION_PROMPT",
    "render_answer_input",
    "resolve_prompt_version",
]

# The strict grounding rules (doc 8 § 5), verbatim in intent — shared by every style.
GROUNDING_RULES = """\
Use only the provided context for source-backed claims.
Do not invent recipes, ingredients, times, or cookbook sources.
If the context is insufficient, say so.
Cite every recommended item using provided citation IDs.
Do not cite sources that were not provided.
Prefer concise answers.
Only reproduce full recipe instructions in authenticated personal contexts when \
requested or clearly needed."""

_DEFAULT_STYLE = "recommendation"


def _build_prompt(*, role: str, task: str, style: str, recommendations_line: str) -> str:
    return f"""\
You are a grounded {role} for a personal recipe library. Given a user query and a \
context pack of retrieved items, {task}, using only the provided context.

Grounding rules:
{GROUNDING_RULES}

Produce a structured answer that conforms to the answer.v1 schema:
- answer.style is "{style}".
- answer.text is a concise, source-backed response.
- answer.citations lists the citation IDs the answer relies on (at least one).
- {recommendations_line}
- citations[] maps each cited citation_id back to its knowledge_item_id, \
source_span_id, and label.

Only use citation IDs and knowledge_item_ids that appear in the context pack below."""


RECOMMENDATION_PROMPT = _build_prompt(
    role="recommendation assistant",
    task="recommend one or more items and explain why each fits",
    style="recommendation",
    recommendations_line=(
        "recommendations[] names each recommended knowledge_item_id with a reason and "
        "its citation_ids (at least one)."
    ),
)

SUMMARY_PROMPT = _build_prompt(
    role="summarization assistant",
    task="summarize the retrieved results",
    style="summary",
    recommendations_line=(
        "recommendations[] may be empty for a summary; any recommendation you include "
        "must name a knowledge_item_id with a reason and its citation_ids (at least one)."
    ),
)

COMPARISON_PROMPT = _build_prompt(
    role="comparison assistant",
    task="compare the retrieved items along the dimensions the query implies",
    style="comparison",
    recommendations_line=(
        "recommendations[] names each compared knowledge_item_id with a reason and its "
        "citation_ids (at least one)."
    ),
)

DIRECT_ANSWER_PROMPT = _build_prompt(
    role="question-answering assistant",
    task="answer the user's specific question from the retrieved context",
    style="direct_answer",
    recommendations_line=(
        "recommendations[] may be empty for a direct answer; any recommendation you "
        "include must name a knowledge_item_id with a reason and its citation_ids "
        "(at least one)."
    ),
)

#: Mirrors ``Settings.answer_prompt_version`` default; a drift test guards the pair.
ANSWER_PROMPT_VERSION = "answer-recommendation-v1"

#: Each answer style → ``(prompt_text, prompt_version)``. The version is resolved per
#: style and flows into ``StructuredOutputRequest`` and the debug payload (doc 8 § 9).
PROMPT_BY_STYLE: dict[str, tuple[str, str]] = {
    "recommendation": (RECOMMENDATION_PROMPT, ANSWER_PROMPT_VERSION),
    "summary": (SUMMARY_PROMPT, "answer-summary-v1"),
    "comparison": (COMPARISON_PROMPT, "answer-comparison-v1"),
    "direct_answer": (DIRECT_ANSWER_PROMPT, "answer-direct-answer-v1"),
}

#: Prompt text keyed by version (derived from ``PROMPT_BY_STYLE``).
PROMPTS_BY_VERSION = {version: text for text, version in PROMPT_BY_STYLE.values()}


def resolve_prompt_version(style: str) -> str:
    """Return the versioned prompt id for ``style`` (falls back to recommendation)."""
    entry = PROMPT_BY_STYLE.get(style)
    return entry[1] if entry is not None else ANSWER_PROMPT_VERSION


def render_answer_input(query: str, pack: ContextPack, *, style: str = _DEFAULT_STYLE) -> str:
    """Compose the full model input for ``style``: the prompt + the serialized pack.

    Deterministic for fixed inputs: the resolved per-style prompt is prepended to the
    indented-JSON pack (dataclass field order is insertion order), followed by the
    explicit lists of allowed ``citation_id``s / ``knowledge_item_id``s the model may
    cite — the same id space Phase 17.2 validates the model's answer against.
    """
    prompt_text = PROMPT_BY_STYLE.get(style, PROMPT_BY_STYLE[_DEFAULT_STYLE])[0]
    pack_json = json.dumps(asdict(pack), indent=2, ensure_ascii=False)

    allowed_citation_ids: list[str] = []
    allowed_item_ids: list[str] = []
    for item in pack.items:
        if item.knowledge_item_id not in allowed_item_ids:
            allowed_item_ids.append(item.knowledge_item_id)
        for citation in item.citations:
            if citation.citation_id not in allowed_citation_ids:
                allowed_citation_ids.append(citation.citation_id)

    return (
        f"{prompt_text}\n\n"
        f"User query: {query}\n\n"
        f"Context pack:\n{pack_json}\n\n"
        f"Allowed citation IDs: {', '.join(allowed_citation_ids) or '(none)'}\n"
        f"Allowed knowledge_item_ids: {', '.join(allowed_item_ids) or '(none)'}"
    )
