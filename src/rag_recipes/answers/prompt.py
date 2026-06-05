"""The query-time answer prompt and context-pack serialization (doc 8 § 5).

The grounding rules (doc 8 § 5) keep the answer derived, not canonical: the model
may explain and compare retrieved items but must cite every recommendation using
only the ``cite_N`` ids present in the context pack, and must never invent recipes,
ingredients, times, or sources. ``render_answer_input`` serializes the pack into a
stable, JSON-ish string and surfaces the allowed ``citation_id``s so the model has
the exact id space its citations are later validated against (Phase 17.2, doc 8 § 7).

The prompt is versioned by ``ANSWER_PROMPT_VERSION`` (mirrors the
``Settings.answer_prompt_version`` default; a drift test guards the pair). Phase
17.3 adds the remaining per-style prompt variants on the same pipeline.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from rag_recipes.answers.context_pack import ContextPack

__all__ = [
    "ANSWER_PROMPT_VERSION",
    "GROUNDING_RULES",
    "PROMPTS_BY_VERSION",
    "RECOMMENDATION_PROMPT",
    "render_answer_input",
]

#: Mirrors ``Settings.answer_prompt_version`` default; a drift test guards the pair.
ANSWER_PROMPT_VERSION = "answer-recommendation-v1"

# The strict grounding rules (doc 8 § 5), verbatim in intent.
GROUNDING_RULES = """\
Use only the provided context for source-backed claims.
Do not invent recipes, ingredients, times, or cookbook sources.
If the context is insufficient, say so.
Cite every recommended item using provided citation IDs.
Do not cite sources that were not provided.
Prefer concise answers.
Only reproduce full recipe instructions in authenticated personal contexts when \
requested or clearly needed."""

RECOMMENDATION_PROMPT = f"""\
You are a grounded recommendation assistant for a personal recipe library. Given a \
user query and a context pack of retrieved items, recommend one or more items and \
explain why each fits, using only the provided context.

Grounding rules:
{GROUNDING_RULES}

Produce a structured answer that conforms to the answer.v1 schema:
- answer.style is "recommendation".
- answer.text is a concise explanation of the recommended items.
- answer.citations lists the citation IDs the answer relies on.
- recommendations[] names each recommended knowledge_item_id with a reason and its \
citation_ids (at least one).
- citations[] maps each cited citation_id back to its knowledge_item_id, \
source_span_id, and label.

Only use citation IDs and knowledge_item_ids that appear in the context pack below."""

#: Prompt text keyed by version so Phase 17.3 can register per-style variants.
PROMPTS_BY_VERSION = {ANSWER_PROMPT_VERSION: RECOMMENDATION_PROMPT}


def render_answer_input(query: str, pack: ContextPack) -> str:
    """Serialize ``pack`` into a stable model-input string and surface allowed ids.

    Deterministic for fixed inputs: the pack is rendered as indented JSON (dataclass
    field order is insertion order), followed by the explicit lists of allowed
    ``citation_id``s and ``knowledge_item_id``s the model may cite — the same id
    space Phase 17.2 validates the model's answer against.
    """
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
        f"User query: {query}\n\n"
        f"Context pack:\n{pack_json}\n\n"
        f"Allowed citation IDs: {', '.join(allowed_citation_ids) or '(none)'}\n"
        f"Allowed knowledge_item_ids: {', '.join(allowed_item_ids) or '(none)'}"
    )
