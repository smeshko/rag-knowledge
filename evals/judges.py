"""LLM-as-judge layer for the subjective extraction fields (Epic 15 Phase 15.2).

A ``Judge`` wraps an **injected** ``LLMProvider`` — it never constructs one and
never reads ``Settings`` or an API key — renders its versioned prompt (loaded
from ``data/fixtures/judge_prompts/<name>.md``), sends the minimal
``JudgeVerdict`` ``{rating, critique}`` JSON schema, and composes the typed
``JudgeRating`` the driver records.

Schema contract: the wire schema is strictified locally (mirroring the
documented ``extraction._strictify`` invariant — ``additionalProperties:
false`` + every key in ``required`` — without importing that private symbol,
DECISIONS #2). The schema is a *request*, not a guarantee: OpenAI enforces it
strictly, but the Anthropic path is non-strict forced tool use, so the response
is **always** re-validated with ``JudgeVerdict.model_validate``. Both failure
modes — ``output_json is None`` (rejection / ``max_tokens`` truncation) and a
validation failure — raise :class:`JudgeError`, never a silent pass/fail
(DECISIONS #4). Keep ``JudgeVerdict`` free of constraint keywords
(``min_length``/``pattern``/bounds): Anthropic's sanitizer strips them, so they
would be silently unenforced.

Prompt rendering substitutes the ``{extracted_output}`` / ``{expected_output}``
/ ``{source_text}`` tokens via ``str.replace`` (never ``str.format`` — source
text may contain literal braces). The judge request's ``prompt_version``
(``<name>-<version>``) and ``schema_version`` (``judge.verdict.v1``) are
distinct from extraction's, so fake/canned hashes can never collide. The prompt
version is load-bearing (cache key + report line): a prompt with no
front-matter version is a hard error, never defaulted.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from evals.fixtures import load_judge_prompt
from evals.models import JudgePrompt
from rag_recipes.providers.llm.base import LLMProvider
from rag_recipes.providers.llm.types import StructuredOutputRequest

__all__ = [
    "JUDGE_SCHEMA_VERSION",
    "Judge",
    "JudgeError",
    "JudgeRating",
    "JudgeVerdict",
    "load_judge",
]

JUDGE_SCHEMA_VERSION = "judge.verdict.v1"

_EXTRACTED_PLACEHOLDER = "{extracted_output}"
_EXPECTED_PLACEHOLDER = "{expected_output}"
_SOURCE_PLACEHOLDER = "{source_text}"


class JudgeError(Exception):
    """A judge call that produced no trustworthy verdict (DECISIONS #4).

    Raised when the provider rejects/truncates the call (``output_json is
    None``) or when the returned object fails ``JudgeVerdict`` validation. The
    driver records the fixture as *un-rated* — a broken judge call must never
    masquerade as a ``pass`` or ``fail``.
    """


class JudgeVerdict(BaseModel):
    """The wire output the LLM must produce — deliberately minimal.

    No constraint keywords (length/pattern/bounds): the Anthropic path strips
    them from the schema, so they would be a false promise.
    """

    rating: Literal["pass", "fail"]
    critique: str


class JudgeRating(BaseModel):
    """A verdict enriched with the harness-known judge identity and provenance."""

    judge_name: str
    judge_version: str
    rating: Literal["pass", "fail"]
    critique: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def _strictify(node: Any) -> Any:
    """Local strict-mode rewrite following the ``extraction._strictify`` invariant.

    For every object node force ``additionalProperties: false`` and list every
    property in ``required``; recurse nested mappings/sequences. Kept local
    rather than importing the private cross-package symbol (DECISIONS #2).
    """
    if isinstance(node, dict):
        if node.get("type") == "object" or "properties" in node:
            properties = node.get("properties", {})
            node["additionalProperties"] = False
            node["required"] = list(properties.keys())
        for value in node.values():
            _strictify(value)
    elif isinstance(node, list):
        for item in node:
            _strictify(item)
    return node


def _build_verdict_json_schema() -> dict[str, Any]:
    """Strict-mode JSON schema for ``JudgeVerdict`` (both keys required)."""
    schema = JudgeVerdict.model_json_schema()
    _strictify(schema)
    return schema


def _render_judge_prompt(
    template: str, *, extracted: str, expected: str, source_text: str
) -> str:
    """Substitute the three placeholder tokens via ``str.replace`` (not ``format``)."""
    return (
        template.replace(_EXTRACTED_PLACEHOLDER, extracted)
        .replace(_EXPECTED_PLACEHOLDER, expected)
        .replace(_SOURCE_PLACEHOLDER, source_text)
    )


class Judge:
    """One named, versioned judge over an injected ``LLMProvider``."""

    def __init__(
        self, name: str, version: str, llm_provider: LLMProvider, prompt: JudgePrompt
    ) -> None:
        self.name = name
        self.version = version
        self._llm_provider = llm_provider
        self._prompt = prompt

    @property
    def prompt_version(self) -> str:
        """The judge request's ``prompt_version`` — distinct from extraction's."""
        return f"{self.name}-{self.version}"

    @property
    def model(self) -> str:
        """The model the injected provider resolves to (feeds the cache key)."""
        return self._llm_provider.default_model

    async def judge(self, extracted: str, expected: str, source_text: str) -> JudgeRating:
        """Rate one fixture's extraction; raises ``JudgeError`` on any bad verdict."""
        request = StructuredOutputRequest(
            provider=self._llm_provider.provider,
            model=self._llm_provider.default_model,
            prompt_version=self.prompt_version,
            schema_version=JUDGE_SCHEMA_VERSION,
            input=_render_judge_prompt(
                self._prompt.text,
                extracted=extracted,
                expected=expected,
                source_text=source_text,
            ),
            json_schema=_build_verdict_json_schema(),
        )
        response = await self._llm_provider.generate_structured_output(request)
        if response.output_json is None:
            raise JudgeError(
                f"judge {self.name!r} ({self.version}): provider returned no output: "
                f"{response.parse_error}"
            )
        try:
            verdict = JudgeVerdict.model_validate(response.output_json)
        except ValidationError as exc:
            raise JudgeError(
                f"judge {self.name!r} ({self.version}): output failed verdict "
                f"validation: {exc}"
            ) from exc
        return JudgeRating(
            judge_name=self.name,
            judge_version=self.version,
            rating=verdict.rating,
            critique=verdict.critique,
            metadata={
                "provider": response.provider,
                "model": response.model,
                "prompt_version": self.prompt_version,
                "schema_version": JUDGE_SCHEMA_VERSION,
            },
        )


def load_judge(name: str, llm_provider: LLMProvider, *, root: Path | None = None) -> Judge:
    """Build a ``Judge`` from its committed prompt; the version is load-bearing.

    Surfaces the loader's ``FileNotFoundError`` for a missing prompt. A prompt
    whose front matter declares no version raises ``JudgeError`` — defaulting
    it would let a prompt edit silently reuse stale cached ratings.
    """
    prompt = load_judge_prompt(name, root=root)
    if prompt.version is None:
        raise JudgeError(
            f"judge prompt {name!r} declares no version; add a '# version: vN' "
            f"front-matter line — the version keys the judge cache and reports"
        )
    return Judge(name=name, version=prompt.version, llm_provider=llm_provider, prompt=prompt)
