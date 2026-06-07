"""Credential helper for opt-in live tests.

Resolves the OpenAI key + model from the environment **or** ``.env`` without
building the full ``Settings`` — which also requires ``database_url`` and
``redis_url`` (both unrelated to these tests and unset in a bare checkout), so a
full ``Settings()`` would raise ``ValidationError`` before the live test could
even skip. Reading the same ``.env`` keeps a ``.env``-only key resolvable while
staying decoupled from DB/Redis.

The ``live`` marker plus ``addopts = "-m 'not live'"`` is what keeps these
real-API calls out of the default suite (``OPENAI_API_KEY`` is a required
``Settings`` field and thus essentially always present locally); the key-presence
skip below is only a secondary net for the opt-in run.
"""

from __future__ import annotations

import pytest
from pydantic_settings import BaseSettings, SettingsConfigDict


class LiveCredentials(BaseSettings):
    # ``env_ignore_empty=True`` so a blank ``EMBEDDING_MODEL=`` / ``LLM_MODEL=``
    # in ``.env`` falls back to the typed default instead of resolving to ``""``
    # (which would 400 the live call with ``model=""``).
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        env_ignore_empty=True,
    )

    openai_api_key: str | None = None
    llm_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    # Anthropic live creds (Epic 19.1). A blank ``ANTHROPIC_LLM_MODEL=`` in
    # ``.env`` falls back to the Sonnet default via ``env_ignore_empty=True``.
    anthropic_api_key: str | None = None
    anthropic_llm_model: str = "claude-sonnet-4-6"


@pytest.fixture
def live_credentials() -> LiveCredentials:
    credentials = LiveCredentials()
    if not credentials.openai_api_key:
        pytest.skip(
            "OPENAI_API_KEY not set (env or .env) — set it to run the live OpenAI test"
        )
    return credentials


@pytest.fixture
def live_anthropic_credentials() -> LiveCredentials:
    # Keyed on the Anthropic key so the OpenAI-key skip does not gate the
    # Anthropic live test (and vice-versa).
    credentials = LiveCredentials()
    if not credentials.anthropic_api_key:
        pytest.skip(
            "ANTHROPIC_API_KEY not set (env or .env) — set it to run the live "
            "Anthropic test"
        )
    return credentials
