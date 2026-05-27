"""Opt-in real-API smoke test for OpenAIEmbeddingProvider (marked ``live``).

Deselected by default (``addopts = "-m 'not live'"``); run via ``just test-live``.
Sends no empty strings — those are unit-tested via the local zero-vector
short-circuit, and a real empty-input call would 400.
"""

from __future__ import annotations

import pytest

from rag_recipes.providers.embeddings.openai import OpenAIEmbeddingProvider
from tests.live.conftest import LiveCredentials


def test_blank_model_env_falls_back_to_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # env_ignore_empty=True: a blank EMBEDDING_MODEL resolves to the typed
    # default rather than "" (which would 400 the live call).
    monkeypatch.setenv("EMBEDDING_MODEL", "")
    credentials = LiveCredentials(_env_file=None)
    assert credentials.embedding_model == "text-embedding-3-small"


@pytest.mark.live
async def test_openai_embeddings_embed_text_and_batch(
    live_credentials: LiveCredentials,
) -> None:
    provider = OpenAIEmbeddingProvider(
        api_key=live_credentials.openai_api_key,
        model=live_credentials.embedding_model,
        dimensions=live_credentials.embedding_dimensions,
    )

    single = await provider.embed_text("hello")
    assert single.dimensions == live_credentials.embedding_dimensions
    assert len(single.vector) == live_credentials.embedding_dimensions
    assert single.provider == "openai"
    assert single.model == live_credentials.embedding_model

    batch = await provider.embed_batch(["alpha", "beta", "gamma"])
    assert len(batch) == 3
    first = await provider.embed_text("alpha")
    assert batch[0].vector == first.vector
