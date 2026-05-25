"""Smoke tests: provider interfaces import clean, are non-instantiable ABCs."""

from __future__ import annotations

import importlib

import pytest

from rag_recipes.providers.embeddings.types import Embedding
from rag_recipes.providers.file_storage.types import StoredObject
from rag_recipes.providers.llm.types import (
    StructuredOutputRequest,
    StructuredOutputResponse,
    TokenUsage,
)
from rag_recipes.providers.pdf_extractor.types import PdfPageText

_INTERFACES = [
    (
        "rag_recipes.providers.file_storage.base",
        "FileStorageProvider",
        frozenset({"put_object", "get_object", "exists", "delete_object"}),
    ),
    (
        "rag_recipes.providers.pdf_extractor.base",
        "PdfTextExtractor",
        frozenset({"extract_pages"}),
    ),
    (
        "rag_recipes.providers.llm.base",
        "LLMProvider",
        frozenset({"generate_structured_output"}),
    ),
    (
        "rag_recipes.providers.embeddings.base",
        "EmbeddingProvider",
        frozenset({"embed_text", "embed_batch"}),
    ),
]


@pytest.mark.parametrize(("module_path", "class_name", "expected"), _INTERFACES)
def test_interface_imports_clean_and_is_abstract(
    module_path: str, class_name: str, expected: frozenset[str]
) -> None:
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    assert cls.__abstractmethods__ == expected
    with pytest.raises(TypeError):
        cls()


def test_stored_object_validates() -> None:
    StoredObject(
        storage_provider="local",
        storage_key="k",
        content_type="application/pdf",
        size_bytes=10,
    )


def test_pdf_page_text_validates_with_and_without_confidence() -> None:
    without = PdfPageText(page_number=1, text="x", extraction_method="embedded_text")
    assert without.confidence is None
    PdfPageText(page_number=2, text="y", extraction_method="ocr", confidence=0.9)


def test_llm_types_validate() -> None:
    StructuredOutputRequest(
        provider="openai",
        model="gpt-4.1",
        prompt_version="recipe-extraction-v1",
        schema_version="recipe.v1",
        input="...",
        json_schema={},
    )
    StructuredOutputResponse(
        output_json={},
        raw_text="",
        usage=TokenUsage(input_tokens=1, output_tokens=2),
        provider="openai",
        model="gpt-4.1",
    )


def test_embedding_validates() -> None:
    Embedding(
        provider="openai",
        model="text-embedding-3-small",
        dimensions=1536,
        vector=[0.1, -0.2],
    )
