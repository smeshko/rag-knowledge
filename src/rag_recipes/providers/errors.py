"""Technical-failure exceptions raised by provider implementations (doc 11 § 3)."""

from __future__ import annotations

__all__ = [
    "EmbeddingTechnicalError",
    "FileStorageError",
    "LLMTechnicalError",
    "PdfExtractionError",
    "ProviderError",
    "RerankerTechnicalError",
]


class ProviderError(Exception):
    """Base for technical failures raised by any provider implementation."""


class FileStorageError(ProviderError):
    """A file-storage operation failed for a technical reason."""


class PdfExtractionError(ProviderError):
    """PDF text extraction failed for a technical reason."""


class LLMTechnicalError(ProviderError):
    """An LLM call failed for a technical reason (transport, timeout, provider error)."""


class EmbeddingTechnicalError(ProviderError):
    """An embedding call failed for a technical reason."""


class RerankerTechnicalError(ProviderError):
    """A reranker call failed for a technical reason (transport, timeout, provider error)."""
