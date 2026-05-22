# Architecture Notes

This folder contains the evolving architecture for the personal knowledge library / RAG system.

We will build this step by step. Each document should explain not only *what* we are designing, but also *why* we are choosing that design.

## Documents

1. [Big Picture](./01-big-picture.md) — the first mental model for the system.
2. [Core Data Model](./02-core-data-model.md) — the first version of `SourceAsset`, `Document`, `SourceSpan`, `KnowledgeItem`, `Chunk`, `ChunkEmbedding`, and `ExtractionRun`.
3. [PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md) — how cookbook PDFs become recipes, chunks, and searchable records.
4. [LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md) — using an LLM during ingestion to extract recipe `KnowledgeItem` records with confidence scores.
5. [Storage and Indexing](./05-storage-and-indexing.md) — where PDFs, metadata, source spans, knowledge items, chunks, embeddings, and search indexes live.
6. [Backend API Shape](./06-backend-api-shape.md) — the first REST API shape for uploads, status, reprocessing, generic search results, and debug details.
7. [Retrieval Behavior](./07-retrieval-behavior.md) — how search combines filters, keyword retrieval, vector retrieval, chunk-type boosts, result merging, `KnowledgeItem` results, and debug details.
8. [Query-Time Answer Layer](./08-query-time-answer-layer.md) — when and how an LLM should synthesize retrieved results into grounded answers with citations.

Implementation support docs:

11. [Configuration and Providers](./11-configuration-and-providers.md) — provider interfaces and configuration for file storage, PDF text extraction, ingestion-time LLM extraction, embeddings, and search tuning.
12. [Evaluation and Testing](./12-evaluation-and-testing.md) — how to test ingestion, extraction quality, retrieval quality, confidence scores, and regressions.

## Learning Approach

We will move slowly through the architecture in layers:

1. What problem are we solving?
2. What are the major system parts?
3. How do documents enter the system?
4. How do we normalize different file types?
5. How do we chunk content?
6. How do we store and index content?
7. How does query routing work?
8. How does retrieval work?
9. How does the answer layer work?
10. How does the frontend interact with the backend?

The goal is not to design the most complex RAG system immediately. The goal is to build a system that teaches the core ideas while still being practical.
