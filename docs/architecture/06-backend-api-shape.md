# 06 — Backend API Shape

## Goal

The backend API is the stable boundary between the knowledge system and any frontend.

Future clients may include:

- web app
- iOS app
- command-line tools
- admin/debug UI

All of them should talk to the same backend.

This document defines the first API shape for:

- uploading PDFs
- checking ingestion status
- reprocessing documents
- searching recipes
- returning generic search results
- returning raw retrieval/debug details

---

## First Principle: Backend Owns the RAG System

The frontend should not know how ingestion, chunking, embeddings, or retrieval work internally.

The frontend should ask for product-level operations:

```text
Upload this cookbook.
Show me ingestion status.
Search my recipe library.
Show this search result.
Show debug retrieval details if requested.
```

The backend owns:

- file storage
- source assets
- documents
- source spans
- extraction runs
- knowledge items
- chunks
- embeddings
- search indexes
- LLM provider calls

This keeps web, iOS, and future clients simpler.

---

## API Style

For the first version, use a REST-style HTTP API.

Suggested prefix:

```text
/api/v1
```

Why REST first?

- easy to learn
- easy to test with curl/Postman/HTTP clients
- works well for web and iOS
- maps cleanly to resources like documents, search, and extraction runs

GraphQL, tRPC, or realtime APIs can come later if they become useful.

---

## Resource Model

The API should expose a small set of concepts:

```text
Document
IngestionStatus
KnowledgeItem
KnowledgeItemResult
SearchResult
DisplayProjection
StructuredPreview
RetrievalDebugInfo
ExtractionRun
SourceSpan
```

Important distinction:

- `KnowledgeItem` is the generic canonical backend object.
- `structured_data` is the canonical type-specific payload inside a `KnowledgeItem`.
- `KnowledgeItemResult` is a generic search-result envelope.
- `DisplayProjection` is a small derived display helper for list UIs.
- `StructuredPreview` is an optional small derived preview for known item types.

This avoids creating a required concrete API object for every future `item_type`. Unknown future item types can still return a generic result with title, summary, matched chunks, and citations.

---

## MVP Endpoint Summary

```text
GET  /api/v1/health

POST /api/v1/documents
GET  /api/v1/documents
GET  /api/v1/documents/{document_id}
GET  /api/v1/documents/{document_id}/status
POST /api/v1/documents/{document_id}/reprocess

POST /api/v1/search
GET  /api/v1/knowledge-items/{item_id}

GET  /api/v1/documents/{document_id}/extraction-runs
GET  /api/v1/extraction-runs/{run_id}
GET  /api/v1/documents/{document_id}/source-spans
```

The last three are mostly for debugging and learning.

---

## 1. Health Check

### `GET /api/v1/health`

Checks whether the backend is alive.

Example response:

```json
{
  "status": "ok"
}
```

Later this can include database and worker status.

---

## 2. Upload / Create Document

### `POST /api/v1/documents`

Uploads one PDF and creates:

```text
SourceAsset + Document(status=queued)
```

The ingestion job starts asynchronously.

### Request

Use multipart form data:

```text
file: cookbook.pdf
category: recipes
subcategory: null
title: Simple Thai Food
author: Leela Punyaratabandhu
language: en
```

For the MVP:

- `file` is required
- `category` defaults to `recipes`
- `subcategory` is optional
- `title` is optional but strongly recommended
- `author` is optional
- `language` is optional

### Response

```json
{
  "document": {
    "id": "doc_123",
    "asset_id": "asset_123",
    "category": "recipes",
    "subcategory": null,
    "title": "Simple Thai Food",
    "author": "Leela Punyaratabandhu",
    "source_type": "pdf",
    "language": "en",
    "active_source_version": null,
    "status": "queued",
    "created_at": "2026-05-21T00:00:00Z",
    "updated_at": "2026-05-21T00:00:00Z"
  },
  "ingestion": {
    "status": "queued"
  }
}
```

### Duplicate upload behavior

Because `source_assets.content_hash` is unique, the backend can detect duplicate files.

Recommended MVP behavior:

```text
If content_hash already exists, return the existing Document instead of creating a duplicate.
```

Later we can add an explicit `force_duplicate` or `reprocess` option if needed.

---

## 3. List Documents

### `GET /api/v1/documents`

Returns documents with optional filters.

Example query params:

```text
category=recipes
status=ready
source_type=pdf
```

Example response:

```json
{
  "documents": [
    {
      "id": "doc_123",
      "category": "recipes",
      "subcategory": null,
      "title": "Simple Thai Food",
      "author": "Leela Punyaratabandhu",
      "source_type": "pdf",
      "status": "ready",
      "active_source_version": 1
    }
  ]
}
```

---

## 4. Get Document

### `GET /api/v1/documents/{document_id}`

Returns document metadata and high-level ingestion information.

Example response:

```json
{
  "document": {
    "id": "doc_123",
    "asset_id": "asset_123",
    "category": "recipes",
    "subcategory": null,
    "title": "Simple Thai Food",
    "author": "Leela Punyaratabandhu",
    "source_type": "pdf",
    "language": "en",
    "active_source_version": 1,
    "status": "ready",
    "created_at": "2026-05-21T00:00:00Z",
    "updated_at": "2026-05-21T00:10:00Z"
  },
  "counts": {
    "source_spans": 240,
    "knowledge_items": 118,
    "ready_items": 115,
    "needs_review_items": 3,
    "chunks": 575
  }
}
```

---

## 5. Get Ingestion Status

### `GET /api/v1/documents/{document_id}/status`

This is the endpoint frontends poll after upload.

Example response:

```json
{
  "document_id": "doc_123",
  "status": "extracting_items",
  "active_source_version": null,
  "current_source_version": 1,
  "progress": {
    "stage": "extracting_items",
    "message": "Extracting recipes from page windows",
    "pages_total": 240,
    "pages_processed": 72
  },
  "terminal": false
}
```

Terminal statuses:

```text
ready
needs_review
failed
```

Important:

- `active_source_version` points to the accepted version.
- `current_source_version` may point to an in-progress version.
- During reprocessing, `active_source_version` can remain on the old ready version until the new version is accepted.

---

## 6. Reprocess Document

### `POST /api/v1/documents/{document_id}/reprocess`

Starts a new ingestion attempt for an existing document.

This can be used after:

- `failed`
- `needs_review`
- `ready`, when manually reprocessing with a better prompt/model/parser

### Request

```json
{
  "mode": "auto",
  "reason": "try improved recipe extraction prompt"
}
```

Initial modes:

```text
auto
reuse_source_spans
new_source_version
```

### Mode meaning

#### `auto`

Backend chooses based on what changed.

- downstream failure only → reuse source spans
- extraction/OCR/parser changed → new source version

#### `reuse_source_spans`

Reuse completed source spans and rerun downstream extraction/chunking/embedding steps.

Useful after:

- LLM prompt change
- extraction failed after source spans were created
- embedding/indexing failure

#### `new_source_version`

Re-extract PDF text and create a new source version.

Useful after:

- PDF text extractor changed
- OCR was added
- previous source text was bad

### Response

```json
{
  "document_id": "doc_123",
  "status": "queued",
  "previous_active_source_version": 1,
  "current_source_version": 2
}
```

---

## 7. Search

### `POST /api/v1/search`

Searches the knowledge library.

For the MVP, this is recipe-focused but still shaped in a way that can expand later.

### Request

```json
{
  "query": "cozy soup with white beans",
  "category": "recipes",
  "subcategory": null,
  "filters": {
    "item_type": "recipe",
    "document_ids": [],
    "exclude_needs_review": true
  },
  "limit": 10,
  "include_debug": true,
  "mode": "hybrid"
}
```

`mode` can be `hybrid`, `keyword`, or `vector`; the default is `hybrid`.

### Response

The response includes product-friendly generic results and, in development/local mode only, raw debug details.

```json
{
  "query": "cozy soup with white beans",
  "results": [
    {
      "type": "knowledge_item_result",
      "item": {
        "id": "item_123",
        "item_type": "recipe",
        "schema": "recipe.v1",
        "title": "Tomato and White Bean Soup",
        "summary": "A simple soup with pantry ingredients.",
        "status": "ready",
        "confidence": {
          "overall": 0.88
        }
      },
      "display": {
        "title": "Tomato and White Bean Soup",
        "subtitle": "Simple Thai Food · page 42",
        "snippet": "White beans, tomato, olive oil...",
        "badges": ["Serves 4", "35 minutes"]
      },
      "structured_preview": {
        "schema": "recipe.preview.v1",
        "yield": "Serves 4",
        "top_ingredients": ["white beans", "tomato", "olive oil"]
      },
      "document": {
        "id": "doc_123",
        "title": "Simple Thai Food",
        "author": "Leela Punyaratabandhu"
      },
      "matched_chunks": [
        {
          "chunk_id": "chunk_123",
          "chunk_type": "recipe_ingredients",
          "score": 0.82
        }
      ],
      "source_citations": [
        {
          "source_span_id": "span_042",
          "label": "page 42",
          "locator": {
            "type": "pdf_page_range",
            "page_start": 42,
            "page_end": 42
          }
        }
      ]
    }
  ],
  "debug": {
    "retrieval_mode": "hybrid",
    "keyword_candidates": [],
    "vector_candidates": [],
    "merged_candidates": [],
    "rerank_applied": false
  }
}
```

`structured_preview` is derived from `KnowledgeItem.structured_data`; it does not replace canonical structured data.

### User-facing filtering rule

Search should only include chunks whose parent knowledge item is:

```text
status = ready
```

and not superseded.

`needs_review` and `superseded` items are excluded by default.

---

## 8. Get Knowledge Item

### `GET /api/v1/knowledge-items/{item_id}`

Returns a single knowledge item.

This endpoint returns the canonical `KnowledgeItem`, including full `structured_data`.

A client can use `structured_data.schema` to render type-specific details. The backend may also include a small derived `display` helper, but `structured_data` remains the source of truth.

Example response:

```json
{
  "knowledge_item": {
    "id": "item_123",
    "document_id": "doc_123",
    "item_type": "recipe",
    "title": "Tomato and White Bean Soup",
    "summary": "A simple soup with pantry ingredients.",
    "status": "ready",
    "source_span_ids": ["span_042", "span_043"],
    "confidence": {
      "overall": 0.88
    },
    "structured_data": {
      "schema": "recipe.v1",
      "yield": "Serves 4",
      "prep_time": null,
      "cook_time": "35 minutes",
      "total_time": null,
      "ingredients": [
        {
          "position": 1,
          "raw_text": "2 tbsp olive oil",
          "quantity_value": 2,
          "unit_normalized": "tablespoon",
          "item_normalized": "olive oil"
        }
      ],
      "steps": [
        {
          "step_number": 1,
          "text": "Heat the oil in a large pot."
        }
      ]
    }
  },
  "display": {
    "title": "Tomato and White Bean Soup",
    "subtitle": "Simple Thai Food · page 42"
  },
  "source_citations": [
    {
      "source_span_id": "span_042",
      "label": "page 42"
    }
  ]
}
```

---

## 9. Debug Endpoints

These endpoints are useful while learning and developing.

MVP rule:

> Implement debug endpoints, but enable them only in development/local mode or behind an explicit dev-only flag.

They should not be exposed publicly in a hosted environment.

### `GET /api/v1/documents/{document_id}/extraction-runs`

Lists extraction runs for a document.

### `GET /api/v1/extraction-runs/{run_id}`

Returns one extraction run, including validation status and optionally raw model output.

### `GET /api/v1/documents/{document_id}/source-spans`

Returns source spans for a document.

For PDFs, these are page-level spans.

Example query params:

```text
source_version=1
page_start=40
page_end=45
```

Important:

> Source span text can contain copyrighted source content. Keep this endpoint private/debug-only.

---

## Error Shape

Use one consistent error response.

Example:

```json
{
  "error": {
    "code": "document_not_found",
    "message": "Document not found",
    "details": {}
  }
}
```

Useful initial error codes:

```text
invalid_request
unsupported_file_type
duplicate_source_asset
document_not_found
ingestion_already_running
search_failed
internal_error
```

---

## Authentication Note

We are designing single-user first, but not local-only.

Before hosting, the API should have at least simple protection.

MVP decision:

```text
personal API token
```

Every hosted API request should include the personal token. Full multi-user auth can come later.

---

## What This Teaches

This step teaches an important product architecture idea:

> RAG internals should not leak into every frontend.

The frontend should not need to know about pgvector, source spans, extraction runs, chunk embeddings, or LLM provider details unless it is showing a debug view.

The backend turns those internals into stable product responses like:

- document status
- generic search results
- knowledge item details
- citations
- raw debug information when requested

---

## Resolved API Choices

For the MVP:

1. Use REST under `/api/v1`.
2. Duplicate PDF uploads return the existing document.
3. Hosted access uses a personal API token.
4. Debug endpoints can exist, but only behind a development/local mode or explicit dev-only flag.
5. Search debug output is also development/local only.

## Next Step

The next architecture document covers retrieval behavior:

> How does `POST /api/v1/search` combine metadata filters, keyword search, vector search, chunk types, and `KnowledgeItem` results?

See [07 — Retrieval Behavior](./07-retrieval-behavior.md).
