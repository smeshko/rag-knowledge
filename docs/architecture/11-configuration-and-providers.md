# 11 — Configuration and Providers

## Goal

This document defines how the system should talk to external or swappable services without hardcoding them everywhere.

For the current architecture focus, this covers the steps up to retrieval:

- file storage provider
- PDF text extraction provider/interface
- LLM provider for ingestion-time recipe extraction
- embedding provider
- database/search configuration

The query-time answer layer is intentionally not the focus here.

---

## First Principle: Code Should Depend on Interfaces, Not Vendors

The system should not be written as if one provider is permanent.

Bad pattern:

```text
Recipe extraction code directly calls one vendor SDK everywhere.
```

Better pattern:

```text
Recipe extraction code → LLMProvider interface → provider implementation
```

This gives us flexibility:

- start local-first
- host later
- switch LLM providers
- switch embedding models
- switch file storage from local disk to S3/R2
- test with mocks instead of real paid APIs

---

## Provider Boundaries

The first implementation should have these provider boundaries:

```text
FileStorageProvider
PdfTextExtractor
LLMProvider
EmbeddingProvider
```

Optional later:

```text
OCRProvider
RerankerProvider
AnswerLLMProvider
```

For now, `AnswerLLMProvider` is out of scope because we are focusing on the system up to retrieval and before query-time answer generation.

---

## 1. FileStorageProvider

The file storage provider stores original uploaded PDFs.

The data model stores:

```text
storage_provider
storage_key
```

as defined in [02 — Core Data Model](./02-core-data-model.md#storage-fields).

### Initial implementations

```text
LocalFileStorage
S3LikeFileStorage later
```

`S3LikeFileStorage` can cover AWS S3, Cloudflare R2, Supabase Storage, or similar object storage.

### Conceptual interface

```text
putObject(key, bytes, contentType) → StoredObject
getObject(key) → bytes/stream
exists(key) → boolean
deleteObject(key) → void
```

For hosted usage, we may also want:

```text
createSignedReadUrl(key, expiresIn) → url
```

But the MVP backend does not need to expose raw PDF files publicly.

### Configuration

Example config keys:

```text
FILE_STORAGE_PROVIDER=local
LOCAL_STORAGE_ROOT=./data/storage

# later
FILE_STORAGE_PROVIDER=s3
S3_BUCKET=rag-recipes
S3_REGION=...
S3_ENDPOINT=...
S3_ACCESS_KEY_ID=...
S3_SECRET_ACCESS_KEY=...
```

### Important rule

Application code should not manually build local paths or S3 URLs.

It should ask the `FileStorageProvider` for storage operations.

---

## 2. PdfTextExtractor

The PDF text extractor converts a PDF file into page-level text.

For the MVP, PDFs are mostly selectable text, so the first extractor can be direct text extraction.

### Conceptual interface

```text
extractPages(file) → PdfPageText[]
```

Where:

```json
{
  "page_number": 42,
  "text": "Tomato and White Bean Soup...",
  "extraction_method": "embedded_text",
  "confidence": null
}
```

### Why this should be an interface

Later we may add:

- OCR fallback
- layout-aware extraction
- table extraction
- scanned-page detection

The ingestion pipeline should not care which extractor produced the page text, as long as it can create `SourceSpan` records.

### Configuration

Example:

```text
PDF_TEXT_EXTRACTOR=embedded_text
PDF_MIN_TEXT_CHARS_FOR_PAGE=20
```

If a page has very little extracted text, we can mark it as suspicious for future OCR support.

---

## 3. LLMProvider

The LLM provider is used during ingestion-time extraction:

```text
PDF SourceSpans → recipe KnowledgeItem candidates
```

This is covered in [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md).

### Conceptual interface

```text
generateStructuredOutput(request) → StructuredOutputResponse
```

Request shape conceptually includes:

```json
{
  "provider": "openai",
  "model": "example-model",
  "prompt_version": "recipe-extraction-v1",
  "schema_version": "recipe.v1",
  "input": "...source span window...",
  "json_schema": {}
}
```

Response shape conceptually includes:

```json
{
  "output_json": {},
  "raw_text": "...",
  "usage": {
    "input_tokens": 1200,
    "output_tokens": 900
  },
  "provider": "openai",
  "model": "example-model"
}
```

### Why return raw text too?

When structured parsing fails, the raw model output is useful for debugging.

It can be stored on `ExtractionRun.output_json` or adjacent debug fields, depending on implementation.

### Configuration

Example:

```text
LLM_PROVIDER=openai
LLM_MODEL=example-model
LLM_API_KEY=...

RECIPE_EXTRACTION_PROMPT_VERSION=recipe-extraction-v1
RECIPE_EXTRACTION_SCHEMA_VERSION=recipe.v1
```

### Cache key

Extraction caching should include:

```text
input_hash
provider
model
prompt_version
schema_version
```

This matches the rule in [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#risks-and-mitigations).

### Failure categories

Provider implementations should distinguish:

```text
technical failure → ExtractionRun.status = failed
hard validation rejection → ExtractionRun.status = rejected
```

The provider interface reports technical failures. The extraction/validation layer decides whether an output is rejected.

---

## 4. EmbeddingProvider

The embedding provider creates vectors for chunks and queries.

It is used in two places:

```text
chunk text → chunk embedding
query text → query embedding
```

The same provider/model pair must be used when comparing vectors.

This follows the `ChunkEmbedding` model in [02 — Core Data Model](./02-core-data-model.md#6-chunkembedding).

### Conceptual interface

```text
embedText(text) → Embedding
embedBatch(texts) → Embedding[]
```

Embedding response:

```json
{
  "provider": "openai",
  "model": "text-embedding-example",
  "dimensions": 1536,
  "vector": [0.01, -0.02]
}
```

### Configuration

Example:

```text
EMBEDDING_PROVIDER=openai
EMBEDDING_MODEL=text-embedding-example
EMBEDDING_DIMENSIONS=1536
```

### Important rule

Never compare embeddings from different models as if they live in the same vector space.

Vector search must filter by:

```text
embedding_provider
embedding_model
```

---

## 5. Database and Search Configuration

The first implementation uses:

```text
PostgreSQL + pgvector
```

Configuration examples:

```text
DATABASE_URL=postgres://...
PGVECTOR_ENABLED=true
```

Search-related tuning constants can start in application config:

```text
SEARCH_DEFAULT_CATEGORY=recipes
SEARCH_DEFAULT_MODE=hybrid
SEARCH_DEFAULT_LIMIT=10
SEARCH_KEYWORD_TOP_K=50
SEARCH_VECTOR_TOP_K=50
SEARCH_RRF_K=60
```

Chunk-type boosts from [07 — Retrieval Behavior](./07-retrieval-behavior.md) can also be config values.

Example:

```text
RECIPE_KEYWORD_BOOST_TITLE=1.40
RECIPE_KEYWORD_BOOST_INGREDIENTS=1.20
RECIPE_VECTOR_BOOST_SUMMARY=1.20
```

They can start as code constants, but they should be easy to tune.

---

## 6. Prompt and Schema Versioning

Prompt and schema versions are part of the system's reproducibility story.

Important version fields:

```text
prompt_version
schema_version
embedding_model
embedding_provider
source_version
```

Why this matters:

- prompts change extraction behavior
- schemas change output shape
- embedding models change vector space
- source versions change extracted text snapshots

If any of these change, results may change.

That is expected, but the system should make it visible and debuggable.

---

## 7. Secrets and Environment Separation

Configuration has two kinds of values:

### Non-secret config

Examples:

```text
SEARCH_DEFAULT_LIMIT=10
RECIPE_EXTRACTION_PROMPT_VERSION=recipe-extraction-v1
FILE_STORAGE_PROVIDER=local
```

These can be checked into example config files.

### Secret config

Examples:

```text
LLM_API_KEY=...
S3_SECRET_ACCESS_KEY=...
PERSONAL_API_TOKEN=...
```

Secrets should not be committed to source control.

Use environment variables or a secret manager when hosted.

---

## 8. Local vs Hosted Configuration

### Local development

Likely config:

```text
FILE_STORAGE_PROVIDER=local
LOCAL_STORAGE_ROOT=./data/storage
DATABASE_URL=postgres://localhost/...
LLM_PROVIDER=...
EMBEDDING_PROVIDER=...
DEBUG_ENDPOINTS_ENABLED=true
```

### Hosted personal deployment

Likely config:

```text
FILE_STORAGE_PROVIDER=s3_like
DATABASE_URL=hosted-postgres-url
PERSONAL_API_TOKEN=...
DEBUG_ENDPOINTS_ENABLED=false
```

Debug endpoints and debug search output should remain local/development-only unless we explicitly revisit that decision.

---

## 9. Testing Providers

Every provider interface should have a fake or mock implementation.

Examples:

```text
FakeFileStorageProvider
FakeLLMProvider
FakeEmbeddingProvider
```

Why?

- tests should not always call paid APIs
- tests should be deterministic
- failure modes can be simulated
- regression tests can use stored fixtures

For example, `FakeLLMProvider` can return a known recipe-extraction JSON payload for a known page-window input.

---

## Initial Provider Decisions

Already decided:

1. Use a file storage interface from the beginning.
2. First file storage implementation is local filesystem.
3. Use an LLM provider interface.
4. Use PostgreSQL + pgvector for storage/search.
5. Use explicit provider/model/version fields for extraction and embeddings.

Additional decisions:

1. Use a cloud LLM provider first for ingestion-time recipe extraction.
2. Use cloud embeddings first.
3. Decide the exact provider/model names during implementation.
4. Decide whether configuration starts as environment-only or typed config plus environment secrets during implementation.
5. Choose the PDF extraction library during implementation or spike work.

---

## What This Teaches

This step teaches that provider choices should be replaceable architecture decisions, not assumptions buried inside code.

Good provider design makes it easier to:

- learn with one setup
- host later
- test without paid calls
- compare providers
- recover from provider changes
- keep extraction/retrieval reproducible

---

## Next Step

The next architecture document should cover evaluation and testing:

> How do we test extraction quality, retrieval quality, provider behavior, confidence scores, and regressions?
