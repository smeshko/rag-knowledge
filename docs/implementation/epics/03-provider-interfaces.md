# Epic 3 — Provider Interfaces & Fakes

**Status**: Blocked (depends on Epic 1)

## Overview

Define the four provider interfaces (`FileStorageProvider`, `PdfTextExtractor`, `LLMProvider`, `EmbeddingProvider`) and ship Fake implementations for each. Fakes are first-class production code (not test scaffolding) and live in the same `providers/<type>/fake.py` files that real implementations will live alongside. With this epic done, every later epic can develop against deterministic Fakes without paid API calls.

## Architecture references

- [11 — Configuration and Providers](../../architecture/11-configuration-and-providers.md) — all four interfaces with conceptual signatures
- [12 — Evaluation and Testing § 2](../../architecture/12-evaluation-and-testing.md#2-provider-contract-tests) — provider contract test requirements
- [13 — Implementation Decisions, topic 9](../../architecture/13-implementation-decisions.md#9-testing-stack-and-fake-provider-scaffolding) — Fake implementations as production code

## Dependencies

- Epic 1 (scaffold)

## Out of scope

- Real implementations (Epics 4 and 5)
- Wiring providers into the DI container (handled when first consumed)

---

## Phase 3.1 — Base interfaces

**Goal**: Define abstract base classes (or Protocols) for all four provider interfaces with full type annotations and docstrings.

### What to build

- **`src/rag_recipes/providers/file_storage/base.py`** — `FileStorageProvider` abstract class per [doc 11 § 1](../../architecture/11-configuration-and-providers.md#1-filestorageprovider):
  - `put_object(key: str, data: bytes, content_type: str) -> StoredObject`
  - `get_object(key: str) -> bytes`
  - `exists(key: str) -> bool`
  - `delete_object(key: str) -> None`
  - (Async or sync — pick async for consistency with the rest of the codebase)
- **`src/rag_recipes/providers/pdf_extractor/base.py`** — `PdfTextExtractor` per [doc 11 § 2](../../architecture/11-configuration-and-providers.md#2-pdftextextractor):
  - `extract_pages(file: bytes) -> list[PdfPageText]`
  - `PdfPageText` dataclass/Pydantic model: `page_number`, `text`, `extraction_method`, `confidence` (nullable)
- **`src/rag_recipes/providers/llm/base.py`** — `LLMProvider` per [doc 11 § 3](../../architecture/11-configuration-and-providers.md#3-llmprovider):
  - `generate_structured_output(request: StructuredOutputRequest) -> StructuredOutputResponse`
  - Pydantic types for request/response: provider, model, prompt_version, schema_version, input (str), json_schema (dict), output_json, raw_text, usage
  - Distinguish technical failures (raise) from rejected output (caller decides — interface just returns)
- **`src/rag_recipes/providers/embeddings/base.py`** — `EmbeddingProvider` per [doc 11 § 4](../../architecture/11-configuration-and-providers.md#4-embeddingprovider):
  - `embed_text(text: str) -> Embedding`
  - `embed_batch(texts: list[str]) -> list[Embedding]`
  - `Embedding` Pydantic model: provider, model, dimensions, vector

All interfaces use Pydantic models for request/response payloads (not dicts).

### Acceptance criteria

- [ ] All four base modules exist with abstract methods and full type annotations
- [ ] Pydantic request/response models defined for each interface
- [ ] mypy passes in strict mode against `rag_recipes.providers.*`
- [ ] Importing any of `rag_recipes.providers.{file_storage,pdf_extractor,llm,embeddings}` succeeds without side effects

### Validation

`uv run python -c "from rag_recipes.providers.llm.base import LLMProvider; ..."` for each interface.

---

## Phase 3.2 — Fake implementations + provider contract tests

**Goal**: Ship Fake implementations and contract tests that any future real implementation must also pass.

### What to build

- **`src/rag_recipes/providers/file_storage/fake.py`** — `FakeFileStorageProvider`:
  - In-memory dict keyed by `storage_key`
  - Supports all four interface methods
  - Idempotent `put_object` (overwrite by default)
- **`src/rag_recipes/providers/pdf_extractor/fake.py`** — `FakePdfTextExtractor`:
  - Constructed with a pre-defined mapping of `content_hash → list[PdfPageText]`
  - Returns the canned page list when called with matching bytes
  - Falls back to a single placeholder page for unknown inputs (configurable)
- **`src/rag_recipes/providers/llm/fake.py`** — `FakeLLMProvider`:
  - Constructed with a mapping of `input_hash → output_json` (cache-like)
  - Records every call so tests can assert what was invoked with which prompt_version/schema_version
  - Configurable failure modes: raise technical failure, return invalid JSON, return non-conforming JSON
- **`src/rag_recipes/providers/embeddings/fake.py`** — `FakeEmbeddingProvider`:
  - Deterministic vector generation from text (e.g., hash → seed for `numpy.random.default_rng`)
  - Returns the configured `dimensions`
  - Supports `embed_batch` with consistent ordering
- **Provider contract tests** in `tests/unit/providers/`:
  - One test module per provider type
  - Tests assert the contract behavior listed in [doc 12 § 2](../../architecture/12-evaluation-and-testing.md#2-provider-contract-tests) (file storage: put/get/exists/delete + duplicate keys; LLM: returns structured output, reports provider/model, surfaces tech failures; embedding: correct dimensions, batch, empty/invalid text; PDF: returns expected page list)
  - Tests run against the Fakes; the same tests will later be reused against real implementations as opt-in smoke tests

### Acceptance criteria

- [ ] All four Fake implementations exist and implement the full interface
- [ ] Contract test modules cover the items in doc 12 § 2
- [ ] `make test-unit` passes
- [ ] Contract tests are parameterizable so they can run against multiple implementations of the same interface (foundational for Epic 5's "opt-in real provider smoke tests")
- [ ] `FakeLLMProvider` exposes a call log usable by tests

### Validation

`make test-unit` — provider contract tests pass against all four Fakes.

---

## Epic-level acceptance criteria

- [ ] All four interfaces defined with full type annotations
- [ ] All four Fake implementations exist and pass contract tests
- [ ] Contract tests structured so real implementations (Epics 4, 5) can re-run them
- [ ] mypy strict mode passes against `rag_recipes.providers.*`
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epics 4 and 5 unblocked
