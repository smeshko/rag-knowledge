# Epic 5 — LLM & Embedding Providers (with Observability)

**Status**: In progress (Epic 3 met; Phase 5.1 merged)

## Overview

Implement the real OpenAI-backed `LLMProvider` (using the OpenAI SDK's native structured-output mode via Pydantic) and `EmbeddingProvider`, both wrapped in Langfuse tracing controlled by a config flag. By the end of this epic, the system can make real LLM and embedding calls behind the same interfaces the rest of the codebase consumes, with full observability surfaced in the local Langfuse instance.

## Architecture references

- [04 — LLM-Assisted Recipe Extraction](../../architecture/04-llm-assisted-recipe-extraction.md) — strict JSON schema, prompt/schema versioning, caching by input_hash
- [11 — Configuration and Providers § 3](../../architecture/11-configuration-and-providers.md#3-llmprovider) — LLM provider interface and failure-category distinction
- [11 — Configuration and Providers § 4](../../architecture/11-configuration-and-providers.md#4-embeddingprovider) — embedding provider interface and provider/model rule
- [13 — Implementation Decisions, topics 4, 4b, 5, 13](../../architecture/13-implementation-decisions.md#4-llm-provider-for-ingestion-time-recipe-extraction) — OpenAI, native `parse()`, text-embedding-3-small, Langfuse self-hosted

## Dependencies

- Epic 3 (interfaces + contract tests + Fakes)

## Out of scope

- Recipe extraction prompt content and schema (Epic 9 — this epic just provides the LLM call mechanism)
- Caching layer (handled in Epic 9 where the input_hash semantics live)
- Anthropic/Gemini providers (deferred behind the interface)

---

## Phase 5.1 — OpenAI LLMProvider with strict JSON-schema mode

**Goal**: A real `LLMProvider` implementation that wraps the OpenAI SDK in strict JSON-schema mode and returns a `StructuredOutputResponse`. This phase ships the call mechanism only — prompt content and the `recipe.v1` schema are Epic 9.

### What to build

- **`src/rag_recipes/providers/llm/openai.py`** — `OpenAILLMProvider`:
  - Constructor `OpenAILLMProvider(api_key, *, default_model, client=None)`: takes `api_key` + `default_model` explicitly (the caller resolves them from `Settings`; the provider never reads `Settings`), and accepts an injectable `client` so unit tests supply a stub without monkeypatching. When `client is None` the default `AsyncOpenAI` is constructed with `max_retries=0`, so the provider issues exactly one external call per request and retry/backoff stays an explicit Epic-9 concern (a post-generation timeout can't trigger duplicate billable generations the audit record never sees)
  - `generate_structured_output(request)`:
    - Calls `client.chat.completions.create(...)` with `response_format={"type": "json_schema", "json_schema": {"name": <derived from schema_version>, "strict": True, "schema": request.json_schema}}` — consuming the **dict** `request.json_schema` directly (not a runtime-built model class)
    - Sends `request.input` as the **sole user message** — no caller-defined system prompt in 5.1 (prompt content is Epic 9)
    - Uses `model=request.model` (the request is authoritative, matching the `ExtractionRun` provider/model record)
    - Returns a `StructuredOutputResponse` with parsed `output_json`, `raw_text` (the model's raw string output for debugging), `usage` (`input_tokens`, `output_tokens`), and echoes provider/model. No post-parse schema validation — strict mode enforces conformance on the wire; `recipe.v1` validation is Epic 9
  - **Failure handling** per [doc 11 § Failure categories](../../architecture/11-configuration-and-providers.md#failure-categories):
    - Technical failures (`APITimeoutError`, `RateLimitError`, `APIConnectionError`, `APIStatusError`, and the `APIError` base) raise a typed `LLMTechnicalError`
    - Un-parseable / non-object / refused / truncated (`finish_reason` `length` or `content_filter`) output is *not* retried inside the provider — it returns `output_json=None` with a non-empty `parse_error` and `raw_text` preserved, letting the caller decide (this preserves the `rejected` vs `failed` distinction the validation layer needs)
- **No instructor, no PydanticAI** — per [doc 13 topic 4b](../../architecture/13-implementation-decisions.md#4b-structured-output-approach-for-the-llm-call), we explicitly use the native SDK behind our own interface
- Bind `OpenAILLMProvider` to the Epic-3 `LLMContract` (against an injected fake client), plus an opt-in real-API test under `tests/live/` behind a `live` marker, run via `just test-live`

### Acceptance criteria

- [x] `OpenAILLMProvider` implements the `LLMProvider` interface
- [x] Uses strict JSON-schema mode (verified by a unit test asserting the outgoing `response_format` carries `strict: True` and `schema == request.json_schema`)
- [x] Technical failures raise a typed exception; un-parseable / non-object / refused / truncated output surfaces without raising (5.1 adds no post-parse JSON-Schema re-validation of a clean object — on-wire conformance is strict mode's job; `recipe.v1` validation is Epic 9)
- [x] mypy passes
- [x] Opt-in real-API test (`just test-live`, i.e. `pytest tests/live -m live`) hits the real API with a tiny prompt and asserts a structured response

### Validation

`just test-unit` exercises the `LLMContract` bound to `OpenAILLMProvider` via an injected fake client; `just test-live` (`pytest tests/live -m live`) hits the real API with `OPENAI_API_KEY` set. The default suite deselects `live` tests via `addopts = "-m 'not live'"`, so real-API calls never run in `just test` even though `OPENAI_API_KEY` is a required `Settings` field.

---

## Phase 5.2 — OpenAI EmbeddingProvider

**Goal**: A real `EmbeddingProvider` implementation that calls OpenAI's embedding API and returns properly-typed `Embedding` records.

### What to build

- **`src/rag_recipes/providers/embeddings/openai.py`** — `OpenAIEmbeddingProvider`:
  - Constructor takes API key, model (default `text-embedding-3-small`), dimensions (default 1536), batch size
  - `embed_text(text)`: single embedding call, returns `Embedding(provider="openai", model=..., dimensions=1536, vector=[...])`
  - `embed_batch(texts)`:
    - Splits into batches of `batch_size` (default 100; OpenAI limit is 2048 inputs but we want small batches for retry safety)
    - Calls the API per batch, preserving input order in the output
    - Returns `list[Embedding]` aligned with input order
  - Handles empty / whitespace-only strings explicitly (OpenAI 400s on empty input, and a single empty string fails a whole batch request): short-circuits locally to a full-dimension zero vector with no API call, never sending empties to the wire — satisfying the frozen contract's "`embed_text("")` returns a full-dimension vector" requirement
  - Raises a typed `EmbeddingTechnicalError` on API failures
- Run the Epic-3 contract tests against `OpenAIEmbeddingProvider` (injected deterministic fake client, no network) plus an opt-in `live`-marked real-API test

### Acceptance criteria

- [x] `OpenAIEmbeddingProvider` implements the `EmbeddingProvider` interface
- [x] Returns vectors at exactly `1536` dimensions for the default model
- [x] Batch output order matches input order (verified by test)
- [x] Empty text handled consistently with the Fake's behavior
- [x] Opt-in smoke test hits the real API

### Validation

`just test-unit` runs the contract suite against the Fake and the injected-client `OpenAIEmbeddingProvider`; `just test-live` runs the opt-in `live`-marked real-API test (deselected from the default `just test`).

---

## Phase 5.3 — Langfuse integration in both providers

**Goal**: Each LLM and embedding call is traced in Langfuse when `langfuse_enabled=true`, with rich metadata (provider, model, prompt_version, schema_version, input source span IDs, token usage, latency) and a session ID grouping calls per ingestion run.

### What to build

- **`src/rag_recipes/providers/_observability.py`** — small helper that:
  - Initializes the Langfuse SDK from `Settings` on first use
  - Provides a context manager / decorator that wraps a provider call and emits a `generation` event with all relevant metadata
  - No-ops cleanly when `langfuse_enabled=false` (zero overhead)
- Apply the wrapper in both `OpenAILLMProvider.generate_structured_output` and `OpenAIEmbeddingProvider.embed_text` / `embed_batch`
- Provide a way to set a **session ID** (e.g., `document_id` or `extraction_run_id`) from the caller — accept it as an optional argument on the provider request types, or via contextvars (designer's choice; document the approach)
- Add the following fields to each trace:
  - For LLM: `provider`, `model`, `prompt_version`, `schema_version`, `input_source_span_ids`, `input_hash`, `output.parsed`, `output.raw`, `usage.input_tokens`, `usage.output_tokens`, `latency_ms`, `status` (success / rejected / failed)
  - For Embedding: `provider`, `model`, `dimensions`, `text_preview` (first N chars), `latency_ms`, `usage` if available, batch size
- Confirm `ExtractionRun` (doc 2) remains the canonical record — Langfuse is *added* observability, not a replacement (per [doc 13 topic 13](../../architecture/13-implementation-decisions.md#13-observability--llm-tracing))

### Acceptance criteria

- [ ] `langfuse_enabled=false` makes all provider calls fully equivalent to non-traced behavior (verified by test that disables tracing and runs contract tests unchanged)
- [ ] `langfuse_enabled=true` produces a trace per call visible in the local Langfuse UI
- [ ] Session ID propagates from caller through to the trace
- [ ] Latency, tokens, prompt_version, schema_version all visible in trace metadata
- [ ] No PII / API key leakage in traces

### Validation

With Langfuse running locally, enable tracing in `.env`, run the Epic-3 smoke tests, and confirm traces appear in the Langfuse UI at `http://localhost:3000` with all metadata fields populated.

---

## Epic-level acceptance criteria

- [x] `OpenAILLMProvider` and `OpenAIEmbeddingProvider` ship and pass all Epic-3 contract tests
- [x] Strict JSON-schema mode confirmed for the LLM provider
- [ ] Langfuse tracing toggled by config flag; no-ops cleanly when disabled
- [x] Real-API smoke tests exist and are opt-in (do not run in default `make test`)
- [x] mypy passes in strict mode against `providers/llm/` and `providers/embeddings/`
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 9 unblocked (also requires Epic 8)
