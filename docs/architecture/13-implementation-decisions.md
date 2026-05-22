# 13 — Implementation Decisions

This document records the concrete implementation-level decisions made on top of the architecture notes in this folder. The architecture docs describe *what* we are building; this doc records *what we picked to build it with* and *why*.

Each decision lists the options considered, the choice, and the reasoning. If a choice is later replaced, leave the old entry in place and add a follow-up entry rather than rewriting history.

---

## 1. Backend language and web framework

**Decision**: Python + FastAPI (with Pydantic; ORM/migration tooling decided separately).

### Options considered

- **Python + FastAPI** — strongest RAG/LLM/PDF ecosystem, Pydantic matches the strict JSON schemas in [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md), async-native, auto OpenAPI.
- **Node/TypeScript + Fastify/Hono** — end-to-end TS types with a future frontend, but PDF extraction libraries are noticeably weaker and the RAG ecosystem is second to Python's.
- **Go + chi/gin** — fast and simple to deploy, but weak AI/PDF ecosystems mean more glue code and slower iteration on prompts and evaluations.
- **Rust + axum** — performance and safety, but the steepest learning curve and the weakest RAG ecosystem of the four.

### Why Python + FastAPI

For a personal RAG library, the interesting work is prompts, schemas, retrieval scoring, and evaluation — not raw request throughput. Python optimizes for that work:

- Every LLM and embedding provider has a first-class Python SDK.
- PDF extraction libraries (`pypdf`, `pdfplumber`, `pymupdf`) lead the space, which matters for the PDF-first MVP in [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md).
- Pydantic models map cleanly to the `recipe.v1` structured data, confidence shapes, and validation rules in [02 — Core Data Model](./02-core-data-model.md) and [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#hard-vs-soft-validation).
- FastAPI is async-native (good fit for the async ingestion model in [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md#sync-vs-async-ingestion)) and produces an OpenAPI schema automatically — useful for the future web/iOS clients described in [06 — Backend API Shape](./06-backend-api-shape.md).

### Why not the alternatives

- **Node/TS** would be the better pick if the priority were a shared-type web frontend shipped quickly; here, weaker PDF tooling outweighs the type-sharing benefit.
- **Go** would be the better pick for a high-throughput ingestion service; this project is single-user.
- **Rust** would be the better pick for performance-critical infra; the iteration speed cost is not justified for a learning-focused project.

---

## 2. Database client and migration tooling

**Decision**: SQLAlchemy 2.0 (async) + Alembic + `pgvector-python`.

### Options considered

- **SQLAlchemy 2.0 + Alembic** — mature async ORM with first-class pgvector support and clear separation between SQLAlchemy DB models and Pydantic API models.
- **SQLModel + Alembic** — one class doubles as DB + API model; less duplication but conflates two layers the architecture deliberately separates.
- **Raw SQL via `asyncpg` + Alembic** — maximum control, but lots of hand-written joins and JSON wiring for a deeply relational FK chain.
- **`timescale-vector` style (raw SQL + helper library)** — the pattern shown in several external RAG transcripts. Simple for vectors but weak for our relational schema with deep FK chains and JSON columns.

### Why SQLAlchemy 2.0 + Alembic

The data model in [02 — Core Data Model](./02-core-data-model.md) is exactly the shape SQLAlchemy was built for: a deep FK chain (`SourceAsset → Document → SourceSpan → KnowledgeItem → Chunk → ChunkEmbedding`), many JSON columns (`structured_data`, `confidence`, `locator`), intentional denormalizations that need to stay in sync, and careful uniqueness constraints. SQLAlchemy 2.0's typed style makes the relationships and JSON columns ergonomic.

Keeping SQLAlchemy models and Pydantic API models as separate layers matches the canonical-vs-derived distinction in [06 — Backend API Shape](./06-backend-api-shape.md#resource-model) and [07 — Retrieval Behavior](./07-retrieval-behavior.md#11-search-result-construction): the DB stores full `structured_data`; the API returns smaller `display` and `structured_preview` projections.

Alembic gives us proper migrations from day one. The schema will churn — extraction prompts, chunk types, confidence fields, and source-version semantics will all evolve as the system is tuned.

### Why not the alternatives

- **SQLModel** is tempting for its brevity, but the docs intentionally separate DB and API shapes (canonical `structured_data` vs derived `structured_preview`). Merging them creates friction the moment they diverge.
- **Raw SQL** would mean hand-writing every join across the FK chain and manually mapping JSON columns to Pydantic. Worth it only if we had very few tables or unusual SQL needs — we have neither.
- **`timescale-vector` style** assumes a flat `(id, metadata, content, embedding)` shape. Our schema is relational and structured; we'd be fighting that helper from day one.

---

## 3. PDF text extraction library

**Decision**: PyMuPDF (`pymupdf` / `fitz`), behind the `PdfTextExtractor` interface from [11 — Configuration and Providers](./11-configuration-and-providers.md#2-pdftextextractor).

### Options considered

- **PyMuPDF** — fastest and highest-quality text extraction on complex layouts; AGPL-3.0 license.
- **pdfplumber** — MIT-licensed, layout-aware, ~5–10× slower than PyMuPDF.
- **pypdf** — minimal dependency, BSD-licensed, but often garbles multi-column cookbook layouts.
- **Docling** — multi-format extractor with built-in hybrid chunking; heavyweight ML dependency whose main value (chunking) doesn't apply to our structured-extraction approach.

### Why PyMuPDF

Per [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md#3-extract-pdf-text), the extractor produces clean per-page text — chunking and structure extraction happen later (the LLM works from raw page text). PyMuPDF is the best fit for that narrow job:

- Fastest by a wide margin, with the best text quality on the multi-column, sidebar-heavy layouts common in cookbooks.
- Page-level API maps directly to the one-span-per-page convention in [02 — Core Data Model](./02-core-data-model.md#3-sourcespan).
- Has OCR hooks (`get_textpage_ocr`, tesseract-backed) that align with the future OCR provider noted in [11 — Configuration and Providers](./11-configuration-and-providers.md#2-pdftextextractor).

### License caveat

PyMuPDF is AGPL-3.0. For the personal-first/self-hosted scope set in [01 — Big Picture](./01-big-picture.md#starting-decisions), this is acceptable. If the project is ever open-sourced publicly or commercialized, swap to `pdfplumber` behind the `PdfTextExtractor` interface — the interface boundary keeps that swap localized.

### Why not the alternatives

- **pdfplumber** is the right pick if license freedom is required upfront; the speed cost is acceptable at single-user volumes.
- **pypdf** is unsuitable for cookbook-style layouts.
- **Docling**'s strengths (multi-format ingestion + hybrid chunking) don't apply: we chunk from `KnowledgeItem` records, not from raw extracted text, and different future source types (videos, transcripts, images) need different `SourceSpan` locator types anyway, so a unified extractor doesn't unify much.

---

## 4. LLM provider for ingestion-time recipe extraction

**Decision**: OpenAI as the first provider. Start with `gpt-4.1` (or `gpt-5` when iterating on extraction quality); the exact model is a tuning constant, not a permanent commitment.

### Options considered

- **OpenAI** — strict JSON-schema mode (`response_format: { type: "json_schema", strict: true }`) guarantees schema-compliant output; most mature Python SDK.
- **Anthropic Claude** — excellent at nuanced extraction and confidence calibration; JSON via tool-use, more verbose than strict mode.
- **Google Gemini** — cheapest (Flash) and longest context, but SDK and schema-strictness lag OpenAI.
- **Open Router / local models** — explicitly deferred by [11 — Configuration and Providers](./11-configuration-and-providers.md#initial-provider-decisions); the `LLMProvider` interface keeps the door open.

### Why OpenAI first

Strict JSON-schema mode is the single feature that most reduces validation failures and `needs_review` items from [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#hard-vs-soft-validation), because the model is constrained to produce schema-compliant output at generation time. The mature Python SDK and ecosystem assume OpenAI as the reference implementation, which simplifies everything downstream.

Anthropic remains the natural A/B-test target later when we have golden fixtures and can compare extraction quality. Gemini is the right pick if cost ever becomes the binding constraint; for a single-user library it doesn't.

The architecture's `LLMProvider` interface ([11 — Configuration and Providers](./11-configuration-and-providers.md#3-llmprovider)) makes the choice reversible — each `ExtractionRun` records provider/model/prompt_version/schema_version, so we can compare runs across providers without breaking the data model.

---

## 4b. Structured-output approach for the LLM call

**Decision**: OpenAI native structured outputs (`client.beta.chat.completions.parse()` with a Pydantic `recipe.v1` model), wrapped behind our own `LLMProvider` interface.

### Options considered

- **OpenAI native + custom `LLMProvider` interface** — minimal dependencies, strict JSON-schema mode used at the source, explicit retry control.
- **Instructor** (multi-provider Pydantic wrapper) — popular in the broader RAG community; auto-retries on validation failure.
- **PydanticAI** (full agent framework) — Pydantic-native but agent-shaped; we have no agents in scope.
- **Raw SDK + manual JSON parsing** — re-implements what the SDK's `parse()` helper already does.

### Why native + our own interface

Two specific reasons this beats Instructor for *our* problem:

1. **Validation outcomes are first-class audit data, not failures to retry away.** [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#hard-vs-soft-validation) and [02 — Core Data Model](./02-core-data-model.md#7-extractionrun) treat schema rejection, `needs_review` items, and `rejected` extraction runs as outcomes we record. Instructor's auto-retry would mask exactly the signals we want to keep.
2. **Provider abstraction is already our `LLMProvider` interface.** Adding Instructor on top duplicates that responsibility. The native SDK + our interface is a thinner, more honest layering.

Instructor remains a strong general-purpose pick — it's just solving a different problem (transparent multi-provider JSON extraction) than ours (audited single-provider extraction with explicit validation semantics).

---

## 5. Embedding provider and model

**Decision**: OpenAI `text-embedding-3-small` at the default 1536 dimensions.

### Options considered

- **OpenAI `text-embedding-3-small` @ 1536 dims** — industry default, cheap, same-vendor as the LLM, good quality for short recipe text.
- **OpenAI `text-embedding-3-large` @ 3072 dims** — higher quality, ~6× cost, 2× storage. Marginal gain on short text.
- **Voyage AI** (`voyage-3`, `voyage-3-large`) — often tops MTEB; extra vendor for marginal real-world gain at MVP scale.
- **Cohere `embed-v4`** — strong multilingual; bigger value-add is their reranker (deferred).
- **Local models** (`nomic-embed`, `bge-m3`, etc.) — privacy/cost wins, but operational complexity and quality lag; deferred per [11 — Configuration and Providers](./11-configuration-and-providers.md#initial-provider-decisions).

### Why `text-embedding-3-small`

Single-user volumes make cost a non-factor, so the deciding criteria are quality on our content and operational simplicity. For short text (titles, summaries, ingredient lists, individual steps), `text-embedding-3-small` is solidly good — the higher-MTEB models add bench-score percentage points that don't translate to noticeable retrieval improvements on recipe-style content. Pairing it with our OpenAI LLM means one vendor, one API key, one SDK.

### Forward compatibility

[02 — Core Data Model](./02-core-data-model.md#re-embedding-rule) keys `chunk_embeddings` by `(chunk_id, provider, model)` and [07 — Retrieval Behavior](./07-retrieval-behavior.md#embedding-model-rule) requires vector search to filter by provider + model. So adding a second embedding model later (for A/B comparison or a content-specific upgrade) just adds rows — no migration. Store `embedding_dimensions = 1536` explicitly on each row so dimension changes stay visible.

---

## 6. Background job approach for async ingestion

**Decision**: `arq` (Redis-backed async task queue) running a worker process separate from the FastAPI app.

### Options considered

- **In-process async task** (FastAPI `BackgroundTasks` or `asyncio.create_task`) — zero new infra; jobs die on API restart.
- **`arq` + Redis** — async-native lightweight queue; worker is a separate process; one extra docker service.
- **Celery** — battle-tested but sync-first and heavyweight; the "production queue" [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md#sync-vs-async-ingestion) says we don't need yet.
- **Dramatiq / RQ** — simpler than Celery but sync-first; fights async FastAPI code.

### Why `arq`

The architecture's intent in [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md#sync-vs-async-ingestion) — "simple single-process background task, no production queue yet" — is to avoid pulling in Celery-class infrastructure prematurely. `arq` honors that spirit: a single Redis container and a small async-native library. The job is still "one async function per ingestion run"; we're just running it in a worker process instead of inside the API process.

Concrete wins this gives us over in-process tasks:

- **Worker survives API restarts.** Active development reloads the API server constantly; in-process tasks lose work each time. Ingestion runs are multi-minute LLM-heavy operations, so this matters.
- **Stuck-job recovery** from [03 — PDF Ingestion Pipeline](./03-pdf-ingestion-pipeline.md#stuck-job-recovery) maps to an `arq` cron job naturally.
- **Extraction caching** from [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#risks-and-mitigations) can use the same Redis if we want.
- **Upgrade path is honest**: the job function is the same async function it would be in the in-process approach. Moving to Celery later, if we ever needed to, is mechanical.

### Why not the alternatives

- **In-process** is the most literal reading of doc 3 and a defensible MVP. The trade-off is real dev-loop pain on restart; this decision accepts one container of complexity to remove it.
- **Celery** brings broker + result backend + Celery Beat — the kind of "production queue" doc 3 explicitly says is premature.
- **Dramatiq / RQ** are sync-first; the impedance mismatch with async FastAPI ingestion code outweighs any simplicity gain.

---

## 7. Local infrastructure setup

**Decision**: Docker Compose stack with `pgvector/pgvector:pg17` and `redis:7-alpine`.

### Options considered

- **Docker Compose with `pgvector/pgvector:pg17` + `redis:7-alpine`** — one command, portable, reproducible.
- **Docker Compose with `timescale/timescaledb-ha:pg17`** — same approach but ships `pg_vectorscale` and TimescaleDB. Useful future-readiness, heavier image, overkill at our scale.
- **Native install** (`brew install postgresql@17 redis`, pgvector via `make install`) — native performance but manual setup, version drift, hard to share with collaborators.
- **Managed services** (Neon for Postgres, Upstash for Redis) — zero local infra but network latency on every dev query, can't work offline.

### Why Docker Compose + vanilla pgvector image

Both [05 — Storage and Indexing](./05-storage-and-indexing.md#first-storage-version) and [11 — Configuration and Providers](./11-configuration-and-providers.md#8-local-vs-hosted-configuration) assume "local first, hostable later." Docker Compose makes that real: the same `docker-compose.yml` can run on a developer Mac or be deployed to a hobby VPS unchanged. Wiping and re-seeding state during ingestion iteration is trivial (`docker compose down -v`).

The vanilla `pgvector/pgvector:pg17` image is the minimal correct choice. `pg_vectorscale`'s DiskANN advantage shows up at corpus sizes well beyond what a personal recipe library will ever reach (millions of vectors); HNSW from stock pgvector is more than enough. If the project ever outgrows that, switching images is a `pg_dump` + restore — annoying but localized.

### What goes in `docker-compose.yml`

- `postgres` — `pgvector/pgvector:pg17`, port `5432`, volume for persistence.
- `redis` — `redis:7-alpine`, port `6379`, volume optional.
- No Adminer / pgAdmin in the compose file — devs use their own GUI (TablePlus, DataGrip, DBeaver, `psql`).

### Why not the alternatives

- **Timescale image** is the right pick if there's near-term intent to scale past HNSW; we don't have that intent.
- **Native install** loses portability and reproducibility for marginal performance gain at our scale.
- **Managed services** are the right move *for production deployment*, not for iterative dev. The local Docker stack can be lifted to any VM later without code changes.

---

## 8. Repository structure

**Decision**: Layer-driven `src/` layout. Top-level modules under `rag_recipes/` map 1:1 to the architectural layers in the docs.

### Layout

```
rag-recipes/
├── pyproject.toml
├── alembic.ini
├── docker-compose.yml
├── .env.example
├── docs/                          # already exists
│   ├── architecture/
│   └── external/
├── src/
│   └── rag_recipes/
│       ├── api/                   # FastAPI app, routes, request/response schemas
│       │   ├── app.py
│       │   ├── routes/
│       │   │   ├── documents.py
│       │   │   ├── search.py
│       │   │   └── debug.py
│       │   ├── schemas/           # Pydantic API request/response models
│       │   └── dependencies.py
│       ├── domain/                # Pydantic domain models (recipe.v1, etc.)
│       ├── ingestion/             # PDF→spans→items→chunks→embeddings
│       │   ├── pipeline.py
│       │   ├── extraction.py
│       │   ├── validation.py
│       │   ├── chunking.py
│       │   ├── embedding.py
│       │   └── jobs.py            # arq tasks
│       ├── retrieval/             # search, keyword, vector, RRF, filters
│       ├── providers/             # interfaces + implementations
│       │   ├── file_storage/
│       │   ├── pdf_extractor/
│       │   ├── llm/
│       │   └── embeddings/
│       ├── storage/               # SQLAlchemy models + repositories
│       └── config.py              # pydantic-settings
├── migrations/                    # Alembic versions and env
├── tests/
│   ├── unit/
│   ├── integration/
│   └── fixtures/
└── scripts/                       # one-off dev scripts
```

### Options considered

- **Layer-driven `src/` layout** — top-level modules per architectural layer.
- **Library-first workspace** (`rag_core` + `rag_api`) — two packages, cleaner separation, more setup overhead.
- **Feature-folder layout** (`recipes/`, `documents/`, `search/`) — group by domain concept; weaker fit for our explicitly layered architecture.
- **Hexagonal / DDD ports-and-adapters** — architecturally pure; heavy ceremony for a solo learning project.

### Why layer-driven

The architecture docs are *already* organized by layer (ingestion, retrieval, storage, providers, API). Mirroring that in the repo makes navigation obvious and keeps the cost of swapping providers low — each provider type has its own folder containing `base.py` (the interface), one or more concrete implementations, and a `fake.py` for tests.

`domain/` holds pure Pydantic models (e.g., `recipe.v1` from [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#strict-output-shape)). They have no IO, no SQLAlchemy. SQLAlchemy ORM models live in `storage/models.py`. The split keeps canonical domain shape and persistence concerns separated, matching the canonical-vs-derived distinction throughout the architecture docs.

### Why not the alternatives

- **Library-first** is right if a CLI, notebook front-end, or shared SDK is on the roadmap. None is.
- **Feature-folder** scatters cross-cutting concerns (validation, embedding, citation building) across feature dirs. Layer-driven keeps each cross-cutting concern in one place.
- **Hexagonal/DDD** is excellent for large team codebases; the ceremony cost isn't justified at MVP solo-dev scale.

---

## 9. Testing stack and fake-provider scaffolding

**Decision**: Opinionated full stack — `pytest` + `pytest-asyncio`, real Postgres in Docker for integration tests, `polyfactory` for test data, `ruff` for lint/format, `mypy` for type checking, `pytest-cov` for coverage (no enforced threshold). Per-provider Fake implementations live in `providers/<type>/fake.py` and are part of production code.

### Options considered

- **Recommended full stack** — pytest + pytest-asyncio + real Postgres + polyfactory + ruff + mypy.
- **Lightweight pytest-only** — skip Postgres-in-tests, polyfactory, type checking. Risks pgvector/`ts_vector` surprises in production.
- **Heavier with Hypothesis + Schemathesis** — property-based + OpenAPI schema testing. Powerful but more upfront learning.
- **Pytest core, defer the rest** — pragmatic but the deferred decisions often never get made.

### Per-provider Fake implementations (already settled in doc 11)

| Interface | Fake behavior |
|---|---|
| `FakeFileStorageProvider` | in-memory bytes keyed by `(provider, key)` |
| `FakePdfExtractor` | returns pre-loaded `PdfPageText` lists for known fixtures |
| `FakeLLMProvider` | returns canned `recipe.v1` JSON keyed by `input_hash`/`prompt_version` |
| `FakeEmbeddingProvider` | returns deterministic vectors (e.g., hash-seeded random or simple bag-of-words projection) at the correct dimension |

Fakes live in `src/rag_recipes/providers/<type>/fake.py` — they are not test-only code. They're also useful for CLI debugging and local-only end-to-end runs without paid APIs, as suggested in [12 — Evaluation and Testing](./12-evaluation-and-testing.md#2-provider-contract-tests).

### Why real Postgres for integration tests

Retrieval depends on pgvector cosine distance, `ts_vector`/`ts_rank_cd` full-text scoring, JSONB column behavior, and our specific GIN/IVFFlat/HNSW index choices. SQLite has none of these. Testing retrieval against SQLite is testing a different system than we deploy.

Strategy:
- Postgres + pgvector container started once per test session (compose or testcontainers).
- Alembic migrations applied at session start.
- Each test runs inside a SQLAlchemy session transaction that rolls back at teardown — fast isolation without truncation.

### Why pytest-asyncio (not anyio)

Our entire async story is `asyncio` (FastAPI, SQLAlchemy 2.0 async, arq, OpenAI SDK async). `pytest-asyncio` matches that directly. `anyio` is preferred when supporting multiple async backends (`trio`); we don't.

### Why ruff + mypy

`ruff` replaces black + isort + flake8 with one fast tool. `mypy` (strict for `domain/` and `providers/`, more relaxed elsewhere) pays for itself given SQLAlchemy 2.0's typed style and our Pydantic-everywhere code. Type errors here catch real bugs — JSON shape drift, schema mismatches, optional/nullable confusion.

### What we are not adopting yet

- **Hypothesis** for property-based tests — valuable later for ingredient-parsing and validation rules; defer until specific properties are worth encoding.
- **Schemathesis** for OpenAPI conformance — useful once the API is stable; add when the API contract is publicly stable.
- **Coverage thresholds** — track but don't gate. Premature thresholds reward ceremony over signal during early development.

---

## Summary of decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Backend language + framework | Python + FastAPI |
| 2 | DB client + migrations | SQLAlchemy 2.0 (async) + Alembic + `pgvector-python` |
| 3 | PDF text extraction | PyMuPDF (behind `PdfTextExtractor` interface) |
| 4 | LLM provider (ingestion) | OpenAI (`gpt-4.1` to start) |
| 4b | Structured-output approach | OpenAI native `parse()` behind own `LLMProvider` interface |
| 5 | Embedding provider + model | OpenAI `text-embedding-3-small` @ 1536 dims |
| 6 | Background job runner | `arq` + Redis |
| 7 | Local infra | Docker Compose: `pgvector/pgvector:pg17` + `redis:7-alpine` |
| 8 | Repo structure | Layer-driven `src/rag_recipes/` layout |
| 9 | Testing stack | `pytest` + `pytest-asyncio` + real Postgres + `polyfactory` + `ruff` + `mypy` |

---

## 10. Evaluation harness layout

**Decision**: Top-level `evals/` directory with a `typer` CLI; shared `data/fixtures/` for inputs used by both pytest tests and eval scripts; per-run reports gitignored under `evals/reports/`; committed regression baselines under `evals/baselines/`.

### Options considered

- **Top-level `evals/` + shared `data/fixtures/`** — clean separation of correctness tests from quality measurements.
- **Evals as pytest tests with `@pytest.mark.eval`** — reuses pytest infra but conflates pass/fail with metric reports.
- **Evals in `scripts/`** — junk-drawer risk; no obvious home for reports/baselines.
- **Jupyter notebooks** — good for exploration, bad for reproducibility and diffing.

### Why a separate `evals/` directory

Tests assert binary correctness; evals measure quality. Conflating them — putting evals inside pytest where "passing" with NDCG@10 = 0.20 looks fine — destroys signal. A separate workflow forces the right mental model: eval runs are quality measurements that produce reports, not gates that block merges.

`data/fixtures/` is intentionally shared between `tests/` and `evals/` so a single small synthetic PDF (or a recipe text fixture) serves both an integration test and an eval run without duplication.

### Layout

```
data/
├── fixtures/                          # shared between tests/ and evals/
│   ├── pdfs/                          # tiny synthetic PDFs (committed)
│   ├── synthetic_recipes/             # markdown + expected recipe.v1 JSON (committed)
│   ├── queries/                       # BEIR-style query/qrels (committed)
│   ├── judge_prompts/                 # see topic 12
│   ├── judge_alignment/               # see topic 12
│   └── private/                       # real cookbooks (gitignored)
└── storage/                           # local FileStorageProvider backend (gitignored)

evals/
├── README.md
├── scripts/
│   ├── extraction_eval.py
│   ├── retrieval_eval.py
│   ├── judge_alignment.py
│   └── confidence_review.py
├── reports/                           # gitignored except baselines/
│   └── <timestamp>-<short-label>/
│       ├── summary.md
│       ├── results.json
│       └── per_item_breakdowns.md
└── baselines/                         # committed regression-comparison points
    ├── extraction.json
    └── retrieval.json
```

### Invocation

A small `typer` CLI installed as a console script in `pyproject.toml`:

```text
rag-evals extraction --fixtures synthetic
rag-evals retrieval --queries synthetic --k 10
rag-evals judge-alignment --judge summary_quality
rag-evals confidence-review
```

Eval scripts import the same `LLMProvider`, `EmbeddingProvider`, and retrieval/extraction code as the app — they are a different *workflow*, not different code.

---

## 11. Retrieval evaluation metrics

**Decision**: NDCG@10 as the headline metric; Recall@5, Recall@10, and MRR as interpretable secondaries; computed via `pytrec_eval`; BEIR-style `query` + `qrels` fixture format.

### Options considered

- **NDCG@10 + Recall@5/Recall@10 + MRR via `pytrec_eval`, BEIR-style fixtures** — field-standard headline plus interpretable secondaries.
- **Recall@5/Recall@10 + MRR only (doc 12 verbatim)** — simplest, but binary relevance only.
- **NDCG@10 only** — gold-standard single number; loses "did we retrieve it at all?" debuggability early.
- **Everything (NDCG, Recall, MRR, Precision@K)** — comprehensive number-soup, invites metric cherry-picking.

### Why this combination

NDCG@10 is the standard headline in modern IR (BEIR, MS MARCO, every external transcript). It rewards correct results in higher positions and supports graded relevance later without changing fixtures. Recall@K answers the most interpretable question — "did the expected item appear in the top K at all?" — and MRR captures whether the top result is useful. Together, three numbers tell the right story: NDCG for overall quality, Recall for retrieval coverage, MRR for top-result usefulness.

### Fixture format

```
data/fixtures/queries/
├── synthetic/
│   ├── queries.tsv         # query_id \t query_text
│   ├── qrels.tsv           # query_id \t knowledge_item_id \t relevance_score
│   └── README.md
└── private/                # gitignored; same structure
```

`relevance_score` is `1` for binary "relevant" today; the format leaves room for graded `0/1/2/3` later without rebuilding fixtures.

The corpus side is implicit — the running database is the corpus, queried via the retrieval API exactly as the API would be called from a frontend.

### Library

`pytrec_eval` is the canonical Python wrapper around TREC's `trec_eval`. ~5 lines to compute all three metrics from a `qrels` dict and a `run` dict (mapping query_id → ranked list of `knowledge_item_id`s with scores). Stable, well-maintained, no surprises.

---

## 12. Extraction evaluation methodology

**Decision**: Field-level objective scoring for measurable fields + LLM-as-judge for subjective fields, with a **human-aligned LLM-judge process** that tracks agreement between human and judge ratings and iterates the judge prompt until agreement is high. Confidence calibration runs as a separate report grouping items by confidence bucket against actual quality.

This is the methodology section that was missing from [12 — Evaluation and Testing](./12-evaluation-and-testing.md); treat this entry as the definitive description until doc 12 is updated.

### Options considered

- **Field-level objective + LLM-judge for subjective + alignment loop** — right tool per field, judge stays honest.
- **Field-level objective only** — fully deterministic but blind to summary/boundary/text quality.
- **LLM-judge for everything** — single methodology but introduces judge variance into measurable fields.
- **Heuristic-only (string similarity, BLEU, ROUGE)** — these proxies for "good recipe summary?" reward surface overlap, not meaning.

### Field categorization

**Objective (exact-match or numeric comparison)**

- `title` (exact + after `normalized_title` normalization)
- `yield` (string match)
- `prep_time` / `cook_time` / `total_time` (parsed minutes comparison)
- Ingredient count, step count
- Per-ingredient: `raw_text` preserved, `quantity_value` parsed, `unit_normalized`, `item_normalized`, `preparation` extracted
- `source_span_ids` (set membership)

**Subjective (LLM-judge, human-aligned)**

- `summary` quality
- Recipe boundary correctness (one recipe vs merged vs split)
- Step-text fidelity (paraphrased without losing instructions)
- Title accuracy when extracted vs ground-truth differ stylistically

### The human-aligned LLM-judge loop

This is the core idea and the most important addition to our evaluation methodology:

1. Pick a small annotated fixture set (~30–50 cases initially).
2. Run extraction over it → model outputs.
3. **You** rate each output as pass/fail with a written critique.
4. The LLM judge rates each output with the same pass/fail + critique format using a prompt under `data/fixtures/judge_prompts/`.
5. Compute agreement (% match between human and judge ratings).
6. Inspect disagreements — the pattern in them reveals what's missing from the judge prompt.
7. Iterate the judge prompt; optionally meta-prompt a strong model with the disagreements to refine it.
8. Re-run, re-check agreement. Stop at the target bar (~90%+ is defensible; track over time).
9. Use the aligned judge for ongoing extraction evals at scale; periodically re-check agreement because data drifts.

The rule: **don't trust the judge until you've shown it agrees with you**. Skipping this is how teams end up with green dashboards full of meaningless numbers.

### Confidence calibration

A separate eval script (`evals/scripts/confidence_review.py`) groups extracted items by `confidence.overall` bucket (`0.9+`, `0.75–0.89`, `0.5–0.74`, `<0.5`) and overlays actual quality scores from the objective + judge scoring. This surfaces miscalibration — e.g., if high-confidence items are routinely judged wrong, the model's confidence signal isn't useful and we need to adjust prompts or build a stronger system confidence score (per the note at the end of [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#confidence-scores)).

### File layout (extension of topic 10)

```
data/fixtures/
├── synthetic_recipes/
│   └── <fixture_name>/
│       ├── source.md             # input text the LLM sees
│       ├── expected.json         # golden recipe.v1
│       └── notes.md
├── judge_prompts/
│   ├── summary_quality.md
│   ├── boundary_correctness.md
│   └── step_text_quality.md
└── judge_alignment/
    └── <fixture_name>.json       # { human_rating, judge_rating, agreement_status, run_metadata }
```

---

## 13. Observability / LLM tracing

**Decision**: Langfuse, self-hosted via docker-compose. Integrate as a thin wrapper inside the `LLMProvider` and `EmbeddingProvider` implementations. Keep `ExtractionRun` as the canonical audit record; Langfuse is the dev-loop lens on top.

### Options considered

- **Langfuse self-hosted** — open-source, local-first, ~5 lines of integration, best dev-loop UI for prompt iteration.
- **Langfuse Cloud** — same product hosted; zero local infra but data leaves the machine.
- **Helicone** — proxy-based, simplest integration but narrower feature set.
- **Defer / roll-your-own** — extend `ExtractionRun` with latency/token fields; build a UI later. Risks reimplementing 80% of Langfuse poorly when you can least afford the context switch.

### `ExtractionRun` vs Langfuse

These are complementary, not alternatives:

- **`ExtractionRun`** (per [02 — Core Data Model](./02-core-data-model.md#7-extractionrun)) is the *canonical record*. Queryable from the API. Source of truth for ingestion debugging in production. Survives observability backend changes.
- **Langfuse** is the *lens*. Browsable trace UI, session grouping (all calls for one document ingestion), prompt-version filtering, side-by-side run comparison, token/latency/cost surfacing. Disposable in principle.

We keep `ExtractionRun` either way. Langfuse adds the dev UI on top.

### Why self-hosted

- Matches the local-first stance in [01 — Big Picture](./01-big-picture.md#starting-decisions). Recipe text and prompts stay on the machine.
- Free.
- Two extra containers in docker-compose (Langfuse server + Clickhouse). ~1 GB memory overhead — acceptable on a dev Mac.
- Switching to Langfuse Cloud later (if you ever want to debug from another machine) is a config change, not a code change.

### Integration shape

The `LLMProvider` interface from [11 — Configuration and Providers](./11-configuration-and-providers.md#3-llmprovider) gets a Langfuse decorator on the `generateStructuredOutput` implementation. Each call records: provider, model, prompt_version, schema_version, input source span IDs, raw output, parsed output, latency, token usage, cost, and a session ID tied to the `Document.id` being ingested. The decorator is conditional on a `LANGFUSE_ENABLED` config flag so production deployments without Langfuse keep working.

`EmbeddingProvider` gets the same treatment.

### What this unlocks

- Browse all LLM calls for one document ingestion in one timeline view.
- Compare two extraction runs across prompt versions, side by side.
- Filter calls by failure mode (rejected validation, low confidence, etc.).
- Surface aggregate cost and token usage during eval runs.
- Verify the input-hash cache from [04 — LLM-Assisted Recipe Extraction](./04-llm-assisted-recipe-extraction.md#risks-and-mitigations) is actually firing.

---

## Summary of decisions

| # | Topic | Decision |
|---|---|---|
| 1 | Backend language + framework | Python + FastAPI |
| 2 | DB client + migrations | SQLAlchemy 2.0 (async) + Alembic + `pgvector-python` |
| 3 | PDF text extraction | PyMuPDF (behind `PdfTextExtractor` interface) |
| 4 | LLM provider (ingestion) | OpenAI (`gpt-4.1` to start) |
| 4b | Structured-output approach | OpenAI native `parse()` behind own `LLMProvider` interface |
| 5 | Embedding provider + model | OpenAI `text-embedding-3-small` @ 1536 dims |
| 6 | Background job runner | `arq` + Redis |
| 7 | Local infra | Docker Compose: `pgvector/pgvector:pg17` + `redis:7-alpine` + Langfuse |
| 8 | Repo structure | Layer-driven `src/rag_recipes/` layout |
| 9 | Testing stack | `pytest` + `pytest-asyncio` + real Postgres + `polyfactory` + `ruff` + `mypy` |
| 10 | Evaluation harness | Top-level `evals/` + shared `data/fixtures/` + `typer` CLI |
| 11 | Retrieval metrics | NDCG@10 + Recall@5/Recall@10 + MRR via `pytrec_eval`, BEIR-style fixtures |
| 12 | Extraction methodology | Field-level objective + LLM-judge with human alignment loop |
| 13 | Observability | Langfuse self-hosted in docker-compose |

## Deferred / explicitly out of scope

- **Frontend architecture** (the missing doc 9/10) — not in scope until retrieval works end-to-end.
- **Answer-layer API shape** — `POST /answers` vs `POST /search { include_answer: true }`; defer per [08 — Query-Time Answer Layer](./08-query-time-answer-layer.md).
- **Reranking** — defer per [05 — Storage and Indexing](./05-storage-and-indexing.md#reranking); the retrieval flow leaves room.
- **`pg_vectorscale` / DiskANN index** — defer until corpus scale demands it; image swap when needed.
- **BM25 via `pg_search` extension** — defer; Postgres FTS (`ts_rank_cd`) is the day-one keyword search.
- **Hypothesis / Schemathesis** — defer until concrete properties / API contracts are worth encoding.
- **Specific synthetic-fixture set, judge prompts, and golden query set** — open per [12 — Evaluation and Testing](./12-evaluation-and-testing.md#open-questions-before-implementation); create alongside the first end-to-end ingestion runs.
- **Production deployment** (managed Postgres, S3-backed file storage, hosted Redis, Langfuse Cloud) — local-first per [01 — Big Picture](./01-big-picture.md#starting-decisions).
