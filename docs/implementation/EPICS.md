# Implementation Epics

This document tracks the implementation work for the **rag-recipes backend**. Each epic is a self-contained unit of functionality; each phase within an epic is sized to fit a small-to-medium pull request.

## Scope

We are building the backend only. The system exposes a REST API and is fully usable in isolation — every feature can be validated against the API (with `curl` or any HTTP client), the database, the eval CLI, and Langfuse traces. A separate frontend repository will consume this API later; no frontend work is planned in these epics.

Architecture grounding:

- [01 — Big Picture](../architecture/01-big-picture.md) — overall system goal, starting decisions, layered architecture
- [06 — Backend API Shape](../architecture/06-backend-api-shape.md) — the API as the stable boundary between this backend and any future client
- [13 — Implementation Decisions](../architecture/13-implementation-decisions.md) — what we picked to build with (language, libraries, infra)

## Status legend

| Status | Meaning |
|---|---|
| Ready for dev | All dependencies met; can be picked up |
| In progress | At least one phase has been merged |
| Done | All phases complete and validated against the acceptance criteria |
| Blocked | Waiting on a prerequisite epic |

## Epic status

| # | Epic | Phases | Dependencies | Status |
|---|------|--------|--------------|--------|
| 1 | [Foundation & Project Scaffolding](./epics/01-foundation.md) | 4 | — | Ready for dev |
| 2 | [Data Model & Migrations](./epics/02-data-model.md) | 3 | Epic 1 | Done |
| 3 | [Provider Interfaces & Fakes](./epics/03-provider-interfaces.md) | 2 | Epic 1 | Done |
| 4 | [Storage & PDF Extraction](./epics/04-storage-and-pdf-extraction.md) | 2 | Epic 3 | Done |
| 5 | [LLM & Embedding Providers](./epics/05-llm-and-embedding-providers.md) | 3 | Epic 3 | Done |
| 6 | [Document Management API](./epics/06-document-api.md) | 3 | Epics 2, 4 | Done |
| 7 | [Async Job Runner](./epics/07-async-job-runner.md) | 2 | Epics 1, 2 | Done |
| 8 | [PDF Ingestion — Text & SourceSpans](./epics/08-pdf-ingestion-spans.md) | 2 | Epics 4, 6, 7 | Done |
| 9 | [LLM Extraction Pipeline](./epics/09-llm-extraction-pipeline.md) | 4 | Epics 5, 8 | In progress |
| 10 | [Chunking, Embedding, Indexing](./epics/10-chunking-embedding-indexing.md) | 3 | Epics 5, 9 | Blocked |
| 11 | [Reprocessing Modes](./epics/11-reprocessing.md) | 2 | Epic 10 | Blocked |
| 12 | [Retrieval Layer](./epics/12-retrieval.md) | 3 | Epic 10 | Blocked |
| 13 | [Search API & Debug Endpoints](./epics/13-search-api.md) | 3 | Epic 12 | Blocked |
| 14 | [Evaluation Harness](./epics/14-evaluation-harness.md) | 2 | Epic 1 | Blocked |
| 15 | [Extraction Evaluation](./epics/15-extraction-evaluation.md) | 3 | Epics 9, 14 | Blocked |
| 16 | [Retrieval Evaluation](./epics/16-retrieval-evaluation.md) | 2 | Epics 13, 14 | Blocked |

## How to work with this plan

Each epic file in [`./epics/`](./epics/) is a self-contained "pickup file": it captures the goal, architecture references, phase-by-phase implementation details, acceptance criteria, and validation steps. To start work on an epic:

1. Confirm dependencies are `Done` in the table above.
2. Open the epic file and start with phase N.1.
3. Each phase should land as its own pull request.
4. Update this table to `In progress` after the first phase of an epic merges.
5. Update to `Done` when all phases are complete and the epic-level acceptance criteria are satisfied.
6. Cross-reference the architecture documents linked in each epic — they are the source of truth for design decisions. Epic files describe *how* to implement those decisions.

## Deferred from these epics

The following are intentionally out of scope and are *not* covered by any epic:

- **Frontend** (web, iOS, CLI client) — separate repository, future work
- **Query-time answer generation** (LLM synthesis on top of retrieval) — see [08 — Query-Time Answer Layer](../architecture/08-query-time-answer-layer.md); deferred until retrieval works end-to-end
- **Reranking** — see [05 — Storage and Indexing](../architecture/05-storage-and-indexing.md#reranking); the retrieval flow leaves room for it
- **`pg_vectorscale` / DiskANN index** — image swap when corpus scale demands it
- **BM25 via `pg_search` extension** — Postgres FTS is the day-one keyword search
- **Production deployment** (managed Postgres, S3 file storage, hosted Redis, Langfuse Cloud) — local-first per doc 1

See [13 — Implementation Decisions](../architecture/13-implementation-decisions.md#deferred--explicitly-out-of-scope) for the full deferred list with rationale.
