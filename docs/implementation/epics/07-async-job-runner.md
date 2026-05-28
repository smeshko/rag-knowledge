# Epic 7 — Async Job Runner

**Status**: In progress (Phase 7.1 complete)

## Overview

Wire `arq` into the project so ingestion jobs run in a worker process separate from the FastAPI app, with status transitions tracked on `Document.status` and a stuck-job recovery cron. This epic ships the runner; Epic 8 ships the first actual ingestion job that uses it.

## Architecture references

- [03 — PDF Ingestion Pipeline § Sync vs Async](../../architecture/03-pdf-ingestion-pipeline.md#sync-vs-async-ingestion) — async ingestion model, stuck-job recovery rule
- [02 — Core Data Model § status](../../architecture/02-core-data-model.md#status) — full `Document.status` transitions
- [13 — Implementation Decisions, topic 6](../../architecture/13-implementation-decisions.md#6-background-job-approach-for-async-ingestion) — arq + Redis choice

## Dependencies

- Epic 1 (Redis service, scaffold)
- Epic 2 (Document model with `status` column)

## Out of scope

- The actual PDF/extraction/chunking jobs (Epic 8+)
- Job result reporting back to the API (the API polls `Document.status`)

---

## Phase 7.1 — arq worker wiring + job registration

**Goal**: An arq worker process can be started, connects to Redis, and is ready to execute registered job functions. The Makefile's `make dev-worker` target becomes functional.

### What to build

- **`src/rag_recipes/ingestion/jobs.py`** — module that:
  - Defines the arq `WorkerSettings` class
  - Lists registered job functions (empty for now; populated in Epic 8+)
  - Configures Redis connection from `Settings`
  - Configures `max_jobs`, `job_timeout`, retry policy per defaults appropriate for ingestion workloads (long-running, low concurrency for MVP)
- **`src/rag_recipes/ingestion/queue.py`** — module exposing an async `enqueue_job` helper used by the API layer (and later by reprocess in Epic 11). Wraps arq's `ArqRedis.enqueue_job` with typing and session-ID propagation for Langfuse tracing.
- **`scripts/run_worker.py`** (or set up an arq entry point in `pyproject.toml`) — the executable invoked by `make dev-worker`
- **`make dev-worker`** target updated to actually start the worker
- **`make dev`** target updated to start both API and worker (e.g., via `make dev-api & make dev-worker &` pattern or a single tool like `honcho`/`overmind`/`concurrently` — document the choice)
- A placeholder no-op job (`ping_job` returning a string) registered so the worker has something to validate against
- Integration test that enqueues `ping_job` from an async helper, runs the worker briefly, and verifies the job completed (use a short-lived worker spawned by the test or `arq.testing.WorkerSettings` pattern)

### Acceptance criteria

- [x] `make dev-worker` starts an arq worker that connects to Redis and logs registration of `ping_job`
- [x] Enqueueing `ping_job` from the API or a test causes the worker to execute it
- [x] Worker configuration (timeout, retries, max_jobs) lives in code and is overridable via `Settings`
- [x] `make dev` starts both API and worker (and stopping one cleanly is documented)
- [x] Integration test for ping_job passes

### Validation

`make dev-worker` shows registration logs; in another terminal, an enqueue helper inserts a `ping_job`; the worker logs execution within seconds.

---

## Phase 7.2 — Stuck-job recovery cron + status transition helpers

**Goal**: A cron-scheduled job sweeps documents stuck in non-terminal statuses past a timeout and marks them `failed`. Status-transition logic is centralized so Epic 8+ uses consistent helpers.

### What to build

- **`src/rag_recipes/ingestion/status.py`** — helpers:
  - `transition_to(document_id, new_status, message=None, increment_progress=None)` — atomic update that validates legal transitions per [doc 2 § status](../../architecture/02-core-data-model.md#status)
  - `mark_failed(document_id, error)` — sets `status="failed"`, records the error message somewhere queryable (column on `documents` or separate `ingestion_failures` table — pick one, document it)
  - `is_terminal(status)` — helper returning True for `ready`, `needs_review`, `failed`
- **Stuck-job recovery cron** registered in `WorkerSettings.cron_jobs`:
  - Runs every N minutes (configurable, default 5)
  - Selects documents where `status NOT IN (ready, needs_review, failed)` AND `updated_at < now() - INTERVAL stuck_timeout` (configurable, default 30 minutes)
  - Calls `mark_failed` with reason `"stuck_job_timeout"`
- Configurable timeouts in `Settings`: `stuck_job_timeout_minutes`, `stuck_job_check_interval_minutes`
- Integration test: insert a Document with `status="extracting_items"` and `updated_at` set to two hours ago; run the cron once; assert status becomes `failed`

### Acceptance criteria

- [ ] Status-transition helper rejects illegal transitions with a clear error
- [ ] Cron registered and visible in worker startup logs
- [ ] Stuck-job detection works for all non-terminal statuses
- [ ] Failures preserve enough context (timestamp, last status, reason) to debug
- [ ] Configurable timeouts respected

### Validation

Integration test demonstrates stuck-job auto-recovery; manual smoke test where a long-running stub job is killed mid-execution and the cron eventually marks the document failed.

---

## Epic-level acceptance criteria

- [x] arq worker runs and executes registered jobs
- [x] `make dev-worker` and `make dev` both work
- [ ] Status transition helpers centralize all `Document.status` updates
- [ ] Stuck-job recovery cron registered and tested
- [ ] Failure mode records enough context to debug stuck jobs
- [ ] Status in [`EPICS.md`](../EPICS.md) updated; Epic 8 unblocked (in combination with Epics 4, 6)
