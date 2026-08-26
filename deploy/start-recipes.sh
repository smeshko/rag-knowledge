#!/usr/bin/env bash
# Start (or confirm running) everything Stove needs on the home server:
# postgres + redis (compose), the API (uvicorn :8004), the ingestion worker
# (arq) and Caddy (:8090, the tunnel origin). Idempotent — re-running only
# starts what is down. Stop a service by killing the PID in its pidfile.
#
# Configuration, all overridable from the environment:
#   RAG_RECIPES_BE        this repo's checkout        (default: the script's parent dir)
#   RAG_RECIPES_FE_DIST   the frontend's built dist/  (default: ../rag-recipes-fe/dist)
#   RAG_RECIPES_LOG_DIR   logs + pidfiles              (default: ~/Developer/logs)
#   RAG_RECIPES_TOKEN     the bearer token Caddy injects; if unset, read from the
#                         macOS login keychain item `rag-recipes-token`:
#                           security add-generic-password -a "$USER" -s rag-recipes-token -w '<token>'
#                         It must equal PERSONAL_API_TOKEN in $RAG_RECIPES_BE/.env.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAG_RECIPES_BE="${RAG_RECIPES_BE:-$(cd "$HERE/.." && pwd)}"
RAG_RECIPES_FE_DIST="${RAG_RECIPES_FE_DIST:-$(cd "$RAG_RECIPES_BE/.." && pwd)/rag-recipes-fe/dist}"
RAG_RECIPES_LOG_DIR="${RAG_RECIPES_LOG_DIR:-$HOME/Developer/logs}"
mkdir -p "$RAG_RECIPES_LOG_DIR"
export RAG_RECIPES_FE_DIST RAG_RECIPES_LOG_DIR

log()  { echo "[start-recipes] $1"; }
warn() { echo "[start-recipes] WARNING: $1" >&2; }
# True if the PID recorded in pidfile $1 is still a live process.
is_running() { [ -f "$1" ] && kill -0 "$(cat "$1")" 2>/dev/null; }
# Roll a log over just before starting its writer (never while one holds the
# fd). Two generations kept.
MAX_LOG_BYTES="${MAX_LOG_BYTES:-10485760}"
rotate() {
  local f="$1" size
  [ -f "$f" ] || return 0
  size="$(stat -f %z "$f" 2>/dev/null || stat -c %s "$f" 2>/dev/null || echo 0)"
  [ "$size" -ge "$MAX_LOG_BYTES" ] || return 0
  rm -f "$f.2"
  [ -f "$f.1" ] && mv "$f.1" "$f.2"
  mv "$f" "$f.1"
}

if [ -z "${RAG_RECIPES_TOKEN:-}" ]; then
  RAG_RECIPES_TOKEN="$(security find-generic-password -a "$USER" -s rag-recipes-token -w 2>/dev/null || true)"
fi
if [ -z "$RAG_RECIPES_TOKEN" ]; then
  warn "no RAG_RECIPES_TOKEN and no keychain item rag-recipes-token; see the header of this script."
  exit 1
fi
export RAG_RECIPES_TOKEN
if [ ! -d "$RAG_RECIPES_FE_DIST" ]; then
  warn "$RAG_RECIPES_FE_DIST missing — run \`just build\` in the frontend repo first."
  exit 1
fi
if ! docker info >/dev/null 2>&1; then
  warn "Docker is not running; postgres and redis need it."
  exit 1
fi

# Postgres + Redis only. The Langfuse stack is opt-in (langfuse_enabled=false
# by default); bring it up by hand with `docker compose up -d` when wanted.
log "Starting datastores (postgres, redis)..."
docker compose -f "$RAG_RECIPES_BE/docker-compose.yml" up -d postgres redis

# API. `.venv/bin/uvicorn` rather than `uv run` so $! is the server's PID (uv
# supervises its child rather than exec'ing it); `nohup env -C DIR` rather
# than `(cd DIR && nohup ...)` so no subshell survives to own the PID.
if is_running "$RAG_RECIPES_LOG_DIR/recipes-be.pid"; then
  log "API already running (PID $(cat "$RAG_RECIPES_LOG_DIR/recipes-be.pid"))."
else
  log "Starting API (uvicorn 127.0.0.1:8004)..."
  rotate "$RAG_RECIPES_LOG_DIR/recipes-be.log"
  nohup env -C "$RAG_RECIPES_BE" .venv/bin/uvicorn rag_recipes.api.app:app \
    --host 127.0.0.1 --port 8004 \
    >> "$RAG_RECIPES_LOG_DIR/recipes-be.log" 2>&1 &
  echo $! > "$RAG_RECIPES_LOG_DIR/recipes-be.pid"
fi

# Ingestion worker — uploads stay queued forever without it.
if is_running "$RAG_RECIPES_LOG_DIR/recipes-worker.pid"; then
  log "worker already running (PID $(cat "$RAG_RECIPES_LOG_DIR/recipes-worker.pid"))."
else
  log "Starting ingestion worker (arq)..."
  rotate "$RAG_RECIPES_LOG_DIR/recipes-worker.log"
  nohup env -C "$RAG_RECIPES_BE" .venv/bin/arq rag_recipes.ingestion.jobs.WorkerSettings \
    >> "$RAG_RECIPES_LOG_DIR/recipes-worker.log" 2>&1 &
  echo $! > "$RAG_RECIPES_LOG_DIR/recipes-worker.pid"
fi

# Caddy — the tunnel origin. Serves the frontend build and proxies /api/*.
if is_running "$RAG_RECIPES_LOG_DIR/recipes-caddy.pid"; then
  log "Caddy already running (PID $(cat "$RAG_RECIPES_LOG_DIR/recipes-caddy.pid"))."
elif ! command -v caddy >/dev/null 2>&1; then
  warn "caddy not installed (\`brew install caddy\`); the frontend will not be served."
  exit 1
else
  log "Starting Caddy (127.0.0.1:8090)..."
  rotate "$RAG_RECIPES_LOG_DIR/recipes-caddy.log"
  nohup caddy run --config "$HERE/Caddyfile" \
    >> "$RAG_RECIPES_LOG_DIR/recipes-caddy.log" 2>&1 &
  echo $! > "$RAG_RECIPES_LOG_DIR/recipes-caddy.pid"
fi

log "done. Logs and pidfiles: $RAG_RECIPES_LOG_DIR/recipes-{be,worker,caddy}.{log,pid}"
