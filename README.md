# Prod RAG Tool

A production-grade retrieval-augmented generation (RAG) system: answers that carry citations, tenant
isolation enforced inside the database, guardrails with explicit refusals, and evals that gate every change.

**Status: v0.1 Foundation.** The runnable part today is the backend foundation: a local stack (API, Postgres
with pgvector, Redis, MLflow), a versioned database schema with row-level security, the OpenAPI contract
(routes return `501 Not Implemented` for now), and the ingest building blocks (parsing, PII redaction,
chunking, embedding adapters). You cannot yet upload a document and ask it a question.

## Install

Requires Docker with Compose v2.

```bash
git clone https://github.com/arnavk2319/prod-rag-tool.git
cd prod-rag-tool
make doctor    # checks Docker, free ports and configuration
make up        # starts the stack
make migrate   # applies the database schema
make urls      # prints where each service listens
```

On macOS, ports 5000 and 11434 are often taken. If `make doctor` reports them, run the commands with
`MLFLOW_PORT=5050 OLLAMA_PORT=11435` set. Stop everything with `make down`.

## Expected output

`make doctor`:

```
doctor: all checks passed
```

`make up` ends with the services healthy (`api`, `worker`, `postgres`, `redis`, `mlflow`):

```
 Container ragplatform-api-1  Healthy
 Container ragplatform-mlflow-1  Healthy
 Container ragplatform-postgres-1  Healthy
 Container ragplatform-redis-1  Healthy
 Container ragplatform-worker-1  Healthy
```

`make migrate`:

```
make migrate: applying migrations as rag_owner to 127.0.0.1:5432/ragapp
```

`curl localhost:8000/healthz`:

```
{"status":"ok"}
```

`curl localhost:8000/v1/config` (routes are stubs in this release):

```
{"error":{"code":"internal_error","message":"not implemented","request_id":"..."}}
```

`make urls` lists the API at `http://127.0.0.1:8000` and MLflow at `http://127.0.0.1:5000`.
