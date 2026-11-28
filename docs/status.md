# Project status

What this repository has built, what is still planned, and why the main design choices were made.
This page is self-contained: everything it states is either written here or generated from tracked
code and config.

**Contributor rule.** Every pull request refreshes its own rows in this page: set the component and
ticket rows it touches to `in progress` or `in review` (with the PR number). The next change that
starts from `dev` flips rows whose PR has merged to `merged`. Run `make status` to refresh the
generated blocks; CI fails (`tests/unit/test_status_doc.py`) when the model, chunker, ports, pins,
migrations, ADR list or eval metrics change without the blocks being refreshed.

States: `planned`, `in progress`, `in review`, `merged`. A `merged` row says only what is merged;
the Choice column states what is not yet built.

## 1. Component map

| Component | Choice | State | Tickets | PR | Decision (section 4 entry) |
| --- | --- | --- | --- | --- | --- |
| Ingest | Idempotent pipeline: raw upload stored in Postgres, worker, copy-forward into a new index version. Only the document registration function exists; no worker, no HTTP upload path yet | planned | T-010 | - | 14, 5 |
| Parser | Text, Markdown, HTML and PDF extraction with limits. No OCR | merged | T-008 | #14 | 1, 13 |
| Redaction | Structured PII (emails, phones, ids, cards and similar) redacted before indexing; person names are out of scope and listed as a gap | merged | T-008 | #14 | 1, 13 |
| Quarantine | Unparseable, scanned, oversize or injection-flagged files are quarantined with a reason, never indexed | merged | T-008 | #14 | 1, 13 |
| Chunker | Parent-child chunking `pc-300-1200-v1`: parents of about 1200 tokens (no overlap, tile the document) are returned; children of about 300 tokens (12% overlap inside one parent, never crossing a parent) are embedded and searched; heading path kept; sizes are settings. Tables can still be split across chunks. The earlier single-level chunker (#13) is replaced | merged | T-009, T-044 | #13, #18 | 19 |
| Embedder | `BAAI/bge-small-en-v1.5` via fastembed (384 dims) and a deterministic hash embedder for tests and gates | merged | T-009 | #13 | 2, 7 |
| Reranker | Cross-encoder adapter merged; not used by any retrieval path yet | merged | T-009 | #13 | 2, 7 |
| Policy gate | Core logic merged (#17) against in-memory fakes; SQL tables (migration 0003), `SqlPolicyRepository`, `SqlEvalRuns` and the additive `versions.policy_hash` contract field are built; an eval run with no metrics counts as not passed, a dangling parent policy or a NUL byte in a field is a validation error, and audit rows are appended only through the `audit_append` function. Not built yet: wiring into an ingest job (no ingest pipeline yet). Postgres authoritative, MLflow mirror best effort | in progress | T-045 | #17 | 20 |
| Index versions | Schema only (policy tables and `index_versions.policy_id` added by 0003; list-partitioned `chunks` and `parent_chunks`; `index_versions`, `index_alias`). No build, gate, alias swap or rollback code yet | planned | T-003, T-011 | #7 | 5 |
| Vector store | Postgres 16 with pgvector 0.8 (per-partition HNSW index DDL exists; nothing is populated or queried yet) | merged | T-003, T-004 | #7, #8 | 3, 6 |
| Retrieval | Hybrid vector plus full-text in one SQL round trip with tenant and ACL filters inside the query. Not built | planned | T-012, T-013 | - | 4, 6 |
| Generation chain | Ollama-first ordered fallback, last tier deterministic extractive, no paid models. No wrapper, no config file, no prompts in use yet | planned | T-016, T-019 | - | 7, 8 |
| Verifier | Code verifier (citation, quote and number checks) and the extractive provider. Not built | planned | T-017 | - | 7, 8 |
| Guardrails | Prompt versions, input and output guards, abstention below the rerank floor. Not built | planned | T-018 | - | 7, 8 |
| API | Contract v1 (OpenAPI snapshot, error envelope, fixtures) and stub routes answering 501. No real handlers | merged | T-043 | #9 | 16 |
| Auth and tenancy | Database roles and row-level security on every tenant table (migration 0004, catalogue-tested) and an append-only, function-only audit log exist in the schema. Login, JWT, RBAC and per-request tenant context are not built | planned | T-023 | - | 4 |
| Web console | Console UI (chat with citation pills and model badge, ingest status, versions, traces). Nothing built | planned | T-042, T-029, T-030 | - | 15 |
| Evals | Harness, deterministic smoke run, `make eval`, case-level regression check, policy check, synthetic corpus. Golden set and thresholds are still being promoted | merged | T-005, T-007 | #10, #11 | 11, 12, 13 |
| Observability | Structured logging and a metrics endpoint skeleton. Traces table exists in the schema; trace writer, dashboards and alert rules are not built | planned | T-020, T-035 | - | 9, 17 |
| CI | Lint, type check, tests, security scan and PR hygiene workflows run on pull requests (merged via earlier PRs); the eval gate workflow exists. Pending: branch protection and the remaining human CI setup | in progress | T-006 | #2, #5 | 11 |
| Deploy | Local Docker Compose stack only (profiles `core`, `obs`, `llm`). No hosted deploy, release candidate or seed demo yet (later tickets) | merged | T-004 | #8 | 17 |

## 2. Tickets

One row per ticket. Generated from the local ticket folder and the pull request list when both are
available; otherwise the block is left as committed.

<!-- generated:tickets begin -->
| Ticket | Title | Phase | State | Depends on | PR |
| --- | --- | --- | --- | --- | --- |
| T-001 | Repo scaffold, tooling, settings, logging and metrics skeleton | P1 | merged | - | [#4](https://github.com/arnavk2319/prod-rag-tool/pull/4) |
| T-002 | Ports, in-memory fakes, error model, import-boundary test | P1 | merged | T-001 | [#6](https://github.com/arnavk2319/prod-rag-tool/pull/6) |
| T-003 | Alembic 0001 DDL (schema contract) | P1 | merged | T-001 | [#7](https://github.com/arnavk2319/prod-rag-tool/pull/7) |
| T-004 | Docker image, Compose profiles, make up/doctor, memory measurement | P1 | merged | T-001, T-003 | [#8](https://github.com/arnavk2319/prod-rag-tool/pull/8) |
| T-005 | Eval harness skeleton, make eval, policy_check, case-level diff | P1 | merged | T-001 | [#10](https://github.com/arnavk2319/prod-rag-tool/pull/10) |
| T-006 | HUMAN: CI workflows, pre-push hook, CODEOWNERS, branch protection | P1 | planned | T-005 | - |
| T-007 | Synthetic corpus and ingest fixtures | P2 | merged | T-001 | [#11](https://github.com/arnavk2319/prod-rag-tool/pull/11) |
| T-008 | Parse, PII redaction, quarantine | P2 | merged | T-002, T-007 | [#14](https://github.com/arnavk2319/prod-rag-tool/pull/14) |
| T-009 | Chunker and embedder adapters | P2 | merged | T-002 | [#13](https://github.com/arnavk2319/prod-rag-tool/pull/13) |
| T-010 | Ingest pipeline: raw_objects store, worker, idempotency, copy-forward staging, delete and re-index | P2 | in progress | T-003, T-008, T-009, T-044, T-045, T-047 | - |
| T-011 | Index version gate, atomic alias swap, rollback, retention | P2 | planned | T-010, T-045 | - |
| T-012 | Hybrid retrieval (one SQL round trip) | P2 | planned | T-009, T-011, T-044 | - |
| T-013 | Tenant, ACL and RLS leak tests with hard-zero metrics | P2 | planned | T-012, T-044 | - |
| T-014 | Retrieval and ingest evals, smoke.yaml proposed, thresholds proposed | P2 | planned | T-005, T-012, T-013 | - |
| T-015 | HUMAN: Gate A2, promote golden set and thresholds | P2 | planned | T-014 | - |
| T-016 | LLM wrapper: OpenAICompatProvider, models.yaml, breaker, mock mode | P3 | planned | T-002 | - |
| T-017 | Verifier and fake-extractive provider | P3 | planned | T-002 | - |
| T-018 | Prompts, per-model profiles, query rewrite, input and output guardrails | P3 | planned | T-017 | - |
| T-019 | Generation service: fallback chain, repair retry, abstention, evidence-only | P3 | planned | T-012, T-016, T-017, T-018 | - |
| T-020 | request_traces writer and LLM metrics | P3 | planned | T-003, T-019 | - |
| T-021 | Answer and safety evals on the fake provider | P3 | planned | T-005, T-015, T-019, T-020 | - |
| T-022 | First real per-model run and ADR locking the model lineup | P3 | planned | T-004, T-016, T-019, T-021 | - |
| T-023 | Auth, tenancy, RBAC, seeded users, production boot guards | P4 | planned | T-002, T-003 | - |
| T-024 | Chat API and trace endpoint | P4 | planned | T-019, T-020, T-023, T-043 | - |
| T-025 | Cache, rate limits, daily token cap, idempotency | P4 | planned | T-024 | - |
| T-026 | Upload and ingest-status API | P4 | planned | T-010, T-023, T-043 | - |
| T-027 | Admin API: versions, commit, rollback, flags, hash-chained audit log, system health | P4 | planned | T-011, T-023, T-043 | - |
| T-028 | MLflow integration: eval run logging, prompt registration, registry_sync | P4 | planned | T-005, T-018, T-027 | - |
| T-029 | Console UI part 1: login, Chat (citation pills, model badge), Source/Trace drawer, Documents with upload and quarantine | P4 | planned | T-024, T-026, T-042 | - |
| T-030 | Console UI part 2: Index versions, Traces and System tabs | P4 | planned | T-027, T-029, T-042 | - |
| T-031 | full.yaml proposed and per-model full runner | P5 | planned | T-015, T-021, T-022 | - |
| T-032 | Red-team suite and threat model | P5 | planned | T-024, T-025, T-026 | - |
| T-033 | Chaos tests and failure matrix | P5 | planned | T-025, T-028 | - |
| T-034 | Performance run, SLOs and capacity model | P5 | planned | T-022, T-025 | - |
| T-035 | Grafana dashboard, Prometheus rules, metric contract and log scan | P5 | planned | T-020, T-004 | - |
| T-036 | policy_check hardening and claims-check script | P5 | planned | T-005, T-032, T-033, T-034 | - |
| T-037 | HUMAN: promote full.yaml and per-model thresholds, wire redteam and claims-check CI, disable failing models | P5 | planned | T-031, T-032, T-036 | - |
| T-038 | README, honesty table and evidence table as review document | P6 | planned | T-034, T-035, T-036 | - |
| T-039 | make seed, demo, llm-pull and the clean-clone test | P6 | planned | T-004, T-029, T-030 | - |
| T-040 | Release candidate: model card, data policy, DEFEND_IT-lite, hostile review, RC report | P6 | planned | T-037, T-038, T-039 | - |
| T-041 | HUMAN: Gate B and D, release tag | P6 | planned | T-040 | - |
| T-042 | Web foundation: Vite/React/TS/Tailwind scaffold, Console tokens, static Console prototype, shell, generated client, MSW mocks | P4 | planned | T-001, T-043 | - |
| T-043 | OpenAPI contract v1: every endpoint, error model, fixtures, snapshot and contract test | P1 | merged | T-001, T-002 | [#9](https://github.com/arnavk2319/prod-rag-tool/pull/9) |
| T-044 | Parent-child chunking: children 300 tokens embedded, parents 1200 tokens returned | P2 | merged | T-003, T-009 | [#18](https://github.com/arnavk2319/prod-rag-tool/pull/18) |
| T-045 | Pipeline policy gate: declare, verify, record in Postgres, mirror to MLflow | P2 | merged | T-003, T-009 | [#17](https://github.com/arnavk2319/prod-rag-tool/pull/17) |
| T-046 | Public status document and generator: what is planned, built and decided | P2 | merged | T-001 | [#16](https://github.com/arnavk2319/prod-rag-tool/pull/16) |
| T-047 | Row-level security on every remaining tenant-scoped table | P2 | merged | T-003, T-045 | [#22](https://github.com/arnavk2319/prod-rag-tool/pull/22) |
<!-- generated:tickets end -->

## 3. Currently built

Generated from code and config, so it cannot drift from the repository.

<!-- generated:built begin -->
**Models and retrieval pieces (from code)**

| Item | Value |
| --- | --- |
| Chunking strategy | single-level `None`: None-None tokens (target None), overlap ratio 0.12 |
| Embedding model | `BAAI/bge-small-en-v1.5`, 384 dims; backend default `hash` (`hash` = deterministic test embedder) |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` adapter exists; not yet wired into any retrieval path |
| Provider default | `fake` (no real model is called by default) |
| LLM fallback chain | not built: `config/models.yaml` does not exist yet |
| Pipeline policy gate | code present |
| Vector store and index | Postgres with pgvector (`pgvector/pgvector:0.8.0-pg16`); tables `index_versions`, `index_alias`, `chunks` exist in the schema; the per-partition HNSW index DDL exists, but no index is populated or queried yet |
| Registry | `registry_pointers` table exists in the schema; nothing writes it yet |
| Database roles | `rag_app`, `rag_owner`, `rag_svc` |
| Highest migration revision | `0004` |

**Settings defaults** (`src/rag_platform/settings.py`, secrets and credentialed URLs omitted)

| Setting | Default |
| --- | --- |
| `app_env` | `local` |
| `mlflow_tracking_uri` | `http://mlflow:5000` |
| `provider` | `fake` |
| `models_file` | `config/models.yaml` |
| `ollama_base_url` | `http://ollama:11434/v1` |
| `user_daily_token_cap` | `200000` |
| `embedding_backend` | `hash` |
| `cache_ttl_s` | `300` |
| `rate_limit_user_rpm` | `30` |
| `rate_limit_tenant_rpm` | `300` |
| `context_token_budget` | `4000` |
| `rerank_min_score` | `0.0` |
| `log_level` | `INFO` |
| `public_base_url` | `http://localhost:8000` |
| `allowed_origins` | `http://localhost:5173` |

**Compose services, images and host ports** (`compose.yaml`)

| Service | Image | Host port (default) |
| --- | --- | --- |
| api | `built from this repo` | 8000 (env API_PORT) |
| postgres | `pgvector/pgvector:0.8.0-pg16` | 5432 (env POSTGRES_PORT) |
| redis | `redis:7.4.2-alpine` | - |
| mlflow | `built from this repo` | 5000 (env MLFLOW_PORT) |
| ollama | `ollama/ollama:0.12.6` | 11434 (env OLLAMA_PORT) |
| ollama-pull | `ollama/ollama:0.12.6` | - |
| prometheus | `prom/prometheus:v3.5.0` | 9090 (env PROMETHEUS_PORT) |
| grafana | `grafana/grafana:12.2.0` | 3000 (env GRAFANA_PORT) |

**Database tables** (all migrations): `documents`, `raw_objects`, `ingest_jobs`, `quarantine_items`, `index_versions`, `chunks`, `index_alias`, `chat_sessions`, `chat_messages`, `control_flags`, `registry_pointers`, `golden_sets`, `gate_runs`, `audit_log`, `role_labels`, `request_traces`, `parent_chunks`, `pipeline_policies`, `policy_documents`

**Pinned dependencies** (`pyproject.toml`)

Runtime: `alembic==1.20.1`, `argon2-cffi==25.1.0`, `beautifulsoup4==4.15.0`, `fastapi==0.165.0`, `fastembed==0.9.1`, `httpx==0.29.0`, `mlflow==3.18.0`, `prometheus-client==0.26.0`, `psycopg[binary]==3.3.9`, `pydantic==2.14.4`, `pydantic-settings==2.16.3`, `pyjwt==2.15.1`, `pypdf==7.2.0`, `rapidfuzz==3.14.6`, `redis==8.2.0`, `sqlalchemy==2.1.9`, `structlog==26.1.0`, `uvicorn==0.54.0`

Dev: `hypothesis==6.170.17`, `mypy==2.4.1`, `pytest==9.2.0`, `pytest-asyncio==1.5.0`, `ruff==0.17.6`
<!-- generated:built end -->

Registered architecture decision records (ADRs) in `docs/decisions/`:

<!-- generated:adrs begin -->
| File | Title |
| --- | --- |
| `ADR-0004-api-contract-and-versioning.md` | ADR-0004: API contract and versioning |
| `ADR-0005-parent-child-chunking.md` | ADR-0005: Parent-child chunking |
| `ADR-0006-pipeline-policy-gate.md` | ADR-0006 Pipeline policy gate |
<!-- generated:adrs end -->

## 4. Decisions

Each entry gives the choice and the reason. Entries marked planned are decided but not built yet.

1. **Lean first version.** Build a small, explainable core and defer everything else. Reason: the full
   long-form design needed far more work than a first version could cover, and a working core that can be
   explained is worth more than a wide half-built system. The core is never cut: tenant and ACL
   filtering in SQL, verifier plus abstention, alias swap with rollback, the eval gate and a
   per-request model trace.
2. **Stack.** Python 3.12, uv with a pinned lockfile, FastAPI and Pydantic v2, SQLAlchemy
   Core with psycopg 3, Alembic, Redis for cache and limits only, a modular monolith (one image, api
   and worker entrypoints) with ports and in-memory fakes. Reason: one image and few moving parts fit a
   16 GB laptop; ports let every piece be built and tested alone without any networked or paid model.
   No LangChain or LlamaIndex, no ORM, no Celery: direct code is smaller and does not fight
   tenant-in-SQL filtering or the verifier. Any new dependency needs its own ADR.
3. **Postgres with pgvector is the only system of record.** Vectors, full text, ACL labels,
   row-level security, traces, raw uploads and registry pointers live in one database. Reason: one
   engine means one transaction for the atomic alias swap and copy-forward versioning, and tenant
   filters share a transaction with the data. A separate vector database or Elasticsearch was rejected
   as an extra container with no shared transaction. The pinned image is pgvector 0.8 because HNSW
   iterative scan needs it.
4. **Access control: label-subset ACL plus row-level security.** A chunk is visible only when all its
   labels are in the user's labels (`acl_labels <@ user_labels`), and an empty label set denies. User
   labels are resolved server-side. Row-level security on tenant is a second layer, and the application
   role is neither table owner nor able to bypass it. Reason: an overlap test would leak
   multi-label documents; filters live inside the query, never after generation.
5. **Index versions: list-partitioned chunks, copy-forward, alias swap.** `chunks` is partitioned by
   index version. A new version copies the parent's chunks (minus deleted documents), adds new ones,
   builds indexes, passes a data gate, then one transaction moves the `live` alias. Rollback points the
   alias at the previous version; the last three versions are kept. Changing embedder or chunker is a
   full re-embed into a new version, and mixed embedding models are refused. Reason: atomic commit,
   instant rollback and no re-embedding of unchanged documents.
6. **Hybrid retrieval in one SQL round trip (planned).** A vector leg (HNSW, cosine,
   iterative scan, with a test that a tenant holding 1% of rows still gets full results) and a
   full-text leg, both with the same tenant and ACL predicate, fused by reciprocal rank fusion (k=60).
   Reason: one leg alone misses either paraphrases or exact identifiers; fusion needs no tuning; one
   round trip keeps retrieval fast. Cost is zero API spend.
7. **Cross-encoder rerank and abstention (planned).** About 20 fused candidates
   are reranked and the top 5 kept. A score floor chosen on a tuning split only decides when to abstain
   without calling any model. Reason: reranking is the main precision lever and costs tens of
   milliseconds on CPU; abstaining below the floor saves tokens and avoids guessing. The floor is never
   tuned on the evaluation split.
8. **Generation chain with no paid models (planned).** An ordered chain of three small local
   models served by Ollama, then a deterministic extractive tier that needs no LLM. A schema failure
   gets one repair retry on the same model; a verification failure moves to the next model; the end of
   the chain always yields a verified, cited answer from the extractive tier. Per-model timeouts and a
   circuit breaker bound latency. Ollama is opt-in and the default stack uses a fake provider with zero
   downloads. Reason: no one should have to pay for reviewers' or CI usage; the last tier guarantees a
   cited answer; an ordered chain is simpler than a router. A per-user daily token cap replaces a spend
   cap. Streaming is deferred because streamed tokens would be unverified.
9. **Every run records which model answered.** One trace row per request with served model, attempts,
   fallback reasons, cache result and per-stage latency, and no question or answer text. Reason: honest
   reporting of how often the fallback tier answers.
10. **MLflow for prompts and eval runs.** MLflow holds eval runs and prompt versions with a `champion`
    alias; serving reads only Postgres pointers (cached, with last-known-good), and a sync step refuses
    a version without a passing gate run. Reason: MLflow is wanted for tracking, but serving must not
    depend on it. It runs with its own artifact storage; CI uses a file store.
11. **PR eval gate.** `make eval` writes the latest results; a local hook blocks stale or failing
    metrics and CI recomputes them, compares with the integration branch and requires an exact match.
    A regression is case-level: a case that passed and now fails blocks the change unless it is listed
    as an approved known flip. Reason: one case is almost 2% of a small set, so metric tolerances are
    noise, and the deterministic fake provider makes an exact match possible.
12. **Golden sets: smoke and full.** `smoke` (about 50-80 cases) gates every PR; `full` (about
    150-200 cases) runs nightly, for releases and per model, with a tune/gate split so tuning never
    grades itself. Reason: small sets are noisy and the larger set backs release claims.
13. **Eval honesty.** The fake provider never reads golden expectations, the verifier score is called
    lexical groundedness, checks are independent, answers get a human review before a release, and PR
    reports are labelled as mechanism proof, not model quality. Reason: avoid circular, rigged-looking
    evals.
14. **Postgres stores raw uploads.** Raw and quarantined files are kept as `bytea` behind an object
    store port, and the S3-compatible container was dropped. Reason: one fewer container and files
    commit transactionally with their rows; this suits 20 MB uploads and a small corpus. Revisit for
    real volumes.
15. **Console UI direction (planned).** A Vite, React and TypeScript console (chat with citation pills
    and model badge, ingest status, index versions, traces, system) built into the api image, driven
    by the versioned OpenAPI contract (ADR-0004). Reason: reviewers should see citations, the serving
    model and index state; the contract is agreed before the UI so shapes do not drift.
16. **API contract first (ADR-0004).** A checked-in OpenAPI snapshot is the contract and a test diffs
    the live app against it; routes live under `/v1`; one error envelope; idempotency keys and
    cursor pagination are specified; stub routes return 501, never fake data. Reason: UI and backend
    are built at different times and must not drift silently.
17. **Local Docker, profiles, no hosted environment.** Compose profiles `core` (default), `obs` and
    `llm`; the api must start with the optional ones down; production-mode boot refuses the fake
    provider and default passwords. Reason: a 16 GB laptop target and a reviewer who installs only
    Docker. Alerting, tracing collectors and a model router are deferred.
18. **Repository policy: ADRs and technical docs are public, planning material is private.** Decision
    records, this page and technical docs are committed; the author's planning and private working
    files stay on the author's machine and are never committed. Reason: the repository should show
    the system and its evidence, not the planning process.
19. **Parent-child chunking.** Children of about 300 tokens are embedded and searched; the
    parent of about 1200 tokens is returned; only children are embedded; at most 3 parents are packed
    into a context budget of about 4000 tokens. Reason: small chunks match precisely while larger
    parents give the generator context, and children stay under the 512-wordpiece embedder limit. The
    cost is a larger prompt for small local models, so sizes and parent count are settings.
20. **Pipeline policy gate (core merged, SQL and wiring pending).** Every index version has a policy row (reason, change kind,
    embedder, chunker, redaction, reranker, prompt and generator versions, documents touched, run and
    eval ids, hash) declared before embedding and verified after. Postgres is authoritative and the
    MLflow registry is a best-effort mirror through an outbox. Reason: every index change is deliberate
    and auditable, and ingest and serving never depend on MLflow.
21. **Public status page.** This page maps components, tickets, built configuration and decisions,
    with generated blocks and a staleness check in CI. Reason: reviewers can see built versus planned
    without any private material; the decision text is written here rather than linked.

## 5. Evals

Latest committed results against `evals/thresholds.json`. These prove the mechanism on a tiny
synthetic set with a fake provider, not model quality.

<!-- generated:evals begin -->
Suite `smoke`, served model `fake-extractive`. mechanism proof, not model quality (provider: fake-extractive).

| Metric | Value | Threshold | Result |
| --- | --- | --- | --- |
| chunk_determinism | 1.0 | none set | n/a |
| injection_block_rate | 1.0 | >= 0.9 | pass |
| mrr | 1.0 | >= 0.6 | pass |
| pii_recall | 1.0 | none set | n/a |
| recall_at_5 | 1.0 | >= 0.8 | pass |

Hard-zero counters: cross_tenant_leakage = 0, orphan_children = 0, pii_leak = 0, policy_violations = 0.

| Class | Cases | Passed |
| --- | --- | --- |
| answerable | 3 | 3 |
| injection | 1 | 1 |
| refusal | 1 | 1 |
| tenancy | 2 | 2 |
<!-- generated:evals end -->

## 6. Known gaps and accepted risks

- Retrieval, generation, verifier, guardrails, auth, the console and observability dashboards are not
  built. The API answers 501 for product routes.
- The committed eval numbers come from a handful of synthetic cases and the hash embedder with a
  deterministic fake provider. They say nothing about semantic quality or real model behaviour.
- Thresholds for answer faithfulness, relevance and citation correctness are set but no metric
  measures them yet.
- Chunker v1 splits by token windows, so a small table can be split across chunks; the bge query
  prefix is not applied; regex token counts only approximate wordpieces (a safety factor is logged).
- Person names are not redacted (structured identifiers only).
- Raw uploads in Postgres cap practical volume; small local models may fail verification often, in
  which case the extractive tier answers (reported as a fallback rate, not hidden).
- Index and model figures in older docs are estimates until measured on the real run.
