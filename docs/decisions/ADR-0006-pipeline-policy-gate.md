# ADR-0006 Pipeline policy gate
Status: accepted

## Context
An index version is expensive to build and hard to explain later. Without a durable record of why a version exists and which embedder, chunker, redaction rules and reranker produced it, mixed-model indexes and unreviewed retrieval changes slip through. The data gate (ADR-0002) checks the data after the build; nothing checks intent before it.

## Decision
**Policy.** A policy is an immutable declaration for one index version: `change_kind` (embedder, chunker, parent_child, redaction, reranker, prompt, corpus_only), a mandatory `change_reason`, embedder id/version/dim, chunker version and sizes, redaction rules version, reranker id, prompt version, generator chain ref, parent policy, ingest run id, the documents touched (doc id, doc version, content hash) and `policy_hash`. Only the lifecycle columns (status, per-transition timestamps, `eval_run_id`, MLflow columns) change after declaration. `eval_run_id` is attached at commit time, so it is not hashed and the hash stays stable.

**Hash.** `policy_hash` = sha256 of canonical JSON of the declared fields: UTF-8, sorted keys, `change_reason` normalised to Unicode NFC with edge whitespace stripped (inner whitespace stays significant), no whitespace, no floats (sizes are ints), documents sorted by (tenant_id, doc_id, doc_version), serialised as UTF-8 without ASCII escaping. Excluded: policy_id, status, timestamps, declared_by, eval_run_id, MLflow columns, hash itself. The hash changes when any declared field changes and is otherwise stable. It is stamped on `index_versions` and exposed in request traces as `versions.policy_hash`.

**State machine** (illegal transitions raise):
| Transition | Triggered by | index_versions.status |
| --- | --- | --- |
| (none) -> declared | ingest job at start, before any embedding | building |
| declared -> verified | job after embedding, `verify()` | building |
| verified -> indexed | job once staged chunks sit in the vector store partition | staged |
| indexed -> committed | commit gate after the data gate passes, on alias swap | committed |
| committed -> rolled_back | operator or automatic rollback (the `champion` alias returns to the previous committed version of the lineage, found through `parent_policy_id` and `committed_at`; with no known previous version the mirror reports not-synced and the outbox retries, it never deletes the alias) | rolled_back |
| declared/verified/indexed -> rejected | verify, data gate or operator refusal | rejected |

Each transition is one Postgres transaction that updates the policy row and the matching `index_versions` row together, plus the outbox flag. The repository port takes the expected policy status and expected index_versions status and raises `IllegalTransition` on any mismatch, so concurrent callers cannot both win. The audit event is written in the same unit of work (same transaction in Postgres): a failing audit write leaves the status unchanged. Commit requires the policy to be `indexed` (so previously `verified`); no verified policy means no commit.

**Rules.**
1. `change_kind` and `change_reason` are mandatory (non-empty after trim).
2. Embedder, chunker, parent_child, redaction or reranker changes force a full rebuild and forbid copy-forward. Only `corpus_only` may copy forward.
3. `verify()` compares every written chunk's `embedding_model`, `chunker_version` and vector dimension to the declared values and fails on any difference, on mixed models in one version, on zero rows, and (when the caller supplies the list) on declared documents with no written chunks.
4. A retrieval-affecting change (embedder, chunker, parent_child, redaction, reranker) and a prompt change need a linked eval run with no failed metric, or commit is refused. This follows the project rule that prompt changes re-run the evals; only `corpus_only` commits without one. Prompt changes may still copy forward.
5. Privacy: policy rows, audit events and mirror tags carry ids and hashes only, never document text or raw reasons containing content. Reasons are short operator text; tags carry a truncated reason.

**Source of truth.** Postgres is authoritative; serving and the commit gate read only Postgres. MLflow is a best-effort mirror through an outbox: each policy row has `mlflow_synced`; declare creates a version of registered model `rag-index-policy` with tags (policy_hash, change_kind, embedder, chunker_version, index_version, reason) and alias `candidate`; commit moves alias `champion`. Failures (server down, import error) set `mlflow_synced=false`, are retried by the worker, and never fail ingest, commit or serving. An outbox runner (`sync_pending`) retries unsynced rows and sets `mlflow_synced`.

**Relation to the data gate.** The policy gate checks intent and provenance (what was declared, what was written, who approved). The data gate checks the data (row parity, vector sanity, self-retrieval). Commit needs both.

**Audit.** declare, commit, rollback and reject write events through the `AuditLog` port with ids and hashes only. The commit event carries `eval_run_id`; reject carries a value from the closed `RejectReason` enum, never free text. The outbox replays unsynced rows ordered by `committed_at` then `declared_at`, and the registered MLflow version is persisted immediately so retries never register a second version.

## Schema plan
Alembic `0003` (down_revision `0002`, the parent_chunks migration) adds:
- `pipeline_policies` with CHECK constraints for change_kind and status, `change_reason text NOT NULL CHECK (btrim(change_reason) <> '')`, unique `policy_hash` per index_version.
- `policy_documents (policy_id, tenant_id, doc_id, doc_version, doc_hash)`, primary key on all but doc_hash, row-level security on `tenant_id` like `documents`.
- `index_versions.policy_id` FK to `pipeline_policies`.
- Grants for `rag_app`: SELECT, INSERT on both tables; UPDATE only on status, timestamp and MLflow columns of `pipeline_policies`; no DELETE; `policy_documents` insert/select only.
Requires the migration-approved label; downgrade drops in reverse order.

## Alternatives rejected
- MLflow as source of truth: couples ingest, commit and serving to an extra service and loses transactional consistency with `index_versions`.
- Gate only after embed: wastes the embedding spend on a change nobody declared or justified, and cannot refuse copy-forward early.
- Free-form text reasons without kinds: cannot be enforced, queried or used to decide rebuild versus copy-forward.

## Consequences
Cost is near zero (two small tables, one mirror call per transition). Every version has a recorded reason and policy. Trace contract gains one additive field. Reversible: tables can be dropped and the mirror disabled without touching serving.
