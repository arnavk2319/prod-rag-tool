# AI design

One line of rationale per component. Pinned ids live in config, never at call sites.

| Component | Choice | Rationale |
| --- | --- | --- |
| Chunking | `pc-300-1200-v1` children: about 300 tokens (bounds 200-350; sizes from `CHILD_TOKENS`, regex word/punctuation approximation), 12% overlap inside one parent, split snapped to paragraph or sentence ends, heading path stored and prepended to the embedded text, chunk id `sha256(doc_hash + ordinal + text)` | Pure and deterministic, so re-ingest is idempotent and ids are stable; small children match precisely and stay well under the 512-wordpiece embedding limit. |
| Parent-child | Parents: about 1200 tokens (bounds 800-1500; `PARENT_TOKENS`), split on headings then paragraphs, no overlap, tiling the document; `parent_id = sha256(doc_hash + "p" + ordinal + text)`. Only children are embedded and searched; parents are returned (best child per parent, top `PARENT_TOP_N=3`, trimmed to `CONTEXT_TOKEN_BUDGET=4000`) | The generator gets a coherent passage instead of a fragment; the cost is a larger prompt (up to about 4000 context tokens), see ADR-0005. |
| Embedding | `BAAI/bge-small-en-v1.5` (384d, fastembed ONNX, batched, lazy load, cache dir via `FASTEMBED_CACHE_PATH`); `EMBEDDING_BACKEND=hash` selects the deterministic 384d `HashEmbedder` for tests and gates | Best quality per CPU-ms at small size and free; the hash embedder proves the mechanism offline but not semantic quality. |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed (the id fastembed 0.9.1 lists for this model) | Reranking about 20 candidates costs tens of ms on CPU and is the main precision lever. |
| Policy gate | Every index version has a declared, hashed policy (change kind, reason, embedder, chunker, redaction, reranker, prompt) stored in Postgres; transitions declared, verified, indexed, committed, rolled back or rejected run in one transaction with the `index_versions` row and a hash-chained audit row; commit needs a passing eval run for retrieval-affecting and prompt changes; MLflow is a best-effort mirror through an outbox (ADR-0006) | Makes every version explainable and blocks undeclared or mixed-model builds before embedding spend; Postgres stays authoritative so MLflow downtime never blocks ingest or serving. |
| Tenancy and audit | Row-level security with a `tenant_isolation` policy (USING and WITH CHECK on `tenant_id = current_setting('app.tenant_id', true)`) on every table with a `tenant_id` column, checked by a catalogue test; `audit_log` is append-only (trigger), tenant sessions read only their own rows, rows are added only through the `SECURITY DEFINER` function `audit_append` (migration 0004) | Database-level isolation holds even if a query forgets its filter; no tenant context means zero rows. |

## Row-level security: enabled, not forced

RLS is ENABLED on the tenant tables, not FORCED. The table owner (`rag_owner`) bypasses it. That is
intended for migrations and for the audit chain verifier, which reads every tenant's rows. The
application must never connect as the owner (or as a superuser): it connects as `rag_svc`, a member
of `rag_app`, which is subject to RLS (a test checks the compose `DATABASE_URL`). Each request sets
`app.tenant_id` with `set_config(..., true)` inside its transaction; without it every read returns zero
rows and every write is rejected.

Audit log: `rag_app` has no INSERT on `audit_log`. `audit_append` runs as the owner with a fixed
`search_path`, takes a two-integer advisory lock so the hash chain stays linear, refuses a tenant
argument that differs from the session's `app.tenant_id`, and records the session tenant when the
argument is NULL. A row with a NULL tenant (system event) is invisible to tenant sessions and can only
be written by a session with no tenant set. Verification runs as the owner role. A 5 s `lock_timeout` on the function stops a session that holds the chain lock from stalling appends forever. `audit_log` rejects UPDATE and DELETE (row trigger) and TRUNCATE (statement trigger); a superuser can still disable triggers.

Documented limits: a tenantless application session can write system (NULL tenant) audit rows, and `app.tenant_id` is chosen by the application session until it is bound to the authenticated identity (authentication work).

## Known gaps

- Tables are not kept whole: chunking is token-window based, so a small table can be split across chunks.
- The bge query prefix ("Represent this sentence for searching relevant passages: ") is not applied to queries yet.
- Wordpiece truncation: bge-small and the MiniLM reranker cut input silently at 512 wordpieces. Children now cap at 350 regex tokens (plus the heading path in `embed_text`); even at the 20-30% higher wordpiece ratio that is about 455 wordpieces, so no cut is expected, but this is an estimate and has not been measured with the real tokenizer (models are not cached offline in CI). Parents are never embedded. The fastembed adapters still log one `embed_input_may_truncate` warning per call (counts only, no text) when an estimate (1.3 x regex tokens) exceeds 512.
- The final chunk is rebalanced so it holds at least 10% new tokens when a window end within the child bounds (200-350) allows it; otherwise it may be mostly overlap.
- Fastembed env: `FASTEMBED_CACHE_PATH` (model cache dir) and `FASTEMBED_THREADS` (ONNX thread cap, positive integer; unset means fastembed default).
