# ADR-0005: Parent-child chunking

Status: proposed

## Context

Single-level chunks of 300-500 tokens force a trade-off. Small chunks match a query precisely but give the generator too little surrounding text. Large chunks give context but blur the embedding and can exceed the embedding model's 512-wordpiece input limit, which silently truncates the tail.

## Decision

Two levels, split deterministically from the same document text.

- Children are about 300 tokens (bounds 200-350), with 12% overlap inside one parent, never crossing a parent boundary. Only children are embedded, indexed (vector and full-text) and searched.
- Parents are about 1200 tokens (bounds 800-1500), split on headings, then paragraphs, with no overlap. Together they reproduce the document text exactly. Parents are never embedded and never searched; they are returned.
- Sizes are settings `CHILD_TOKENS` (300) and `PARENT_TOKENS` (1200). They are part of `chunker_version`, spelled `pc-<child>-<parent>-v<n>`, for example `pc-300-1200-v1`. Changing a size changes the version and requires a re-index.
- `parent_id = sha256(doc_hash + "p" + parent_ordinal + parent_text)`. The child `chunk_id` formula is unchanged; each child carries its `parent_id`.

Why: small children keep match precision, parents give the generator a coherent passage, and 300-token children stay well under the 512-wordpiece limit, so nothing is truncated at embedding time.

## Schema

- New table `parent_chunks`, list-partitioned by `index_version` like `chunks`. Primary key `(index_version, tenant_id, parent_id)`. Columns: tenant_id, doc_id, doc_version, ordinal, text, token_count, acl_labels, heading_path, doc_hash, plus the version and chunker columns needed for copy-forward.
- New column `chunks.parent_id`, NOT NULL.
- Row-level security on `parent_chunks` uses the same tenant policy as `chunks`. The application role has no ownership and no BYPASSRLS, and partitions get no grants.
- The application role gets SELECT, INSERT and DELETE on `parent_chunks` (DELETE mirrors `chunks` and is needed for delete and re-index); parents are immutable, so there is no UPDATE.
- `CHILD_TOKENS`, `PARENT_TOKENS` and `PARENT_TOP_N` are read from the environment by a separate settings class (`chunk_settings`); `CONTEXT_TOKEN_BUDGET` stays in `Settings` and its default rises from 3000 to 4000.
- Copy-forward copies parents together with their children.

## Retrieval and security

- The parent is fetched in the same SQL statement as the child match. It carries the same tenant, ACL-subset, `as_of` and expiry predicates, and runs under RLS. Nothing is filtered after the query.
- The join key is `(index_version, tenant_id, parent_id)`. A child can only resolve to a parent in its own tenant and index version, so a forged or stale `parent_id` resolves to nothing and the child is dropped, not served.
- Collapse rule: keep the best-scoring child per parent, then at most `PARENT_TOP_N=3` parents, then trim to `CONTEXT_TOKEN_BUDGET=4000` tokens. The result carries the parent text plus the matching child ids and scores. Duplicate hits of one child merge (best score). If a parent exceeds the remaining budget it is cut to it and marked truncated, children outside the kept prefix are dropped, and the budget is then used up: when the first parent alone exceeds `CONTEXT_TOKEN_BUDGET`, it is truncated and later parents get nothing.

## Citations

The passage unit shown to the generator is the parent. A citation names the document, the parent id and the child ids inside it that matched. The verifier checks quoted spans against the parent text, so a quote may come from any part of the passage the model actually saw. Retrieval-time checks on tenant and ACL still apply to the parent.

## Migration plan (Alembic 0002)

1. Create `parent_chunks` (partitioned), its indexes (tenant, ACL GIN), RLS policy and grants.
2. Add `chunks.parent_id` as nullable, check that `chunks` is empty (the project has no production data yet), fail with a clear "re-index required" message otherwise, then set NOT NULL. There is no default value, because a fake parent would be a silent integrity bug.
3. Redefine `create_chunks_partition(v)` with CREATE OR REPLACE so it creates both the chunk and the parent partition (idempotent, IF NOT EXISTS), and call it for every existing index version so existing partitions gain a parent partition.
4. Downgrade drops `chunks.parent_id`, drops `parent_chunks`, and restores the 0001 function body.

This is a schema change and needs the migration-approved review.

## Consequences

- The generator prompt grows: up to 3 parents of about 1200 tokens is roughly 3600-4000 tokens of context, versus about 1500 before. Small local models pay in prompt-processing latency and may need a smaller `PARENT_TOP_N` or `PARENT_TOKENS`; both are settings. Quality and latency must be re-measured by the eval suite.
- Storage grows by one copy of the text per version (parents), and copy-forward copies it too. Acceptable at corpus size.
- Embedding cost is lower than embedding large chunks, since only children are embedded.
- Re-chunking is required whenever sizes change.

## Alternatives rejected

- Single-level chunks: either imprecise (large) or context-poor (small).
- Sentence-window retrieval: window is computed at read time from neighbours, citations and spans become unstable, and it needs ordering joins across chunks.
- Embedding parents as well: doubles embedding cost and index size, and long parents risk truncation at 512 wordpieces.
