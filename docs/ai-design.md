# AI design

One line of rationale per component. Pinned ids live in config, never at call sites.

| Component | Choice | Rationale |
| --- | --- | --- |
| Chunking | `chunker-v1`: 300-500 tokens (regex word/punctuation approximation), 12% overlap, split snapped to paragraph or sentence ends, heading path stored and prepended to the embedded text, chunk id `sha256(doc_hash + ordinal + text)` | Pure and deterministic, so re-ingest is idempotent and ids are stable; windows are large enough for context and small enough for precise citation. |
| Embedding | `BAAI/bge-small-en-v1.5` (384d, fastembed ONNX, batched, lazy load, cache dir via `FASTEMBED_CACHE_PATH`); `EMBEDDING_BACKEND=hash` selects the deterministic 384d `HashEmbedder` for tests and gates | Best quality per CPU-ms at small size and free; the hash embedder proves the mechanism offline but not semantic quality. |
| Reranker | `Xenova/ms-marco-MiniLM-L-6-v2` cross-encoder via fastembed (the id fastembed 0.9.1 lists for this model) | Reranking about 20 candidates costs tens of ms on CPU and is the main precision lever. |

## Known gaps

- Tables are not kept whole: chunking is token-window based, so a small table can be split across chunks.
- The bge query prefix ("Represent this sentence for searching relevant passages: ") is not applied to queries yet.
- Wordpiece truncation: bge-small and the MiniLM reranker truncate silently at 512 wordpieces. The chunker caps at 500 regex tokens (plus the heading path in `embed_text`), and the regex-to-wordpiece ratio can be 20-30% higher; it has never been measured with the real tokenizer (models are not cached offline in CI). The fastembed adapters log one `embed_input_may_truncate` warning per call (counts only, no text) when an estimate (1.3 x regex tokens) exceeds 512.
- The final chunk is rebalanced so it holds at least 10% new tokens when a window end in 300-500 allows it; otherwise it may be mostly overlap.
- Fastembed env: `FASTEMBED_CACHE_PATH` (model cache dir) and `FASTEMBED_THREADS` (ONNX thread cap, positive integer; unset means fastembed default).
