"""Embedder and Reranker adapters (no network at import; models load lazily on first use).

HashEmbedder is the deterministic gate embedder (feature hashing of lowercase word tokens).
Fastembed adapters read the model cache dir from FASTEMBED_CACHE_PATH (fastembed's own variable).
"""

import asyncio
import hashlib
import math
import os
import re
import threading
from collections.abc import Sequence
from dataclasses import replace
from typing import Any

import structlog

from rag_platform.domain.ports import Embedder, ScoredChunk
from rag_platform.settings import Settings

HASH_DIM = 384
FASTEMBED_EMBED_MODEL = "BAAI/bge-small-en-v1.5"
FASTEMBED_RERANK_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"
_WORD = re.compile(r"\w+")
_TOKEN = re.compile(r"\w+|[^\w\s]")
_log = structlog.get_logger(__name__)

# bge-small and MiniLM truncate silently at 512 wordpieces. The regex token count (chunker) is an
# unmeasured approximation; wordpieces are assumed up to 1.3x regex tokens (+heading path).
WORDPIECE_LIMIT = 512
WORDPIECE_FACTOR = 1.3


def warn_if_truncation_risk(component: str, texts: Sequence[str]) -> int:
    """Log one warning (counts only, never text) when inputs may exceed the wordpiece limit."""
    over = 0
    worst = 0
    for t in texts:
        est = int(len(_TOKEN.findall(t)) * WORDPIECE_FACTOR)
        worst = max(worst, est)
        over += est > WORDPIECE_LIMIT
    if over:
        _log.warning(
            "embed_input_may_truncate",
            component=component,
            over_limit=over,
            batch=len(texts),
            max_est_wordpieces=worst,
            limit=WORDPIECE_LIMIT,
        )
    return over


class HashEmbedder:
    """Deterministic 384d, L2-normalised bag-of-words hashing embedder. No network."""

    dim = HASH_DIM

    @property
    def model_id(self) -> str:
        return f"hash-{self.dim}-v1"

    def _one(self, text: str) -> tuple[float, ...]:
        vec = [0.0] * self.dim
        for w in _WORD.findall(text.lower()):
            d = hashlib.sha256(w.encode()).digest()
            vec[int.from_bytes(d[:4], "big") % self.dim] += 1.0 if d[4] & 1 else -1.0
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0.0:
            vec[0], norm = 1.0, 1.0
        return tuple(v / norm for v in vec)

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        return [self._one(t) for t in texts]


def _cache_dir() -> str | None:
    return os.environ.get("FASTEMBED_CACHE_PATH") or None


def _threads() -> int | None:
    """ONNX thread cap from FASTEMBED_THREADS; unset or invalid means fastembed default."""
    raw = os.environ.get("FASTEMBED_THREADS", "")
    return int(raw) if raw.isdigit() and int(raw) > 0 else None


def _model_kwargs() -> dict[str, Any]:
    kw: dict[str, Any] = {"cache_dir": _cache_dir()}
    if (t := _threads()) is not None:
        kw["threads"] = t
    return kw


class FastembedEmbedder:
    """BAAI/bge-small-en-v1.5 (384d) via fastembed ONNX; batched, lazy model load."""

    dim = 384

    def __init__(self, model_name: str = FASTEMBED_EMBED_MODEL, batch_size: int = 32) -> None:
        self._name = model_name
        self._batch = batch_size
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def model_id(self) -> str:
        return self._name

    def _load(self) -> Any:
        if self._model is None:
            with self._lock:
                if self._model is None:  # double-checked: load once under concurrency
                    from fastembed import TextEmbedding

                    self._model = TextEmbedding(self._name, **_model_kwargs())
        return self._model

    def _embed_sync(self, texts: list[str]) -> list[tuple[float, ...]]:
        model = self._load()
        return [tuple(float(x) for x in v) for v in model.embed(texts, batch_size=self._batch)]

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        if not texts:
            return []
        warn_if_truncation_risk("FastembedEmbedder", texts)
        return await asyncio.to_thread(self._embed_sync, list(texts))


class FastembedReranker:
    """Cross-encoder reranker (Xenova/ms-marco-MiniLM-L-6-v2) via fastembed; lazy load."""

    def __init__(self, model_name: str = FASTEMBED_RERANK_MODEL) -> None:
        self._name = model_name
        self._model: Any = None
        self._lock = threading.Lock()

    def _score_sync(self, query: str, docs: list[str]) -> list[float]:
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from fastembed.rerank.cross_encoder import TextCrossEncoder

                    self._model = TextCrossEncoder(self._name, **_model_kwargs())
        return [float(s) for s in self._model.rerank(query, docs)]

    async def rerank(self, query: str, candidates: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        if not candidates:
            return []
        # Score the same heading-prefixed text that was embedded when the chunk carries it.
        docs = [str(getattr(c.chunk, "embed_text", c.chunk.text)) for c in candidates]
        warn_if_truncation_risk("FastembedReranker", docs)
        scores = await asyncio.to_thread(self._score_sync, query, docs)
        scored = [replace(c, score=s) for c, s in zip(candidates, scores, strict=True)]
        return sorted(scored, key=lambda c: (-c.score, c.chunk.chunk_id))


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedding_backend == "fastembed":
        return FastembedEmbedder()
    return HashEmbedder()
