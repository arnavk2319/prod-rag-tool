import math

import pytest

from rag_platform.adapters.embedders import (
    FASTEMBED_EMBED_MODEL,
    FASTEMBED_RERANK_MODEL,
    FastembedEmbedder,
    FastembedReranker,
    HashEmbedder,
    build_embedder,
)
from rag_platform.settings import Settings
from tests.contract import ports_contract as c


async def test_hash_embedder_contract_and_shape() -> None:
    e = HashEmbedder()
    await c.check_embedder(e)
    v = (await e.embed(["Credit limits for cards", ""]))[0]
    assert len(v) == 384
    assert math.isclose(sum(x * x for x in v), 1.0, rel_tol=1e-9)
    empty = (await e.embed([""]))[0]
    assert math.isclose(sum(x * x for x in empty), 1.0, rel_tol=1e-9)


async def test_hash_embedder_lexical_similarity() -> None:
    e = HashEmbedder()
    q, near, far = await e.embed(["card limit", "the card limit is 5000", "mortgage tenor"])
    dot = lambda a, b: sum(x * y for x, y in zip(a, b, strict=True))  # noqa: E731
    assert dot(q, near) > dot(q, far)


def test_factory_selects_backend() -> None:
    assert isinstance(
        build_embedder(Settings(_env_file=None, embedding_backend="hash")), HashEmbedder
    )  # type: ignore[call-arg]
    fe = build_embedder(Settings(_env_file=None, embedding_backend="fastembed"))  # type: ignore[call-arg]
    assert isinstance(fe, FastembedEmbedder)
    assert fe.model_id == FASTEMBED_EMBED_MODEL
    assert fe._model is None  # lazy: nothing loaded at construction


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")


def _require_cached(kind: str, name: str) -> None:
    try:
        if kind == "embed":
            from fastembed import TextEmbedding

            TextEmbedding(name, local_files_only=True)
        else:
            from fastembed.rerank.cross_encoder import TextCrossEncoder

            TextCrossEncoder(name, local_files_only=True)
    except Exception as e:  # noqa: BLE001 - any failure means model unavailable offline
        pytest.skip(f"fastembed model {name} not cached ({type(e).__name__})")


@pytest.mark.model
async def test_fastembed_embedder_contract(offline: None) -> None:
    _require_cached("embed", FASTEMBED_EMBED_MODEL)
    e = FastembedEmbedder()
    await c.check_embedder(e)
    assert len((await e.embed(["x"]))[0]) == 384


@pytest.mark.model
async def test_fastembed_reranker_contract(offline: None) -> None:
    _require_cached("rerank", FASTEMBED_RERANK_MODEL)
    await c.check_reranker(FastembedReranker())
