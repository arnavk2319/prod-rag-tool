"""Embedder/Reranker probes with injected stub models; no network, no model downloads."""

import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_platform.adapters import embedders as emb
from rag_platform.adapters.embedders import (
    FastembedEmbedder,
    FastembedReranker,
    HashEmbedder,
    build_embedder,
)
from rag_platform.adapters.fakes import FakeEmbedder, FakeReranker
from rag_platform.domain.ports import ScoredChunk
from rag_platform.settings import Settings
from tests.contract import ports_contract as c

ROOT = Path(__file__).resolve().parents[2]


def norm(v: tuple[float, ...]) -> float:
    return math.sqrt(sum(x * x for x in v))


@pytest.mark.parametrize("e", [HashEmbedder(), FakeEmbedder()], ids=["hash", "fake"])
async def test_embedder_contract(e) -> None:
    await c.check_embedder(e)


@pytest.mark.parametrize("r", [FakeReranker()], ids=["fake"])
async def test_reranker_contract(r) -> None:
    await c.check_reranker(r)


async def test_hash_shape_norm_and_batch_equals_single() -> None:
    e = HashEmbedder()
    texts = ["alpha beta", "", "日本語 😀", "x " * 20000, "Alpha Beta"]
    batch = await e.embed(texts)
    for t, v in zip(texts, batch, strict=True):
        assert len(v) == 384 and abs(norm(v) - 1.0) < 1e-9
        assert (await e.embed([t]))[0] == v
    assert batch[0] == batch[4]  # case-insensitive by design
    assert await e.embed([]) == []


async def test_hash_empty_is_unit_e0_and_texts_differ() -> None:
    e = HashEmbedder()
    (z,) = await e.embed([""])
    assert z[0] == 1.0 and sum(abs(x) for x in z[1:]) == 0.0
    a, b = await e.embed(["red fox jumps", "quarterly revenue forecast"])
    assert a != b


async def test_hash_deterministic_across_processes_and_hashseed() -> None:
    code = (
        "import asyncio,hashlib;from rag_platform.adapters.embedders import HashEmbedder;"
        "v=asyncio.run(HashEmbedder().embed(['red fox 日本','']));"
        "print(hashlib.sha256(repr(v).encode()).hexdigest())"
    )
    outs = set()
    for seed in ("0", "7", "999"):
        r = subprocess.run(  # noqa: S603, ASYNC221
            [sys.executable, "-c", code],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[-400:]
        outs.add(r.stdout.strip())
    assert len(outs) == 1


def test_build_embedder_selection_without_loading(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*a, **k):
        raise AssertionError("model loaded")

    monkeypatch.setattr(FastembedEmbedder, "_load", boom)
    assert isinstance(build_embedder(Settings()), HashEmbedder)
    assert isinstance(build_embedder(Settings(embedding_backend="hash")), HashEmbedder)
    fe = build_embedder(Settings(embedding_backend="fastembed"))
    assert isinstance(fe, FastembedEmbedder) and fe.model_id == emb.FASTEMBED_EMBED_MODEL


def test_invalid_backend_rejected_clearly() -> None:
    with pytest.raises(ValueError, match="embedding_backend"):
        Settings(embedding_backend="openai")  # type: ignore[arg-type]


def test_invalid_backend_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("EMBEDDING_BACKEND", "fake")  # the stale value in .env.example
    with pytest.raises(ValueError, match="embedding_backend"):
        Settings()


class StubTE:
    instances: list["StubTE"] = []

    def __init__(self, name: str, cache_dir: str | None = None) -> None:
        self.name, self.cache_dir, self.calls = name, cache_dir, []
        StubTE.instances.append(self)

    def embed(self, texts, batch_size=256):
        self.calls.append((list(texts), batch_size))
        for t in texts:
            yield [float(len(t))] + [0.0] * 383


class StubCE:
    def __init__(self, name: str, cache_dir: str | None = None) -> None:
        self.name, self.cache_dir = name, cache_dir

    def rerank(self, query, docs):
        return [float(len(set(query.split()) & set(d.split()))) for d in docs]


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch):
    import fastembed
    from fastembed.rerank import cross_encoder

    StubTE.instances.clear()
    monkeypatch.setattr(fastembed, "TextEmbedding", StubTE)
    monkeypatch.setattr(cross_encoder, "TextCrossEncoder", StubCE)
    monkeypatch.delenv("FASTEMBED_CACHE_PATH", raising=False)


async def test_fastembed_lazy_load_batching_and_cache(
    stub_fastembed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTEMBED_CACHE_PATH", "fe-cache-test")
    e = FastembedEmbedder(batch_size=7)
    assert StubTE.instances == []  # constructing does not load
    assert await e.embed([]) == [] and StubTE.instances == []
    out = await e.embed(["a", "bb", "ccc"])
    assert [v[0] for v in out] == [1.0, 2.0, 3.0] and all(len(v) == 384 for v in out)
    (m,) = StubTE.instances
    assert m.name == "BAAI/bge-small-en-v1.5" and m.cache_dir == "fe-cache-test"
    assert m.calls == [(["a", "bb", "ccc"], 7)]
    await e.embed(["x"])
    assert len(StubTE.instances) == 1  # loaded once


async def test_fastembed_default_cache_is_none(stub_fastembed: None) -> None:
    await FastembedEmbedder().embed(["a"])
    assert StubTE.instances[0].cache_dir is None


async def test_fastembed_embed_preserves_order_for_many_texts(stub_fastembed: None) -> None:
    texts = ["t" * i for i in range(1, 80)]
    out = await FastembedEmbedder().embed(texts)
    assert [v[0] for v in out] == [float(i) for i in range(1, 80)]


@pytest.mark.xfail(
    strict=True, reason="known gap: bge query instruction prefix not applied to queries"
)
async def test_query_prefix_applied_for_queries(stub_fastembed: None) -> None:
    e = FastembedEmbedder()
    assert hasattr(e, "embed_query")


async def test_fastembed_missing_model_is_clean_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import fastembed

    def missing(*a, **k):
        raise ValueError("Could not load model")

    monkeypatch.setattr(fastembed, "TextEmbedding", missing)
    with pytest.raises(ValueError, match="Could not load"):
        await FastembedEmbedder().embed(["x"])


def scored(cid: str, text: str, score: float = 0.5) -> ScoredChunk:
    return ScoredChunk(c.make_chunk(cid, "t", {"a"}, text=text), score)


async def test_fastembed_reranker_order_ties_and_lazy(stub_fastembed: None) -> None:
    r = FastembedReranker()
    assert await r.rerank("q", []) == []
    cands = [scored("c9", "red"), scored("c1", "red fox"), scored("c5", "red"), scored("c0", "no")]
    out = await r.rerank("red fox", cands)
    assert [x.chunk.chunk_id for x in out] == ["c1", "c5", "c9", "c0"]  # ties by chunk_id
    assert [x.score for x in out] == [2.0, 1.0, 1.0, 0.0]
    again = await r.rerank("red fox", list(reversed(cands)))
    assert [x.chunk.chunk_id for x in again] == [x.chunk.chunk_id for x in out]


async def test_fastembed_reranker_passes_contract(stub_fastembed: None) -> None:
    await c.check_reranker(FastembedReranker())


async def test_reranker_length_mismatch_is_error(
    stub_fastembed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(StubCE, "rerank", lambda self, q, d: [1.0])
    with pytest.raises(ValueError):
        await FastembedReranker().rerank("q", [scored("a", "x"), scored("b", "y")])


async def test_load_once_under_concurrency_and_threads_env(
    stub_fastembed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    import asyncio
    import time

    import fastembed

    seen: list[dict] = []

    class Slow(StubTE):
        def __init__(self, name, cache_dir=None, threads=None):
            time.sleep(0.05)
            seen.append({"threads": threads})
            super().__init__(name, cache_dir)

    monkeypatch.setattr(fastembed, "TextEmbedding", Slow)
    monkeypatch.setenv("FASTEMBED_THREADS", "2")
    e = FastembedEmbedder()
    await asyncio.gather(*(e.embed(["a"]) for _ in range(8)))
    assert len(StubTE.instances) == 1 and seen == [{"threads": 2}]


async def test_invalid_threads_env_ignored(
    stub_fastembed: None, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FASTEMBED_THREADS", "abc")
    await FastembedEmbedder().embed(["a"])  # StubTE rejects a threads kwarg, so none was passed


async def test_truncation_warning_counts_only(
    stub_fastembed: None, capsys: pytest.CaptureFixture[str]
) -> None:
    from rag_platform.adapters.embedders import warn_if_truncation_risk

    secret = "canarysecret " * 450
    assert warn_if_truncation_risk("t", ["short", secret]) == 1
    assert warn_if_truncation_risk("t", ["short"]) == 0
    await FastembedEmbedder().embed([secret])
    out = capsys.readouterr()
    assert "canarysecret" not in out.out + out.err


async def test_reranker_scores_embed_text_when_present(stub_fastembed: None) -> None:
    class WithPath:
        def __init__(self, c):
            self.__dict__.update(c.__dict__)
            self.embed_text = "Policy > Limits\n" + c.text

    base = scored("c1", "fox").chunk
    sc = ScoredChunk(WithPath(base), 0.5)  # type: ignore[arg-type]
    out = await FastembedReranker().rerank("Limits fox", [sc])
    assert out[0].score == 2.0
