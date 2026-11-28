"""register_document probes: tenancy, status transitions, idempotency."""

import asyncio
import itertools
import json
from pathlib import Path

import evals.metrics_chunking as mc
import pytest

from rag_platform.adapters.fakes import FakeDocumentRepository
from rag_platform.domain.documents import Document
from rag_platform.services.ingest.register import content_hash, register_document

ROOT = Path(__file__).resolve().parents[2]


async def reg(repo, tenant="ta", h="h1", doc="d1", title="T"):
    return await register_document(repo, tenant_id=tenant, doc_hash=h, doc_id=doc, title=title)


async def test_created_then_duplicate_returns_same_row() -> None:
    repo = FakeDocumentRepository()
    a = await reg(repo)
    b = await reg(repo, doc="other-id", title="other")
    assert (a.outcome, b.outcome) == ("created", "duplicate")
    assert b.document == a.document and b.document.doc_id == "d1"


async def test_retired_reactivates_same_row_status_only() -> None:
    repo = FakeDocumentRepository()
    a = await reg(repo)
    await repo.set_status("ta", "d1", "retired")
    b = await reg(repo, doc="new-id", title="new title")
    assert b.outcome == "reactivated"
    assert b.document.doc_id == "d1" and b.document.title == "T"
    assert b.document.status == "active" and b.document.version == a.document.version
    assert len(repo._rows) == 1
    assert (await reg(repo)).outcome == "duplicate"  # active again


async def test_two_tenants_same_content_independent() -> None:
    repo = FakeDocumentRepository()
    a = await reg(repo, tenant="ta", doc="d1")
    b = await reg(repo, tenant="tb", doc="d1")  # even same doc_id: keys are tenant-scoped
    assert a.outcome == b.outcome == "created" and len(repo._rows) == 2
    await repo.set_status("ta", "d1", "retired")
    assert (await repo.get_by_hash("tb", "h1")).status == "active"


async def test_tenant_cannot_see_or_reactivate_other_tenants_row() -> None:
    repo = FakeDocumentRepository()
    await reg(repo, tenant="tb", doc="b1")
    await repo.set_status("tb", "b1", "retired")
    assert await repo.get_by_hash("ta", "h1") is None
    r = await reg(repo, tenant="ta", doc="a1")
    assert r.outcome == "created" and r.document.tenant_id == "ta"
    assert (await repo.get_by_hash("tb", "h1")).status == "retired"  # untouched


async def test_same_content_different_doc_ids_in_tenant_is_duplicate() -> None:
    repo = FakeDocumentRepository()
    await reg(repo, doc="a")
    assert (await reg(repo, doc="b")).outcome == "duplicate"
    assert await repo.get_by_hash("ta", "h1") is not None and len(repo._rows) == 1


async def test_different_content_same_doc_id_is_rejected_not_overwritten() -> None:
    repo = FakeDocumentRepository()
    await reg(repo, h="h1", doc="d1")
    with pytest.raises(ValueError):
        await reg(repo, h="h2", doc="d1")
    assert (await repo.get_by_hash("ta", "h1")).doc_hash == "h1"


async def test_repeated_calls_idempotent() -> None:
    repo = FakeDocumentRepository()
    outs = [(await reg(repo)).outcome for _ in range(5)]
    assert outs == ["created"] + ["duplicate"] * 4 and len(repo._rows) == 1


async def test_concurrent_double_call_one_created() -> None:
    repo = FakeDocumentRepository()
    res = await asyncio.gather(reg(repo, doc="a"), reg(repo, doc="b"), return_exceptions=True)
    assert all(not isinstance(r, Exception) for r in res)
    assert sorted(r.outcome for r in res) == ["created", "duplicate"]


async def test_fake_repo_enforces_unique() -> None:
    repo = FakeDocumentRepository()
    await repo.insert(Document("d1", "ta", "h", "t"))
    with pytest.raises(ValueError):
        await repo.insert(Document("d2", "ta", "h", "t"))
    with pytest.raises(KeyError):
        await repo.set_status("tb", "d1", "retired")


def test_content_hash_is_sha256_hex() -> None:
    assert content_hash(b"") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"


# ---- chunk_determinism metric ----


def test_metric_detects_nondeterminism(monkeypatch: pytest.MonkeyPatch) -> None:
    real = mc.chunk_document
    counter = itertools.count()

    def flaky(doc_hash: str, text: str):
        return real(doc_hash + str(next(counter)), text)

    monkeypatch.setattr(mc, "chunk_document", flaky)
    assert mc.chunk_determinism(ROOT) == 0.0


def test_metric_ignores_non_text_and_scores_one_on_tmp_corpus(tmp_path: Path) -> None:
    d = tmp_path / "data" / "corpus"
    d.mkdir(parents=True)
    (d / "a.md").write_text("# T\n\nhello world")
    (d / "b.pdf").write_bytes(b"%PDF-binary")
    assert mc.chunk_determinism(tmp_path) == 1.0
    (tmp_path / "data" / "corpus" / "a.md").unlink()
    assert mc.chunk_determinism(tmp_path) == 0.0  # only a pdf left: nothing chunked


def test_metric_tolerates_invalid_utf8(tmp_path: Path) -> None:
    d = tmp_path / "data" / "corpus"
    d.mkdir(parents=True)
    (d / "a.txt").write_bytes(b"ok \xff\xfe bytes")
    assert mc.chunk_determinism(tmp_path) == 1.0


def test_report_byte_identical_and_includes_chunk_determinism(tmp_path: Path) -> None:
    from evals.run import main

    outs = []
    for i in range(2):
        out = tmp_path / f"r{i}.json"
        assert main(["--suite", "smoke", "--out", str(out)]) == 0
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]
    assert json.loads(outs[0])["metrics"]["chunk_determinism"] == 1.0
