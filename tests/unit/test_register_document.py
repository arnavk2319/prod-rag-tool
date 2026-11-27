from rag_platform.adapters.fakes import FakeDocumentRepository
from rag_platform.services.ingest.register import content_hash, register_document


async def test_reactivates_retired_instead_of_inserting() -> None:
    repo = FakeDocumentRepository()
    h = content_hash(b"policy text")
    first = await register_document(repo, tenant_id="a", doc_hash=h, doc_id="d1", title="P")
    assert first.outcome == "created"
    again = await register_document(repo, tenant_id="a", doc_hash=h, doc_id="d2", title="P")
    assert again.outcome == "duplicate" and again.document.doc_id == "d1"
    await repo.set_status("a", "d1", "retired")
    back = await register_document(repo, tenant_id="a", doc_hash=h, doc_id="d3", title="P")
    assert back.outcome == "reactivated"
    assert back.document.doc_id == "d1" and back.document.status == "active"
    assert len(repo._rows) == 1


async def test_same_content_two_tenants_independent_rows() -> None:
    repo = FakeDocumentRepository()
    h = content_hash(b"shared")
    a = await register_document(repo, tenant_id="a", doc_hash=h, doc_id="d1", title="P")
    b = await register_document(repo, tenant_id="b", doc_hash=h, doc_id="d1", title="P")
    assert a.outcome == b.outcome == "created"
    await repo.set_status("a", "d1", "retired")
    assert (await repo.get_by_hash("b", h)) is not None
    assert (await repo.get_by_hash("b", h)).status == "active"  # type: ignore[union-attr]
    again_b = await register_document(repo, tenant_id="b", doc_hash=h, doc_id="x", title="P")
    assert again_b.outcome == "duplicate"


async def test_fake_enforces_unique_hash() -> None:
    import pytest

    from rag_platform.domain.documents import Document

    repo = FakeDocumentRepository()
    await repo.insert(Document("d1", "a", "h", "t"))
    with pytest.raises(ValueError):
        await repo.insert(Document("d2", "a", "h", "t"))


async def test_register_atomic_under_threads() -> None:
    import asyncio
    from concurrent.futures import ThreadPoolExecutor

    repo = FakeDocumentRepository()

    def one(i: int) -> str:
        return asyncio.run(
            register_document(repo, tenant_id="a", doc_hash="h", doc_id=f"d{i}", title="t")
        ).outcome

    with ThreadPoolExecutor(32) as ex:
        outs = list(ex.map(one, range(200)))
    assert outs.count("created") == 1 and outs.count("duplicate") == 199
    assert len(repo._rows) == 1
    await repo.set_status("a", next(iter(repo._rows))[1], "retired")

    with ThreadPoolExecutor(32) as ex:
        outs = list(ex.map(one, range(200, 300)))
    assert outs.count("reactivated") == 1 and outs.count("duplicate") == 99
    assert len(repo._rows) == 1
