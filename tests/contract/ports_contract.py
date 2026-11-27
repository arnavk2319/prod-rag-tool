"""Reusable contract checks per port; real adapters import and run these too."""

from datetime import UTC, datetime

from rag_platform.domain.ports import (
    AuditEvent,
    AuditLog,
    Cache,
    Chunk,
    Embedder,
    IndexRepository,
    ModelProvider,
    ObjectStore,
    RateLimiter,
    Registry,
    Reranker,
    ScoredChunk,
)


def make_chunk(cid: str, tenant: str, labels: set[str], text: str = "t", doc: str = "d1") -> Chunk:
    return Chunk(
        chunk_id=cid,
        doc_id=doc,
        tenant_id=tenant,
        text=text,
        embedding=(1.0, 0.0),
        acl_labels=frozenset(labels),
        index_version="v1",
        embedding_model="m",
        chunker_version="c1",
    )


async def check_object_store(s: ObjectStore) -> None:
    sha = await s.put("t1", b"hello", "text/plain")
    assert await s.put("t1", b"hello", "text/plain") == sha  # idempotent
    assert await s.get("t1", sha) == b"hello"
    assert await s.get("t2", sha) is None  # tenant isolation
    assert await s.delete("t1", sha) is True
    assert await s.delete("t1", sha) is False
    assert await s.get("t1", sha) is None


async def check_embedder(e: Embedder) -> None:
    assert e.model_id
    a, b = await e.embed(["x", "y"])
    assert len(a) == len(b) > 0
    assert (await e.embed(["x"]))[0] == a  # deterministic
    assert a != b


async def check_reranker(r: Reranker) -> None:
    cands = [
        ScoredChunk(make_chunk(f"c{i}", "t", {"a"}, text=t), 0.5)
        for i, t in enumerate(["no", "red fox"])
    ]
    out = await r.rerank("red fox", cands)
    assert {c.chunk.chunk_id for c in out} == {"c0", "c1"}
    assert out[0].chunk.chunk_id == "c1"


async def check_index(repo: IndexRepository) -> None:
    mine = make_chunk("c1", "t1", {"a"})
    await repo.add_chunks([mine])
    await repo.add_chunks([mine])  # idempotent
    other_tenant = make_chunk("c2", "t2", {"a"})
    restricted = make_chunk("c3", "t1", {"a", "secret"})
    unreadable = make_chunk("c4", "t1", set())
    await repo.add_chunks([other_tenant, restricted, unreadable])
    hits = await repo.search(
        (1.0, 0.0), tenant_id="t1", user_labels=frozenset({"a"}), index_version="v1", k=10
    )
    assert [h.chunk.chunk_id for h in hits] == ["c1"]  # tenant + subset ACL, empty = unreadable
    none = await repo.search(
        (1.0, 0.0), tenant_id="t1", user_labels=frozenset({"a"}), index_version="v2", k=10
    )
    assert none == []
    assert await repo.delete_document("t1", "d1") == 3
    assert await repo.delete_document("t1", "d1") == 0


async def check_provider(p: ModelProvider) -> None:
    c = await p.complete("hello world", max_tokens=16)
    assert c.text and c.tokens_in >= 0 and c.usd >= 0 and c.model
    assert (await p.complete("hello world", max_tokens=16)).text == c.text


async def check_cache(c: Cache, advance: "Advance") -> None:
    assert await c.get("k") is None
    await c.set("k", "v", 10)
    assert await c.get("k") == "v"
    advance(11)
    assert await c.get("k") is None
    await c.set("k", "v", 10)
    await c.delete("k")
    assert await c.get("k") is None


async def check_rate_limiter(r: RateLimiter, advance: "Advance") -> None:
    assert await r.allow("u", 2, 60)
    assert await r.allow("u", 2, 60)
    assert not await r.allow("u", 2, 60)
    assert await r.allow("other", 2, 60)
    advance(61)
    assert await r.allow("u", 2, 60)


async def check_registry(r: Registry) -> None:
    assert await r.get_pointer("index") is None
    await r.set_pointer("index", "v1")
    await r.set_pointer("index", "v2")
    assert await r.get_pointer("index") == "v2"


async def check_audit(a: AuditLog) -> None:
    ev = AuditEvent(datetime(2026, 1, 1, tzinfo=UTC), "u", "login", "t1", {"ok": True})
    await a.record(ev)
    await a.record(AuditEvent(ev.at, "u", "login", "t2"))
    assert await a.list_events("t1") == [ev]


from collections.abc import Callable  # noqa: E402

Advance = Callable[[float], None]
