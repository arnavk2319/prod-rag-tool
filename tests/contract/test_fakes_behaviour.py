"""Boundary behaviour of fakes plus proof the contract checks catch broken impls."""

import pytest

from rag_platform.adapters import fakes
from rag_platform.domain.ports import Chunk, ScoredChunk

from . import ports_contract as c


async def test_cache_expiry_boundary() -> None:
    clk = fakes.FakeClock()
    cache = fakes.FakeCache(clk)
    await cache.set("k", "v", 10)
    clk.advance(9.999)
    assert await cache.get("k") == "v"
    clk.advance(0.001)
    assert await cache.get("k") is None


async def test_rate_limiter_window_boundary() -> None:
    clk = fakes.FakeClock()
    rl = fakes.FakeRateLimiter(clk)
    assert await rl.allow("u", 1, 10)
    clk.advance(9)
    assert not await rl.allow("u", 1, 10)
    clk.advance(1)
    assert await rl.allow("u", 1, 10)


async def test_rate_limiter_denied_calls_do_not_extend_window() -> None:
    clk = fakes.FakeClock()
    rl = fakes.FakeRateLimiter(clk)
    assert await rl.allow("u", 1, 10)
    for _ in range(5):
        clk.advance(1)
        assert not await rl.allow("u", 1, 10)
    clk.advance(5)
    assert await rl.allow("u", 1, 10)


async def test_index_search_orders_and_limits() -> None:
    repo = fakes.FakeIndexRepository()
    await repo.add_chunks([c.make_chunk(f"c{i}", "t", {"a"}) for i in range(3)])
    hits = await repo.search(
        (1.0, 0.0), tenant_id="t", user_labels=frozenset({"a"}), index_version="v1", k=2
    )
    assert [h.chunk.chunk_id for h in hits] == ["c0", "c1"]


async def test_index_cross_tenant_never_returned() -> None:
    repo = fakes.FakeIndexRepository()
    await repo.add_chunks([c.make_chunk("x", "other", {"a"})])
    hits = await repo.search(
        (1.0, 0.0), tenant_id="t", user_labels=frozenset({"a", "b"}), index_version="v1", k=5
    )
    assert hits == []
    assert await repo.delete_document("t", "d1") == 0


async def test_audit_log_tenant_scoped_and_ordered() -> None:
    from datetime import UTC, datetime

    from rag_platform.domain.ports import AuditEvent

    log = fakes.FakeAuditLog()
    at = datetime(2026, 1, 1, tzinfo=UTC)
    evs = [AuditEvent(at, "u", f"a{i}", "t1") for i in range(3)]
    for ev in evs:
        await log.record(ev)
    assert await log.list_events("t1") == evs
    assert await log.list_events("none") == []


# --- contract checks must fail against deliberately broken fakes ---


class BrokenCache(fakes.FakeCache):
    async def get(self, key: str) -> str | None:  # never expires
        item = self._data.get(key)
        return item[0] if item else None


class BrokenRateLimiter(fakes.FakeRateLimiter):
    async def allow(self, key: str, limit: int, window_s: int) -> bool:
        return True


class LeakyStore(fakes.FakeObjectStore):
    async def get(self, tenant_id: str, sha256: str) -> bytes | None:
        for (_, sha), data in self._data.items():
            if sha == sha256:
                return data
        return None


class LeakyIndex(fakes.FakeIndexRepository):
    async def search(self, embedding, *, tenant_id, user_labels, index_version, k):  # type: ignore[no-untyped-def]
        return [ScoredChunk(ch, 1.0) for ch in self._chunks.values()][:k]


class NonDeterministicProvider(fakes.FakeModelProvider):
    n = 0

    async def complete(self, prompt: str, *, max_tokens: int):  # type: ignore[no-untyped-def]
        self.n += 1
        comp = await super().complete(prompt + str(self.n), max_tokens=max_tokens)
        return comp


class StickyRegistry(fakes.FakeRegistry):
    async def set_pointer(self, name: str, value: str) -> None:
        self._ptrs.setdefault(name, value)


async def test_contract_rejects_non_expiring_cache() -> None:
    clk = fakes.FakeClock()
    with pytest.raises(AssertionError):
        await c.check_cache(BrokenCache(clk), clk.advance)


async def test_contract_rejects_unlimited_rate_limiter() -> None:
    clk = fakes.FakeClock()
    with pytest.raises(AssertionError):
        await c.check_rate_limiter(BrokenRateLimiter(clk), clk.advance)


async def test_contract_rejects_cross_tenant_store() -> None:
    with pytest.raises(AssertionError):
        await c.check_object_store(LeakyStore())


async def test_contract_rejects_tenant_leaking_index() -> None:
    with pytest.raises(AssertionError):
        await c.check_index(LeakyIndex())


async def test_contract_rejects_nondeterministic_provider() -> None:
    with pytest.raises(AssertionError):
        await c.check_provider(NonDeterministicProvider())


async def test_contract_rejects_sticky_registry() -> None:
    with pytest.raises(AssertionError):
        await c.check_registry(StickyRegistry())


def test_chunk_type_is_importable() -> None:
    assert Chunk is not None
