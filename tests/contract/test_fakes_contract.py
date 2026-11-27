from datetime import UTC, datetime

from rag_platform.adapters import fakes
from rag_platform.domain.ports import Clock

from . import ports_contract as c


async def test_clock() -> None:
    clk: Clock = fakes.FakeClock()
    assert clk.now() == datetime(2026, 1, 1, tzinfo=UTC)
    fakes.FakeClock().advance(5)


async def test_object_store() -> None:
    await c.check_object_store(fakes.FakeObjectStore())


async def test_embedder() -> None:
    await c.check_embedder(fakes.FakeEmbedder())


async def test_reranker() -> None:
    await c.check_reranker(fakes.FakeReranker())


async def test_index() -> None:
    await c.check_index(fakes.FakeIndexRepository())


async def test_provider() -> None:
    await c.check_provider(fakes.FakeModelProvider())


async def test_cache() -> None:
    clk = fakes.FakeClock()
    await c.check_cache(fakes.FakeCache(clk), clk.advance)


async def test_rate_limiter() -> None:
    clk = fakes.FakeClock()
    await c.check_rate_limiter(fakes.FakeRateLimiter(clk), clk.advance)


async def test_registry() -> None:
    await c.check_registry(fakes.FakeRegistry())


async def test_audit() -> None:
    await c.check_audit(fakes.FakeAuditLog())
