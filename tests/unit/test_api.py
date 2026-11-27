import httpx
import pytest

from rag_platform.api.app import create_app


@pytest.fixture
async def client() -> httpx.AsyncClient:
    transport = httpx.ASGITransport(app=create_app())
    return httpx.AsyncClient(transport=transport, base_url="http://test")


async def test_healthz_and_trace_echo(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz", headers={"X-Trace-Id": "trace-12345"})
    assert r.status_code == 200
    assert r.headers["X-Trace-Id"] == "trace-12345"


async def test_trace_generated_when_missing_or_invalid(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz", headers={"X-Trace-Id": "bad id\t!"})
    assert r.headers["X-Trace-Id"] != "bad id\t!"
    assert len(r.headers["X-Trace-Id"]) == 32


async def test_metrics(client: httpx.AsyncClient) -> None:
    await client.get("/healthz")
    r = await client.get("/metrics")
    assert r.status_code == 200
    assert "http_requests_total" in r.text


async def test_trace_generated_when_absent(client: httpx.AsyncClient) -> None:
    a = await client.get("/healthz")
    b = await client.get("/healthz")
    assert len(a.headers["X-Trace-Id"]) == 32
    assert a.headers["X-Trace-Id"] != b.headers["X-Trace-Id"]


@pytest.mark.parametrize("bad", ["short", "x" * 65, "has space 123"])
async def test_trace_invalid_replaced(client: httpx.AsyncClient, bad: str) -> None:
    r = await client.get("/healthz", headers={"X-Trace-Id": bad})
    assert r.headers["X-Trace-Id"] != bad


async def test_healthz_body(client: httpx.AsyncClient) -> None:
    r = await client.get("/healthz")
    assert r.json() == {"status": "ok"}


async def test_metrics_names_and_content_type(client: httpx.AsyncClient) -> None:
    await client.get("/healthz")
    r = await client.get("/metrics")
    assert r.headers["content-type"].startswith("text/plain")
    assert "http_request_duration_seconds" in r.text
    assert 'path="/healthz"' in r.text


async def test_unknown_path_label_is_bounded(client: httpx.AsyncClient) -> None:
    r = await client.get("/nope-123")
    assert r.status_code == 404
    assert "X-Trace-Id" in r.headers
    assert "nope-123" not in (await client.get("/metrics")).text
