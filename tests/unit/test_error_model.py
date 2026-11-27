import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_platform.api.errors import install_error_handlers
from rag_platform.domain import errors as e

CASES: list[tuple[type[e.AppError], int, str]] = [
    (e.ValidationFailed, 422, "validation_failed"),
    (e.Unauthenticated, 401, "unauthenticated"),
    (e.Forbidden, 403, "forbidden"),
    (e.NotFound, 404, "not_found"),
    (e.Conflict, 409, "conflict"),
    (e.PayloadTooLarge, 413, "payload_too_large"),
    (e.RateLimited, 429, "rate_limited"),
    (e.UpstreamUnavailable, 503, "upstream_unavailable"),
    (e.UpstreamTimeout, 504, "upstream_timeout"),
    (e.Internal, 500, "internal_error"),
]


def make_client(exc: Exception) -> TestClient:
    app = FastAPI()
    install_error_handlers(app)

    @app.get("/boom")
    async def boom() -> None:
        raise exc

    @app.get("/typed")
    async def typed(n: int) -> dict[str, int]:
        return {"n": n}

    return TestClient(app, raise_server_exceptions=False)


@pytest.mark.parametrize(("cls", "status", "code"), CASES)
def test_exception_maps_to_status_and_envelope(
    cls: type[e.AppError], status: int, code: str
) -> None:
    r = make_client(cls("msg")).get("/boom")
    assert r.status_code == status
    body = r.json()
    assert set(body) == {"error"}
    assert body["error"]["code"] == code
    assert body["error"]["message"] == "msg"
    assert "trace_id" in body["error"]


def test_unhandled_exception_hides_detail() -> None:
    r = make_client(RuntimeError("secret detail")).get("/boom")
    assert r.status_code == 500
    assert "secret" not in r.text
    assert r.json()["error"]["code"] == "internal_error"


def test_request_validation_uses_envelope() -> None:
    r = make_client(RuntimeError()).get("/typed?n=x")
    assert r.status_code == 422
    assert r.json()["error"]["code"] == "validation_failed"


def test_unknown_route_uses_envelope() -> None:
    r = make_client(RuntimeError()).get("/nope")
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


def test_unhandled_500_carries_trace_id_in_body_and_header() -> None:
    from rag_platform.api.app import create_app

    app = create_app()

    @app.get("/_boom")
    async def boom() -> None:
        raise RuntimeError("secret detail")

    r = TestClient(app, raise_server_exceptions=False).get("/_boom")
    assert r.status_code == 500
    tid = r.json()["error"]["trace_id"]
    assert tid
    assert r.headers["X-Trace-Id"] == tid
    assert "secret" not in r.text


def test_fake_index_rejects_dimension_mismatch() -> None:
    import asyncio

    from rag_platform.adapters.fakes import FakeIndexRepository
    from tests.contract.ports_contract import make_chunk

    async def run() -> None:
        repo = FakeIndexRepository()
        await repo.add_chunks([make_chunk("c", "t", {"a"})])
        await repo.search(
            (1.0,), tenant_id="t", user_labels=frozenset({"a"}), index_version="v1", k=1
        )

    with pytest.raises(ValueError):
        asyncio.run(run())
