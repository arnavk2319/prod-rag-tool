import pytest
import structlog
from fastapi import FastAPI
from fastapi.testclient import TestClient

from rag_platform.api.errors import install_error_handlers
from rag_platform.domain import errors as e

from .test_error_model import CASES, make_client


def test_unhandled_500_leaks_no_text_or_stack() -> None:
    r = make_client(RuntimeError("db password=hunter2 at /srv/app.py")).get("/boom")
    assert r.status_code == 500
    assert "hunter2" not in r.text
    assert "Traceback" not in r.text
    assert "RuntimeError" not in r.text
    assert "/srv/app.py" not in r.text
    assert set(r.json()["error"]) == {"code", "message", "trace_id"}


@pytest.mark.parametrize(("cls", "status", "code"), CASES)
def test_every_error_body_has_exact_keys(cls: type[e.AppError], status: int, code: str) -> None:
    r = make_client(cls("m")).get("/boom")
    assert set(r.json()["error"]) == {"code", "message", "trace_id"}
    assert isinstance(r.json()["error"]["trace_id"], str)


def test_trace_id_comes_from_context() -> None:
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(trace_id="abc123")
    try:
        app = FastAPI()
        install_error_handlers(app)

        @app.get("/boom")
        async def boom() -> None:
            raise e.Conflict("x")

        r = TestClient(app, raise_server_exceptions=False).get("/boom")
        assert r.json()["error"]["trace_id"] == "abc123"
    finally:
        structlog.contextvars.clear_contextvars()


def test_validation_and_404_have_trace_id_key() -> None:
    c = make_client(RuntimeError())
    assert "trace_id" in c.get("/typed?n=x").json()["error"]
    assert "trace_id" in c.get("/nope").json()["error"]


def test_non_404_http_error_uses_envelope() -> None:
    r = make_client(RuntimeError()).post("/boom")
    assert r.status_code == 405
    assert r.json()["error"]["code"] == "http_error"


def test_all_errors_are_app_errors() -> None:
    assert all(issubclass(cls, e.AppError) for cls, _, _ in CASES)
