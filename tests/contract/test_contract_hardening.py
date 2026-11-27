"""Extra T-043 contract tests: mutation-sensitive checks, live stubs, runtime/contract drift."""

import copy
import json
import re
from typing import Any

import pytest
from fastapi.testclient import TestClient

from rag_platform.api.app import create_app
from rag_platform.api.errors import envelope
from rag_platform.domain import errors as domain_errors

from . import test_openapi_snapshot as snap
from .schema_validator import validate

SNAPSHOT = snap.SNAPSHOT
OPS = snap.operations()
ERROR_ENVELOPE = {"$ref": "#/components/schemas/ErrorEnvelope"}
STUB_OPS = [(m, p, o) for m, p, o in OPS if p not in {"/healthz", "/metrics"}]


def _fill(path: str) -> str:
    return re.sub(r"\{[^}]+\}", "abc12345", path)


def _client() -> TestClient:
    return TestClient(create_app(), raise_server_exceptions=False)


# --- snapshot structure -------------------------------------------------------------------


def test_operation_ids_match_snake_case_regex() -> None:
    for _, _, op in OPS:
        assert re.fullmatch(r"[a-z][a-z0-9]*(_[a-z0-9]+)*", op["operationId"]), op["operationId"]


def test_every_non_2xx_response_has_json_envelope_only() -> None:
    for method, path, op in OPS:
        for status, resp in op["responses"].items():
            if status.startswith("2"):
                continue
            assert list(resp["content"]) == ["application/json"], (method, path, status)
            assert resp["content"]["application/json"]["schema"] == ERROR_ENVELOPE


def test_429_retry_after_is_integer_header_on_every_operation() -> None:
    for _, _, op in OPS:
        hdr = op["responses"]["429"]["headers"]["Retry-After"]
        assert hdr["schema"]["type"] == "integer"


def test_security_only_where_required() -> None:
    for _, path, op in OPS:
        oid = op["operationId"]
        if oid in snap.NO_AUTH_OPS:
            assert not op.get("security"), path
        else:
            assert op["security"] == [{"BearerAuth": []}], path
    assert SNAPSHOT["paths"]["/v1/config"]["get"].get("security") is None
    assert SNAPSHOT["paths"]["/v1/auth/login"]["post"].get("security") is None
    assert not SNAPSHOT.get("security")  # no global default that would cover public paths


def test_idempotency_key_only_on_three_posts() -> None:
    with_key = {(m, p) for m, p, o in OPS if "Idempotency-Key" in snap.header_names(o)}
    assert with_key == {
        ("post", "/v1/documents"),
        ("post", "/v1/chats"),
        ("post", "/v1/chats/{id}/messages"),
    }


def test_idempotency_and_request_id_headers_are_optional_strings() -> None:
    for _, _, op in OPS:
        for p in op.get("parameters", []):
            if p["in"] == "header":
                assert p.get("required", False) is False, p["name"]
                types = [p["schema"].get("type")] + [
                    a.get("type") for a in p["schema"].get("anyOf", [])
                ]
                assert "string" in types, p["name"]


def test_pagination_params_only_on_list_collections() -> None:
    for m, p, op in OPS:
        names = {q["name"] for q in op.get("parameters", []) if q["in"] == "query"}
        if m == "get" and op["operationId"].startswith("list_"):
            assert {"limit", "cursor"} <= names, p
        else:
            assert not ({"limit", "cursor"} & names) or op["operationId"].startswith("list_"), p


def test_trace_list_has_limit_and_cursor() -> None:
    names = {q["name"] for q in SNAPSHOT["paths"]["/v1/traces"]["get"]["parameters"]}
    assert {"limit", "cursor"} <= names


def test_fixture_filenames_reference_real_responses_no_orphans() -> None:
    by_id = {o["operationId"]: o for _, _, o in OPS}
    for f in snap.FIXTURES.glob("*"):
        assert f.suffix == ".json", f.name
        oid, status, _ = f.name.split(".")
        assert oid in by_id, f"orphan fixture {f.name}"
        assert status in by_id[oid]["responses"], f"orphan status {f.name}"


def test_error_fixtures_use_matching_status_semantics() -> None:
    code_for = {"401": "unauthenticated", "404": "not_found", "422": "validation_failed",
                "503": "dependency_unavailable"}  # fmt: skip
    for f in snap.FIXTURES.glob("*.json"):
        status = f.name.split(".")[1]
        if status in code_for:
            body = json.loads(f.read_text())
            assert body["error"]["code"] == code_for[status], f.name


# --- fixture / validator mutation checks ---------------------------------------------------


def test_fixture_with_dropped_required_field_is_rejected() -> None:
    body = json.loads((snap.FIXTURES / "create_message.200.json").read_text())
    schema = SNAPSHOT["paths"]["/v1/chats/{id}/messages"]["post"]["responses"]["200"]["content"][
        "application/json"
    ]["schema"]
    assert validate(body, schema, SNAPSHOT) == []
    for key in ("content", "answer_status", "citations", "trace_id", "latency_ms"):
        broken = {k: v for k, v in body.items() if k != key}
        assert validate(broken, schema, SNAPSHOT), key
    assert validate({**body, "answer_status": "maybe"}, schema, SNAPSHOT)


def test_error_fixture_with_wrong_code_is_rejected() -> None:
    body = {"error": {"code": "teapot", "message": "x", "request_id": "r"}}
    assert validate(body, ERROR_ENVELOPE, SNAPSHOT)
    assert validate({"error": {"code": "not_found", "message": "x"}}, ERROR_ENVELOPE, SNAPSHOT)


ROOT: dict[str, Any] = {
    "components": {
        "schemas": {
            "Strict": {
                "type": "object",
                "required": ["a"],
                "additionalProperties": False,
                "properties": {"a": {"type": "string", "enum": ["x", "y"]}},
            }
        }
    }
}
STRICT = {"$ref": "#/components/schemas/Strict"}


def test_validator_negative_cases() -> None:
    assert validate({"a": "x"}, STRICT, ROOT) == []
    assert validate({"a": 1}, STRICT, ROOT)  # wrong type
    assert validate({}, STRICT, ROOT)  # missing required
    assert validate({"a": "z"}, STRICT, ROOT)  # bad enum
    assert any("unexpected" in e for e in validate({"a": "x", "b": 1}, STRICT, ROOT))


def test_validator_unresolved_ref_fails_loudly() -> None:
    with pytest.raises((KeyError, ValueError)):
        validate({}, {"$ref": "#/components/schemas/Nope"}, ROOT)
    with pytest.raises(ValueError):
        validate({}, {"$ref": "http://example.com/x.json"}, ROOT)


def test_validator_null_and_float_edges() -> None:
    assert validate(None, {"type": "string"}, ROOT)
    assert validate(1.5, {"type": "integer"}, ROOT)
    assert validate(5, {"type": "integer", "maximum": 4}, ROOT)
    assert validate("x", {"const": "y"}, ROOT)


# --- live snapshot diff --------------------------------------------------------------------


def _drift_message(mutated: dict[str, Any]) -> str:
    live = snap.canonical(mutated)
    stored = snap.SNAPSHOT_PATH.read_text()
    assert live != stored
    import difflib

    return "".join(difflib.unified_diff(stored.splitlines(True), live.splitlines(True), n=1))


def test_mutated_live_spec_is_detected_with_readable_diff() -> None:
    live = copy.deepcopy(create_app().openapi())
    assert snap.canonical(live) == snap.SNAPSHOT_PATH.read_text()
    del live["paths"]["/v1/me"]
    diff = _drift_message(live)
    assert "/v1/me" in diff and diff.startswith("---")


def test_each_mutation_kind_changes_canonical_form() -> None:
    base = snap.canonical(create_app().openapi())
    for mutate in (
        lambda d: d["paths"]["/v1/me"]["get"].update(operationId="whoami"),
        lambda d: d["paths"]["/v1/me"]["get"]["responses"].pop("404"),
        lambda d: d["paths"]["/v1/documents"]["post"].update(
            parameters=[
                p
                for p in d["paths"]["/v1/documents"]["post"]["parameters"]
                if p["name"] != "Idempotency-Key"
            ]
        ),  # fmt: skip
    ):
        doc = copy.deepcopy(create_app().openapi())
        mutate(doc)
        assert snap.canonical(doc) != base


def test_snapshot_docstring_documents_regeneration() -> None:
    assert "make contract-update" in (snap.test_live_matches_snapshot.__doc__ or "")


# --- live app ------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("method", "path"),
    [(m, p) for m, p, _ in STUB_OPS],
    ids=[f"{m}-{p}" for m, p, _ in STUB_OPS],
)
def test_stub_returns_501_with_envelope(method: str, path: str) -> None:
    resp = _client().request(method.upper(), _fill(path), json={} if method != "get" else None)
    # Body validation may legitimately reject the empty JSON before the stub runs.
    assert resp.status_code in (501, 422), (resp.status_code, resp.text)
    if resp.status_code == 501:
        body = resp.json()
        assert validate(body, ERROR_ENVELOPE, SNAPSHOT) == [], body
        assert body["error"]["request_id"] == resp.headers["X-Request-ID"]


def test_colon_path_stubs_are_routed() -> None:
    colon = [(m, p) for m, p, _ in STUB_OPS if ":" in p]
    assert colon, "expected audit:verify style path"
    for m, p in colon:
        assert _client().request(m.upper(), p).status_code == 501


def test_stub_echoes_valid_request_id() -> None:
    resp = _client().get("/v1/me", headers={"X-Request-ID": "req-12345678"})
    assert resp.status_code == 501
    assert resp.headers["X-Request-ID"] == "req-12345678"
    assert resp.json()["error"]["request_id"] == "req-12345678"


def test_probes() -> None:
    c = _client()
    r = c.get("/healthz")
    assert r.status_code == 200 and r.json() == {"status": "ok"}
    m = c.get("/metrics")
    assert m.status_code == 200 and m.headers["content-type"].startswith("text/plain")
    assert c.get("/readyz").status_code == 501


# --- runtime error handler vs contract  ---------------------


def test_runtime_envelope_has_request_id_key() -> None:
    err = domain_errors.NotFound("x")
    body = envelope(err)
    assert body["error"].keys() >= {"code", "message", "request_id"}, body


def test_runtime_error_bodies_validate_against_contract() -> None:
    c = _client()
    for resp in (c.get("/no/such/route"), c.get("/v1/documents?limit=0")):
        assert validate(resp.json(), ERROR_ENVELOPE, SNAPSHOT) == [], resp.text


def test_runtime_503_code_is_in_contract_enum() -> None:
    assert domain_errors.UpstreamUnavailable.code in snap.ERROR_CODES


def test_all_runtime_error_codes_are_in_contract_enum() -> None:
    codes = {
        c.code
        for c in vars(domain_errors).values()
        if isinstance(c, type) and issubclass(c, domain_errors.AppError)
    }
    assert codes <= snap.ERROR_CODES, sorted(codes - snap.ERROR_CODES)


def test_runtime_429_carries_retry_after_and_request_id_headers() -> None:
    from fastapi import FastAPI

    from rag_platform.api.errors import install_error_handlers

    app = FastAPI()
    install_error_handlers(app)

    @app.get("/limited")
    async def limited() -> None:
        raise domain_errors.RateLimited("slow down", retry_after=7)

    r = TestClient(app, raise_server_exceptions=False).get("/limited")
    assert r.status_code == 429 and r.headers["Retry-After"] == "7"
    assert validate(r.json(), ERROR_ENVELOPE, SNAPSHOT) == []
