"""Contract tests: live OpenAPI vs the checked-in snapshot, plus fixtures (ADR-0004)."""

import difflib
import json
from pathlib import Path
from typing import Any

import pytest

from rag_platform.api.app import create_app

from .schema_validator import validate

HERE = Path(__file__).parent
SNAPSHOT_PATH = HERE / "openapi.snapshot.json"
FIXTURES = HERE / "fixtures"
HTTP_METHODS = {"get", "put", "post", "delete", "patch"}
ERROR_CODES = {
    "unauthenticated",
    "forbidden",
    "not_found",
    "validation_failed",
    "conflict",
    "rate_limited",
    "payload_too_large",
    "dependency_unavailable",
    "internal_error",
}
IDEMPOTENT_OPS = {"create_document", "create_chat", "create_message"}
NO_AUTH_OPS = {"get_config", "login", "healthz", "readyz", "metrics"}


def canonical(doc: dict[str, Any]) -> str:
    return json.dumps(doc, sort_keys=True, indent=2) + "\n"


SNAPSHOT: dict[str, Any] = json.loads(SNAPSHOT_PATH.read_text())


def operations() -> list[tuple[str, str, dict[str, Any]]]:
    return [
        (method, path, op)
        for path, item in SNAPSHOT["paths"].items()
        for method, op in item.items()
        if method in HTTP_METHODS
    ]


def op_id(op: dict[str, Any]) -> str:
    return str(op["operationId"])


def test_live_matches_snapshot() -> None:
    """The FastAPI app must produce exactly the checked-in OpenAPI snapshot.

    On a deliberate contract change run `make contract-update`, then regenerate the UI
    client and add an ADR note in the same PR (docs/api-contract.md).
    """
    live = canonical(create_app().openapi())
    snap = SNAPSHOT_PATH.read_text()
    diff = "".join(
        difflib.unified_diff(
            snap.splitlines(keepends=True),
            live.splitlines(keepends=True),
            "openapi.snapshot.json",
            "live app",
            n=2,
        )
    )
    assert live == snap, "OpenAPI drifted; run `make contract-update` if intended:\n" + diff[:4000]


def test_snapshot_is_canonical() -> None:
    assert SNAPSHOT_PATH.read_text() == canonical(SNAPSHOT)
    assert SNAPSHOT["openapi"].startswith("3.1")


def test_snapshot_covers_plan_endpoints() -> None:
    expected = {
        line.strip()
        for line in (HERE / "plan_endpoints.txt").read_text().splitlines()
        if line.strip()
    }
    actual = {f"{m.upper()} {p}" for m, p, _ in operations()}
    assert expected <= actual, f"missing from snapshot: {sorted(expected - actual)}"
    assert actual <= expected, f"not in plan_endpoints.txt: {sorted(actual - expected)}"


def test_operation_ids_unique_snake_case() -> None:
    ids = [op_id(op) for _, _, op in operations()]
    assert len(ids) == len(set(ids))
    assert all(i.islower() and " " not in i and "-" not in i for i in ids)


def test_versioning_rule_paths() -> None:
    for _, path, _ in operations():
        assert path in {"/healthz", "/readyz", "/metrics"} or path.startswith("/v1/")


def test_no_default_validation_error_component() -> None:
    assert "HTTPValidationError" not in SNAPSHOT["components"]["schemas"]
    assert {"ErrorEnvelope", "ErrorBody"} <= set(SNAPSHOT["components"]["schemas"])


def test_every_non_2xx_response_uses_error_envelope() -> None:
    for method, path, op in operations():
        errors = [s for s in op["responses"] if not s.startswith("2")]
        assert {"401", "403", "404", "409", "413", "422", "429", "503", "500"} <= set(errors)
        for status in errors:
            ref = op["responses"][status]["content"]["application/json"]["schema"]["$ref"]
            assert ref == "#/components/schemas/ErrorEnvelope", f"{method} {path} {status}"
        assert "Retry-After" in op["responses"]["429"]["headers"]


def test_error_codes_enum() -> None:
    body = SNAPSHOT["components"]["schemas"]["ErrorBody"]
    assert set(SNAPSHOT["components"]["schemas"]["ErrorCode"]["enum"]) == ERROR_CODES
    assert {"code", "message", "request_id"} <= set(body["required"])
    assert "details" in body["properties"] and "details" not in body["required"]


def header_names(op: dict[str, Any]) -> set[str]:
    return {p["name"] for p in op.get("parameters", []) if p["in"] == "header"}


def test_headers_declared_where_required() -> None:
    for _, _, op in operations():
        oid = op_id(op)
        assert "X-Request-ID" in header_names(op), oid
        assert ("Idempotency-Key" in header_names(op)) == (oid in IDEMPOTENT_OPS), oid
        assert ("security" in op) == (oid not in NO_AUTH_OPS), oid
        for status, resp in op["responses"].items():
            if status.startswith("2") or status == "429":
                assert "X-Request-ID" in resp["headers"], (oid, status)
    scheme = SNAPSHOT["components"]["securitySchemes"]["BearerAuth"]
    assert scheme == {"type": "http", "scheme": "bearer"}


def test_list_endpoints_paginate() -> None:
    lists = [(p, op) for m, p, op in operations() if m == "get" and op_id(op).startswith("list_")]
    assert len(lists) >= 6
    for _, op in lists:
        params = {p["name"]: p for p in op["parameters"] if p["in"] == "query"}
        limit = params["limit"]["schema"]
        assert (limit["default"], limit["minimum"], limit["maximum"]) == (50, 1, 200)
        assert "cursor" in params
        ref = op["responses"]["200"]["content"]["application/json"]["schema"]["$ref"]
        page = SNAPSHOT["components"]["schemas"][ref.rsplit("/", 1)[1]]
        assert set(page["required"]) == {"items", "next_cursor"}


def test_documents_status_filter() -> None:
    op = SNAPSHOT["paths"]["/v1/documents"]["get"]
    assert "status" in {p["name"] for p in op["parameters"]}


def test_message_and_trace_schema_fields() -> None:
    schemas = SNAPSHOT["components"]["schemas"]
    assert set(schemas["Message"]["required"]) >= {
        "content", "answer_status", "citations", "versions", "served_model",
        "cache", "trace_id", "latency_ms",
    }  # fmt: skip
    assert set(schemas["AnswerStatus"]["enum"]) == {"answered", "abstained", "evidence_only"}
    assert set(schemas["Citation"]["required"]) == {"document_id", "chunk_id", "span"}
    assert set(schemas["TraceDetail"]["required"]) >= {
        "attempts", "fallback_reasons", "stage_latency", "retrieved", "served_model",
    }  # fmt: skip
    assert set(schemas["RetrievedChunk"]["required"]) >= {"chunk_id", "score"}


def fixture_files() -> dict[str, list[Path]]:
    out: dict[str, list[Path]] = {}
    for f in sorted(FIXTURES.glob("*.json")):
        out.setdefault(f.name.split(".")[0], []).append(f)
    return out


def test_every_operation_has_success_and_error_fixture() -> None:
    files = fixture_files()
    assert set(files) == {op_id(op) for _, _, op in operations()}
    for oid, paths in files.items():
        statuses = [int(p.name.split(".")[1]) for p in paths]
        assert any(200 <= s < 300 for s in statuses), oid
        assert any(s >= 400 for s in statuses), oid


@pytest.mark.parametrize("fixture", sorted(FIXTURES.glob("*.json")), ids=lambda p: p.name)
def test_fixture_validates_against_snapshot(fixture: Path) -> None:
    oid, status, _ = fixture.name.split(".")
    op = next(op for _, _, op in operations() if op_id(op) == oid)
    content = op["responses"][status]["content"]
    media = "text/plain" if "text/plain" in content else "application/json"
    errors = validate(json.loads(fixture.read_text()), content[media]["schema"], SNAPSHOT)
    assert errors == [], errors
    if int(status) >= 400:
        body = json.loads(fixture.read_text())
        assert body["error"]["code"] in ERROR_CODES
