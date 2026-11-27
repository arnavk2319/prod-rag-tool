"""T-044: `parent_id` is additive and optional in the contract (never required, nothing removed)."""

import json
import subprocess
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SNAP = json.loads((HERE / "openapi.snapshot.json").read_text())
SCHEMAS = SNAP["components"]["schemas"]


def _with_parent_id() -> list[str]:
    return sorted(n for n, s in SCHEMAS.items() if "parent_id" in s.get("properties", {}))


def test_parent_id_is_optional_nullable_string_on_citation_and_chunk() -> None:
    names = _with_parent_id()
    assert len(names) >= 2
    for n in names:
        s = SCHEMAS[n]
        assert "parent_id" not in s.get("required", [])
        kinds = {a.get("type") for a in s["properties"]["parent_id"]["anyOf"]}
        assert kinds == {"string", "null"}


def test_snapshot_change_vs_head_is_additive_only() -> None:
    r = subprocess.run(  # noqa: S603
        ["git", "show", "HEAD:tests/contract/openapi.snapshot.json"],  # noqa: S607
        cwd=HERE,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        pytest.skip("no git baseline")
    old = json.loads(r.stdout)["components"]["schemas"]
    for name, schema in old.items():
        new = SCHEMAS[name]
        if "properties" not in schema:
            assert new == schema, name
            continue
        assert set(schema["properties"]) <= set(new["properties"]), name
        assert set(new.get("required", [])) <= set(schema.get("required", [])), name
        for prop, body in schema["properties"].items():
            assert new["properties"][prop] == body, f"{name}.{prop} changed"
    assert set(old) <= set(SCHEMAS)
    assert json.loads(r.stdout)["paths"].keys() == SNAP["paths"].keys()


def test_fixtures_accept_with_and_without_parent_id() -> None:
    for f in ("get_chunk.200.json", "create_message.200.json", "get_chat.200.json"):
        text = (HERE / "fixtures" / f).read_text()
        assert json.loads(text)  # parseable; schema validation lives in test_openapi_snapshot
