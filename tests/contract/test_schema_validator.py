from typing import Any

from .schema_validator import validate

ROOT: dict[str, Any] = {
    "components": {
        "schemas": {
            "Color": {"type": "string", "enum": ["red", "blue"]},
            "Box": {
                "type": "object",
                "required": ["color", "n"],
                "properties": {
                    "color": {"$ref": "#/components/schemas/Color"},
                    "n": {"type": "integer", "minimum": 1},
                    "tag": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "items": {"type": "array", "items": {"type": "number"}},
                    "meta": {"type": "object", "additionalProperties": {"type": "number"}},
                },
                "additionalProperties": False,
            },
        }
    }
}
BOX = {"$ref": "#/components/schemas/Box"}


def errs(value: Any) -> list[str]:
    return validate(value, BOX, ROOT)


def test_valid_document() -> None:
    assert errs({"color": "red", "n": 2, "tag": None, "items": [1, 2.5], "meta": {"a": 1}}) == []


def test_missing_required_and_enum() -> None:
    out = errs({"color": "green"})
    assert any("missing required 'n'" in e for e in out)
    assert any("not in enum" in e for e in out)


def test_type_mismatch_and_bool_is_not_integer() -> None:
    assert errs({"color": "red", "n": "x"})
    assert errs({"color": "red", "n": True})


def test_minimum_and_anyof_and_items() -> None:
    assert errs({"color": "red", "n": 0})
    assert errs({"color": "red", "n": 1, "tag": 5})
    assert errs({"color": "red", "n": 1, "items": ["a"]})


def test_additional_properties() -> None:
    assert errs({"color": "red", "n": 1, "zzz": 1})
    assert errs({"color": "red", "n": 1, "meta": {"a": "x"}})


def test_non_object_rejected() -> None:
    assert errs([])
