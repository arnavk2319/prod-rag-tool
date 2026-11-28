"""Small stdlib JSON Schema subset validator (no jsonschema dependency, see ADR-0004).

Supports $ref (local "#/..."), type, enum, const, properties, required, items,
anyOf, allOf, additionalProperties (bool or schema), minimum, maximum.
"""

from typing import Any

_TYPES: dict[str, tuple[type, ...]] = {
    "string": (str,),
    "integer": (int,),
    "number": (int, float),
    "boolean": (bool,),
    "array": (list,),
    "object": (dict,),
    "null": (type(None),),
}


def _resolve(ref: str, root: dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"unsupported $ref {ref}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part.replace("~1", "/").replace("~0", "~")]
    return node


def _type_ok(value: Any, name: str) -> bool:
    if name in ("integer", "number") and isinstance(value, bool):
        return False
    return isinstance(value, _TYPES[name])


def validate(
    value: Any, schema: dict[str, Any], root: dict[str, Any], path: str = "$"
) -> list[str]:
    """Return a list of error strings; empty means valid."""
    if "$ref" in schema:
        return validate(value, _resolve(schema["$ref"], root), root, path)
    errors: list[str] = []
    if "anyOf" in schema and not any(not validate(value, s, root, path) for s in schema["anyOf"]):
        errors.append(f"{path}: matches none of anyOf")
    for sub in schema.get("allOf", []):
        errors += validate(value, sub, root, path)
    if "type" in schema:
        types = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_type_ok(value, t) for t in types):
            return [*errors, f"{path}: expected {types}, got {type(value).__name__}"]
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: {value!r} not in enum")
    if "const" in schema and value != schema["const"]:
        errors.append(f"{path}: expected const {schema['const']!r}")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: below minimum")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: above maximum")
    if isinstance(value, dict):
        props: dict[str, Any] = schema.get("properties", {})
        errors += [
            f"{path}: missing required '{k}'" for k in schema.get("required", []) if k not in value
        ]
        extra = schema.get("additionalProperties", True)
        for key, item in value.items():
            if key in props:
                errors += validate(item, props[key], root, f"{path}.{key}")
            elif extra is False:
                errors.append(f"{path}: unexpected property '{key}'")
            elif isinstance(extra, dict):
                errors += validate(item, extra, root, f"{path}.{key}")
    if isinstance(value, list) and "items" in schema:
        for i, item in enumerate(value):
            errors += validate(item, schema["items"], root, f"{path}[{i}]")
    return errors
