"""domain/ and services/ must not import adapters/ or api/ (ports and adapters)."""

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src" / "rag_platform"
FORBIDDEN = {
    "domain": ("rag_platform.adapters", "rag_platform.api", "rag_platform.worker"),
    "services": ("rag_platform.adapters", "rag_platform.api"),
}


def violations(root: Path, layer: str) -> list[str]:
    bad: list[str] = []
    for f in (root / layer).rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text())):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module] + [f"{node.module}.{a.name}" for a in node.names]
            bad += [f"{f.name}: {n}" for n in names if n.startswith(FORBIDDEN[layer])]
    return bad


def test_layers_respect_boundaries() -> None:
    for layer in FORBIDDEN:
        assert violations(SRC, layer) == []


def test_negative_case_detects_domain_importing_adapters(tmp_path: Path) -> None:
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "bad.py").write_text(
        "from rag_platform.adapters.fakes import FakeClock\n"
    )
    assert "bad.py: rag_platform.adapters.fakes" in violations(tmp_path, "domain")


def test_negative_case_detects_plain_import(tmp_path: Path) -> None:
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "bad.py").write_text("import rag_platform.adapters\n")
    assert violations(tmp_path, "domain")


def test_negative_case_detects_from_package_import_name(tmp_path: Path) -> None:
    (tmp_path / "domain").mkdir()
    (tmp_path / "domain" / "bad.py").write_text("from rag_platform import adapters\n")
    assert violations(tmp_path, "domain")
