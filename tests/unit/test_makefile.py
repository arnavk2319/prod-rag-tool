from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_makefile_targets() -> None:
    text = (ROOT / "Makefile").read_text()
    for target in ("lint", "test", "typecheck", "verify"):
        assert f"\n{target}:" in "\n" + text


def test_layout_exists() -> None:
    for layer in ("api", "domain", "services", "adapters", "worker", "cli"):
        assert (ROOT / "src" / "rag_platform" / layer / "__init__.py").is_file()
    for d in ("config", "prompts", "migrations", "evals", "data", "observability", "tests"):
        assert (ROOT / d).is_dir()
