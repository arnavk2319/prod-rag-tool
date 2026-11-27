from pathlib import Path

from evals.metrics_chunking import chunk_determinism

ROOT = Path(__file__).resolve().parents[2]


def test_corpus_is_deterministic() -> None:
    assert chunk_determinism(ROOT) == 1.0


def test_empty_root_is_zero(tmp_path: Path) -> None:
    assert chunk_determinism(tmp_path) == 0.0
