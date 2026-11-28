from pathlib import Path

import pytest
from evals.metrics_chunking import chunk_determinism, orphan_children

ROOT = Path(__file__).resolve().parents[2]


def test_corpus_is_deterministic() -> None:
    assert chunk_determinism(ROOT) == 1.0


def test_empty_root_is_zero(tmp_path: Path) -> None:
    assert chunk_determinism(tmp_path) == 0.0


def test_no_orphan_children_on_corpus() -> None:
    from evals.metrics_chunking import orphan_children

    assert orphan_children(ROOT) == 0
    assert orphan_children(ROOT / "nope") == 0  # nothing chunked, nothing orphaned


def test_orphan_metric_detects_a_dangling_parent(monkeypatch: pytest.MonkeyPatch) -> None:
    import json
    from dataclasses import replace

    from evals import check, metrics_chunking

    real = metrics_chunking.chunk_document

    def broken(doc_hash: str, text: str):  # type: ignore[no-untyped-def]
        doc = real(doc_hash, text)
        if not doc.children:
            return doc
        kids = (replace(doc.children[0], parent_id="missing"), *doc.children[1:])
        return replace(doc, children=kids)

    monkeypatch.setattr(metrics_chunking, "chunk_document", broken)
    n = orphan_children(ROOT)
    assert n > 0
    report = json.loads((ROOT / "evals" / "results" / "latest.json").read_text())
    report["hard_zero"]["orphan_children"] = n
    assert any("orphan_children" in e for e in check.check_hard_zero(report))
