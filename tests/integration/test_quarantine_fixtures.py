"""Every ingest fixture lands in its expected outcome (no DB needed)."""

import json
from pathlib import Path

import pytest

from rag_platform.services.ingest.quarantine import assess

FIX = Path(__file__).resolve().parents[2] / "evals/fixtures/ingest"
KINDS = sorted(p.name for p in FIX.iterdir() if p.is_dir())


def _expected(kind: str) -> dict[str, object]:
    return json.loads((FIX / kind / "expected.yaml").read_text())  # type: ignore[no-any-return]


def test_all_ten_kinds_present() -> None:
    assert len(KINDS) == 10


@pytest.mark.parametrize("kind", KINDS)
def test_fixture_outcomes(kind: str) -> None:
    exp = _expected(kind)
    max_bytes = int(exp.get("max_upload_bytes", 20 * 1024 * 1024))  # type: ignore[call-overload]
    seen: set[str] = set()
    for f in exp["files"]:  # type: ignore[attr-defined]
        gen = exp.get("generate_at_test_time")
        if gen:
            data = str(gen["filler"]).encode() * int(gen["size_bytes"])  # type: ignore[index,call-overload]
        else:
            data = (FIX / kind / f["file"]).read_bytes()
        d = assess(data, f["file"], known_hashes=seen, max_bytes=max_bytes)
        seen.add(d.doc_hash) if d.outcome == "accepted" else None
        assert d.outcome.value == f["outcome"], (kind, f["file"])
        assert (d.reason.value if d.reason else None) == f["reason"], (kind, f["file"])


def test_pii_seeded_text_has_no_seeded_values() -> None:
    d = assess((FIX / "pii_seeded/pii.md").read_bytes(), "pii.md")
    for v in ("alex.mockridge0@example.test", "4000 0003 5945 7043", "212-555-0100", "000-776-254"):
        assert v not in d.text
    assert "[CARD]" in d.text and "[EMAIL]" in d.text
