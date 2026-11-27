import hashlib
import json
import re
import tempfile
from pathlib import Path

from data import generate as gen

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
FIX = ROOT / "evals" / "fixtures"


def _load(path: Path) -> dict:  # type: ignore[type-arg]
    return json.loads(path.read_text(encoding="utf-8"))


def test_manifest_matches_files_and_hashes() -> None:
    docs = _load(CORPUS / "manifest.yaml")["documents"]
    listed = {d["source_file"] for d in docs}
    on_disk = {p.relative_to(CORPUS).as_posix() for p in CORPUS.rglob("*") if p.is_file()}
    assert on_disk - {"manifest.yaml"} == listed
    for d in docs:
        assert hashlib.sha256((CORPUS / d["source_file"]).read_bytes()).hexdigest() == d["sha256"]


def test_manifest_fields_and_superseded_versions() -> None:
    docs = _load(CORPUS / "manifest.yaml")["documents"]
    assert docs
    for d in docs:
        assert d["tenant"] in {"bank-a", "bank-b"}
        assert d["labels"] and set(d["labels"]) <= set(gen.LABELS)
        assert d["format"] in {"md", "pdf", "html", "txt"}
        assert d["status"] == ("superseded" if d["expires_date"] else "active")
    assert {d["tenant"] for d in docs} == {"bank-a", "bank-b"}
    assert {x for d in docs for x in d["labels"]} == set(gen.LABELS)
    assert {d["format"] for d in docs} == {"md", "pdf", "html", "txt"}
    sup = [d for d in docs if d["status"] == "superseded"]
    assert sup and all(d["effective_date"] < d["expires_date"] for d in sup)


def test_pii_seed_allow_list_and_luhn() -> None:
    items = _load(FIX / "pii_seed.yaml")["items"]
    assert len(items) == 100
    cards = [i["value"] for i in items if i["kind"] == "card"]
    assert cards and all(gen.luhn_valid(c) for c in cards)
    for i in items:
        v, k = i["value"], i["kind"]
        if k == "email":
            assert v.endswith("@" + gen.EMAIL_DOMAIN)
        elif k == "phone":
            assert re.fullmatch(r"(212|416)-555-01\d\d", v)
        elif k in {"ssn", "sin"}:
            assert v.startswith("000-")
        elif k == "name":
            first, last = v.split(" ")
            assert first in gen.FIRST_NAMES and last in gen.LAST_NAMES
        elif k == "card":
            assert v.replace(" ", "")[:6] in gen.CARD_PREFIXES
        else:
            raise AssertionError(k)


def test_corpus_has_no_foreign_emails() -> None:
    for p in list(CORPUS.rglob("*")) + list(FIX.rglob("*")):
        if p.is_file() and p.suffix != ".pdf":
            for m in re.findall(r"[\w.+-]+@([\w.-]+)", p.read_text(encoding="utf-8")):
                assert m.rstrip(".") == gen.EMAIL_DOMAIN or m.startswith("attacker.example.test"), p


def test_injection_corpus_and_ingest_fixture_counts() -> None:
    inj = _load(FIX / "injection_corpus.yaml")["items"]
    assert len(inj) == 30
    assert {i["kind"] for i in inj} == {
        "direct", "indirect_in_document", "exfiltration", "role_override", "encoding",
    }  # fmt: skip
    kinds = sorted(p.name for p in (FIX / "ingest").iterdir() if p.is_dir())
    assert len(kinds) == 10
    for k in kinds:
        exp = _load(FIX / "ingest" / k / "expected.yaml")
        for f in exp["files"]:
            assert f["outcome"] in {"accepted", "quarantined", "rejected"}
            if f["outcome"] != "accepted":
                assert f["reason"]


def _hashes(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def test_generator_is_deterministic_and_committed_output_is_current() -> None:
    with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
        gen.generate(Path(a))
        gen.generate(Path(b))
        ha, hb = _hashes(Path(a)), _hashes(Path(b))
    assert ha == hb and ha
    for rel, h in ha.items():
        assert hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() == h, rel
