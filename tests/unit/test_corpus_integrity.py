"""T-007 extra integrity probes for the synthetic corpus and ingest fixtures."""

import json
import re
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from data import generate as gen

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "corpus"
FIX = ROOT / "evals" / "fixtures"
INGEST = FIX / "ingest"


def _load(path: Path):  # type: ignore[no-untyped-def]
    return json.loads(path.read_text(encoding="utf-8"))


def _docs() -> list[dict]:  # type: ignore[type-arg]
    return _load(CORPUS / "manifest.yaml")["documents"]


def _files(root: Path) -> list[Path]:
    return [p for p in sorted(root.rglob("*")) if p.is_file()]


def test_ids_unique_per_version_and_files_unique() -> None:
    docs = _docs()
    keys = [(d["doc_id"], d["version"]) for d in docs]
    assert len(keys) == len(set(keys))
    assert len({d["source_file"] for d in docs}) == len(docs)
    for d in docs:
        assert d["source_file"].startswith(d["tenant"] + "/")


def test_version_chain_has_no_gaps_or_overlaps() -> None:
    chains: dict[tuple[str, str], list[dict]] = defaultdict(list)  # type: ignore[type-arg]
    for d in _docs():
        chains[(d["tenant"], d["doc_id"])].append(d)
    for key, vs in chains.items():
        vs.sort(key=lambda d: d["version"])
        assert [v["version"] for v in vs] == list(range(1, len(vs) + 1)), key
        for a, b in zip(vs, vs[1:], strict=False):
            assert a["expires_date"] is not None and a["effective_date"] < a["expires_date"], key
            assert b["effective_date"] == a["expires_date"], key  # no gap, no overlap
        assert vs[-1]["expires_date"] is None, key  # exactly one current version
        assert all(v["expires_date"] is not None for v in vs[:-1]), key


def test_every_tenant_has_every_label() -> None:
    for t in gen.TENANTS:
        seen = {x for d in _docs() if d["tenant"] == t for x in d["labels"]}
        assert seen == set(gen.LABELS), t


def test_no_identical_hash_across_tenants() -> None:
    by_hash: dict[str, set[str]] = defaultdict(set)
    for d in _docs():
        by_hash[d["sha256"]].add(d["tenant"])
    assert all(len(t) == 1 for t in by_hash.values())


def test_tenant_markers_do_not_leak() -> None:
    docs = _docs()
    names = gen.BANK_NAMES
    for d in docs:
        text = (CORPUS / d["source_file"]).read_bytes().decode("latin-1")
        other = "bank-b" if d["tenant"] == "bank-a" else "bank-a"
        assert names[other] not in text, d["source_file"]
        assert other not in text, d["source_file"]
        for o in docs:  # the other tenant's key_fact must not appear in this doc
            if o["tenant"] == other and o["key_fact"] != d["key_fact"]:
                assert o["key_fact"] not in text, (d["source_file"], o["source_file"])


def test_key_fact_present_in_own_file_and_tenant_values_differ() -> None:
    docs = _docs()
    for d in docs:
        if d["format"] != "pdf":
            assert d["key_fact"].split(".")[0][:30] in (CORPUS / d["source_file"]).read_text(
                "utf-8"
            )
    facts: dict[tuple[str, int], dict[str, str]] = defaultdict(dict)
    for d in docs:
        topic = d["doc_id"].split("-", 2)[2]
        facts[(topic, d["version"])][d["tenant"]] = d["key_fact"]
    for k, v in facts.items():
        if len(v) == 2:  # every version (including superseded) is tenant-distinct
            assert v["bank-a"] != v["bank-b"], k


def test_pdfs_structurally_sane() -> None:
    pdfs = [p for p in _files(CORPUS) if p.suffix == ".pdf"]
    assert pdfs
    for p in pdfs:
        b = p.read_bytes()
        assert b.startswith(b"%PDF-"), p
        assert b.rstrip().endswith(b"%%EOF"), p
        assert b"xref" in b and b"trailer" in b and b"startxref" in b, p
        off = int(re.search(rb"startxref\s+(\d+)\s+%%EOF", b).group(1))  # type: ignore[union-attr]
        assert b[off : off + 4] == b"xref", p
        # each xref entry points at "N 0 obj"
        table = b[off:].split(b"trailer")[0].split(b"\n")
        n = int(table[1].split()[1])
        entries = [ln for ln in table[2:] if len(ln.strip()) >= 18]
        assert len(entries) == n, p
        for i, ln in enumerate(entries[1:], start=1):
            o = int(ln[:10])
            assert b[o:].startswith(f"{i} 0 obj".encode()), (p, i)


def test_text_files_utf8_lf_only() -> None:
    for root in (CORPUS, FIX):
        for p in _files(root):
            if p.suffix == ".pdf":
                continue
            raw = p.read_bytes()
            raw.decode("utf-8")
            assert b"\r" not in raw, p
            assert not raw.startswith(b"\xef\xbb\xbf"), p


def test_no_timestamps_or_absolute_paths_in_generated_files() -> None:
    bad = re.compile(
        rb"(/Users/|/home/|/tmp/|/private/|[A-Z]:\\\\|\d{4}-\d\d-\d\dT\d\d:\d\d|CreationDate|ModDate)"
    )
    for root in (CORPUS, FIX):
        for p in _files(root):
            assert not bad.search(p.read_bytes()), p


def test_generate_py_has_no_network_imports() -> None:
    src = (ROOT / "data" / "generate.py").read_text(encoding="utf-8")
    mods = re.findall(r"^\s*(?:import|from)\s+([\w.]+)", src, re.M)
    banned = {
        "socket",
        "urllib",
        "http",
        "httpx",
        "requests",
        "aiohttp",
        "ftplib",
        "smtplib",
        "ssl",
    }
    assert not {m.split(".")[0] for m in mods} & banned


def test_generator_from_other_cwd_with_out_matches_committed() -> None:
    with tempfile.TemporaryDirectory() as out, tempfile.TemporaryDirectory() as cwd:
        env = {"PYTHONPATH": str(ROOT), "PATH": "/usr/bin:/bin"}
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-m", "data.generate", "--out", out],
            cwd=cwd, env=env, capture_output=True, text=True, check=False,
        )  # fmt: skip
        assert r.returncode == 0, r.stderr
        for p in _files(Path(out)):
            rel = p.relative_to(out)
            assert p.read_bytes() == (ROOT / rel).read_bytes(), rel


def test_ingest_fixture_kinds_expected_files_and_referenced_files() -> None:
    expected_kinds = {
        "clean", "scanned_like", "garbled", "duplicate", "superseded",
        "malformed", "injected", "pii_seeded", "oversized", "wrong_extension",
    }  # fmt: skip
    assert {p.name for p in INGEST.iterdir() if p.is_dir()} == expected_kinds
    for k in expected_kinds:
        exp = _load(INGEST / k / "expected.yaml")
        assert exp["kind"] == k and exp["tenant"] in gen.TENANTS
        assert exp["files"]
        for f in exp["files"]:
            assert f["outcome"] in {"accepted", "quarantined", "rejected"}
            if f["outcome"] == "accepted":
                assert f["reason"] is None or f["reason"] == "pii_redacted"
            else:
                assert isinstance(f["reason"], str) and f["reason"].strip()
            if k != "oversized":
                assert (INGEST / k / f["file"]).is_file(), (k, f["file"])


def test_ingest_fixture_content_matches_kind() -> None:
    inj = [i["text"] for i in _load(FIX / "injection_corpus.yaml")["items"]]
    injected = (INGEST / "injected" / "injected.md").read_text("utf-8")
    canary = {i["canary"] for i in _load(FIX / "injection_corpus.yaml")["items"]}
    assert canary and any(c in injected for c in canary)
    # pattern lead-in ("NOTE TO AI ASSISTANT", "ignore ...") from the corpus appears in the fixture
    leads = {t.split(":")[0] for t in inj if ":" in t and t.split(":")[0].isupper()}
    assert any(lead in injected for lead in leads)
    assert any(t in injected for t in inj)  # exact corpus text, not a paraphrase
    pii = (INGEST / "pii_seeded" / "pii.md").read_text("utf-8")
    assert "@" + gen.EMAIL_DOMAIN in pii and re.search(r"\b000-\d\d-\d{4}\b", pii)
    dup = INGEST / "duplicate"
    assert (dup / "original.md").read_bytes() == (dup / "copy.md").read_bytes()
    assert (INGEST / "wrong_extension" / "report.pdf").read_bytes()[:4] != b"%PDF"
    mal = (INGEST / "malformed" / "truncated.pdf").read_bytes()
    assert mal.startswith(b"%PDF") and not mal.rstrip().endswith(b"%%EOF")
    scan = (INGEST / "scanned_like" / "scanned.pdf").read_bytes()
    assert scan.startswith(b"%PDF") and b"BT" not in scan  # no text layer
    sup = INGEST / "superseded"
    assert (sup / "card-limit.v1.md").read_bytes() != (sup / "card-limit.v2.md").read_bytes()
    letters = re.sub(
        r"[^a-z ]", "", (INGEST / "garbled" / "garbled.txt").read_text("utf-8").lower()
    )
    words = letters.split()
    common = {"the", "and", "of", "to", "is", "a", "for", "per"}
    assert not common & set(words)  # garbled: no ordinary stopwords


def test_pii_seeded_fixture_contains_listed_seeds() -> None:
    seeds = [i["value"] for i in _load(FIX / "pii_seed.yaml")["items"]]
    pii = (INGEST / "pii_seeded" / "pii.md").read_text("utf-8")
    items = _load(FIX / "pii_seed.yaml")["items"]
    for kind in ("name", "email", "phone", "card"):
        assert sum(i["value"] in pii for i in items if i["kind"] == kind) >= 3, kind
    assert sum(s in pii for s in seeds) >= 12


def test_pii_seed_unique_and_ids() -> None:
    items = _load(FIX / "pii_seed.yaml")["items"]
    assert len(items) == 100
    assert len({i["value"] for i in items}) == 100
    assert len({i["id"] for i in items}) == 100
    assert all(i["value"].strip() for i in items)


def test_pii_seed_ssn_sin_never_real_issuable_and_noncards_not_luhn() -> None:
    for i in _load(FIX / "pii_seed.yaml")["items"]:
        v, k = i["value"], i["kind"]
        digits = re.sub(r"\D", "", v)
        if k == "ssn":
            assert re.fullmatch(r"000-\d\d-\d{4}", v)  # area 000 is never issued
        elif k == "sin":
            assert re.fullmatch(r"000-\d{3}-\d{3}", v)  # SINs never start with 0
        elif k == "card":
            assert len(digits) == 16 and gen.luhn_valid(digits)
        if k in {"ssn", "sin", "phone"} and len(digits) >= 9:
            # must not be accidentally Luhn-valid (would pass a real SIN/card checksum)
            assert not gen.luhn_valid(digits), (i["id"], v)


def test_pii_card_seeds_distinct_and_test_bins_only() -> None:
    cards = [
        re.sub(r"\D", "", i["value"])
        for i in _load(FIX / "pii_seed.yaml")["items"]
        if i["kind"] == "card"
    ]
    assert len(cards) == 20 and len(set(cards)) == 20
    assert {c[:6] for c in cards} <= {"400000", "555555"}


def test_injection_corpus_unique_non_empty() -> None:
    items = _load(FIX / "injection_corpus.yaml")["items"]
    assert len(items) == 30
    assert len({i["text"] for i in items}) == 30
    assert len({i["id"] for i in items}) == 30
    for i in items:
        assert i["text"].strip() and i["kind"] and i["canary"] and i["expected"]


def test_no_real_looking_pii_anywhere() -> None:
    allowed_cards = {re.sub(r"\D", "", i["value"]) for i in _load(FIX / "pii_seed.yaml")["items"]}
    for root in (CORPUS, FIX):
        for p in _files(root):
            if p.suffix == ".pdf" or p.name == "manifest.yaml":  # manifest holds sha256 hex
                continue
            t = p.read_text("utf-8")
            for m in re.findall(r"\b(\d{3})-(\d{2})-(\d{4})\b", t):
                assert m[0] == "000", (p, m)
            for m in re.findall(r"(?<![\w-])(?:\d[ -]?){13,19}(?![\w-])", t):
                assert re.sub(r"\D", "", m) in allowed_cards, (p, m)
            for m in re.findall(r"(?<![\w.])\d{10,}(?![\w.])", t):
                assert m in allowed_cards, (p, m)  # long digit runs
            for m in re.findall(r"\(?\b(\d{3})\)?[ -](\d{3})-(\d{4})\b", t):
                assert m[1] == "555" and m[2].startswith("01"), (p, m)
