"""Hypothesis properties for the parent-child chunker (T-044)."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag_platform.chunk_settings import ChunkSettings
from rag_platform.services.ingest.chunker import (
    CHUNKER_VERSION,
    MAX_TOKENS,
    MIN_TOKENS,
    PARENT_MAX_TOKENS,
    PARENT_MIN_TOKENS,
    Sizes,
    chunk_document,
    count_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
SETTINGS = settings(
    max_examples=40,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)
word = st.text(alphabet="abcdefghij", min_size=1, max_size=8)
sep = st.sampled_from([" ", " ", " ", ". ", "\n\n", "\n", ", ", "\r\n\r\n"])
heading = st.builds(lambda t: f"\n\n## {t}\n\n", word)
piece = st.one_of(st.tuples(word, sep).map("".join), heading)
docs = st.lists(piece, min_size=0, max_size=4500).map("".join)
DH = hashlib.sha256(b"doc").hexdigest()


@SETTINGS
@given(docs)
def test_parents_partition_text(text: str) -> None:
    doc = chunk_document(DH, text)
    assert "".join(p.text for p in doc.parents) == (text if text.strip() else "")
    cursor = 0
    for i, p in enumerate(doc.parents):
        assert p.ordinal == i and p.span[0] == cursor and text[p.span[0] : p.span[1]] == p.text
        assert p.tokens == count_tokens(p.text)
        cursor = p.span[1]
    assert cursor == (len(text) if doc.parents else 0)
    assert len({p.parent_id for p in doc.parents}) == len(doc.parents)


@SETTINGS
@given(docs)
def test_every_child_in_exactly_one_parent(text: str) -> None:
    doc = chunk_document(DH, text)
    by_id = {p.parent_id: p for p in doc.parents}
    for c in doc.children:
        parent = by_id[c.parent_id]  # exactly one parent id, and it exists
        assert c.text in parent.text
        assert parent.span[0] <= c.span[0] and c.span[1] <= parent.span[1]
        assert text[c.span[0] : c.span[1]] == c.text
        containing = [p for p in doc.parents if p.span[0] <= c.span[0] and c.span[1] <= p.span[1]]
        assert [p.parent_id for p in containing] == [c.parent_id]
    assert {c.parent_id for c in doc.children} == set(by_id)  # no childless parent
    assert [c.ordinal for c in doc.children] == list(range(len(doc.children)))


@SETTINGS
@given(docs)
def test_bounds(text: str) -> None:
    doc = chunk_document(DH, text)
    for p in doc.parents[:-1]:
        assert PARENT_MIN_TOKENS <= p.tokens <= PARENT_MAX_TOKENS
    for p in doc.parents:
        assert p.tokens <= PARENT_MAX_TOKENS
    for p in doc.parents:
        kids = [c for c in doc.children if c.parent_id == p.parent_id]
        assert all(c.tokens <= MAX_TOKENS and c.tokens == count_tokens(c.text) for c in kids)
        assert all(c.tokens >= MIN_TOKENS for c in kids[:-1])
        assert kids[0].overlap_tokens == 0
        for prev, nxt in zip(kids, kids[1:], strict=False):
            assert nxt.overlap_tokens == round(0.12 * prev.tokens)
            assert count_tokens(text[nxt.span[0] : prev.span[1]]) == nxt.overlap_tokens


@SETTINGS
@given(docs)
def test_no_text_lost_in_children(text: str) -> None:
    doc = chunk_document(DH, text)
    covered = [False] * len(text)
    for c in doc.children:
        for i in range(*c.span):
            covered[i] = True
    assert all(covered[i] for i, ch in enumerate(text) if not ch.isspace())


@SETTINGS
@given(docs)
def test_deterministic_and_id_formulas(text: str) -> None:
    a = chunk_document(DH, text)
    assert a == chunk_document(DH, text)
    assert a.chunker_version == CHUNKER_VERSION == "pc-300-1200-v1"
    for p in a.parents:
        assert (
            p.parent_id == hashlib.sha256((DH + "p" + str(p.ordinal) + p.text).encode()).hexdigest()
        )
    for c in a.children:
        assert c.chunk_id == hashlib.sha256((DH + str(c.ordinal) + c.text).encode()).hexdigest()


def test_heading_path_on_parents_and_children() -> None:
    body = " ".join(f"w{i}" for i in range(900))
    text = f"# Policy\n\nintro\n\n## Limits\n\n{body}\n\n## Fees\n\n{body}"
    doc = chunk_document("h", text)
    assert len(doc.parents) >= 2
    assert doc.parents[0].heading_path == ("Policy",)
    assert doc.parents[-1].heading_path == ("Policy", "Fees")  # split on the heading boundary
    assert doc.parents[-1].text.startswith("## Fees")
    assert doc.children[-1].heading_path == ("Policy", "Fees")


def test_crlf_and_code_fence() -> None:
    lf = "# A\n\n" + " ".join(f"w{i}." if i % 20 == 19 else f"w{i}" for i in range(1800))
    lf += "\n\n## B\n\n```bash\n# not a heading\necho hi\n```\n\n" + " ".join(
        f"z{i}" for i in range(900)
    )
    crlf = lf.replace("\n", "\r\n")
    a, b = chunk_document(DH, lf), chunk_document(DH, crlf)
    assert [p.tokens for p in a.parents] == [p.tokens for p in b.parents]
    assert "".join(p.text for p in b.parents) == crlf
    assert all("not a heading" not in c.heading_path for c in a.children)
    assert all("\r" not in t for c in b.children for t in c.heading_path)


def test_empty_doc_and_version_follows_sizes() -> None:
    assert chunk_document(DH, " \n ").parents == ()
    other = chunk_document(DH, "a b c", Sizes(child=150, parent=600))
    assert other.chunker_version == "pc-150-600-v1"
    assert Sizes(child=150, parent=600).child_max == 175 and Sizes().parent_min == 800


def test_chunk_settings_validation() -> None:
    assert ChunkSettings(_env_file=None).parent_top_n == 3  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="PARENT_TOKENS"):
        ChunkSettings(child_tokens=300, parent_tokens=400)


_SNIPPET = (
    "import json\nfrom pathlib import Path\n"
    "from rag_platform.services.ingest.chunker import chunk_document\n"
    "out=[]\n"
    "for p in sorted(Path('data/corpus').rglob('*')):\n"
    "    if p.suffix in {'.md','.txt'}:\n"
    "        d=chunk_document('h',p.read_bytes().decode('utf-8','replace'))\n"
    "        out.append([[x.parent_id for x in d.parents],\n"
    "                    [(c.chunk_id,c.parent_id) for c in d.children]])\n"
    "print(json.dumps(out))\n"
)


def test_stable_under_pythonhashseed() -> None:
    outs = []
    for seed in ("0", "1", "999"):
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _SNIPPET],
            cwd=ROOT,
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
        )
        assert r.returncode == 0, r.stderr[-500:]
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2] and len(outs[0]) > 10
