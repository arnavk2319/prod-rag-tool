"""Edge-case, corpus and cross-process probes for the chunker (T-009)."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import pytest

from rag_platform.services.ingest.chunker import (
    MAX_TOKENS,
    MIN_TOKENS,
    chunk_id,
    chunk_text,
    count_tokens,
)

ROOT = Path(__file__).resolve().parents[2]
CORPUS = sorted(p for p in (ROOT / "data" / "corpus").rglob("*") if p.suffix in {".md", ".txt"})
DH = "d" * 64


def words(n: int, prefix: str = "w") -> str:
    return " ".join(f"{prefix}{i}" for i in range(n))


def prose(n: int) -> str:
    return " ".join(f"word{i}." if i % 15 == 14 else f"word{i}" for i in range(n))


def check_invariants(text: str, chunks: list) -> None:
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))
    assert len({c.chunk_id for c in chunks}) == len(chunks)
    for i, c in enumerate(chunks):
        assert text[c.span[0] : c.span[1]] == c.text
        assert c.chunk_id == hashlib.sha256((DH + str(i) + c.text).encode()).hexdigest()
        assert count_tokens(c.text) == c.tokens
        assert c.tokens <= MAX_TOKENS
        if i < len(chunks) - 1:
            assert c.tokens >= MIN_TOKENS
    # no text lost: stitching spans (minus overlap) reproduces every non-space char
    rebuilt, cursor = "", 0
    for c in chunks:
        assert c.span[0] <= cursor or not text[cursor : c.span[0]].strip()
        rebuilt += text[max(cursor, c.span[0]) : c.span[1]]
        cursor = max(cursor, c.span[1])
    assert "".join(rebuilt.split()) == "".join(text.split())
    # overlap really is shared text
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        shared = text[nxt.span[0] : prev.span[1]]
        assert count_tokens(shared) == nxt.overlap_tokens
        assert prev.text.endswith(shared) and nxt.text.startswith(shared)


@pytest.mark.parametrize("text", ["", "   ", "\n\n\t \r\n"])
def test_empty_and_whitespace(text: str) -> None:
    assert chunk_text(DH, text) == []


def test_short_doc_single_chunk() -> None:
    cs = chunk_text(DH, "# T\n\nhello world")
    assert len(cs) == 1 and cs[0].heading_path == ("T",) and cs[0].overlap_tokens == 0


def test_one_huge_token() -> None:
    text = "x" * 200_000
    cs = chunk_text(DH, text)
    assert len(cs) == 1 and cs[0].text == text


def test_no_headings_long_doc() -> None:
    text = words(2500)
    cs = chunk_text(DH, text)
    assert len(cs) > 3 and all(c.heading_path == () for c in cs)
    check_invariants(text, cs)


def test_long_paragraph_without_sentence_ends() -> None:
    text = words(5000)
    cs = chunk_text(DH, text)
    check_invariants(text, cs)
    assert all(c.tokens == 400 for c in cs[:-1])


@pytest.mark.parametrize("n", [299, 300, 499, 500, 501, 520, 700, 1000])
def test_boundary_sizes(n: int) -> None:
    text = prose(n)
    check_invariants(text, chunk_text(DH, text))


def test_overlap_ratio_about_12_percent() -> None:
    text = words(3000)
    for c in chunk_text(DH, text)[1:]:
        assert c.overlap_tokens == 48  # round(0.12 * 400)


def test_nested_headings_path() -> None:
    body = words(450)
    text = f"# A\n\n## B\n\n### C\n\n{body}\n\n## D\n\n{words(450, 'z')}\n"
    assert len(chunk_text(DH, text)) >= 3
    cs = chunk_text(DH, text)
    assert cs[0].heading_path == ("A",)  # path is what is in effect at chunk start
    assert cs[1].heading_path == ("A", "B", "C")  # second chunk starts inside C's body
    check_invariants(text, cs)


def test_heading_path_pops_to_sibling() -> None:
    text = f"# A\n## B\n{words(320)}\n## D\n{words(320, 'q')}\n"
    cs = chunk_text(DH, text)
    # second chunk starts after overlap; the path in effect at its start is B or D, never both
    for c in cs:
        assert not ("B" in c.heading_path and "D" in c.heading_path)


def test_heading_only_doc() -> None:
    cs = chunk_text(DH, "# Title\n## Sub\n")
    assert len(cs) == 1 and cs[0].heading_path in {("Title", "Sub"), ("Title",)}
    assert cs[0].text.startswith("#")


def test_embed_text_prepends_path() -> None:
    c = chunk_text(DH, "# A\n## B\nbody text")[0]
    assert c.embed_text == " > ".join(c.heading_path) + "\n" + c.text and c.text.startswith("# A")
    plain = chunk_text(DH, "no heading here")[0]
    assert plain.embed_text == plain.text


@pytest.mark.parametrize(
    "text",
    [
        "日本語のテキストです。" * 400,
        "emoji 😀🎉 " * 600,
        "naïve café Ünïcode " * 400,
        "العربية " * 700,
        "á" * 900,
    ],
)
def test_unicode_inputs(text: str) -> None:
    cs = chunk_text(DH, text)
    assert cs
    check_invariants(text, cs)


def test_crlf_matches_lf_structure() -> None:
    lf = f"# A\n\n{prose(900)}\n\n## B\n\n{prose(900)}\n"
    crlf = lf.replace("\n", "\r\n")
    a, b = chunk_text(DH, lf), chunk_text(DH, crlf)
    assert [c.tokens for c in a] == [c.tokens for c in b]
    check_invariants(crlf, b)


def test_crlf_heading_has_no_carriage_return() -> None:
    cs = chunk_text(DH, "# Title\r\n\r\nbody text here\r\n")
    assert cs[0].heading_path == ("Title",)


def test_table_and_code_fence_text_preserved() -> None:
    table = "\n".join(f"| c{i} | v{i} | x{i} |" for i in range(200))
    fence = "```python\n" + "\n".join(f"x{i} = {i}" for i in range(300)) + "\n```"
    text = f"# Doc\n\n{table}\n\n{fence}\n"
    check_invariants(text, chunk_text(DH, text))


def test_hash_comment_in_code_fence_is_not_heading() -> None:
    text = "# Real\n\n```bash\n# not a heading\necho hi\n```\n\n" + words(900)
    cs = chunk_text(DH, text)
    assert all("not a heading" not in c.heading_path for c in cs)


@pytest.mark.xfail(
    strict=True, reason="known gap: tables are split across chunks (no whole-table retention)"
)
def test_small_table_not_split() -> None:
    table = "\n".join(f"| c{i} | v{i} |" for i in range(90))
    text = words(150, "p") + "\n\n" + table + "\n\n" + words(400, "q")
    cs = chunk_text(DH, text)
    assert any(table in c.text for c in cs)


def test_chunk_id_formula_and_sensitivity() -> None:
    base = chunk_id(DH, 3, "text")
    assert base == hashlib.sha256((DH + "3" + "text").encode()).hexdigest()
    assert (
        len({base, chunk_id("e" * 64, 3, "text"), chunk_id(DH, 4, "text"), chunk_id(DH, 3, "texu")})
        == 4
    )


def test_doc_hash_changes_every_id() -> None:
    t = words(1200)
    a, b = chunk_text(DH, t), chunk_text("e" * 64, t)
    assert not {c.chunk_id for c in a} & {c.chunk_id for c in b}
    assert [c.text for c in a] == [c.text for c in b]


def test_text_edit_changes_only_later_ids_when_early_chunks_unchanged() -> None:
    t = words(1500)
    edited = t[:-5] + "ZZZZZ"
    a, b = chunk_text(DH, t), chunk_text(DH, edited)
    assert a[0].chunk_id == b[0].chunk_id and a[-1].chunk_id != b[-1].chunk_id


@pytest.mark.parametrize("path", CORPUS, ids=lambda p: p.relative_to(ROOT).as_posix())
def test_real_corpus_invariants(path: Path) -> None:
    text = path.read_bytes().decode("utf-8", errors="replace")
    cs = chunk_text(hashlib.sha256(path.read_bytes()).hexdigest(), text)
    dh = hashlib.sha256(path.read_bytes()).hexdigest()
    assert cs or not text.strip()
    for i, c in enumerate(cs):
        assert c.chunk_id == hashlib.sha256((dh + str(i) + c.text).encode()).hexdigest()
        assert text[c.span[0] : c.span[1]] == c.text
        assert c.tokens <= MAX_TOKENS and (i == len(cs) - 1 or c.tokens >= MIN_TOKENS)
    # reconstruct by dropping overlap: nothing but whitespace is missing
    rebuilt, cursor = "", 0
    for c in cs:
        assert (
            text[cursor : c.span[0]].strip() == "" or cursor == 0 and not text[: c.span[0]].strip()
        )
        rebuilt += text[max(cursor, c.span[0]) : c.span[1]]
        cursor = max(cursor, c.span[1])
    assert "".join(rebuilt.split()) == "".join(text.split())


_SNIPPET = (
    "import json,sys\n"
    "from pathlib import Path\n"
    "from rag_platform.services.ingest.chunker import chunk_text\n"
    "out=[]\n"
    "for p in sorted(Path('data/corpus').rglob('*')):\n"
    "    if p.suffix in {'.md','.txt'}:\n"
    "        t=p.read_bytes().decode('utf-8','replace')\n"
    "        out.append([(c.chunk_id,c.span,c.heading_path) for c in chunk_text('h',t)])\n"
    "print(json.dumps(out))\n"
)


def test_stable_under_pythonhashseed() -> None:
    outs = []
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed}
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-c", _SNIPPET], cwd=ROOT, env=env, capture_output=True, text=True
        )
        assert r.returncode == 0, r.stderr[-500:]
        outs.append(r.stdout)
    assert outs[0] == outs[1] == outs[2] and len(outs[0]) > 10
