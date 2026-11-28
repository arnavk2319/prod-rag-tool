import hashlib

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from rag_platform.services.ingest.chunker import (
    MAX_TOKENS,
    MIN_TOKENS,
    OVERLAP_RATIO,
    chunk_id,
    chunk_text,
    count_tokens,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)

word = st.text(alphabet="abcdefghij", min_size=1, max_size=8)
sep = st.sampled_from([" ", " ", " ", ". ", "\n\n", "\n", ", "])
heading = st.builds(lambda t: f"\n\n## {t}\n\n", word)
piece = st.one_of(st.tuples(word, sep).map("".join), heading)
docs = st.lists(piece, min_size=0, max_size=2200).map("".join)
DOC_HASH = hashlib.sha256(b"doc").hexdigest()


@SETTINGS
@given(docs)
def test_bounds(text: str) -> None:
    chunks = chunk_text(DOC_HASH, text)
    for c in chunks:
        assert c.tokens == count_tokens(c.text) <= MAX_TOKENS
    for c in chunks[:-1]:
        assert c.tokens >= MIN_TOKENS
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


@SETTINGS
@given(docs)
def test_overlap(text: str) -> None:
    chunks = chunk_text(DOC_HASH, text)
    for prev, nxt in zip(chunks, chunks[1:], strict=False):
        expected = round(OVERLAP_RATIO * prev.tokens)
        assert nxt.overlap_tokens == expected
        shared = text[nxt.span[0] : prev.span[1]]
        assert count_tokens(shared) == expected
        assert expected >= int(0.11 * prev.tokens)


@SETTINGS
@given(docs)
def test_no_text_lost(text: str) -> None:
    chunks = chunk_text(DOC_HASH, text)
    covered = [False] * len(text)
    for c in chunks:
        assert text[c.span[0] : c.span[1]] == c.text
        for i in range(*c.span):
            covered[i] = True
    assert all(covered[i] for i, ch in enumerate(text) if not ch.isspace())


@SETTINGS
@given(docs)
def test_deterministic_and_ids(text: str) -> None:
    a = chunk_text(DOC_HASH, text)
    assert a == chunk_text(DOC_HASH, text)
    for c in a:
        assert (
            c.chunk_id == hashlib.sha256((DOC_HASH + str(c.ordinal) + c.text).encode()).hexdigest()
        )
    assert len({c.chunk_id for c in a}) == len(a)


def test_heading_path_kept() -> None:
    body = " ".join(f"w{i}" for i in range(700))
    text = f"# Policy\n\nintro\n\n## Limits\n\n{body}\n\n## Fees\n\n{body}"
    chunks = chunk_text("h", text)
    assert chunks[0].heading_path == ("Policy",)
    assert chunks[1].heading_path == ("Policy", "Limits")
    assert chunks[-1].heading_path == ("Policy", "Fees")
    assert chunks[1].embed_text.startswith("Policy > Limits\n")


def test_empty_and_chunk_id_formula() -> None:
    assert chunk_text("h", "  \n ") == []
    assert chunk_id("h", 3, "t") == hashlib.sha256(b"h3t").hexdigest()


def test_closing_hash_sequence_only() -> None:
    assert chunk_text("h", "# Learn C#\n\nbody")[0].heading_path == ("Learn C#",)
    assert chunk_text("h", "## Title ##\n\nbody")[0].heading_path == ("Title",)
    assert chunk_text("h", "# C# #\n\nbody")[0].heading_path == ("C#",)


def test_last_chunk_has_enough_new_tokens() -> None:
    for extra in range(351, 410, 3):
        text = " ".join(f"w{i}" for i in range(extra))
        cs = chunk_text("h", text)
        last = cs[-1]
        assert last.tokens - last.overlap_tokens >= 0.1 * last.tokens, extra
        assert all(c.tokens <= MAX_TOKENS for c in cs)
        assert all(c.tokens >= MIN_TOKENS for c in cs[:-1])
