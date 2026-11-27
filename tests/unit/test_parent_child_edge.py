"""Edge-input, id, settings and collapse probes for parent-child chunking (T-044, test-writer)."""

import asyncio
import hashlib
import itertools
import re
from collections.abc import Sequence
from datetime import date
from pathlib import Path

import pytest

from rag_platform.chunk_settings import ChunkSettings, get_chunk_settings
from rag_platform.domain.ports import Chunk, ParentChunk, ScoredChunk
from rag_platform.services.ingest.build import DocMeta, build_records
from rag_platform.services.ingest.chunker import (
    ChunkedDocument,
    Sizes,
    chunk_document,
    chunk_id,
    count_tokens,
    parent_id,
)
from rag_platform.services.retrieval.collapse import collapse_to_parents

ROOT = Path(__file__).resolve().parents[2]
S = Sizes()
DH = hashlib.sha256(b"doc").hexdigest()


def w(n: int) -> str:
    return " ".join(f"w{i}" for i in range(n))


def check(text: str, sz: Sizes = S, dh: str = DH) -> ChunkedDocument:
    """Assert every structural invariant of a chunked document and return it."""
    d = chunk_document(dh, text, sz)
    ps, cs = d.parents, d.children
    assert d.chunker_version == f"pc-{sz.child}-{sz.parent}-v1"
    if count_tokens(text) == 0:
        assert ps == () and cs == ()
        return d
    assert "".join(p.text for p in ps) == text  # exact reproduction
    assert ps[0].span[0] == 0 and ps[-1].span[1] == len(text)
    for a, b in itertools.pairwise(ps):
        assert a.span[1] == b.span[0]  # no gap, no overlap
    for i, p in enumerate(ps):
        assert text[p.span[0] : p.span[1]] == p.text and p.ordinal == i
        assert p.parent_id == hashlib.sha256((dh + "p" + str(i) + p.text).encode()).hexdigest()
        assert p.tokens == count_tokens(p.text)
        if i < len(ps) - 1:
            assert sz.parent_min <= p.tokens <= sz.parent_max
        else:  # documented exception: the last parent may be short, never oversize
            assert p.tokens <= sz.parent_max
    assert len({p.parent_id for p in ps}) == len(ps)
    by = {p.parent_id: p for p in ps}
    assert [c.ordinal for c in cs] == list(range(len(cs)))
    assert len({c.chunk_id for c in cs}) == len(cs)
    kids: dict[str, list[object]] = {p.parent_id: [] for p in ps}
    for c in cs:
        p = by[c.parent_id]  # exactly one existing parent
        kids[p.parent_id].append(c)
        assert p.span[0] <= c.span[0] and c.span[1] <= p.span[1]  # never crosses a boundary
        assert text[c.span[0] : c.span[1]] == c.text  # document-relative span
        assert p.text[c.span[0] - p.span[0] : c.span[1] - p.span[0]] == c.text  # parent-relative
        assert c.text in p.text
        assert c.chunk_id == chunk_id(dh, c.ordinal, c.text)  # formula unchanged
        assert 0 < c.tokens <= sz.child_max
    for p in ps:
        mine = [c for c in cs if c.parent_id == p.parent_id]
        assert mine, "every parent with tokens has at least one child"
        assert mine[0].overlap_tokens == 0 and mine[0].span[0] == _first_tok(p.text, p.span[0])
        assert mine[-1].span[1] == _last_tok_end(p.text, p.span[0])
        for prev, cur in itertools.pairwise(mine):
            assert cur.overlap_tokens == round(0.12 * prev.tokens)  # ~12% inside one parent
            assert cur.span[0] < prev.span[1]  # windows overlap
            assert prev.tokens >= sz.child_min  # only the last child may be short
    return d


def _first_tok(ptext: str, base: int) -> int:
    m = re.search(r"\w|[^\w\s]", ptext)
    assert m
    return base + m.start()


def _last_tok_end(ptext: str, base: int) -> int:
    ms = list(re.finditer(r"\w+|[^\w\s]", ptext))
    return base + ms[-1].end()


CASES = {
    "empty": "",
    "whitespace": "  \n\n \t ",
    "giant_paragraph": "alpha beta. " * 3000,
    "giant_no_punct": "alpha " * 5000,
    "no_headings": w(3000),
    "nested_headings": "".join(f"# A{i}\n## B\n### C\n{w(200)}\n\n" for i in range(20)),
    "heading_only": "# T\n## U\n### V",
    "heading_only_big": "\n".join(f"## h{i}" for i in range(1500)),
    "crlf": ("# H\r\n\r\n" + w(400) + "\r\n\r\n") * 8,
    "code_fence": "```\n" + w(2000) + "\n```\n",
    "code_fence_with_headings": "```\n# not a heading\n" + w(1700) + "\n```\n# real\n" + w(100),
    "table": "| a | b |\n|---|---|\n" * 400,
    "cjk": "你好世界。" * 1500,
    "emoji_mixed": "héllo wörld 😀 " * 1200,
    "shorter_than_child": "hello",
    "one_token": "x",
    "just_under_child": w(199),
    "just_over_child_max": w(351),
    "parent_bound": w(1500),
    "just_over_parent_bound": w(1501),
    "just_over_parent_bound2": w(1599),
    "two_parents_exact": w(2400),
    "leading_ws": "\n\n   " + w(2500),
    "trailing_ws": w(2500) + "\n\n\n   ",
    "repeated_para": "same paragraph here.\n\n" * 600,
}


@pytest.mark.parametrize("name", sorted(CASES))
def test_edge_inputs_hold_all_invariants(name: str) -> None:
    check(CASES[name])


def test_chunker_is_pure_and_repeatable() -> None:
    t = CASES["nested_headings"]
    assert chunk_document(DH, t) == chunk_document(DH, t)


def test_real_corpus_invariants() -> None:
    files = [p for p in sorted((ROOT / "data/corpus").rglob("*")) if p.suffix in {".md", ".txt"}]
    assert len(files) >= 8
    seen = set()
    for f in files:
        text = f.read_bytes().decode("utf-8", "replace")
        check(text)
        seen.add(f.suffix)
    assert seen == {".md", ".txt"}


@pytest.mark.parametrize("sz", [Sizes(30, 90), Sizes(100, 400), Sizes(150, 600), Sizes(300, 1200)])
def test_other_sizes_hold_invariants(sz: Sizes) -> None:
    check("# T\n\n" + w(2500), sz)
    check("para one is here.\n\n" * 700, sz)


def test_repeated_parents_get_unique_ids_by_ordinal() -> None:
    d = check(CASES["repeated_para"])
    assert len(d.parents) >= 2
    assert len({p.parent_id for p in d.parents}) == len(d.parents)


def test_parent_id_formula_and_sensitivity() -> None:
    t = w(3000)
    base = chunk_document(DH, t)
    assert chunk_document(DH, t) == base
    other_hash = chunk_document(hashlib.sha256(b"other").hexdigest(), t)
    assert not {p.parent_id for p in base.parents} & {p.parent_id for p in other_hash.parents}
    assert not {c.chunk_id for c in base.children} & {c.chunk_id for c in other_hash.children}
    changed = chunk_document(DH, "X" + t[1:])  # edit inside the first parent only
    assert changed.parents[0].parent_id != base.parents[0].parent_id
    p0 = base.parents[0]
    assert parent_id(DH, 0, p0.text) == p0.parent_id
    assert parent_id(DH, 1, p0.text) != p0.parent_id  # ordinal matters
    assert parent_id(DH, 0, p0.text + " ") != p0.parent_id  # text matters
    assert parent_id(DH, 0, p0.text) != chunk_id(DH, 0, p0.text)  # "p" separates the namespaces


def test_chunk_id_unchanged_formula_and_parent_independent() -> None:
    d = chunk_document(DH, w(3000))
    for c in d.children:
        assert c.chunk_id == hashlib.sha256((DH + str(c.ordinal) + c.text).encode()).hexdigest()
        assert c.chunk_id != c.parent_id


def test_version_string_follows_sizes() -> None:
    assert chunk_document(DH, "a b").chunker_version == "pc-300-1200-v1"
    assert chunk_document(DH, "").chunker_version == "pc-300-1200-v1"
    versions = {Sizes(c, p).version for c, p in [(300, 1200), (200, 1200), (300, 1000), (100, 400)]}
    assert len(versions) == 4 and all(re.fullmatch(r"pc-\d+-\d+-v1", v) for v in versions)


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    for k in ("CHILD_TOKENS", "PARENT_TOKENS", "PARENT_TOP_N"):
        monkeypatch.delenv(k, raising=False)
    get_chunk_settings.cache_clear()
    yield monkeypatch
    get_chunk_settings.cache_clear()


def test_sizes_from_env(clean_env: pytest.MonkeyPatch) -> None:
    assert Sizes.from_env() == Sizes(300, 1200)
    clean_env.setenv("CHILD_TOKENS", "100")
    clean_env.setenv("PARENT_TOKENS", "400")
    get_chunk_settings.cache_clear()
    assert Sizes.from_env() == Sizes(100, 400)
    assert chunk_document(DH, w(900)).chunker_version == "pc-100-400-v1"


@pytest.mark.parametrize(
    ("child", "parent", "needle"),
    [
        ("0", "1200", "child_tokens"),
        ("-5", "1200", "child_tokens"),
        ("abc", "1200", "child_tokens"),
        ("", "1200", "child_tokens"),
        ("300", "0", "parent_tokens"),
        ("300", "-1", "parent_tokens"),
        ("300", "x", "parent_tokens"),
        ("300", "300", "PARENT_TOKENS"),  # child == parent
        ("1200", "300", "PARENT_TOKENS"),  # child > parent
        ("300", "599", "PARENT_TOKENS"),  # parent < 2 x child
        ("30.5", "1200", "child_tokens"),
    ],
)
def test_invalid_sizes_raise_clear_error(
    clean_env: pytest.MonkeyPatch, child: str, parent: str, needle: str
) -> None:
    clean_env.setenv("CHILD_TOKENS", child)
    clean_env.setenv("PARENT_TOKENS", parent)
    get_chunk_settings.cache_clear()
    with pytest.raises(ValueError, match=needle):
        Sizes.from_env()
    with pytest.raises(ValueError, match=needle):
        chunk_document(DH, "a b c")


def test_invalid_top_n_raises() -> None:
    for bad in (0, -1):
        with pytest.raises(ValueError, match="parent_top_n"):
            ChunkSettings(parent_top_n=bad)
    assert ChunkSettings(parent_top_n=5).parent_top_n == 5


# ---- build_records: only children are embedded ----


class Counting:
    model_id = "counting"

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.calls.append(list(texts))
        return [(1.0, 0.0) for _ in texts]


META = DocMeta("t", "d", DH, "v1", frozenset({"a"}), date(2026, 1, 1))


@pytest.mark.parametrize("name", ["empty", "whitespace"])
def test_empty_documents_never_call_embedder(name: str) -> None:
    emb = Counting()
    parents, chunks = asyncio.run(build_records(chunk_document(DH, CASES[name]), META, emb))
    assert parents == [] and chunks == [] and emb.calls == []


def test_parents_never_embedded_on_edge_docs() -> None:
    for name in ("giant_paragraph", "cjk", "crlf", "heading_only", "table"):
        doc = chunk_document(DH, CASES[name])
        emb = Counting()
        parents, chunks = asyncio.run(build_records(doc, META, emb))
        sent = [t for call in emb.calls for t in call]
        assert sent == [c.embed_text for c in doc.children]
        assert len(parents) == len(doc.parents) and len(chunks) == len(doc.children)
        assert len(emb.calls) == 1
        for p in doc.parents:
            if len(doc.parents) > 1:
                assert p.text not in sent
        assert all(p.token_count > 0 and p.doc_hash == DH for p in parents)
        assert {c.parent_id for c in chunks} <= {p.parent_id for p in parents}


# ---- collapse_to_parents ----


def _parent(pid: str, tokens: int = 10, tenant: str = "t", version: str = "v1") -> ParentChunk:
    text = " ".join(f"w{i}" for i in range(tokens))
    return ParentChunk(
        parent_id=pid, doc_id="d", tenant_id=tenant, text=text, acl_labels=frozenset({"a"}),
        index_version=version, chunker_version="c", ordinal=0, token_count=tokens, doc_hash="h",
        effective_date=date(2026, 1, 1), span=(0, len(text)),
    )  # fmt: skip


def _hit(cid: str, pid: str, score: float, tenant: str = "t", version: str = "v1") -> ScoredChunk:
    c = Chunk(
        chunk_id=cid, doc_id="d", tenant_id=tenant, text="x", embedding=(1.0,),
        acl_labels=frozenset({"a"}), index_version=version, embedding_model="m",
        chunker_version="c", parent_id=pid, span=(0, 2),
    )  # fmt: skip
    return ScoredChunk(c, score)


def test_collapse_empty_inputs() -> None:
    assert collapse_to_parents([], {}, top_n=3, token_budget=100) == []
    assert collapse_to_parents([_hit("c", "p", 1.0)], {}, top_n=3, token_budget=100) == []


def test_collapse_is_input_order_independent() -> None:
    parents = {p: _parent(p) for p in ("pa", "pb", "pc")}
    hits = [
        _hit("c1", "pa", 0.5), _hit("c2", "pa", 0.5), _hit("c3", "pb", 0.5),
        _hit("c4", "pc", 0.9), _hit("c5", "pb", 0.2),
    ]  # fmt: skip
    ref = collapse_to_parents(hits, parents, top_n=3, token_budget=1000)
    assert [o.parent_id for o in ref] == ["pc", "pa", "pb"]
    for perm in itertools.permutations(hits):
        assert collapse_to_parents(list(perm), parents, top_n=3, token_budget=1000) == ref
    pa = ref[1]
    assert [m.chunk_id for m in pa.children] == ["c1", "c2"]  # tie on score -> chunk id


def test_collapse_top_n_and_best_score() -> None:
    parents = {p: _parent(p) for p in ("pa", "pb", "pc")}
    hits = [_hit("c1", "pa", 0.1), _hit("c2", "pa", 0.8), _hit("c3", "pb", 0.7)]
    out = collapse_to_parents(hits, parents, top_n=1, token_budget=1000)
    assert [(o.parent_id, o.score) for o in out] == [("pa", 0.8)]
    assert collapse_to_parents(hits, parents, top_n=-3, token_budget=1000) == []


def test_collapse_first_parent_larger_than_budget_is_trimmed_not_dropped() -> None:
    """Reported behaviour: an oversize best parent is truncated to the budget (truncated=True)."""
    parents = {"pa": _parent("pa", 50), "pb": _parent("pb", 5)}
    out = collapse_to_parents(
        [_hit("c1", "pa", 0.9), _hit("c2", "pb", 0.5)], parents, top_n=3, token_budget=20
    )
    assert [(o.parent_id, o.token_count, o.truncated) for o in out] == [("pa", 20, True)]
    assert count_tokens(out[0].text) == 20  # budget fully used, second parent gets nothing
    exact = collapse_to_parents([_hit("c1", "pa", 0.9)], parents, top_n=3, token_budget=50)
    assert exact[0].truncated is False and exact[0].text == parents["pa"].text


def test_collapse_total_tokens_never_exceed_budget() -> None:
    parents = {f"p{i}": _parent(f"p{i}", 7 + i) for i in range(6)}
    hits = [_hit(f"c{i}", f"p{i}", 1.0 - i / 10) for i in range(6)]
    for budget in range(0, 60, 3):
        out = collapse_to_parents(hits, parents, top_n=6, token_budget=budget)
        assert sum(count_tokens(o.text) for o in out) <= budget


def test_collapse_drops_wrong_index_version_and_tenant() -> None:
    parents = {"pa": _parent("pa", version="v2"), "pb": _parent("pb", tenant="other")}
    hits = [_hit("c1", "pa", 0.9), _hit("c2", "pb", 0.9)]
    assert collapse_to_parents(hits, parents, top_n=3, token_budget=100) == []


def test_collapse_duplicate_child_ids_are_merged() -> None:
    parents = {"pa": _parent("pa")}
    out = collapse_to_parents(
        [_hit("c1", "pa", 0.9), _hit("c1", "pa", 0.9)], parents, top_n=3, token_budget=100
    )
    assert [m.chunk_id for m in out[0].children] == ["c1"]
