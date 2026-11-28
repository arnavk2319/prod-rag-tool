"""Collapse, only-children-embedded and copy-forward SQL (T-044)."""

import asyncio
from collections.abc import Sequence
from datetime import date

from rag_platform.adapters.fakes import FakeIndexRepository
from rag_platform.domain.ports import Chunk, ParentChunk, ScoredChunk
from rag_platform.services.ingest.build import DocMeta, build_records
from rag_platform.services.ingest.chunker import chunk_document
from rag_platform.services.ingest.copy_forward import (
    COPY_FORWARD_CHUNKS_SQL,
    COPY_FORWARD_PARENTS_SQL,
    COPY_FORWARD_STATEMENTS,
)
from rag_platform.services.retrieval.collapse import collapse_to_parents


def parent(pid: str, tokens: int = 10, tenant: str = "t", start: int = 0) -> ParentChunk:
    text = " ".join(f"w{i}" for i in range(tokens))
    return ParentChunk(
        parent_id=pid, doc_id="d", tenant_id=tenant, text=text, acl_labels=frozenset({"a"}),
        index_version="v1", chunker_version="c", ordinal=0, token_count=tokens, doc_hash="h",
        effective_date=date(2026, 1, 1), span=(start, start + len(text)),
    )  # fmt: skip


def hit(cid: str, pid: str, score: float, tenant: str = "t", span: tuple[int, int] = (0, 2)):  # type: ignore[no-untyped-def]
    c = Chunk(
        chunk_id=cid, doc_id="d", tenant_id=tenant, text="x", embedding=(1.0,),
        acl_labels=frozenset({"a"}), index_version="v1", embedding_model="m",
        chunker_version="c", parent_id=pid, span=span,
    )  # fmt: skip
    return ScoredChunk(c, score)


def test_best_child_per_parent_top_n_and_tiebreak() -> None:
    parents = {p: parent(p) for p in ("pa", "pb", "pc", "pd")}
    hits = [
        hit("c1", "pb", 0.5), hit("c2", "pb", 0.9), hit("c3", "pa", 0.9),
        hit("c4", "pc", 0.7), hit("c5", "pd", 0.1),
    ]  # fmt: skip
    out = collapse_to_parents(hits, parents, top_n=3, token_budget=1000)
    assert [o.parent_id for o in out] == ["pa", "pb", "pc"]  # tie on 0.9 -> parent id
    assert [m.chunk_id for m in out[1].children] == ["c2", "c1"]
    assert out[1].score == 0.9 and out[0].text == parents["pa"].text
    assert out == collapse_to_parents(list(reversed(hits)), parents, top_n=3, token_budget=1000)


def test_orphan_and_cross_tenant_children_are_dropped() -> None:
    parents = {"p1": parent("p1"), "px": parent("px", tenant="other")}
    hits = [hit("c1", "missing", 0.9), hit("c2", "px", 0.8), hit("c3", "p1", 0.1)]
    out = collapse_to_parents(hits, parents, top_n=3, token_budget=1000)
    assert [o.parent_id for o in out] == ["p1"]


def test_budget_trims_deterministically() -> None:
    parents = {"pa": parent("pa", 30), "pb": parent("pb", 30)}
    hits = [hit("c1", "pa", 0.9), hit("c2", "pb", 0.8)]
    out = collapse_to_parents(hits, parents, top_n=3, token_budget=40)
    assert [o.token_count for o in out] == [30, 10] and [o.truncated for o in out] == [False, True]
    assert out[1].text == " ".join(f"w{i}" for i in range(10))
    assert collapse_to_parents(hits, parents, top_n=3, token_budget=40) == out
    assert collapse_to_parents(hits, parents, top_n=3, token_budget=0) == []
    assert collapse_to_parents(hits, parents, top_n=0, token_budget=100) == []


def test_trimmed_passage_drops_invisible_children() -> None:
    parents = {"pa": parent("pa", 30)}
    hits = [hit("late", "pa", 0.9, span=(150, 160))]  # beyond the 5-token visible prefix
    assert collapse_to_parents(hits, parents, top_n=3, token_budget=5) == []


class CountingEmbedder:
    model_id = "counting"

    def __init__(self) -> None:
        self.seen: list[str] = []

    async def embed(self, texts: Sequence[str]) -> list[tuple[float, ...]]:
        self.seen.extend(texts)
        return [(1.0, 0.0) for _ in texts]


def test_only_children_are_embedded() -> None:
    text = "# T\n\n" + " ".join(f"w{i}." if i % 20 == 19 else f"w{i}" for i in range(3000))
    doc = chunk_document("h", text)
    assert len(doc.parents) >= 2
    emb = CountingEmbedder()
    meta = DocMeta("t", "d", "h", "v1", frozenset({"a"}), date(2026, 1, 1))
    parents, chunks = asyncio.run(build_records(doc, meta, emb))
    assert len(emb.seen) == len(doc.children) == len(chunks)
    assert emb.seen == [c.embed_text for c in doc.children]
    parent_texts = {p.text for p in doc.parents}
    assert not parent_texts & set(emb.seen)
    assert {c.parent_id for c in chunks} == {p.parent_id for p in parents}
    assert all(c.chunker_version == "pc-300-1200-v1" for c in chunks)

    async def roundtrip() -> int:
        repo = FakeIndexRepository()
        await repo.add_parents(parents)
        got = await repo.get_parents(
            tenant_id="t", index_version="v1", parent_ids=[c.parent_id for c in chunks]
        )
        return len(got)

    assert asyncio.run(roundtrip()) == len(parents)


def test_copy_forward_copies_parents_with_children() -> None:
    assert COPY_FORWARD_STATEMENTS == (COPY_FORWARD_PARENTS_SQL, COPY_FORWARD_CHUNKS_SQL)
    for sql in COPY_FORWARD_STATEMENTS:
        assert "INSERT INTO" in sql and "SELECT" in sql and "tenant_id = %(tenant_id)s" in sql
    assert "INSERT INTO parent_chunks" in COPY_FORWARD_PARENTS_SQL
    assert (
        "INSERT INTO chunks" in COPY_FORWARD_CHUNKS_SQL and "parent_id" in COPY_FORWARD_CHUNKS_SQL
    )


def test_parent_of_other_document_or_acl_is_dropped() -> None:
    other_doc = parent("pd")
    other_doc = ParentChunk(**{**other_doc.__dict__, "doc_id": "other"})
    other_acl = ParentChunk(**{**parent("pl").__dict__, "acl_labels": frozenset({"a", "secret"})})
    parents = {"pd": other_doc, "pl": other_acl, "ok": parent("ok")}
    hits = [hit("c1", "pd", 0.9), hit("c2", "pl", 0.8), hit("c3", "ok", 0.1)]
    out = collapse_to_parents(hits, parents, top_n=3, token_budget=1000)
    assert [o.parent_id for o in out] == ["ok"]
