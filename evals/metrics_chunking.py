"""Chunking metrics over `data/corpus` text files (parents and children).

chunk_determinism: two independent chunking runs are identical (parent and child ids, texts, spans).
orphan_children: hard zero. A child is an orphan when its parent id is not among its document's
parents, or its text is not inside that parent's text.
"""

import hashlib
from collections.abc import Iterator
from pathlib import Path

from rag_platform.services.ingest.chunker import ChunkedDocument, chunk_document

TEXT_SUFFIXES = {".md", ".txt"}  # text formats only; binary parsing is not this metric's job


def _documents(root: Path) -> Iterator[tuple[str, ChunkedDocument]]:
    for f in sorted((root / "data" / "corpus").rglob("*")):
        if f.suffix not in TEXT_SUFFIXES or not f.is_file():
            continue
        raw = f.read_bytes()
        doc = chunk_document(hashlib.sha256(raw).hexdigest(), raw.decode("utf-8", errors="replace"))
        yield f.relative_to(root).as_posix(), doc


def _run(root: Path) -> list[tuple[object, ...]]:
    return [
        (
            name,
            tuple((p.parent_id, p.text, p.span) for p in doc.parents),
            tuple((c.chunk_id, c.parent_id, c.text, c.span) for c in doc.children),
        )
        for name, doc in _documents(root)
    ]


def chunk_determinism(root: Path) -> float:
    """1.0 when two runs agree and at least one file was chunked, else 0.0."""
    first = _run(root)
    return 1.0 if first and first == _run(root) else 0.0


def orphan_children(root: Path) -> int:
    """Number of children without a valid parent in the same document (must be 0)."""
    bad = 0
    for _, doc in _documents(root):
        parents = {p.parent_id: p for p in doc.parents}
        for c in doc.children:
            p = parents.get(c.parent_id)
            if p is None or c.text not in p.text:
                bad += 1
    return bad
