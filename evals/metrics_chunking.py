"""chunk_determinism: two independent chunking runs over the corpus text files are identical."""

import hashlib
from pathlib import Path

from rag_platform.services.ingest.chunker import chunk_text

TEXT_SUFFIXES = {".md", ".txt"}  # text formats only; binary parsing is not this metric's job


def _run(root: Path) -> list[tuple[str, tuple[tuple[str, str, tuple[int, int]], ...]]]:
    out = []
    for f in sorted((root / "data" / "corpus").rglob("*")):
        if f.suffix not in TEXT_SUFFIXES or not f.is_file():
            continue
        raw = f.read_bytes()
        chunks = chunk_text(hashlib.sha256(raw).hexdigest(), raw.decode("utf-8", errors="replace"))
        out.append(
            (f.relative_to(root).as_posix(), tuple((c.chunk_id, c.text, c.span) for c in chunks))
        )
    return out


def chunk_determinism(root: Path) -> float:
    """1.0 when two runs agree and at least one file was chunked, else 0.0."""
    first = _run(root)
    return 1.0 if first and first == _run(root) else 0.0
