"""Deterministic, pure chunker: 300-500 tokens, 12% overlap, heading path, content-hash ids.

Token approximation (documented, deterministic, no model download): a token is a maximal run of
word characters (`\\w+`) or a single non-space, non-word character. This is close to a BPE count
for English prose within about 20-30%, which is enough for sizing windows.

Windows slide over the whole text. A window is 300-500 tokens, its end snapped back to a
paragraph break or sentence end when one exists in range (else 400 tokens). The next window
starts `round(0.12 * window)` tokens before the previous end. Only the final chunk may be
shorter than 300 tokens. Chunk text is the exact source slice, so no non-whitespace text is lost.

Last chunk: if it would hold fewer than 10% new (non-overlap) tokens, the previous window end
is moved back (never below 300 tokens) until it holds at least 10%; if no end satisfies this the
original end is kept. Merging into the previous chunk is impossible within the 500 bound, because
a window that could absorb the tail would already have been the final one.

Headings: only a CommonMark closing sequence (whitespace then `#`s at the end) is stripped, so
`# Learn C#` keeps the title `Learn C#`.

Line endings: `\n` and `\r\n` are both recognised when parsing headings (a trailing `\r` is never
part of a title). Chunk text and spans are NOT normalised: they are exact slices of the input,
so CRLF input yields CRLF text and the no-text-lost property holds without any modulo.
Headings inside fenced code blocks (``` or ~~~, closed by the same char and at least as long)
are ignored, so a `# comment` in code is not a heading.
"""

import hashlib
import re
from dataclasses import dataclass

CHUNKER_VERSION = "chunker-v1"
MIN_TOKENS = 300
TARGET_TOKENS = 400
MAX_TOKENS = 500
OVERLAP_RATIO = 0.12

_TOKEN = re.compile(r"\w+|[^\w\s]")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*\r?$", re.M)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})", re.M)
_SENTENCE_END = frozenset(".!?")


@dataclass(frozen=True)
class ChunkSpec:
    ordinal: int
    chunk_id: str
    text: str
    span: tuple[int, int]
    tokens: int
    overlap_tokens: int
    heading_path: tuple[str, ...]

    @property
    def embed_text(self) -> str:
        """Text to embed: heading path prepended (the stored `text` stays the raw slice)."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def chunk_id(doc_hash: str, ordinal: int, text: str) -> str:
    return hashlib.sha256((doc_hash + str(ordinal) + text).encode()).hexdigest()


def _fenced_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    open_at: int | None = None
    marker = ""
    for m in _FENCE.finditer(text):
        fence = m.group(1)
        if open_at is None:
            open_at, marker = m.start(), fence
        elif fence[0] == marker[0] and len(fence) >= len(marker):
            ranges.append((open_at, m.end()))
            open_at = None
    if open_at is not None:
        ranges.append((open_at, len(text)))
    return ranges


def _heading_marks(text: str) -> list[tuple[int, int, str]]:
    fenced = _fenced_ranges(text)
    return [
        (m.start(), len(m.group(1)), m.group(2))
        for m in _HEADING.finditer(text)
        if not any(a <= m.start() < b for a, b in fenced)
    ]


def _path_at(marks: list[tuple[int, int, str]], offset: int) -> tuple[str, ...]:
    stack: list[tuple[int, str]] = []
    for start, level, title in marks:
        if start > offset:
            break
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
    return tuple(t for _, t in stack)


def _snap_end(text: str, toks: list[tuple[int, int]], pos: int) -> int:
    """Token index (exclusive) to end a window that starts at `pos`; 300 <= end-pos <= 500."""
    for end in range(pos + MAX_TOKENS, pos + MIN_TOKENS - 1, -1):
        prev_end = toks[end - 1][1]
        if text[prev_end - 1] in _SENTENCE_END or "\n\n" in text[prev_end : toks[end][0]]:
            return end
    return pos + TARGET_TOKENS


MIN_NEW_RATIO = 0.10


def _tail_ok(n: int, pos: int, end: int) -> bool:
    ov = round(OVERLAP_RATIO * (end - pos))
    tail = n - (end - ov)
    return tail > MAX_TOKENS or (n - end) >= MIN_NEW_RATIO * tail


def _balanced_end(n: int, pos: int, end: int) -> int:
    if _tail_ok(n, pos, end):
        return end
    for e in range(end - 1, pos + MIN_TOKENS - 1, -1):
        if _tail_ok(n, pos, e):
            return e
    return end


def chunk_text(doc_hash: str, text: str) -> list[ChunkSpec]:
    toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    n = len(toks)
    if n == 0:
        return []
    marks = _heading_marks(text)
    out: list[ChunkSpec] = []
    pos = 0
    overlap = 0
    while True:
        end = n if n - pos <= MAX_TOKENS else _balanced_end(n, pos, _snap_end(text, toks, pos))
        start_c, end_c = toks[pos][0], toks[end - 1][1]
        piece = text[start_c:end_c]
        ordinal = len(out)
        out.append(
            ChunkSpec(
                ordinal=ordinal,
                chunk_id=chunk_id(doc_hash, ordinal, piece),
                text=piece,
                span=(start_c, end_c),
                tokens=end - pos,
                overlap_tokens=overlap,
                heading_path=_path_at(marks, start_c),
            )
        )
        if end >= n:
            return out
        overlap = round(OVERLAP_RATIO * (end - pos))
        pos = end - overlap
