"""Deterministic, pure parent-child chunker: ~1200-token parents, ~300-token children.

Token approximation (documented, deterministic, no model download): a token is a maximal run of
word characters (`\\w+`) or a single non-space, non-word character. This is close to a BPE count
for English prose within about 20-30%, which is enough for sizing windows.

Sizes come from CHILD_TOKENS and PARENT_TOKENS (defaults 300 and 1200, see `chunk_settings`).
Child bounds are 2/3..7/6 of the child target (200-350), parent bounds 2/3..5/4 of the parent
target (800-1500). The sizes are part of `chunker_version` (`pc-<child>-<parent>-v1`).

Parents: the document is split on heading boundaries first, then blank-line (paragraph)
boundaries, then sentence ends, then the target size, always inside the parent bounds and the
break closest to the target. Parents do not overlap and tile the document: parent spans are
contiguous from 0 to len(text), so concatenating parent texts reproduces the input exactly. Only
the final parent may be shorter than the minimum. `parent_id = sha256(doc_hash + "p" +
parent_ordinal + parent_text)`.

Children: windows slide inside ONE parent, never across a parent boundary. A window is 200-350
tokens (at the default size), its end snapped back to a paragraph break or sentence end when one
exists in range (else the target). The next window starts `round(0.12 * window)` tokens before the
previous end. Only the last child of a parent may be shorter than the minimum. Child text is the
exact source slice. Child ordinals run across the whole document and the child `chunk_id` formula
is unchanged: sha256(doc_hash + ordinal + child_text). Only children are embedded.

Last child of a parent: if it would hold fewer than 10% new (non-overlap) tokens, the previous
window end is moved back (never below the minimum) until it holds at least 10%; if no end
satisfies this the original end is kept.

`chunk_text` is the same child windowing over a whole text with no parents (parent_id is empty);
it is the building block and is kept for single-level callers and tests.

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

from rag_platform.chunk_settings import get_chunk_settings

OVERLAP_RATIO = 0.12
MIN_NEW_RATIO = 0.10

_TOKEN = re.compile(r"\w+|[^\w\s]")
_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)(?:[ \t]+#+)?[ \t]*\r?$", re.M)
_FENCE = re.compile(r"^ {0,3}(`{3,}|~{3,})", re.M)
_BLANK = re.compile(r"\r?\n[ \t]*\r?\n")
_SENTENCE_END = frozenset(".!?")


@dataclass(frozen=True)
class Sizes:
    """Child and parent target sizes in tokens; bounds derive from them."""

    child: int = 300
    parent: int = 1200

    @property
    def child_min(self) -> int:
        return (2 * self.child) // 3

    @property
    def child_max(self) -> int:
        return (7 * self.child) // 6

    @property
    def parent_min(self) -> int:
        return (2 * self.parent) // 3

    @property
    def parent_max(self) -> int:
        return (5 * self.parent) // 4

    @property
    def version(self) -> str:
        return f"pc-{self.child}-{self.parent}-v1"

    @classmethod
    def from_env(cls) -> "Sizes":
        cfg = get_chunk_settings()
        return cls(child=cfg.child_tokens, parent=cfg.parent_tokens)


_DEFAULT = Sizes()
CHUNKER_VERSION = _DEFAULT.version
MIN_TOKENS = _DEFAULT.child_min
TARGET_TOKENS = _DEFAULT.child
MAX_TOKENS = _DEFAULT.child_max
PARENT_MIN_TOKENS = _DEFAULT.parent_min
PARENT_TARGET_TOKENS = _DEFAULT.parent
PARENT_MAX_TOKENS = _DEFAULT.parent_max


@dataclass(frozen=True)
class ChunkSpec:
    """A child chunk: the unit that is embedded and searched."""

    ordinal: int
    chunk_id: str
    text: str
    span: tuple[int, int]
    tokens: int
    overlap_tokens: int
    heading_path: tuple[str, ...]
    parent_id: str = ""

    @property
    def embed_text(self) -> str:
        """Text to embed: heading path prepended (the stored `text` stays the raw slice)."""
        if not self.heading_path:
            return self.text
        return " > ".join(self.heading_path) + "\n" + self.text


@dataclass(frozen=True)
class ParentSpec:
    """A parent chunk: returned to the generator, never embedded or searched."""

    ordinal: int
    parent_id: str
    text: str
    span: tuple[int, int]
    tokens: int
    heading_path: tuple[str, ...]


@dataclass(frozen=True)
class ChunkedDocument:
    parents: tuple[ParentSpec, ...]
    children: tuple[ChunkSpec, ...]
    chunker_version: str


def count_tokens(text: str) -> int:
    return len(_TOKEN.findall(text))


def truncate_tokens(text: str, limit: int) -> str:
    """Longest prefix of `text` holding at most `limit` tokens, cut after the last kept token."""
    if limit <= 0:
        return ""
    for i, m in enumerate(_TOKEN.finditer(text), start=1):
        if i == limit:
            return text[: m.end()]
    return text


def chunk_id(doc_hash: str, ordinal: int, text: str) -> str:
    return hashlib.sha256((doc_hash + str(ordinal) + text).encode()).hexdigest()


def parent_id(doc_hash: str, ordinal: int, text: str) -> str:
    return hashlib.sha256((doc_hash + "p" + str(ordinal) + text).encode()).hexdigest()


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


def _snap_end(text: str, toks: list[tuple[int, int]], pos: int, sz: Sizes) -> int:
    """Token index (exclusive) to end a window that starts at `pos`."""
    for end in range(pos + sz.child_max, pos + sz.child_min - 1, -1):
        prev_end = toks[end - 1][1]
        if text[prev_end - 1] in _SENTENCE_END or "\n\n" in text[prev_end : toks[end][0]]:
            return end
    return pos + sz.child


def _tail_ok(hi: int, pos: int, end: int, sz: Sizes) -> bool:
    ov = round(OVERLAP_RATIO * (end - pos))
    tail = hi - (end - ov)
    return tail > sz.child_max or (hi - end) >= MIN_NEW_RATIO * tail


def _balanced_end(hi: int, pos: int, end: int, sz: Sizes) -> int:
    if _tail_ok(hi, pos, end, sz):
        return end
    for e in range(end - 1, pos + sz.child_min - 1, -1):
        if _tail_ok(hi, pos, e, sz):
            return e
    return end


def _windows(
    text: str,
    toks: list[tuple[int, int]],
    marks: list[tuple[int, int, str]],
    lo: int,
    hi: int,
    sz: Sizes,
    *,
    doc_hash: str,
    first_ordinal: int,
    parent: str,
) -> list[ChunkSpec]:
    """Child windows over token range [lo, hi); never reads or emits outside it."""
    out: list[ChunkSpec] = []
    pos = lo
    overlap = 0
    while pos < hi:
        end = (
            hi
            if hi - pos <= sz.child_max
            else _balanced_end(hi, pos, _snap_end(text, toks, pos, sz), sz)
        )
        start_c, end_c = toks[pos][0], toks[end - 1][1]
        piece = text[start_c:end_c]
        ordinal = first_ordinal + len(out)
        out.append(
            ChunkSpec(
                ordinal=ordinal,
                chunk_id=chunk_id(doc_hash, ordinal, piece),
                text=piece,
                span=(start_c, end_c),
                tokens=end - pos,
                overlap_tokens=overlap,
                heading_path=_path_at(marks, start_c),
                parent_id=parent,
            )
        )
        if end >= hi:
            break
        overlap = round(OVERLAP_RATIO * (end - pos))
        pos = end - overlap
    return out


def _closest(cands: list[int], target: int) -> int:
    return min(cands, key=lambda e: (abs(e - target), e))


def _parent_end(
    text: str,
    toks: list[tuple[int, int]],
    heading_starts: frozenset[int],
    pos: int,
    n: int,
    sz: Sizes,
) -> int:
    """Exclusive token end of the parent starting at `pos` (requires n - pos > parent_max)."""
    lo, hi = pos + sz.parent_min, pos + sz.parent_max
    hi = min(hi, n - sz.parent_min)  # keep the rest at least parent_min when possible
    if hi < lo:
        hi = lo
    ends = range(lo, hi + 1)
    headings = [e for e in ends if toks[e][0] in heading_starts]
    if headings:
        return _closest(headings, pos + sz.parent)
    paras = [e for e in ends if _BLANK.search(text, toks[e - 1][1], toks[e][0])]
    if paras:
        return _closest(paras, pos + sz.parent)
    sentences = [e for e in ends if text[toks[e - 1][1] - 1] in _SENTENCE_END]
    if sentences:
        return _closest(sentences, pos + sz.parent)
    return min(max(pos + sz.parent, lo), hi)


def chunk_text(doc_hash: str, text: str, sizes: Sizes | None = None) -> list[ChunkSpec]:
    """Single-level child windows over the whole text (no parents; `parent_id` is empty)."""
    sz = sizes or Sizes.from_env()
    toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    if not toks:
        return []
    return _windows(
        text,
        toks,
        _heading_marks(text),
        0,
        len(toks),
        sz,
        doc_hash=doc_hash,
        first_ordinal=0,
        parent="",
    )


def chunk_document(doc_hash: str, text: str, sizes: Sizes | None = None) -> ChunkedDocument:
    """Split into parents (tiling the text) and children (inside exactly one parent)."""
    sz = sizes or Sizes.from_env()
    toks = [(m.start(), m.end()) for m in _TOKEN.finditer(text)]
    n = len(toks)
    if n == 0:
        return ChunkedDocument((), (), sz.version)
    marks = _heading_marks(text)
    heading_starts = frozenset(m[0] for m in marks)
    bounds = [0]
    while n - bounds[-1] > sz.parent_max:
        bounds.append(_parent_end(text, toks, heading_starts, bounds[-1], n, sz))
    bounds.append(n)
    parents: list[ParentSpec] = []
    children: list[ChunkSpec] = []
    for ordinal, (lo, hi) in enumerate(zip(bounds, bounds[1:], strict=False)):
        start = 0 if ordinal == 0 else toks[lo][0]
        end = len(text) if hi == n else toks[hi][0]
        ptext = text[start:end]
        pid = parent_id(doc_hash, ordinal, ptext)
        parents.append(
            ParentSpec(
                ordinal=ordinal,
                parent_id=pid,
                text=ptext,
                span=(start, end),
                tokens=hi - lo,
                heading_path=_path_at(marks, toks[lo][0]),
            )
        )
        children.extend(
            _windows(
                text,
                toks,
                marks,
                lo,
                hi,
                sz,
                doc_hash=doc_hash,
                first_ordinal=len(children),
                parent=pid,
            )
        )
    return ChunkedDocument(tuple(parents), tuple(children), sz.version)
