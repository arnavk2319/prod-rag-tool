"""Parsers for md, txt, html and pdf with size, type-sniffing and decompression limits."""

import time
import zlib
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePath

from bs4 import BeautifulSoup
from bs4.element import Comment
from pypdf import PdfReader
from pypdf.errors import PyPdfError

from rag_platform.services.ingest.reasons import IngestError, Reason

MAX_UPLOAD_BYTES = 20 * 1024 * 1024
MAX_PDF_PAGES = 2000
MAX_PDF_OBJECTS = 200_000
MAX_STREAM_BYTES = 32 * 1024 * 1024  # decompressed, per stream
MAX_TOTAL_DECOMPRESSED = 64 * 1024 * 1024
MAX_PARSE_SECONDS = 20.0
_CHUNK = 1024 * 1024
_IN_SLICE = 64 * 1024

_EXT_KIND = {
    ".md": "md",
    ".markdown": "md",
    ".txt": "txt",
    ".html": "html",
    ".htm": "html",
    ".pdf": "pdf",
}


@dataclass(frozen=True)
class ParsedDoc:
    kind: str
    text: str
    hidden_text: str = ""  # html comments etc.: scanned for injection, never indexed


def sniff(data: bytes) -> str:
    """Content type from bytes: pdf | html | text. Raises parse_error for binary data."""
    if data[:1024].lstrip().startswith(b"%PDF-"):
        return "pdf"
    if b"\x00" in data[:8192]:
        raise IngestError(Reason.PARSE_ERROR)
    head = data[:1024].lstrip().lower()
    if head.startswith((b"<!doctype html", b"<html")):
        return "html"
    return "text"


def check_type(filename: str, data: bytes) -> str:
    kind = _EXT_KIND.get(PurePath(filename).suffix.lower())
    if kind is None:
        raise IngestError(Reason.TYPE_MISMATCH)
    sniffed = sniff(data)
    if kind == "pdf" and sniffed != "pdf":
        raise IngestError(Reason.TYPE_MISMATCH)
    if kind != "pdf" and sniffed == "pdf":
        raise IngestError(Reason.TYPE_MISMATCH)
    if kind in ("md", "txt") and sniffed == "html":
        raise IngestError(Reason.TYPE_MISMATCH)
    return kind


def _decode(data: bytes) -> str:
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise IngestError(Reason.PARSE_ERROR) from e


def _parse_html(data: bytes) -> ParsedDoc:
    soup = BeautifulSoup(_decode(data), "html.parser")
    hidden = "\n".join(str(c) for c in soup.find_all(string=lambda s: isinstance(s, Comment)))
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text("\n")
    return ParsedDoc(
        "html", "\n".join(ln.strip() for ln in text.splitlines() if ln.strip()), hidden
    )


def _check_deadline(deadline: float) -> None:
    if time.monotonic() > deadline:
        raise IngestError(Reason.RESOURCE_LIMIT)


def guard_pdf_streams(data: bytes, deadline: float | None = None) -> None:
    """Cap decompressed stream sizes before pypdf touches the file (zip-bomb defence).

    Works on offsets and a memoryview (no per-stream copies); compressed input is fed in
    bounded slices so `unconsumed_tail` copies stay small.
    """
    deadline = deadline if deadline is not None else time.monotonic() + MAX_PARSE_SECONDS
    view = memoryview(data)
    total = 0
    pos = 0
    while (start := data.find(b"stream", pos)) != -1:
        _check_deadline(deadline)
        if data[max(0, start - 3) : start] == b"end":
            pos = start + 6
            continue
        body = start + 6
        if data[body : body + 2] == b"\r\n":
            body += 2
        elif data[body : body + 1] == b"\n":
            body += 1
        end = data.find(b"endstream", body)
        if end == -1:
            end = len(data)
        pos = end + 9
        d = zlib.decompressobj()
        size = 0
        try:
            off = body
            while off < end and not d.eof:
                piece: bytes | memoryview = view[off : min(off + _IN_SLICE, end)]
                off += _IN_SLICE
                while piece and not d.eof:
                    out = d.decompress(piece, _CHUNK)
                    size += len(out)
                    if size > MAX_STREAM_BYTES or total + size > MAX_TOTAL_DECOMPRESSED:
                        raise IngestError(Reason.RESOURCE_LIMIT)
                    _check_deadline(deadline)
                    piece = d.unconsumed_tail
        except zlib.error:
            continue  # not a flate stream (or damaged): pypdf decides later
        total += size


def _parse_pdf(data: bytes) -> ParsedDoc:
    deadline = time.monotonic() + MAX_PARSE_SECONDS
    guard_pdf_streams(data, deadline)
    try:
        reader = PdfReader(BytesIO(data), strict=False)
        if reader.is_encrypted:
            raise IngestError(Reason.ENCRYPTED_PDF)
        _check_deadline(deadline)
        n_objects = sum(len(v) for v in reader.xref.values()) + len(reader.xref_objStm)
        if len(reader.pages) > MAX_PDF_PAGES or n_objects > MAX_PDF_OBJECTS:
            raise IngestError(Reason.RESOURCE_LIMIT)
        parts: list[str] = []
        for page in reader.pages:
            if time.monotonic() > deadline:
                raise IngestError(Reason.RESOURCE_LIMIT)
            parts.append(page.extract_text() or "")
    except IngestError:
        raise
    except (
        PyPdfError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        RecursionError,
        OSError,
    ) as e:
        raise IngestError(Reason.PARSE_ERROR) from e
    text = "\n\n".join(p.strip() for p in parts if p.strip())
    if not text:
        raise IngestError(Reason.NO_TEXT_LAYER)
    return ParsedDoc("pdf", text)


def parse(data: bytes, filename: str, *, max_bytes: int = MAX_UPLOAD_BYTES) -> ParsedDoc:
    """Validate size and type, then parse. Raises IngestError with a reason code."""
    if len(data) > max_bytes:
        raise IngestError(Reason.FILE_TOO_LARGE)
    kind = check_type(filename, data)
    if kind == "pdf":
        return _parse_pdf(data)
    if kind == "html":
        return _parse_html(data)
    return ParsedDoc(kind, _decode(data))
