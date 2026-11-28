import time
import tracemalloc
import zlib

import pytest

from rag_platform.services.ingest.parsers import MAX_UPLOAD_BYTES, parse
from rag_platform.services.ingest.reasons import IngestError, Reason


def _reason(data: bytes, name: str, **kw: int) -> Reason:
    with pytest.raises(IngestError) as ei:
        parse(data, name, **kw)
    return ei.value.reason


def test_md_and_txt() -> None:
    assert parse(b"# Title\n\nbody", "a.md").text.startswith("# Title")
    assert parse(b"plain", "a.txt").kind == "txt"


def test_html_strips_scripts_and_keeps_comments_hidden() -> None:
    doc = parse(b"<html><script>x()</script><p>Hi</p><!-- secret --></html>", "a.html")
    assert doc.text == "Hi"
    assert "secret" in doc.hidden_text


def _pdf(objs: list[bytes]) -> bytes:
    out = b"%PDF-1.4\n"
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += b"%d 0 obj\n" % i + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offs)
    return out + b"trailer<</Size %d/Root 1 0 R>>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)


def _page_objs(content_obj: bytes) -> list[bytes]:
    return [
        b"<</Type/Catalog/Pages 2 0 R>>",
        b"<</Type/Pages/Kids[3 0 R]/Count 1>>",
        b"<</Type/Page/Parent 2 0 R/MediaBox[0 0 200 200]/Contents 4 0 R"
        b"/Resources<</Font<</F1 5 0 R>>>>>>",
        content_obj,
        b"<</Type/Font/Subtype/Type1/BaseFont/Helvetica>>",
    ]


def test_pdf_text_extracted() -> None:
    body = b"BT /F1 12 Tf 20 100 Td (Hello) Tj ET"
    pdf = _pdf(_page_objs(b"<</Length %d>>\nstream\n" % len(body) + body + b"\nendstream"))
    assert "Hello" in parse(pdf, "a.pdf").text


def test_size_limit_default_is_20mb() -> None:
    assert MAX_UPLOAD_BYTES == 20 * 1024 * 1024
    assert _reason(b"A" * (MAX_UPLOAD_BYTES + 1), "a.txt") is Reason.FILE_TOO_LARGE
    assert len(parse(b"A" * MAX_UPLOAD_BYTES, "a.txt").text) == MAX_UPLOAD_BYTES


def test_type_mismatch_and_unknown_extension() -> None:
    assert _reason(b"# markdown", "a.pdf") is Reason.TYPE_MISMATCH
    assert _reason(b"%PDF-1.4 x", "a.md") is Reason.TYPE_MISMATCH
    assert _reason(b"<!DOCTYPE html><p>x</p>", "a.txt") is Reason.TYPE_MISMATCH
    assert _reason(b"x", "a.exe") is Reason.TYPE_MISMATCH


def test_binary_and_bad_utf8_are_parse_errors() -> None:
    assert _reason(b"ab\x00cd", "a.txt") is Reason.PARSE_ERROR
    assert _reason(b"\xff\xfe\xfa bad", "a.txt") is Reason.PARSE_ERROR


def test_truncated_pdf_is_parse_error() -> None:
    assert _reason(b"%PDF-1.4\n1 0 obj\n<< /Type /Catalog", "a.pdf") is Reason.PARSE_ERROR


def test_zip_bomb_pdf_rejected_within_time_and_memory_bound() -> None:
    # ~200 MB of zeros deflates to ~200 KB; must be refused without inflating it.
    comp = zlib.compress(bytes(200 * 1024 * 1024), 9)
    pdf = _pdf(
        _page_objs(
            b"<</Filter/FlateDecode/Length %d>>\nstream\n" % len(comp) + comp + b"\nendstream"
        )
    )
    assert len(pdf) < 1024 * 1024
    tracemalloc.start()
    t0 = time.monotonic()
    reason = _reason(pdf, "bomb.pdf")
    elapsed = time.monotonic() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert reason is Reason.RESOURCE_LIMIT
    assert elapsed < 5.0
    assert peak < 96 * 1024 * 1024
