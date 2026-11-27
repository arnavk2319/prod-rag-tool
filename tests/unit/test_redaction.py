import json
from pathlib import Path

import pytest

from rag_platform.services.ingest.redaction import luhn_ok, redact

ROOT = Path(__file__).resolve().parents[2]


def test_card_luhn_checked() -> None:
    assert redact("card 4000 0003 5945 7043 ok").text == "card [CARD] ok"
    assert luhn_ok("4000000359457043")
    r = redact("order 4000 0003 5945 7044 shipped")  # Luhn-invalid, not a card
    assert "[CARD]" not in r.text


def test_typed_placeholders() -> None:
    r = redact(
        "mail a.b@example.test, call (212) 555-0100 or 416-555-0105, "
        "SIN 000-776-254, SSN 000-10-5438, account number 12345678901"
    )
    assert r.text == (
        "mail [EMAIL], call [PHONE] or [PHONE], SIN [SIN], SSN [SIN], account number [ACCOUNT]"
    )
    assert r.counts == {"[EMAIL]": 1, "[PHONE]": 2, "[SIN]": 2, "[ACCOUNT]": 1}
    assert r.total == 6


def test_clean_text_untouched() -> None:
    s = "The fee is 10 dollars from 2024-07-01. Section 3.2 limit 5000."
    assert redact(s).text == s


def test_idempotent() -> None:
    once = redact("x a@example.test y").text
    assert redact(once).text == once


@pytest.mark.parametrize("kind", ["email", "phone", "ssn", "sin", "card"])
def test_pii_seed_kinds_fully_redacted(kind: str) -> None:
    items = json.loads((ROOT / "evals/fixtures/pii_seed.yaml").read_text())["items"]
    vals = [i["value"] for i in items if i["kind"] == kind]
    assert vals
    for v in vals:
        out = redact(f"Customer value {v} on file.").text
        assert v not in out
        assert "[" in out
