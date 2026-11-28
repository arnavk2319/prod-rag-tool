"""Adversarial 1 MB inputs must not cause super-linear regex time (redact, detect, assess)."""

import time

import pytest
from evals import metrics_pii

from rag_platform.services.ingest.injection import find_injection
from rag_platform.services.ingest.quarantine import assess
from rag_platform.services.ingest.reasons import MAX_TEXT_CHARS, IngestError, Outcome, Reason
from rag_platform.services.ingest.redaction import redact

N = 1_000_000
SHAPES = {
    "digits": "1" * N,
    "digit_space": "1 " * (N // 2),
    "digit_dash": "1-" * (N // 2),
    "digit_dash_end": "1" * N + "-",
    "spaces": " " * N,
    "word": "a" * N,
    "at": "@" * N,
    "a_at": "a@" * (N // 2),
    "plus": "+" * N,
    "plus_one": "+1" * (N // 2),
    "domain_dots": "a@" + "b." * (N // 2),
    "many_emails_noTLD": "x@y " * (N // 4),
    "account_kw": "account " * (N // 8),
    "account_digits": "account " + "1 " * (N // 2),
    "account_spaces": "account" + " " * N,
    "ignore": "ignore " * (N // 7),
    "ignore_punct": "ignore" + "!" * N,
    "letter_spaced": "i " * (N // 2),
    "letter_spaced_tail": "a " * (N // 2) + "1",
    "zero_width": "​" * N,
    "display_none": "display:none " * (N // 13),
    "md_image": "![" * (N // 2),
    "md_image_url": "![x](http://a" * (N // 13),
    "phone_like": "(123) " * (N // 6),
}


# Per-call bound (worst case measured under 1 s); 5 s keeps it stable under heavy load.
LINEAR_BOUND_S = 5.0


@pytest.mark.parametrize("shape", sorted(SHAPES))
def test_redact_and_detect_linear(shape: str) -> None:
    text = SHAPES[shape]
    t0 = time.monotonic()
    redact(text)
    find_injection(text)
    assert time.monotonic() - t0 < LINEAR_BOUND_S, shape


@pytest.mark.parametrize("shape", ["digits", "word", "a_at", "ignore", "letter_spaced", "spaces"])
def test_assess_linear(shape: str) -> None:
    t0 = time.monotonic()
    assess(SHAPES[shape].encode(), "a.txt")
    assert time.monotonic() - t0 < LINEAR_BOUND_S, shape


def test_text_over_cap_is_refused_not_skipped() -> None:
    big = "a" * (MAX_TEXT_CHARS + 1)
    with pytest.raises(IngestError) as ei:
        redact(big)
    assert ei.value.reason is Reason.RESOURCE_LIMIT
    with pytest.raises(IngestError):
        find_injection(big)
    d = assess(big.encode(), "a.txt")
    assert (d.outcome, d.reason, d.text) == (Outcome.REJECTED, Reason.RESOURCE_LIMIT, "")


def test_pii_metrics_meet_thresholds() -> None:
    res = metrics_pii.evaluate()
    assert res["metrics"]["pii_recall"] >= 0.98
    assert res["hard_zero"]["pii_leak"] == 0
    assert res["pii"]["out_of_scope_kinds"] == ["name"]
