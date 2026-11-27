# Ingest safety (parse, redact, quarantine)

Code: `src/rag_platform/services/ingest/`. Pure functions, no DB or network I/O.

## Stored text and citations

Text that is stored and indexed is the redacted text after NFKC normalisation and removal of
zero-width/soft-hyphen characters. Citation spans refer to this stored text. Offsets into the
original document are not preserved (normalisation and placeholders change lengths).

## Limits

- Upload 20 MiB; extracted text over 8,000,000 characters is refused as `resource_limit`
  (never partly scanned, because skipping would leak PII). All regexes are linear-time
  (1 MB adversarial timing tests in `tests/unit/test_ingest_redos.py`).
- PDF: 2000 pages, 200k objects, 32 MiB per decompressed stream, 64 MiB total, 20 s deadline.
- Density: more than 5 redactions per 1,000 tokens (documents of 200+ tokens), or 25+ redactions in any document.

## Reason codes

| Code | Outcome | Meaning |
| ---- | ------- | ------- |
| pii_redacted | accepted | indexed after redaction |
| no_text_layer | quarantined | scanned-like PDF, no text |
| garbled_text | quarantined | mojibake or control characters |
| prompt_injection_suspected | quarantined | injection pattern matched |
| pii_density_high | quarantined | too much PII |
| encrypted_pdf | quarantined | needs a human |
| parse_error | rejected | unreadable or truncated |
| file_too_large | rejected | over upload limit |
| type_mismatch | rejected | extension and content disagree |
| duplicate_hash | rejected | same tenant and content hash already known |
| resource_limit | rejected | zip-bomb-like or oversized text |

## Accepted risks

- Personal names are not redacted (no safe regex); the PII eval covers the structured kinds only
  (85 of 100 seeds).
- CSS-hidden HTML text is extracted and indexed (it is injection-scanned).
- Luhn-invalid 10+ digit runs become `[ACCOUNT]` (conservative over-redaction).
- A bare 9-digit SIN without a keyword is not redacted.
- Injection detection is heuristic, not a guarantee.
