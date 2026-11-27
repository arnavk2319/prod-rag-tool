# Synthetic data

All data in `data/corpus/` and `evals/fixtures/` is synthetic. Banks ("Alder Test Bank", "Birch Test Credit Union"),
people, emails (`example.test`), phones (reserved `555-01xx`), SSN/SIN (`000-` prefix, invalid by construction) and
card numbers (Luhn-valid, `400000`/`555555` test prefixes) are invented. No real people, companies or confidential text.

Regenerate (deterministic, stdlib only, no timestamps):

    python -m data.generate            # writes into the repo
    python -m data.generate --out DIR  # writes data/corpus and evals/fixtures under DIR

Structured files (`manifest.yaml`, `pii_seed.yaml`, `injection_corpus.yaml`, `expected.yaml`) are JSON text, which is
valid YAML; read them with `json.load`. Manifest fields mirror the `documents` table: `tenant`, `labels`
(`acl_labels`), `effective_date`, `expires_date`, `status` (`active`/`superseded`).

Oversized fixture: not stored; generate `max_upload_bytes + 1` bytes at test time (limit 10 MiB, see
`evals/fixtures/ingest/oversized/expected.yaml`).
