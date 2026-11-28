# Changelog

## 0.1.0 - Foundation (unreleased until tagged)

No end-to-end ingest, retrieval, generation or UI yet; see [docs/status.md](docs/status.md).
Changes are tracked in pull requests (numbers below).

### Tooling and CI
- #1 Track evals, tests and scripts
- #2 CI hygiene and security hardening
- #3 Repository private-mode housekeeping
- #5 CI migration-approval check ignores placeholder files
- #15 Fix security scan finding in migration 0001

### Schema, stack and contract
- #6 Ports, in-memory fakes, error model, import-boundary test
- #7 Alembic 0001 schema (row-level security, partitioned chunks)
- #8 Docker image, Compose profiles, `make up` and `make doctor`, memory measurement
- #9 OpenAPI v1 contract with stub routes
- #12 Document MLflow tracking URI in the environment template

### Evals
- #10 Eval harness, `make eval`, policy check, case-level diff
- #11 Synthetic corpus and ingest fixtures

### Ingest building blocks
- #13 Chunker and embedder adapters
- #14 Parsing, PII redaction and quarantine
- #17 Pipeline policy gate (core logic)
- #18 Parent-child chunking
- #21 Pipeline policy gate: SQL tables, repository and hash-chained audit function

### Security
- #19 Refuse default secrets when the environment is production
- #20 Explicit MLflow allowed hosts instead of a wildcard
- #22 Row-level security on every tenant table and a hardened audit append
- #23 Suppress a semgrep false positive on role switching

### Documentation
- #16 Public status document and generator
- #24 Skip the git-history status test on shallow clones
