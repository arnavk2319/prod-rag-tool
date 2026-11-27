# ADR-0004: API contract and versioning

Status: accepted

## Context

The web UI, the ops views and the backend are built at different times. Without a single agreed contract, request and response shapes drift and the UI breaks silently. The contract must exist before the routes are implemented, and must stay enforceable afterwards.

## Decision

1. **Source of truth.** A checked-in OpenAPI 3.1 snapshot (`tests/contract/openapi.snapshot.json`) is the contract. The FastAPI app must produce the same document. A contract test builds the live OpenAPI from the app and diffs it against the snapshot, failing with a readable diff. The snapshot is regenerated only deliberately (`make contract-update`).
2. **Coverage.** A checked-in list of path+method pairs is compared to the snapshot, so a dropped endpoint fails a test rather than disappearing quietly.
3. **Versioning.** All product routes live under `/v1`. Ops probes (`/healthz`, `/readyz`, `/metrics`) are unversioned. Any contract change (path, parameter, schema, error code) requires, in the same change: the snapshot update, a regenerated UI client, and a note in this ADR (or a superseding one). Additive, optional fields stay in `/v1`; removals or semantic changes need `/v2`.
4. **Error envelope.** Every non-2xx response references one shared component: `{error:{code,message,request_id,details?}}`. Codes: `unauthenticated`, `forbidden`, `not_found`, `validation_failed`, `conflict`, `rate_limited`, `payload_too_large`, `dependency_unavailable`, `internal_error`. `rate_limited` responses carry a `Retry-After` header (seconds). `request_id` equals the echoed `X-Request-ID`. Messages never include raw user content or PII.
5. **Headers.** `Authorization: Bearer <jwt>` on all authenticated routes. `X-Request-ID` is accepted and always echoed. `Idempotency-Key` is declared on creating POSTs (documents, chats, messages).
6. **Idempotency.** The key is scoped per tenant and user. Replaying the same key with the same body returns the original result (same status and body). The same key with a different body returns `conflict`. Keys expire after 24 hours.
7. **Pagination.** List endpoints use `limit` (default 50, max 200) and an opaque `cursor`. Responses are `{items, next_cursor}`; `next_cursor` is null on the last page. Ordering is stable (newest first, id as tiebreaker).
8. **Stub routes.** In P1 routes may be stubs, as long as the generated OpenAPI matches the snapshot. Stubs return `501 Not Implemented` with the shared error envelope (code `internal_error`, message "not implemented"), never fake success data. Real behaviour replaces them in later phases without contract changes.
9. **Fixtures.** Each operation has an example success payload and at least one error payload in `tests/contract/fixtures/`, keyed by operationId. Tests validate every fixture against the matching response schema in the snapshot (JSON Schema validation with `$ref` resolution), and error fixtures against the shared envelope.

## Implementation notes

- Document upload is JSON with `content_base64` (no multipart dependency).
- Deletes return 200 with `{id, status}`.
- Index version actions are `POST /v1/admin/index/versions/{version}/{stage|commit|rollback}`.
- HTTP statuses without a dedicated code (for example 405) map to `validation_failed`; 504 maps to `dependency_unavailable` and keeps its status.
- `request_id` takes a valid incoming `X-Request-ID` (8-64 chars of `[A-Za-z0-9._-]`), else a generated id.
- `Authorization` is declared in the contract but not yet enforced; stubs answer 501 without credentials.

## Alternatives considered

- Code-first only, no snapshot: cheapest, but drift is invisible until the UI breaks.
- Hand-written spec, generated server: heavier tooling and a new dependency; FastAPI already emits OpenAPI.
- Consumer-driven contract tools (Pact): more setup than a single-team product justifies.
- Version in a header rather than the path: harder to cache, test and read in logs.

## Consequences

- Contract changes are visible in review as a snapshot diff and cannot merge unnoticed.
- The UI client is generated from the snapshot, so UI and API cannot diverge.
- Small recurring cost: each contract change must update the snapshot and client.
- Streaming messages are deferred; adding them later is an additive `/v1` change.
