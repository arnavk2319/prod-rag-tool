# API contract

The contract is the OpenAPI 3.1 snapshot `tests/contract/openapi.snapshot.json`. The decision and its rationale are in [ADR-0004](decisions/ADR-0004-api-contract-and-versioning.md).

## Versioning rule

Any contract change (path, method, parameter, header, schema, error code) must land in one PR with all three of:

1. the updated snapshot (`make contract-update`, then review the diff);
2. the regenerated UI client;
3. a note in the ADR (or a superseding ADR).

Additive optional fields stay in `/v1`. Removals or semantic changes need `/v2`. `/healthz`, `/readyz` and `/metrics` are unversioned.

## Checks

- `tests/contract/test_openapi_snapshot.py` diffs the live app against the snapshot, checks coverage against `plan_endpoints.txt`, error envelope use, headers, pagination, and validates every fixture in `tests/contract/fixtures/` (`<operationId>.<status>.json`).
- Schema validation uses a small stdlib validator (`tests/contract/schema_validator.py`); no extra dependency.

## Conventions

- Errors: `{error:{code,message,request_id,details?}}`; `429` carries an integer `Retry-After` header.
- Headers: `Authorization: Bearer` on authenticated routes, `X-Request-ID` accepted and echoed, `Idempotency-Key` (optional) on `create_document`, `create_chat`, `create_message`. Idempotency and `Authorization` are declared in the contract but not yet enforced; stubs answer 501 without credentials.
- Lists: `limit` (default 50, max 200) and `cursor`; response `{items, next_cursor}`.
- Until real handlers exist, routes return `501` with the error envelope (`internal_error`, "not implemented").
- `versions` on a message carries `index_version`, `embedding_model`, `prompt_version` and the optional `policy_hash` (pipeline policy of the serving index version, ADR-0006; absent for versions built before the policy gate).
