# Note: pinned pgvector image

Integration tests and local Compose use `pgvector/pgvector:0.8.0-pg16` (Postgres 16, pgvector 0.8.0).
pgvector 0.8 is the floor because HNSW iterative scan (ADR-0002) needs it; Alembic revision `0001`
raises an error and aborts if the installed extension is older than 0.8.
Bump the tag deliberately and re-run the integration tests when upgrading.
