"""Parent-child chunking: parent_chunks table, chunks.parent_id, partition function (ADR-0005).

Revision ID: 0002
Revises: 0001
"""

from alembic import op

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None

PARENT_TABLE = """
CREATE TABLE parent_chunks (
    index_version text NOT NULL,
    tenant_id text NOT NULL,
    parent_id text NOT NULL,
    doc_id text NOT NULL,
    doc_version int NOT NULL DEFAULT 1,
    ordinal int NOT NULL,
    text text NOT NULL,
    token_count int NOT NULL,
    acl_labels text[] NOT NULL DEFAULT '{}',
    heading_path text,
    doc_hash text NOT NULL,
    span_start int NOT NULL,
    span_end int NOT NULL,
    effective_date date NOT NULL,
    expires_date date,
    chunker_version text NOT NULL,
    PRIMARY KEY (index_version, tenant_id, parent_id)
) PARTITION BY LIST (index_version);
CREATE INDEX parent_chunks_tenant_idx ON parent_chunks (tenant_id);
CREATE INDEX parent_chunks_acl_idx ON parent_chunks USING gin (acl_labels);
CREATE INDEX parent_chunks_doc_idx ON parent_chunks (index_version, tenant_id, doc_id);
"""

PARENT_RLS = """
-- parents are immutable (no UPDATE). DELETE mirrors chunks: needed for delete and re-index.
-- Partitions get no grants (access only via the parent, so RLS applies).
GRANT SELECT, INSERT, DELETE ON parent_chunks TO rag_app;
ALTER TABLE parent_chunks ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON parent_chunks
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
"""

# No default on purpose: a fake parent id would be a silent integrity bug. No FK: the
# orphan_children eval enforces the link (a FK across partitioned tables would also be costly).
CHUNKS_PARENT_ID = """
ALTER TABLE chunks ADD COLUMN parent_id text;
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM chunks) THEN
        RAISE EXCEPTION 'chunks not empty: re-index required';
    END IF;
END $$;
ALTER TABLE chunks ALTER COLUMN parent_id SET NOT NULL;
CREATE INDEX chunks_parent_idx ON chunks (index_version, tenant_id, parent_id);
"""

# 0002 body: creates both the chunk partition (with HNSW, ADR-0002) and the parent partition.
PARTITION_FN = """
CREATE OR REPLACE FUNCTION create_chunks_partition(v text) RETURNS void
LANGUAGE plpgsql AS $fn$
DECLARE
    part text := 'chunks_' || regexp_replace(v, '[^a-zA-Z0-9]', '_', 'g');
    ppart text := 'parent_chunks_' || regexp_replace(v, '[^a-zA-Z0-9]', '_', 'g');
BEGIN
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF chunks FOR VALUES IN (%L)', part, v);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (embedding vector_cosine_ops) '
        'WITH (m = 16, ef_construction = 64)', part || '_hnsw_idx', part);
    EXECUTE format(
        'CREATE TABLE IF NOT EXISTS %I PARTITION OF parent_chunks FOR VALUES IN (%L)', ppart, v);
END $fn$;
"""

# 0001 body, restored on downgrade.
PARTITION_FN_0001 = """
CREATE OR REPLACE FUNCTION create_chunks_partition(v text) RETURNS void
LANGUAGE plpgsql AS $fn$
DECLARE part text := 'chunks_' || regexp_replace(v, '[^a-zA-Z0-9]', '_', 'g');
BEGIN
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF chunks FOR VALUES IN (%L)', part, v);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (embedding vector_cosine_ops) '
        'WITH (m = 16, ef_construction = 64)', part || '_hnsw_idx', part);
END $fn$;
"""

# Existing chunk partitions gain a parent partition (the function is idempotent).
BACKFILL_PARTITIONS = """
DO $$ DECLARE r record; v text;
BEGIN
    FOR r IN
        SELECT pg_get_expr(c.relpartbound, c.oid) AS bound
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        WHERE i.inhparent = 'chunks'::regclass
    LOOP
        v := replace((regexp_match(r.bound, '^FOR VALUES IN \\(''(.*)''\\)$'))[1], '''''', '''');
        IF v IS NOT NULL THEN
            PERFORM create_chunks_partition(v);
        END IF;
    END LOOP;
END $$;
"""

DROP_PARENT_ID = "ALTER TABLE chunks DROP COLUMN IF EXISTS parent_id"
DROP_PARENTS = "DROP TABLE IF EXISTS parent_chunks CASCADE"


def upgrade() -> None:
    op.execute(PARENT_TABLE)
    op.execute(PARENT_RLS)
    op.execute(CHUNKS_PARENT_ID)
    op.execute(PARTITION_FN)
    op.execute(BACKFILL_PARTITIONS)


def downgrade() -> None:
    op.execute(PARTITION_FN_0001)
    op.execute(DROP_PARENT_ID)
    op.execute(DROP_PARENTS)
