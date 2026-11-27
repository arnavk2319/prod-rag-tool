"""Initial schema (contract): all spec tables, RLS, partitioned chunks.

Revision ID: 0001
Revises:
"""

from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None

EMBEDDING_DIM = 384  # BAAI/bge-small-en-v1.5 (ADR-0002)

VERSION_CHECK = """
DO $$ DECLARE v text; mm text[];
BEGIN
    SELECT extversion INTO v FROM pg_extension WHERE extname = 'vector';
    mm := regexp_match(coalesce(v, ''), '^(\\d+)\\.(\\d+)');
    IF mm IS NULL OR (mm[1]::int, mm[2]::int) < (0, 8) THEN
        RAISE EXCEPTION 'pgvector >= 0.8 required (iterative scan), found %', coalesce(v, 'none');
    END IF;
END $$;
"""

TABLES = """
CREATE TABLE documents (
    doc_id text NOT NULL,
    tenant_id text NOT NULL,
    version int NOT NULL DEFAULT 1,
    title text NOT NULL,
    source_uri text,
    doc_hash text NOT NULL,
    acl_labels text[] NOT NULL DEFAULT '{}',
    effective_date date NOT NULL DEFAULT current_date,
    expires_date date,
    pii_class text NOT NULL DEFAULT 'none',
    status text NOT NULL DEFAULT 'active',
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, doc_id, version),
    UNIQUE (tenant_id, doc_hash)
);
CREATE INDEX documents_tenant_status_idx ON documents (tenant_id, status);

CREATE TABLE raw_objects (
    sha256 text NOT NULL,
    tenant_id text NOT NULL,
    mime text NOT NULL,
    size bigint NOT NULL,
    status text NOT NULL DEFAULT 'raw' CHECK (status IN ('raw', 'quarantined')),
    content bytea NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (tenant_id, sha256)
);

CREATE TABLE ingest_jobs (
    job_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    doc_id text,
    raw_sha256 text,
    status text NOT NULL DEFAULT 'queued',
    error text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX ingest_jobs_tenant_status_idx ON ingest_jobs (tenant_id, status);

CREATE TABLE quarantine_items (
    item_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    job_id text,
    raw_sha256 text,
    reason text NOT NULL,
    pii_class text,
    status text NOT NULL DEFAULT 'pending',
    created_at timestamptz NOT NULL DEFAULT now(),
    resolved_at timestamptz
);
CREATE INDEX quarantine_items_tenant_idx ON quarantine_items (tenant_id, status);

CREATE TABLE index_versions (
    index_version text PRIMARY KEY,
    parent_version text REFERENCES index_versions (index_version),
    embedding_model text NOT NULL,
    chunker_version text NOT NULL,
    status text NOT NULL DEFAULT 'building',
    row_count bigint,
    gate_run_id text,
    created_at timestamptz NOT NULL DEFAULT now(),
    committed_at timestamptz
);

CREATE TABLE chunks (
    chunk_id text NOT NULL,
    index_version text NOT NULL,
    tenant_id text NOT NULL,
    doc_id text NOT NULL,
    doc_version int NOT NULL DEFAULT 1,
    ordinal int NOT NULL,
    text text NOT NULL,
    span_start int NOT NULL,
    span_end int NOT NULL,
    section text,
    page int,
    acl_labels text[] NOT NULL DEFAULT '{}',
    effective_date date NOT NULL,
    expires_date date,
    embedding_model text NOT NULL,
    chunker_version text NOT NULL,
    embedding vector(__DIM__) NOT NULL,
    tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED,
    PRIMARY KEY (index_version, tenant_id, chunk_id)
) PARTITION BY LIST (index_version);
CREATE INDEX chunks_tenant_idx ON chunks (tenant_id);
CREATE INDEX chunks_tsv_idx ON chunks USING gin (tsv);
CREATE INDEX chunks_acl_idx ON chunks USING gin (acl_labels);

CREATE TABLE index_alias (
    alias text PRIMARY KEY CHECK (alias = 'live'),
    index_version text NOT NULL REFERENCES index_versions (index_version),
    previous_version text REFERENCES index_versions (index_version),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE chat_sessions (
    session_id text PRIMARY KEY,
    tenant_id text NOT NULL,
    user_id text NOT NULL,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chat_sessions_tenant_user_idx ON chat_sessions (tenant_id, user_id);

CREATE TABLE chat_messages (
    message_id text PRIMARY KEY,
    session_id text NOT NULL REFERENCES chat_sessions (session_id) ON DELETE CASCADE,
    tenant_id text NOT NULL,
    role text NOT NULL,
    content text NOT NULL,
    citations jsonb NOT NULL DEFAULT '[]',
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX chat_messages_session_idx ON chat_messages (session_id, created_at);

CREATE TABLE control_flags (
    flag text PRIMARY KEY,
    value jsonb NOT NULL DEFAULT 'false',
    updated_by text,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE registry_pointers (
    name text PRIMARY KEY,
    prompt_version text,
    model_id text,
    chain_version text,
    canary_percent int NOT NULL DEFAULT 0 CHECK (canary_percent BETWEEN 0 AND 100),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE golden_sets (
    golden_id text PRIMARY KEY,
    name text NOT NULL,
    sha256 text NOT NULL,
    approved_by text,
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE gate_runs (
    gate_run_id text PRIMARY KEY,
    kind text NOT NULL,
    index_version text,
    golden_id text REFERENCES golden_sets (golden_id),
    passed boolean NOT NULL,
    metrics jsonb NOT NULL DEFAULT '{}',
    created_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE audit_log (
    audit_id bigserial PRIMARY KEY,
    at timestamptz NOT NULL DEFAULT now(),
    tenant_id text,
    actor text NOT NULL,
    action text NOT NULL,
    target text,
    detail jsonb NOT NULL DEFAULT '{}',
    prev_hash bytea,
    hash bytea
);
CREATE INDEX audit_log_at_idx ON audit_log (at);

CREATE FUNCTION audit_log_append_only() RETURNS trigger
LANGUAGE plpgsql AS $t$
BEGIN
    RAISE EXCEPTION 'audit_log is append-only' USING ERRCODE = 'insufficient_privilege';
END $t$;
CREATE TRIGGER audit_log_no_change
    BEFORE UPDATE OR DELETE ON audit_log
    FOR EACH ROW EXECUTE FUNCTION audit_log_append_only();

CREATE TABLE role_labels (
    role text NOT NULL,
    label text NOT NULL,
    PRIMARY KEY (role, label)
);

-- ids and scores only: deliberately no question or answer text column (ADR-0003)
CREATE TABLE request_traces (
    trace_id text PRIMARY KEY,
    message_id text,
    tenant_id text NOT NULL,
    at timestamptz NOT NULL DEFAULT now(),
    outcome text NOT NULL,
    cache text,
    versions jsonb NOT NULL DEFAULT '{}',
    retrieval jsonb NOT NULL DEFAULT '{}',
    llm jsonb NOT NULL DEFAULT '{}',
    tokens jsonb NOT NULL DEFAULT '{}',
    latency_ms jsonb NOT NULL DEFAULT '{}'
);
CREATE INDEX request_traces_tenant_at_idx ON request_traces (tenant_id, at);
"""

# Partition + per-partition HNSW (ADR-0002: m=16, ef_construction=64, cosine).
PARTITION_FN = """
CREATE FUNCTION create_chunks_partition(v text) RETURNS void
LANGUAGE plpgsql AS $fn$
DECLARE part text := 'chunks_' || regexp_replace(v, '[^a-zA-Z0-9]', '_', 'g');
BEGIN
    EXECUTE format('CREATE TABLE IF NOT EXISTS %I PARTITION OF chunks FOR VALUES IN (%L)', part, v);
    EXECUTE format(
        'CREATE INDEX IF NOT EXISTS %I ON %I USING hnsw (embedding vector_cosine_ops) '
        'WITH (m = 16, ef_construction = 64)', part || '_hnsw_idx', part);
END $fn$;
"""

RLS = """
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_app') THEN
        CREATE ROLE rag_app NOLOGIN NOBYPASSRLS;
    END IF;
END $$;
GRANT USAGE ON SCHEMA public TO rag_app;
-- least privilege per table; partitions get no grants (access only via the parent, so RLS applies)
GRANT SELECT, INSERT, UPDATE, DELETE ON documents, raw_objects, ingest_jobs, quarantine_items,
    chat_sessions, chat_messages, control_flags, registry_pointers TO rag_app;
GRANT SELECT, INSERT, UPDATE ON index_versions TO rag_app;
GRANT SELECT, INSERT, UPDATE ON index_alias TO rag_app;
GRANT SELECT, INSERT, DELETE ON chunks TO rag_app;
GRANT SELECT, INSERT ON request_traces, gate_runs, golden_sets, audit_log TO rag_app;
GRANT SELECT ON role_labels TO rag_app;
GRANT USAGE ON SEQUENCE audit_log_audit_id_seq TO rag_app;
ALTER TABLE chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON chunks
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
CREATE POLICY tenant_isolation ON documents
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
"""

TABLE_NAMES = (
    "request_traces, role_labels, audit_log, gate_runs, golden_sets, registry_pointers, "
    "control_flags, chat_messages, chat_sessions, index_alias, chunks, index_versions, "
    "quarantine_items, ingest_jobs, raw_objects, documents"
)


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")
    op.execute(VERSION_CHECK)
    op.execute(TABLES.replace("__DIM__", str(EMBEDDING_DIM)))
    op.execute(PARTITION_FN)
    op.execute(RLS)


def downgrade() -> None:
    op.execute("DROP FUNCTION IF EXISTS create_chunks_partition(text)")
    op.execute("DROP FUNCTION IF EXISTS audit_log_append_only() CASCADE")
    # TABLE_NAMES is a module-level constant tuple of this migration's own tables; no external input
    # reaches the statement, so the formatted-SQL rules do not apply.
    # nosemgrep
    op.execute(f"DROP TABLE IF EXISTS {TABLE_NAMES} CASCADE")
