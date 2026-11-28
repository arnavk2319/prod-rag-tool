"""Pipeline policy gate: pipeline_policies, policy_documents, index_versions.policy_id (ADR-0006).

Revision ID: 0003
Revises: 0002
"""

from alembic import op

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None

POLICY_TABLES = """
CREATE TABLE pipeline_policies (
    policy_id uuid PRIMARY KEY,
    policy_hash text NOT NULL CHECK (policy_hash ~ '^[0-9a-f]{64}$'),
    ingest_run_id text NOT NULL,
    index_version text NOT NULL REFERENCES index_versions (index_version),
    parent_policy_id uuid REFERENCES pipeline_policies (policy_id),
    change_kind text NOT NULL CHECK (change_kind IN
        ('embedder', 'chunker', 'parent_child', 'redaction', 'reranker', 'prompt', 'corpus_only')),
    change_reason text NOT NULL CHECK (change_reason ~ '\\S'),
    embedder_id text NOT NULL,
    embedder_version text NOT NULL,
    embedder_dim int NOT NULL CHECK (embedder_dim > 0),
    chunker_version text NOT NULL,
    chunk_tokens int NOT NULL CHECK (chunk_tokens > 0),
    chunk_overlap int NOT NULL CHECK (chunk_overlap >= 0),
    parent_tokens int CHECK (parent_tokens IS NULL OR parent_tokens > 0),
    redaction_rules_version text NOT NULL,
    reranker_id text,
    prompt_version text NOT NULL,
    generator_chain_ref text NOT NULL,
    -- tenants whose documents the run touched: lets the repository read policy_documents
    -- tenant by tenant under RLS (fixed at declaration, never updated)
    doc_tenants text[] NOT NULL DEFAULT '{}',
    eval_run_id text,
    status text NOT NULL DEFAULT 'declared' CHECK (status IN
        ('declared', 'verified', 'indexed', 'committed', 'rolled_back', 'rejected')),
    declared_by text NOT NULL,
    declared_at timestamptz NOT NULL DEFAULT now(),
    verified_at timestamptz,
    indexed_at timestamptz,
    committed_at timestamptz,
    rolled_back_at timestamptz,
    rejected_at timestamptz,
    mlflow_model_version text,
    mlflow_synced boolean NOT NULL DEFAULT false,
    UNIQUE (index_version, policy_hash)
);
CREATE INDEX pipeline_policies_version_idx ON pipeline_policies (index_version, declared_at);
CREATE INDEX pipeline_policies_parent_idx ON pipeline_policies (parent_policy_id);
-- outbox: unsynced rows in promotion order (committed_at NULLS LAST, declared_at)
CREATE INDEX pipeline_policies_unsynced_idx
    ON pipeline_policies (committed_at NULLS LAST, declared_at) WHERE NOT mlflow_synced;

CREATE TABLE policy_documents (
    policy_id uuid NOT NULL REFERENCES pipeline_policies (policy_id),
    tenant_id text NOT NULL,
    doc_id text NOT NULL,
    doc_version int NOT NULL,
    doc_hash text NOT NULL,
    PRIMARY KEY (policy_id, tenant_id, doc_id, doc_version)
);
CREATE INDEX policy_documents_tenant_idx ON policy_documents (tenant_id, doc_id);
"""

# The declared (hashed) columns are immutable for every role; the trigger backs the column grants.
IMMUTABLE_GUARD = """
CREATE FUNCTION pipeline_policies_immutable() RETURNS trigger
LANGUAGE plpgsql AS $t$
BEGIN
    IF (OLD.policy_id, OLD.policy_hash, OLD.ingest_run_id, OLD.index_version, OLD.parent_policy_id,
        OLD.change_kind, OLD.change_reason, OLD.embedder_id, OLD.embedder_version, OLD.embedder_dim,
        OLD.chunker_version, OLD.chunk_tokens, OLD.chunk_overlap, OLD.parent_tokens,
        OLD.redaction_rules_version, OLD.reranker_id, OLD.prompt_version, OLD.generator_chain_ref,
        OLD.doc_tenants, OLD.declared_by, OLD.declared_at)
       IS DISTINCT FROM
       (NEW.policy_id, NEW.policy_hash, NEW.ingest_run_id, NEW.index_version, NEW.parent_policy_id,
        NEW.change_kind, NEW.change_reason, NEW.embedder_id, NEW.embedder_version, NEW.embedder_dim,
        NEW.chunker_version, NEW.chunk_tokens, NEW.chunk_overlap, NEW.parent_tokens,
        NEW.redaction_rules_version, NEW.reranker_id, NEW.prompt_version, NEW.generator_chain_ref,
        NEW.doc_tenants, NEW.declared_by, NEW.declared_at)
    THEN
        RAISE EXCEPTION 'declared policy fields are immutable'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    RETURN NEW;
END $t$;
CREATE TRIGGER pipeline_policies_no_edit
    BEFORE UPDATE ON pipeline_policies
    FOR EACH ROW EXECUTE FUNCTION pipeline_policies_immutable();
"""

# 0001 left index_versions.status unconstrained; the policy states map onto these values.
INDEX_VERSIONS = """
ALTER TABLE index_versions ADD COLUMN policy_id uuid REFERENCES pipeline_policies (policy_id);
ALTER TABLE index_versions ADD COLUMN policy_hash text;
ALTER TABLE index_versions ADD CONSTRAINT index_versions_status_check CHECK (status IN
    ('building', 'staged', 'committed', 'rolled_back', 'rejected', 'failed'));
CREATE INDEX index_versions_policy_idx ON index_versions (policy_id);
"""

# Hash input is a JSON array (UTC timestamp in a fixed format, NULL kept distinct from ''), so it
# does not depend on the session TimeZone and field boundaries cannot be shifted.
# Hash-chained audit append (audit_log.prev_hash / hash from 0001). One writer at a time:
# the advisory lock lasts until the caller's transaction ends, so chain order is commit order.
AUDIT_APPEND = """
CREATE FUNCTION audit_append(
    p_at timestamptz, p_tenant text, p_actor text, p_action text, p_target text, p_detail jsonb
) RETURNS bigint
LANGUAGE plpgsql AS $fn$
DECLARE prev bytea; h bytea; new_id bigint;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtext('audit_log_chain'));
    SELECT hash INTO prev FROM audit_log ORDER BY audit_id DESC LIMIT 1;
    h := sha256(coalesce(prev, ''::bytea) || convert_to(jsonb_build_array(
        to_char(p_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'),
        p_tenant, p_actor, p_action, p_target, p_detail)::text, 'UTF8'));
    INSERT INTO audit_log (at, tenant_id, actor, action, target, detail, prev_hash, hash)
    VALUES (p_at, p_tenant, p_actor, p_action, p_target, p_detail, prev, h)
    RETURNING audit_id INTO new_id;
    RETURN new_id;
END $fn$;
"""

# Notes. pipeline_policies and policy_documents hold system-level metadata readable by rag_app
# (admin API only); tenant ids reach rag_app through doc_tenants: accepted and documented, to be
# revisited. The downgrade is destructive for policies and documents but leaves audit_log
# untouched. audit_log still has no RLS and rag_app still has INSERT on it (follow-up RLS ticket).
GRANTS = """
REVOKE EXECUTE ON FUNCTION audit_append(timestamptz, text, text, text, text, jsonb) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION audit_append(timestamptz, text, text, text, text, jsonb) TO rag_app;
GRANT SELECT, INSERT ON pipeline_policies, policy_documents TO rag_app;
-- lifecycle columns only; every declared (hashed) column stays immutable. No DELETE.
GRANT UPDATE (status, verified_at, indexed_at, committed_at, rolled_back_at, rejected_at,
    eval_run_id, mlflow_model_version, mlflow_synced) ON pipeline_policies TO rag_app;
ALTER TABLE policy_documents ENABLE ROW LEVEL SECURITY;
CREATE POLICY tenant_isolation ON policy_documents
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true));
"""

DOWNGRADE = (
    "DROP FUNCTION IF EXISTS audit_append(timestamptz, text, text, text, text, jsonb)",
    "ALTER TABLE index_versions DROP CONSTRAINT IF EXISTS index_versions_status_check",
    "ALTER TABLE index_versions DROP COLUMN IF EXISTS policy_hash",
    "ALTER TABLE index_versions DROP COLUMN IF EXISTS policy_id",
    "DROP TABLE IF EXISTS policy_documents",
    "DROP TABLE IF EXISTS pipeline_policies",
    "DROP FUNCTION IF EXISTS pipeline_policies_immutable()",
)


def upgrade() -> None:
    op.execute(POLICY_TABLES)
    op.execute(IMMUTABLE_GUARD)
    op.execute(INDEX_VERSIONS)
    op.execute(AUDIT_APPEND)
    op.execute(GRANTS)


def downgrade() -> None:
    for stmt in DOWNGRADE:
        op.execute(stmt)
