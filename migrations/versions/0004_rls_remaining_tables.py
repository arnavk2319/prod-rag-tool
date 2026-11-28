"""Row-level security on every remaining tenant-scoped table; audit_log append via function only.

Revision ID: 0004
Revises: 0003

Tables that carry a tenant_id column and had no policy: raw_objects, ingest_jobs,
quarantine_items, chat_sessions, chat_messages, request_traces, audit_log. chunks, documents,
parent_chunks and policy_documents already have one (partitions are covered through the parent).

RLS is ENABLED, not FORCED: the table owner (rag_owner) bypasses it. That is the intended path
for the migration role and for the audit chain verifier, which must read every tenant's rows.
The application must never connect as the owner (it connects as rag_svc, a member of rag_app).

audit_log: a tenant session reads only its own rows. A row with a NULL tenant_id (system event)
is invisible to tenant sessions. rag_app loses INSERT (and the id sequence): rows are added only
through audit_append, which is SECURITY DEFINER (runs as the owner, bypassing RLS). So that the
definer rights cannot be used to forge another tenant's row, audit_append refuses a tenant
argument that differs from the session's app.tenant_id; a NULL argument takes the session's
tenant, or stays NULL (system event) when the session has none; an empty string counts as
NULL. The hash formula and chain semantics are unchanged from 0003; only the advisory lock
key becomes two fixed integers.
"""

from alembic import op

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None

POLICY_TABLES = (
    "raw_objects",
    "ingest_jobs",
    "quarantine_items",
    "chat_sessions",
    "chat_messages",
    "request_traces",
    "audit_log",
)

# Same expression as chunks/documents (0001), parent_chunks (0002) and policy_documents (0003).
# Written out literally per table (no formatted SQL).
ENABLE = (
    "ALTER TABLE raw_objects ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE ingest_jobs ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE quarantine_items ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE chat_messages ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE request_traces ENABLE ROW LEVEL SECURITY",
    "ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY",
)
POLICY = """
CREATE POLICY tenant_isolation ON {t}
    USING (tenant_id = current_setting('app.tenant_id', true))
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true))
"""
CREATE_POLICIES = tuple(  # nosemgrep: table names come from the constant tuple above
    POLICY.replace("{t}", t) for t in POLICY_TABLES
)
DROP_POLICIES = (
    "DROP POLICY IF EXISTS tenant_isolation ON raw_objects",
    "DROP POLICY IF EXISTS tenant_isolation ON ingest_jobs",
    "DROP POLICY IF EXISTS tenant_isolation ON quarantine_items",
    "DROP POLICY IF EXISTS tenant_isolation ON chat_sessions",
    "DROP POLICY IF EXISTS tenant_isolation ON chat_messages",
    "DROP POLICY IF EXISTS tenant_isolation ON request_traces",
    "DROP POLICY IF EXISTS tenant_isolation ON audit_log",
)
DISABLE = (
    "ALTER TABLE raw_objects DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE ingest_jobs DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE quarantine_items DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE chat_sessions DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE chat_messages DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE request_traces DISABLE ROW LEVEL SECURITY",
    "ALTER TABLE audit_log DISABLE ROW LEVEL SECURITY",
)

AUDIT_APPEND_DEFINER = """
CREATE OR REPLACE FUNCTION public.audit_append(
    p_at timestamptz, p_tenant text, p_actor text, p_action text, p_target text, p_detail jsonb
) RETURNS bigint
LANGUAGE plpgsql SECURITY DEFINER SET search_path = pg_catalog, public
SET lock_timeout = '5s' AS $fn$
DECLARE
    sess text := nullif(pg_catalog.current_setting('app.tenant_id', true), '');
    eff text;
    prev bytea; h bytea; new_id bigint;
BEGIN
    p_tenant := nullif(p_tenant, '');  -- an empty tenant is a system event
    IF p_tenant IS NOT NULL AND p_tenant IS DISTINCT FROM sess THEN
        RAISE EXCEPTION 'audit tenant does not match the session tenant'
            USING ERRCODE = 'insufficient_privilege';
    END IF;
    eff := coalesce(p_tenant, sess);
    PERFORM pg_catalog.pg_advisory_xact_lock(1096041805, 1633837924);
    SELECT a.hash INTO prev FROM public.audit_log a ORDER BY a.audit_id DESC LIMIT 1;
    h := pg_catalog.sha256(coalesce(prev, ''::bytea) || pg_catalog.convert_to(
        pg_catalog.jsonb_build_array(
            pg_catalog.to_char(p_at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'),
            eff, p_actor, p_action, p_target, p_detail)::text, 'UTF8'));
    INSERT INTO public.audit_log (at, tenant_id, actor, action, target, detail, prev_hash, hash)
    VALUES (p_at, eff, p_actor, p_action, p_target, p_detail, prev, h)
    RETURNING audit_id INTO new_id;
    RETURN new_id;
END $fn$;
"""

# The 0003 body, restored on downgrade (invoker rights, hashtext lock key).
AUDIT_APPEND_INVOKER = """
CREATE OR REPLACE FUNCTION public.audit_append(
    p_at timestamptz, p_tenant text, p_actor text, p_action text, p_target text, p_detail jsonb
) RETURNS bigint
LANGUAGE plpgsql SECURITY INVOKER RESET search_path RESET lock_timeout AS $fn$
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

# 0001's trigger is FOR EACH ROW and does not fire on a table-wipe statement; this statement-level
# trigger reuses the same raising function. A superuser can still disable triggers.
STATEMENT_GUARD = """
CREATE TRIGGER audit_log_no_wipe
    BEFORE TRUNCATE ON audit_log
    FOR EACH STATEMENT EXECUTE FUNCTION audit_log_append_only()
"""

GRANTS = (
    "REVOKE INSERT ON audit_log FROM rag_app",
    "REVOKE USAGE ON SEQUENCE audit_log_audit_id_seq FROM rag_app",
    "REVOKE EXECUTE ON FUNCTION audit_append(timestamptz, text, text, text, text, jsonb) "
    "FROM PUBLIC",
    "GRANT EXECUTE ON FUNCTION audit_append(timestamptz, text, text, text, text, jsonb) TO rag_app",
)
UNGRANTS = (
    "GRANT INSERT ON audit_log TO rag_app",
    "GRANT USAGE ON SEQUENCE audit_log_audit_id_seq TO rag_app",
)


def upgrade() -> None:
    for stmt in ENABLE + CREATE_POLICIES:
        op.execute(stmt)
    op.execute(AUDIT_APPEND_DEFINER)
    op.execute(STATEMENT_GUARD)
    for stmt in GRANTS:
        op.execute(stmt)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_log_no_wipe ON audit_log")
    op.execute(AUDIT_APPEND_INVOKER)
    for stmt in UNGRANTS:
        op.execute(stmt)
    for stmt in DROP_POLICIES + DISABLE:
        op.execute(stmt)
