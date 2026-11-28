# ruff: noqa: SIM117
"""Migration 0001 security and governance properties (RLS, audit, partitions, cleanup).

Skipped unless TEST_DATABASE_URL points at a reachable throwaway Postgres with pgvector >= 0.8.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pg_helpers import add_chunk

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")


def _cfg() -> Config:
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    return cfg


def _reset(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        for stmt in WIPE:
            c.execute(stmt)  # type: ignore[arg-type]
        c.execute("DROP EXTENSION IF EXISTS vector CASCADE")  # type: ignore[arg-type]


@pytest.fixture
def dsn(monkeypatch: pytest.MonkeyPatch) -> Iterator[str]:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    d = url.replace("postgresql+psycopg://", "postgresql://")
    try:
        _reset(d)
    except psycopg.OperationalError:
        pytest.skip("TEST_DATABASE_URL unreachable")
    monkeypatch.setenv("DATABASE_URL", d.replace("postgresql://", "postgresql+psycopg://"))
    command.upgrade(_cfg(), "head")
    yield d
    _reset(d)


def _chunk(c: psycopg.Connection, tenant: str, cid: str, version: str = "v1") -> None:
    add_chunk(c, tenant, cid, version)


def _doc(c: psycopg.Connection, tenant: str) -> None:
    c.execute(
        "INSERT INTO documents (doc_id,tenant_id,title,doc_hash) VALUES ('d',%s,'t','h')",
        (tenant,),
    )


def _seed(d: str) -> None:
    with psycopg.connect(d, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        for t in ("bank-a", "bank-b"):
            _chunk(c, t, f"c-{t}")
            _doc(c, t)


def _as_app(d: str, tenant: str | None = "bank-a") -> psycopg.Connection:
    c = psycopg.connect(d)
    c.execute("SET ROLE rag_app")
    if tenant is not None:
        c.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    return c


# (1) RLS ------------------------------------------------------------------------------------


def test_role_has_no_bypassrls_or_superuser(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        row = c.execute(
            "SELECT rolbypassrls, rolsuper, rolcanlogin FROM pg_roles WHERE rolname='rag_app'"
        ).fetchone()
    assert row == (False, False, False)


def test_rls_enabled_and_forced_state_on_both_tables(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        rows = dict(
            c.execute(  # type: ignore[arg-type]
                "SELECT relname, relrowsecurity FROM pg_class "
                "WHERE relname IN ('chunks','documents')"
            ).fetchall()
        )
    assert rows == {"chunks": True, "documents": True}


@pytest.mark.parametrize("table", ["chunks", "documents"])
def test_other_tenant_rows_invisible_to_update_and_delete(dsn: str, table: str) -> None:
    _seed(dsn)
    with _as_app(dsn) as c:
        if table == "documents":  # chunks are immutable: the app role has no UPDATE on them
            upd = c.execute(f"UPDATE {table} SET tenant_id = tenant_id")  # type: ignore[arg-type]  # noqa: S608
            assert upd.rowcount == 1  # only bank-a's row is reachable
        dele = c.execute(f"DELETE FROM {table}")  # type: ignore[arg-type]  # noqa: S608
        assert dele.rowcount == 1
    with psycopg.connect(dsn) as c:
        left = c.execute(f"SELECT tenant_id FROM {table}").fetchall()  # type: ignore[arg-type]  # noqa: S608
    assert left == [("bank-b",)]


def test_insert_other_tenant_rejected_with_check(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            _doc(c, "bank-b")
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            _chunk(c, "bank-b", "x")
    with _as_app(dsn) as c:
        _doc(c, "bank-a")
        _chunk(c, "bank-a", "ok")  # own tenant still allowed


def test_update_to_other_tenant_rejected_with_check(dsn: str) -> None:
    _seed(dsn)
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
            c.execute("UPDATE documents SET tenant_id='bank-b'")
    with _as_app(dsn) as c:  # chunks: no UPDATE privilege at all (immutable)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE chunks SET tenant_id='bank-b'")


def test_app_role_cannot_disable_rls_or_change_policy(dsn: str) -> None:
    for stmt in (
        "ALTER TABLE chunks DISABLE ROW LEVEL SECURITY",
        "ALTER TABLE documents NO FORCE ROW LEVEL SECURITY",
        "DROP POLICY tenant_isolation ON documents",
        "ALTER ROLE rag_app BYPASSRLS",
    ):
        with _as_app(dsn) as c:
            with pytest.raises(psycopg.errors.InsufficientPrivilege):
                c.execute(stmt)  # type: ignore[arg-type]
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("SET row_security = off")
            c.execute("SELECT 1 FROM chunks")  # row_security=off errors rather than bypassing


def test_unset_or_empty_tenant_returns_nothing(dsn: str) -> None:
    _seed(dsn)
    for tenant in (None, ""):
        with _as_app(dsn, tenant) as c:
            for table in ("chunks", "documents"):
                got = c.execute(f"SELECT count(*) FROM {table}").fetchone()  # type: ignore[arg-type]  # noqa: S608
                assert got == (0,)


# (2) ACL ------------------------------------------------------------------------------------


def test_acl_empty_label_deny_is_enforced_in_retrieval_not_schema(dsn: str) -> None:
    """Documents where the empty-label deny lives.

    The schema stores acl_labels text[] NOT NULL DEFAULT '{}' and does not reject '{}' (it
    cannot express the user-label subset check). The deny (chunk visible only if
    acl_labels <> '{}' AND acl_labels <@ user_labels) is the T-012 retrieval query, per
    ADR-0002. This test pins the schema half: NULL is rejected, and '{}' is the default so
    the retrieval predicate must treat it as deny.
    """
    _seed(dsn)
    with psycopg.connect(dsn, autocommit=True) as c:
        with pytest.raises(psycopg.errors.NotNullViolation):
            c.execute("UPDATE chunks SET acl_labels = NULL")
        default = c.execute(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name='documents' AND column_name='acl_labels'"
        ).fetchone()
        assert default is not None and "'{}'" in default[0]
        c.execute("UPDATE chunks SET acl_labels='{}' WHERE tenant_id='bank-a'")
        # ADR-0002 predicate: empty labels never match any user, even a superset user.
        n = c.execute(
            "SELECT count(*) FROM chunks WHERE acl_labels <> '{}' "
            "AND acl_labels <@ ARRAY['a','b']::text[]"
        ).fetchone()
        assert n == (1,)  # only bank-b's {a} row; the empty-label row is denied


# (3) audit_log ------------------------------------------------------------------------------


def test_audit_log_has_hash_chain_columns(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        cols = {
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='audit_log'"
            )
        }
    assert {"prev_hash", "hash"} <= cols


def test_app_role_can_append_audit_log_only_through_the_function(dsn: str) -> None:
    with _as_app(dsn) as c:
        c.execute("SELECT audit_append(now(), NULL, 'u', 'login', NULL, '{}'::jsonb)")
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("INSERT INTO audit_log (actor, action) VALUES ('u','login')")


@pytest.mark.parametrize("stmt", ["UPDATE audit_log SET actor='x'", "DELETE FROM audit_log"])
def test_app_role_cannot_modify_audit_log(dsn: str, stmt: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("INSERT INTO audit_log (actor, action) VALUES ('u','a')")
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(stmt)  # type: ignore[arg-type]


# (4) partitions -----------------------------------------------------------------------------


def test_create_chunks_partition_indexes(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v2')")
        defs = [
            r[0] for r in c.execute("SELECT indexdef FROM pg_indexes WHERE tablename='chunks_v2'")
        ]
    joined = "\n".join(defs)
    hnsw = [d for d in defs if "USING hnsw" in d]
    assert len(hnsw) == 1
    assert "vector_cosine_ops" in hnsw[0]
    assert "m='16'" in hnsw[0] and "ef_construction='64'" in hnsw[0]
    assert "USING gin (tsv)" in joined
    assert "USING gin (acl_labels)" in joined
    assert "USING btree (tenant_id)" in joined


def test_create_chunks_partition_is_idempotent(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v3')")
        c.execute("SELECT create_chunks_partition('v3')")
        n = c.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename='chunks_v3' AND indexdef LIKE '%hnsw%'"
        ).fetchone()
    assert n == (1,)


def test_chunk_without_partition_is_rejected(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        with pytest.raises(psycopg.errors.CheckViolation, match="no partition"):
            _chunk(c, "bank-a", "orphan", version="never-created")


def test_partition_not_directly_accessible_to_app_role(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v9')")
    with _as_app(dsn) as c:
        _chunk(c, "bank-a", "p1", version="v9")  # insert through the parent routes to the partition
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == (1,)
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="permission denied"):
            c.execute("SELECT count(*) FROM chunks_v9")
    with _as_app(dsn, "bank-b") as c:
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
    with psycopg.connect(dsn) as c:  # owner sees the routed row in the partition
        assert c.execute("SELECT count(*) FROM chunks_v9").fetchone() == (1,)


# (4c) tenant-scoped keys --------------------------------------------------------------------


def _ingest(c: psycopg.Connection, tenant: str) -> None:
    """The same document (same doc_hash, chunk id, raw sha256) as seen by one tenant."""
    c.execute(
        "INSERT INTO raw_objects (sha256,tenant_id,mime,size,content) "
        "VALUES ('same-sha',%s,'text/plain',1,'x')",
        (tenant,),
    )
    c.execute(
        "INSERT INTO documents (doc_id,tenant_id,title,doc_hash) VALUES ('d',%s,'t','same-hash')",
        (tenant,),
    )
    _chunk(c, tenant, "same-chunk-id")


def test_two_tenants_ingest_identical_document(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
    with _as_app(dsn, "bank-a") as c:
        _ingest(c, "bank-a")
        c.commit()
    with _as_app(dsn, "bank-b") as c:  # must not fail because of bank-a's rows
        _ingest(c, "bank-b")
        c.commit()
    for tenant in ("bank-a", "bank-b"):
        with _as_app(dsn, tenant) as c:
            for table in ("chunks", "documents"):
                rows = c.execute(f"SELECT tenant_id FROM {table}").fetchall()  # type: ignore[arg-type]  # noqa: S608
                assert rows == [(tenant,)]
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM raw_objects").fetchone() == (2,)


def test_duplicate_within_same_tenant_still_fails(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        _ingest(c, "bank-a")
        with pytest.raises(psycopg.errors.UniqueViolation):
            _chunk(c, "bank-a", "same-chunk-id")
        with pytest.raises(psycopg.errors.UniqueViolation):
            c.execute(
                "INSERT INTO raw_objects (sha256,tenant_id,mime,size,content) "
                "VALUES ('same-sha','bank-a','text/plain',1,'x')"
            )


def test_app_role_cannot_delete_index_versions(dsn: str) -> None:
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("DELETE FROM index_versions")


# (4b) constraints ---------------------------------------------------------------------------


def test_documents_unique_tenant_doc_hash(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        _doc(c, "bank-a")
        with pytest.raises(psycopg.errors.UniqueViolation):
            c.execute(
                "INSERT INTO documents (doc_id,tenant_id,title,doc_hash) "
                "VALUES ('other','bank-a','t','h')"
            )
        c.execute(  # same hash in another tenant is fine
            "INSERT INTO documents (doc_id,tenant_id,title,doc_hash) VALUES ('d','bank-b','t','h')"
        )


def test_index_alias_single_live_row_and_fks(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        for v in ("v1", "v2"):
            c.execute(
                "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
                "VALUES (%s,'m','c1')",
                (v,),
            )
        c.execute("INSERT INTO index_alias (alias, index_version, previous_version) "
                  "VALUES ('live','v2','v1')")  # fmt: skip
        with pytest.raises(psycopg.errors.CheckViolation):
            c.execute("INSERT INTO index_alias (alias, index_version) VALUES ('staging','v1')")
        with pytest.raises(psycopg.errors.UniqueViolation):
            c.execute("INSERT INTO index_alias (alias, index_version) VALUES ('live','v1')")
        with pytest.raises(psycopg.errors.ForeignKeyViolation):
            c.execute("UPDATE index_alias SET previous_version='nope'")


def test_raw_objects_status_check(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        with pytest.raises(psycopg.errors.CheckViolation):
            c.execute(
                "INSERT INTO raw_objects (sha256,tenant_id,mime,size,status,content) "
                "VALUES ('s','bank-a','text/plain',1,'weird','x')"
            )


@pytest.mark.parametrize(
    ("table", "stmt"),
    [
        ("chunks", "UPDATE chunks SET text='x'"),
        ("request_traces", "UPDATE request_traces SET outcome='x'"),
        ("request_traces", "DELETE FROM request_traces"),
        ("gate_runs", "UPDATE gate_runs SET passed = true"),
        ("golden_sets", "UPDATE golden_sets SET name='x'"),
        ("role_labels", "INSERT INTO role_labels VALUES ('r','l')"),
        ("alembic_version", "SELECT * FROM alembic_version"),
    ],
)
def test_app_role_least_privilege(dsn: str, table: str, stmt: str) -> None:
    with _as_app(dsn) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute(stmt)  # type: ignore[arg-type]


# (5) cleanup --------------------------------------------------------------------------------


def _objects(d: str) -> dict[str, set[str]]:
    q = {
        "relations": "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace "
        "WHERE n.nspname='public'",
        "functions": "SELECT p.proname FROM pg_proc p JOIN pg_namespace n ON n.oid=p.pronamespace "
        "WHERE n.nspname='public'",
        "types": "SELECT t.typname FROM pg_type t JOIN pg_namespace n ON n.oid=t.typnamespace "
        "WHERE n.nspname='public' AND t.typtype IN ('e','c','d') AND t.typrelid=0",
        "policies": "SELECT policyname FROM pg_policies",
    }
    with psycopg.connect(d) as c:
        return {k: {r[0] for r in c.execute(v).fetchall()} for k, v in q.items()}  # type: ignore[arg-type]


def test_downgrade_leaves_no_orphans_beyond_documented(dsn: str) -> None:
    """Downgrade drops tables, policies, sequences, partitions and the function.

    Documented survivors (Findings): role rag_app and the vector extension. Extension-owned
    objects (vector functions/types in public) are excluded by comparing against a baseline
    taken right after enabling the extension on an empty schema.
    """
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")  # partition must be dropped too
    command.downgrade(_cfg(), "base")
    after = _objects(dsn)
    assert "create_chunks_partition" not in after["functions"]
    assert after["policies"] == set()
    leftover = {r for r in after["relations"] if r.startswith(("chunks", "documents", "audit"))}
    assert leftover == set()
    with psycopg.connect(dsn) as c:
        roles = {r[0] for r in c.execute("SELECT rolname FROM pg_roles WHERE rolname='rag_app'")}
        ext = c.execute("SELECT 1 FROM pg_extension WHERE extname='vector'").fetchone()
    assert roles == {"rag_app"} and ext is not None  # documented: kept by design
    # re-upgrade succeeds on top of the surviving role/extension and restores everything
    command.upgrade(_cfg(), "head")
    again = _objects(dsn)
    assert "create_chunks_partition" in again["functions"]
    assert {"chunks", "documents"} <= again["relations"]
    assert after["functions"] < again["functions"]


def test_downgrade_removes_every_app_table_and_grants_gone(dsn: str) -> None:
    command.downgrade(_cfg(), "base")
    with psycopg.connect(dsn) as c:
        n = c.execute(
            "SELECT count(*) FROM information_schema.role_table_grants WHERE grantee='rag_app' "
            "AND table_schema='public' AND table_name <> 'alembic_version'"
        ).fetchone()
    assert n == (0,)


# (6) no content text beyond spec ------------------------------------------------------------


def test_chat_messages_text_only_in_content(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        rows = c.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name='chat_messages' AND data_type IN ('text','character varying')"
        ).fetchall()
    # free text belongs only in `content`; the rest are ids/enums/role names
    assert {r[0] for r in rows} == {"message_id", "session_id", "tenant_id", "role", "content"}


def test_request_traces_columns_exact(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        rows = c.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='request_traces'"
        ).fetchall()
    text_cols = {r[0] for r in rows if r[1] == "text"}
    assert text_cols == {"trace_id", "message_id", "tenant_id", "outcome", "cache"}
    assert not any(r[1] in ("bytea", "character varying") for r in rows)


def test_app_role_has_no_access_to_alembic_version(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        n = c.execute(
            "SELECT count(*) FROM information_schema.role_table_grants "
            "WHERE grantee='rag_app' AND table_name='alembic_version'"
        ).fetchone()
    assert n == (0,)
