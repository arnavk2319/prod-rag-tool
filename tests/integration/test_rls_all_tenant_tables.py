# ruff: noqa: SIM117
"""Row-level security on every tenant-scoped table, audit_log append-only via function (T-047).

DB tests are skipped unless TEST_DATABASE_URL points at a reachable throwaway Postgres with
pgvector >= 0.8. The compose test needs no database.
"""

import re
import time
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from psycopg import sql
from test_policy_sql_extra import APPEND, bad_rows
from test_policy_tables import _cfg, dsn

__all__ = ["dsn"]

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
T0 = datetime(2026, 1, 1, tzinfo=UTC)
A, B = "tenant-a", "tenant-b"

# Tables allowed to carry tenant_id without a policy. Deliberately empty.
EXCEPTIONS: frozenset[str] = frozenset()
ALREADY_PROTECTED = {"chunks", "documents", "parent_chunks", "policy_documents"}
NEW_TABLES = [
    "raw_objects",
    "ingest_jobs",
    "quarantine_items",
    "chat_sessions",
    "chat_messages",
    "request_traces",
    "audit_log",
]
ALL_TABLES = sorted(ALREADY_PROTECTED | set(NEW_TABLES))

# table -> (seed/insert statement with %(t)s tenant and %(k)s key, UPDATE statement or None)
ROWS: dict[str, tuple[str, str | None]] = {
    "raw_objects": (
        "INSERT INTO raw_objects (sha256, tenant_id, mime, size, content) "
        "VALUES (%(k)s, %(t)s, 'text/plain', 1, 'x')",
        "UPDATE raw_objects SET mime = 'text/html'",
    ),
    "ingest_jobs": (
        "INSERT INTO ingest_jobs (job_id, tenant_id) VALUES (%(k)s, %(t)s)",
        "UPDATE ingest_jobs SET status = 'done'",
    ),
    "quarantine_items": (
        "INSERT INTO quarantine_items (item_id, tenant_id, reason) VALUES (%(k)s, %(t)s, 'r')",
        "UPDATE quarantine_items SET status = 'resolved'",
    ),
    "chat_sessions": (
        "INSERT INTO chat_sessions (session_id, tenant_id, user_id) VALUES (%(k)s, %(t)s, 'u')",
        "UPDATE chat_sessions SET user_id = 'v'",
    ),
    "chat_messages": (
        "INSERT INTO chat_messages (message_id, session_id, tenant_id, role, content) "
        "VALUES (%(k)s, 'sess-b', %(t)s, 'user', 'hi')",
        "UPDATE chat_messages SET content = 'changed'",
    ),
    "request_traces": (
        "INSERT INTO request_traces (trace_id, tenant_id, outcome) VALUES (%(k)s, %(t)s, 'ok')",
        None,
    ),
    "audit_log": (
        "INSERT INTO audit_log (tenant_id, actor, action) VALUES (%(t)s, 'a', 'x')",
        None,
    ),
}
CAN_DELETE = {"raw_objects", "ingest_jobs", "quarantine_items", "chat_sessions", "chat_messages"}


@pytest.fixture
def db(dsn: str) -> Iterator[str]:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "DO $$ BEGIN IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'rag_svc') THEN "
            "CREATE ROLE rag_svc NOLOGIN NOSUPERUSER NOBYPASSRLS IN ROLE rag_app; END IF; END $$"
        )
    yield dsn


def svc(d: str, tenant: str | None) -> psycopg.Connection:
    """A transaction as the application login (rag_svc) with app.tenant_id set locally."""
    c = psycopg.connect(d)
    c.execute("SET ROLE rag_svc")
    if tenant is not None:
        c.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    return c


def seed(d: str) -> None:
    """As the owner (bypasses RLS): one row per tenant per table. chat_sessions keys are the
    session ids chat_messages refers to."""
    with psycopg.connect(d) as c:
        for table in ["chat_sessions", *[t for t in ROWS if t != "chat_sessions"]]:
            insert = ROWS[table][0]
            for tenant, key, sess in ((A, "a1", "sess-a"), (B, "b1", "sess-b")):
                params = {"t": tenant, "k": sess if table == "chat_sessions" else key}
                c.execute(insert.replace("sess-b", sess), params)  # type: ignore[call-overload]


def count(c: psycopg.Connection, table: str, where: str = "") -> int:
    q = sql.SQL("SELECT count(*) FROM {}").format(sql.Identifier(table))
    if where:
        q = q + sql.SQL(" WHERE tenant_id = %s")
        row = c.execute(q, (where,)).fetchone()
    else:
        row = c.execute(q).fetchone()
    assert row is not None
    return int(row[0])


# (1) catalogue ------------------------------------------------------------------------------

CATALOGUE = """
SELECT n.nspname, c.relname, c.relrowsecurity,
       (SELECT count(*) FROM pg_policies p
         WHERE p.schemaname = n.nspname AND p.tablename = c.relname
           AND p.qual LIKE '%current_setting(''app.tenant_id''%'
           AND p.with_check LIKE '%current_setting(''app.tenant_id''%')
FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
  AND n.nspname NOT LIKE 'pg_temp%' AND n.nspname NOT LIKE 'pg_toast_temp%'
  AND c.relkind IN ('r', 'p') AND NOT c.relispartition AND c.relname <> 'alembic_version'
  AND EXISTS (SELECT 1 FROM information_schema.columns k
               WHERE k.table_schema = n.nspname AND k.table_name = c.relname
                 AND k.column_name = 'tenant_id')
ORDER BY 1, 2
"""


def catalogue(c: psycopg.Connection) -> list[tuple[str, str, bool, int]]:
    return c.execute(CATALOGUE).fetchall()  # type: ignore[call-overload,no-any-return]


def catalogue_offenders(rows: list[tuple[str, str, bool, int]]) -> list[str]:
    """Tables with a tenant_id column in any user schema that lack RLS or a policy whose USING
    and WITH CHECK both reference current_setting('app.tenant_id' (USING (true) does not count)."""
    names = ((f"{r[0]}.{r[1]}", r) for r in rows)
    return [n for n, r in names if n not in EXCEPTIONS and not (r[2] and r[3] >= 1)]


def test_every_table_with_tenant_id_has_rls_and_a_policy(db: str) -> None:
    with psycopg.connect(db) as c:
        c.execute("SELECT create_chunks_partition('v1')")  # partitions must not trip the check
        rows = catalogue(c)
    assert {f"{r[0]}.{r[1]}" for r in rows} == {f"public.{t}" for t in ALL_TABLES}
    assert catalogue_offenders(rows) == []


def test_catalogue_checker_flags_missing_rls_and_permissive_policies(db: str) -> None:
    """Negative test of the checker on scratch tables in a rolled-back transaction."""
    good = "tenant_id = current_setting('app.tenant_id', true)"
    with psycopg.connect(db) as c:
        c.execute("CREATE SCHEMA scratch")
        for name in ("no_rls", "no_policy", "open_policy", "good"):
            c.execute(f"CREATE TABLE scratch.{name} (tenant_id text)")  # type: ignore[arg-type]
            if name != "no_rls":
                c.execute(f"ALTER TABLE scratch.{name} ENABLE ROW LEVEL SECURITY")  # type: ignore[arg-type]
        c.execute("CREATE POLICY p ON scratch.open_policy USING (true) WITH CHECK (true)")
        c.execute(
            f"CREATE POLICY p ON scratch.good USING ({good}) WITH CHECK ({good})"  # type: ignore[arg-type]
        )
        offenders = catalogue_offenders(catalogue(c))
        c.rollback()
    assert sorted(offenders) == ["scratch.no_policy", "scratch.no_rls", "scratch.open_policy"]


def test_partitions_are_covered_through_their_parent(db: str) -> None:
    """A partition has no policy of its own; it is unreachable for rag_app except via the parent
    (no grants), where the parent's policy applies."""
    with psycopg.connect(db) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        parts = c.execute(
            "SELECT c.relname, c.relrowsecurity FROM pg_class c WHERE c.relispartition"
        ).fetchall()
        assert parts  # chunks_v1 (and any parent_chunks partition)
        assert c.execute(
            "SELECT has_table_privilege('rag_app', 'chunks_v1', 'SELECT')"
        ).fetchone() == (False,)
    with svc(db, A) as c:
        assert count(c, "chunks") == 0


def test_every_policy_uses_the_same_expression(db: str) -> None:
    with psycopg.connect(db) as c:
        rows = c.execute(
            "SELECT tablename, qual, with_check, cmd, permissive FROM pg_policies "
            "WHERE schemaname = 'public' AND policyname = 'tenant_isolation'"
        ).fetchall()
    assert {r[0] for r in rows} == set(ALL_TABLES)
    for _, qual, check, cmd, permissive in rows:
        assert qual == check == "(tenant_id = current_setting('app.tenant_id'::text, true))"
        assert cmd == "ALL" and permissive == "PERMISSIVE"


# (2) per-table isolation --------------------------------------------------------------------


@pytest.mark.parametrize("table", NEW_TABLES)
def test_tenant_a_cannot_read_tenant_b(db: str, table: str) -> None:
    seed(db)
    with svc(db, A) as c:
        assert count(c, table, B) == 0
        assert count(c, table) == 1  # only its own row
        assert count(c, table, A) == 1


@pytest.mark.parametrize("table", NEW_TABLES)
def test_no_tenant_context_reads_nothing(db: str, table: str) -> None:
    seed(db)
    with svc(db, None) as c:  # never set
        assert count(c, table) == 0
    with svc(db, "") as c:  # set to empty
        assert count(c, table) == 0
    with svc(db, A) as c:
        pass
    with svc(db, None) as c:  # the previous (local) setting did not leak
        assert count(c, table) == 0


@pytest.mark.parametrize("table", [t for t in NEW_TABLES if ROWS[t][1]])
def test_tenant_a_cannot_update_tenant_b(db: str, table: str) -> None:
    seed(db)
    update = ROWS[table][1]
    assert update is not None
    with svc(db, A) as c:
        cur = c.execute(update + " WHERE tenant_id = %s", (B,))  # type: ignore[operator]
        assert cur.rowcount == 0
    with psycopg.connect(db) as c:  # owner: B's row is untouched
        assert count(c, table, B) == 1


@pytest.mark.parametrize("table", sorted(CAN_DELETE))
def test_tenant_a_cannot_delete_tenant_b(db: str, table: str) -> None:
    seed(db)
    with svc(db, A) as c:
        cur = c.execute(
            sql.SQL("DELETE FROM {} WHERE tenant_id = %s").format(sql.Identifier(table)), (B,)
        )
        assert cur.rowcount == 0
    with psycopg.connect(db) as c:
        assert count(c, table, B) == 1


@pytest.mark.parametrize("table", [t for t in NEW_TABLES if t != "audit_log"])
def test_insert_for_another_tenant_is_rejected_by_with_check(db: str, table: str) -> None:
    seed(db)
    insert = ROWS[table][0].replace("sess-b", "sess-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with svc(db, A) as c:
            c.execute(insert, {"t": B, "k": "new"})  # type: ignore[call-overload]
    with svc(db, A) as c:  # the same insert for its own tenant works
        c.execute(insert, {"t": A, "k": "new"})  # type: ignore[call-overload]
        assert count(c, table, A) == 2


@pytest.mark.parametrize("table", [t for t in NEW_TABLES if t != "audit_log"])
def test_insert_without_tenant_context_is_rejected(db: str, table: str) -> None:
    insert = ROWS[table][0].replace("sess-b", "sess-a")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with svc(db, None) as c:
            c.execute(insert, {"t": A, "k": "new"})  # type: ignore[call-overload]


def test_owner_bypasses_rls_because_it_is_enabled_not_forced(db: str) -> None:
    seed(db)
    with psycopg.connect(db) as c:
        rows = c.execute(
            "SELECT relname, relforcerowsecurity FROM pg_class WHERE relname = ANY(%s) "
            "AND NOT relispartition",
            (ALL_TABLES,),
        ).fetchall()
        assert rows and not any(r[1] for r in rows)
        for table in NEW_TABLES:
            assert count(c, table) == 2  # owner sees both tenants


# (3) audit_log ------------------------------------------------------------------------------


def _append(c: psycopg.Connection, n: int, tenant: str | None) -> int:
    row = c.execute(
        APPEND, (T0 + timedelta(seconds=n), tenant, "u", "act", f"t{n}", "{}")
    ).fetchone()
    assert row is not None
    return int(row[0])


def test_direct_insert_into_audit_log_is_denied(db: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with svc(db, A) as c:
            c.execute(
                "INSERT INTO audit_log (tenant_id, actor, action) VALUES (%s, 'a', 'x')", (A,)
            )
    with psycopg.connect(db) as c:
        assert c.execute(
            "SELECT has_table_privilege('rag_svc', 'audit_log', 'INSERT'), "
            "has_sequence_privilege('rag_svc', 'audit_log_audit_id_seq', 'USAGE')"
        ).fetchone() == (False, False)


def test_audit_append_works_and_chain_verifies(db: str) -> None:
    with svc(db, A) as c:
        ids = [_append(c, n, A) for n in range(3)]
    with svc(db, B) as c:
        ids.append(_append(c, 3, B))
    assert ids == sorted(ids)
    assert bad_rows(db) == []


def test_audit_append_refuses_a_foreign_or_unscoped_tenant(db: str) -> None:
    for session, arg in ((A, B), (None, B), ("", A)):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with svc(db, session) as c:
                _append(c, 1, arg)
    with psycopg.connect(db) as c:
        assert count(c, "audit_log") == 0


def test_audit_append_null_tenant_takes_session_or_stays_system(db: str) -> None:
    with svc(db, A) as c:
        _append(c, 1, None)
    with svc(db, None) as c:
        _append(c, 2, None)  # system event
    with psycopg.connect(db) as c:
        rows = c.execute("SELECT tenant_id FROM audit_log ORDER BY audit_id").fetchall()
    assert rows == [(A,), (None,)]
    assert bad_rows(db) == []  # the hash covers the stored (effective) tenant
    with svc(db, A) as c:  # a tenant never sees the NULL-tenant row
        assert c.execute("SELECT count(*) FROM audit_log WHERE tenant_id IS NULL").fetchone() == (
            0,
        )
        assert count(c, "audit_log") == 1


def test_tenant_a_cannot_read_tenant_b_audit_rows_but_owner_reads_all(db: str) -> None:
    with svc(db, A) as c:
        _append(c, 1, A)
    with svc(db, B) as c:
        _append(c, 2, B)
        _append(c, 3, None)
    with svc(db, A) as c:
        assert count(c, "audit_log", B) == 0 and count(c, "audit_log") == 1
    with svc(db, None) as c:
        assert count(c, "audit_log") == 0
    with psycopg.connect(db) as c:  # owner verifier path
        assert count(c, "audit_log") == 3
    assert bad_rows(db) == []


def test_audit_log_remains_append_only_under_rls(db: str) -> None:
    with svc(db, A) as c:
        _append(c, 1, A)
    for stmt in ("UPDATE audit_log SET actor = 'evil'", "DELETE FROM audit_log"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with svc(db, A) as c:
                c.execute(stmt)  # type: ignore[call-overload]
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
        with psycopg.connect(db) as c:  # even the owner: the trigger stays
            c.execute("DELETE FROM audit_log")


def test_concurrent_appends_keep_one_linear_chain(db: str) -> None:
    def work(n: int) -> int:
        tenant = A if n % 2 else B
        with svc(db, tenant) as c:
            i = _append(c, n, tenant)
            c.execute("SELECT pg_sleep(0.01)")
            return i

    with ThreadPoolExecutor(20) as ex:
        ids = list(ex.map(work, range(20)))
    assert len(set(ids)) == 20
    with psycopg.connect(db) as c:
        assert c.execute("SELECT count(*), count(DISTINCT hash) FROM audit_log").fetchone() == (
            20,
            20,
        )
        assert c.execute("SELECT count(*) FROM audit_log WHERE prev_hash IS NULL").fetchone() == (
            1,
        )
        assert c.execute(
            "SELECT count(*) FROM (SELECT prev_hash FROM audit_log GROUP BY prev_hash "
            "HAVING count(*) > 1) x"
        ).fetchone() == (0,)
    assert bad_rows(db) == []


def test_audit_append_is_definer_with_fixed_search_path_and_no_public_execute(db: str) -> None:
    with psycopg.connect(db) as c:
        row = c.execute(
            "SELECT prosecdef, proconfig, pg_get_userbyid(proowner) = current_user, "
            "pg_get_functiondef(oid) LIKE '%%pg_advisory_xact_lock(1096041805, 1633837924)%%', "
            "pg_get_functiondef(oid) LIKE '%%hashtext%%' "
            "FROM pg_proc WHERE proname = 'audit_append'"
        ).fetchone()
        assert row is not None
        definer, config, owned, two_int, hashtext = row
        assert definer is True
        assert sorted(config) == ["lock_timeout=5s", "search_path=pg_catalog, public"]
        assert owned is True and two_int is True and hashtext is False
        sig = "audit_append(timestamptz,text,text,text,text,jsonb)"
        assert c.execute(
            "SELECT has_function_privilege('public', %s, 'EXECUTE'), "
            "has_function_privilege('rag_svc', %s, 'EXECUTE')",
            (sig, sig),
        ).fetchone() == (False, True)


def test_search_path_cannot_hijack_the_definer_function(db: str) -> None:
    """Shadow tables in the caller's temp schema (searched first) and a rogue search_path must
    not capture the function's reads or writes."""
    with svc(db, A) as c:
        c.execute("CREATE TEMP TABLE audit_log (LIKE public.audit_log INCLUDING DEFAULTS)")
        c.execute("SET LOCAL search_path = pg_temp, public")
        _append(c, 1, A)
        assert c.execute("SELECT count(*) FROM pg_temp.audit_log").fetchone() == (0,)
    with psycopg.connect(db) as c:
        assert count(c, "audit_log") == 1
    assert bad_rows(db) == []


# (4) downgrade / upgrade --------------------------------------------------------------------

SEVEN = NEW_TABLES


def test_downgrade_restores_0003_and_upgrade_reapplies(db: str) -> None:
    with svc(db, A) as c:
        _append(c, 1, A)
    command.downgrade(_cfg(), "0003")
    with psycopg.connect(db) as c:
        rls = c.execute(
            "SELECT relname, relrowsecurity FROM pg_class WHERE relname = ANY(%s)", (SEVEN,)
        ).fetchall()
        assert rls and not any(r[1] for r in rls)
        assert c.execute(
            "SELECT count(*) FROM pg_policies WHERE tablename = ANY(%s)", (SEVEN,)
        ).fetchone() == (0,)
        assert c.execute(
            "SELECT prosecdef, proconfig, pg_get_functiondef(oid) LIKE '%%hashtext%%' "
            "FROM pg_proc WHERE proname = 'audit_append'"
        ).fetchone() == (False, None, True)
        assert c.execute(
            "SELECT has_table_privilege('rag_app', 'audit_log', 'INSERT')"
        ).fetchone() == (True,)
        # the policies that existed before 0004 are kept
        assert c.execute(
            "SELECT count(*) FROM pg_policies WHERE tablename = ANY(%s)",
            (sorted(ALREADY_PROTECTED),),
        ).fetchone() == (4,)
    with svc(db, A) as c:  # the 0003 function: any tenant argument is accepted
        _append(c, 2, B)
    assert bad_rows(db) == []
    command.upgrade(_cfg(), "head")
    with svc(db, A) as c:
        _append(c, 3, A)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("INSERT INTO audit_log (actor, action) VALUES ('a', 'b')")
    assert bad_rows(db) == []  # the chain continues across down and up
    with svc(db, A) as c:
        assert count(c, "audit_log", B) == 0


# (5) compose --------------------------------------------------------------------------------


def test_compose_app_dsn_uses_the_rag_svc_role() -> None:
    text = (ROOT / "compose.yaml").read_text()
    urls: list[Any] = re.findall(r"^\s*DATABASE_URL:\s*(\S+)", text, flags=re.M)
    assert urls, "compose.yaml defines no DATABASE_URL"
    for url in urls:
        assert re.match(r"postgresql(\+psycopg)?://rag_svc:", url), url
    assert not re.search(r"(DATABASE_URL|_DSN)\S*:\s*\S*://(rag_owner|postgres)[:@]", text)


def test_second_append_fails_fast_while_another_session_holds_the_chain_lock(db: str) -> None:
    with psycopg.connect(db) as holder:
        holder.execute("SELECT pg_advisory_xact_lock(1096041805, 1633837924)")
        start = time.monotonic()
        with pytest.raises(psycopg.errors.LockNotAvailable):
            with svc(db, A) as c:
                _append(c, 1, A)
        assert time.monotonic() - start < 6.5  # lock_timeout is 5 s, not an endless wait
    with svc(db, A) as c:  # the lock is free again
        _append(c, 2, A)
    assert bad_rows(db) == []
