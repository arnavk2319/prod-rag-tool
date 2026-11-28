# ruff: noqa: SIM117, S105, S608
"""T-047 adversarial probes under real LOGIN roles (not SET ROLE from a superuser).

t047_svc mimics rag_svc (LOGIN, NOSUPERUSER, NOBYPASSRLS, member of rag_app). t047_owner mimics
rag_owner: a non-superuser that owns the tables and audit_append, as in compose. DB tests are
skipped without TEST_DATABASE_URL.
"""

import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import psycopg
import pytest
from alembic import command
from psycopg import sql
from psycopg.conninfo import make_conninfo
from test_policy_sql_extra import APPEND, bad_rows
from test_policy_tables import _cfg, dsn
from test_rls_all_tenant_tables import (
    ALL_TABLES,
    CAN_DELETE,
    NEW_TABLES,
    ROOT,
    ROWS,
    T0,
    A,
    B,
    catalogue,
    catalogue_offenders,
    count,
    seed,
)

__all__ = ["dsn"]

pytestmark = pytest.mark.integration
PW = "t047-probe-pw"
UPDATABLE = [t for t in NEW_TABLES if ROWS[t][1]]

SETUP = """
DO $$ DECLARE r record; BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 't047_svc') THEN
    CREATE ROLE t047_svc LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD 't047-probe-pw' IN ROLE rag_app;
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 't047_owner') THEN
    CREATE ROLE t047_owner LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD 't047-probe-pw';
  END IF;
  ALTER SCHEMA public OWNER TO t047_owner;
  FOR r IN SELECT oid::regclass AS n FROM pg_class
           WHERE relnamespace = 'public'::regnamespace AND relkind IN ('r', 'p') LOOP
    EXECUTE format('ALTER TABLE %s OWNER TO t047_owner', r.n);
  END LOOP;
  ALTER FUNCTION public.audit_append(timestamptz, text, text, text, text, jsonb)
    OWNER TO t047_owner;
  ALTER FUNCTION public.audit_log_append_only() OWNER TO t047_owner;
END $$
"""


@pytest.fixture
def real(dsn: str) -> Iterator[str]:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(SETUP)  # type: ignore[arg-type]
    yield dsn


def login(d: str, role: str, tenant: str | None = None, local: bool = True) -> psycopg.Connection:
    c = psycopg.connect(make_conninfo(d, user=role, password=PW))
    if tenant is not None:
        c.execute("SELECT set_config('app.tenant_id', %s, %s)", (tenant, local))
    return c


def append(c: psycopg.Connection, n: int, tenant: str | None, detail: str = "{}") -> int:
    row = c.execute(
        APPEND, (T0 + timedelta(seconds=n), tenant, "u", "act", f"t{n}", detail)
    ).fetchone()
    assert row is not None
    return int(row[0])


# --- roles ----------------------------------------------------------------------------------


def test_probe_roles_are_not_privileged(real: str) -> None:
    with login(real, "t047_svc") as c:
        assert c.execute(
            "SELECT rolsuper, rolbypassrls, pg_has_role(current_user, 'rag_app', 'member') "
            "FROM pg_roles WHERE rolname = current_user"
        ).fetchone() == (False, False, True)
    with login(real, "t047_owner") as c:
        assert c.execute(
            "SELECT rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone() == (False, False)


# --- CRUD matrix over real connections -------------------------------------------------------


@pytest.mark.parametrize("table", NEW_TABLES)
def test_crud_matrix_real_login(real: str, table: str) -> None:
    seed(real)
    with login(real, "t047_svc", A) as c:
        assert count(c, table) == 1 and count(c, table, B) == 0
        update = ROWS[table][1]
        if update:
            assert c.execute(update + " WHERE tenant_id = %s", (B,)).rowcount == 0  # type: ignore[operator]
        if table in CAN_DELETE:
            q = sql.SQL("DELETE FROM {} WHERE tenant_id = %s").format(sql.Identifier(table))
            assert c.execute(q, (B,)).rowcount == 0
    with login(real, "t047_owner") as c:  # the owner (RLS not forced) sees everything
        assert count(c, table) == 2


@pytest.mark.parametrize("table", UPDATABLE)
def test_update_cannot_move_a_row_to_another_tenant(real: str, table: str) -> None:
    seed(real)
    q = sql.SQL("UPDATE {} SET tenant_id = %s WHERE tenant_id = %s").format(sql.Identifier(table))
    with pytest.raises(psycopg.errors.InsufficientPrivilege, match="row-level security"):
        with login(real, "t047_svc", A) as c:
            c.execute(q, (B, A))
    with login(real, "t047_owner") as c:
        assert count(c, table, A) == 1


VARIANTS = [
    "TENANT-A", "Tenant-A", "tenant-a ", " tenant-a", "tenant-a\n", "tenant-a\t", "tenant-",
    "tenant-%", "%", "_", "tenant-a' OR '1'='1", "tenant-a'; --", "' OR tenant_id IS NOT NULL --",
    '"tenant-a"', "tenant-a\\", "tenant-a\u200b", "tenant\u2011a", "tenant-b ", "",
]  # fmt: skip


@pytest.mark.parametrize("table", NEW_TABLES)
def test_tenant_value_variants_never_match(real: str, table: str) -> None:
    seed(real)
    for value in VARIANTS:
        with login(real, "t047_svc", value) as c:
            assert count(c, table) == 0, repr(value)
    with login(real, "t047_svc", "tenant-a") as c:  # the exact value does match
        assert count(c, table) == 1


# --- session setting lifetime ----------------------------------------------------------------


def test_set_local_does_not_leak_but_session_set_does_until_reset(real: str) -> None:
    """Pooled-connection behaviour, documented: set_config(.., true) / SET LOCAL end with the
    transaction; a plain SET (or set_config(.., false)) survives COMMIT and is inherited by the
    next borrower until RESET / DISCARD ALL. The app must set the tenant per transaction."""
    seed(real)
    with psycopg.connect(make_conninfo(real, user="t047_svc", password=PW), autocommit=True) as c:
        with c.transaction():
            c.execute("SET LOCAL app.tenant_id = 'tenant-a'")
            assert count(c, "chat_sessions") == 1
        assert count(c, "chat_sessions") == 0  # LOCAL reverted at commit
        c.execute("SET app.tenant_id = 'tenant-a'")  # session scope
        assert count(c, "chat_sessions") == 1  # LEAK: the next borrower inherits tenant A
        c.execute("RESET app.tenant_id")
        assert count(c, "chat_sessions") == 0
        c.execute("SET app.tenant_id = 'tenant-a'")
        c.execute("DISCARD ALL")
        assert count(c, "chat_sessions") == 0
        c.execute("SET app.tenant_id = 'tenant-a'")
        c.execute("SET app.tenant_id TO DEFAULT")
        assert count(c, "chat_sessions") == 0


def test_tenant_changed_mid_transaction_is_self_asserted(real: str) -> None:
    """Documented gap (binding the setting to the authenticated identity is T-023): any rag_app
    session may switch its own tenant, so RLS isolates by the setting, not by identity."""
    seed(real)
    with login(real, "t047_svc", A) as c:
        assert count(c, "chat_sessions", B) == 0
        c.execute("SELECT set_config('app.tenant_id', %s, true)", (B,))
        assert count(c, "chat_sessions", B) == 1


# --- NULL tenant rows -------------------------------------------------------------------------


def test_null_tenant_audit_rows_visible_only_to_owner(real: str) -> None:
    with login(real, "t047_svc") as c:  # no tenant: system row
        append(c, 1, None)
    for tenant in (A, "", "null", "NULL", None):
        with login(real, "t047_svc", tenant) as c:
            assert count(c, "audit_log") == 0, repr(tenant)
            nulls = c.execute("SELECT count(*) FROM audit_log WHERE tenant_id IS NULL")
            assert nulls.fetchone() == (0,)
    with login(real, "t047_owner") as c:
        nulls = c.execute("SELECT count(*) FROM audit_log WHERE tenant_id IS NULL")
        assert nulls.fetchone() == (1,)


def test_system_rows_can_be_forged_by_any_tenantless_rag_app_session(real: str) -> None:
    """Assessment (not a fix): a rag_app session with no tenant creates NULL-tenant rows with a
    free-form actor/action. They are invisible to tenants and the chain stays valid, so this
    cannot leak or break the chain, but system-event provenance is only as trustworthy as the
    app code (actor is caller-supplied everywhere). Pinned so a change is deliberate."""
    with login(real, "t047_svc") as c:
        c.execute(APPEND, (T0, None, "system", "policy.approve", "p1", '{"forged": true}'))
    with login(real, "t047_owner") as c:
        row = c.execute("SELECT actor, action, tenant_id FROM audit_log").fetchone()
        assert row == ("system", "policy.approve", None)
    assert bad_rows(real) == []


# --- ON CONFLICT, COPY, TRUNCATE -------------------------------------------------------------

CONFLICT = {
    "chat_sessions": (
        "(session_id, tenant_id, user_id)",
        "(%s, %s, 'evil')",
        "session_id",
        "sess-b",
    ),
    "ingest_jobs": ("(job_id, tenant_id)", "(%s, %s)", "job_id", "b1"),
}


@pytest.mark.parametrize("table", sorted(CONFLICT))
def test_on_conflict_cannot_take_over_another_tenants_row(real: str, table: str) -> None:
    seed(real)
    cols, vals, key_col, key = CONFLICT[table]
    base = f"INSERT INTO {table} {cols} VALUES {vals}"
    upd = f"{base} ON CONFLICT ({key_col}) DO UPDATE SET tenant_id = EXCLUDED.tenant_id"
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with login(real, "t047_svc", A) as c:
            c.execute(upd, (key, A))  # type: ignore[call-overload]
    with login(real, "t047_svc", A) as c:  # DO NOTHING must not touch the foreign row either
        c.execute(base + " ON CONFLICT DO NOTHING", (key, A))  # type: ignore[call-overload]
    with login(real, "t047_owner") as c:
        assert count(c, table, B) == 1 and count(c, table, A) == 1


@pytest.mark.parametrize("table", NEW_TABLES)
def test_copy_respects_rls_and_copy_from_is_refused(real: str, table: str) -> None:
    seed(real)
    ident = sql.Identifier(table)
    with login(real, "t047_svc", A) as c:
        with c.cursor() as cur:
            with cur.copy(sql.SQL("COPY {} (tenant_id) TO STDOUT").format(ident)) as cp:
                out = b"".join(bytes(x) for x in cp)
        assert out.decode().split() == [A]
    with pytest.raises(psycopg.Error):
        with login(real, "t047_svc", A) as c:
            with c.cursor() as cur:
                with cur.copy(sql.SQL("COPY {} (tenant_id) FROM STDIN").format(ident)) as cp:
                    cp.write(f"{B}\n".encode())
    with login(real, "t047_owner") as c:
        assert count(c, table, B) == 1


@pytest.mark.parametrize("table", ALL_TABLES)
def test_truncate_denied_for_the_app_role(real: str, table: str) -> None:
    seed(real)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with login(real, "t047_svc", A) as c:
            c.execute(sql.SQL("TRUNCATE {}").format(sql.Identifier(table)))
    if table in NEW_TABLES:
        with login(real, "t047_owner") as c:
            assert count(c, table) == 2


# --- functions and views that could bypass RLS -----------------------------------------------


def test_only_audit_append_is_security_definer_and_no_view_exposes_tenant_data(real: str) -> None:
    with psycopg.connect(real) as c:
        definers = c.execute(
            "SELECT p.proname FROM pg_proc p WHERE p.prosecdef AND p.pronamespace NOT IN "
            "('pg_catalog'::regnamespace, 'information_schema'::regnamespace) AND NOT EXISTS "
            "(SELECT 1 FROM pg_depend d WHERE d.objid = p.oid AND d.deptype = 'e')"
        ).fetchall()
        assert definers == [("audit_append",)]
        views = c.execute(
            "SELECT c.relname FROM pg_class c WHERE c.relnamespace = 'public'::regnamespace "
            "AND c.relkind IN ('v','m','f')"
        ).fetchall()
        assert views == []  # a view would run with its owner's rights and bypass RLS


# --- audit_append hijack attempts ------------------------------------------------------------


def test_audit_append_ignores_hostile_search_path_and_temp_shadows(real: str) -> None:
    with login(real, "t047_svc", A) as c:
        c.execute("CREATE TEMP TABLE audit_log (LIKE public.audit_log INCLUDING DEFAULTS)")
        c.execute(
            "CREATE FUNCTION pg_temp.sha256(bytea) RETURNS bytea LANGUAGE sql AS "
            "$$ SELECT '\\x00'::bytea $$"
        )
        c.execute(
            "CREATE FUNCTION pg_temp.to_char(timestamptz, text) RETURNS text LANGUAGE sql AS "
            "$$ SELECT 'x' $$"
        )
        c.execute("SET LOCAL search_path = pg_temp, public, pg_catalog")
        append(c, 1, A)
        append(c, 2, None)
    with login(real, "t047_owner") as c:
        assert count(c, "audit_log") == 2
    assert bad_rows(real) == []


@pytest.mark.parametrize("arg", ["TENANT-A", "Tenant-A", "tenant-a ", " tenant-a", "tenant-b", "*"])
def test_audit_append_requires_the_exact_session_tenant(real: str, arg: str) -> None:
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with login(real, "t047_svc", "tenant-a") as c:
            append(c, 1, arg)
    with login(real, "t047_owner") as c:
        assert count(c, "audit_log") == 0


def test_audit_append_tenant_with_quotes_and_unicode_round_trips(real: str) -> None:
    weird = "t'\"\\;--\u00e9\u4e2d"
    with login(real, "t047_svc", weird) as c:
        append(c, 1, weird)
        append(c, 2, "")
        assert count(c, "audit_log", weird) == 2
    assert bad_rows(real) == []


def test_audit_append_huge_detail_keeps_chain_valid(real: str) -> None:
    big = '{"k": "' + "x" * 3_000_000 + '"}'
    with login(real, "t047_svc", A) as c:
        append(c, 1, A, big)
        append(c, 2, A)
    assert bad_rows(real) == []


def test_concurrent_cross_tenant_appends_with_real_logins_one_chain(real: str) -> None:
    def work(n: int) -> int:
        t = A if n % 2 else B
        with login(real, "t047_svc", t) as c:
            return append(c, n, t)

    with ThreadPoolExecutor(12) as ex:
        ids = list(ex.map(work, range(24)))
    assert len(set(ids)) == 24
    with login(real, "t047_owner") as c:  # verifier role reads every tenant's rows
        assert count(c, "audit_log") == 24
    assert bad_rows(real) == []


# --- definer owner is not a superuser ---------------------------------------------------------


def test_non_superuser_definer_appends_enforces_tenant_rule_and_verifier_reads_all(
    real: str,
) -> None:
    with login(real, "t047_svc", A) as c:
        append(c, 1, A)
    with login(real, "t047_svc", A) as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="session tenant"):
            append(c, 2, B)
    with login(real, "t047_svc", B) as c:
        append(c, 3, B)
    with login(real, "t047_owner") as c:  # owner is neither superuser nor BYPASSRLS
        assert count(c, "audit_log") == 2
    assert bad_rows(real) == []


def test_audit_log_immutable_even_for_owner_and_tamper_is_detected(real: str) -> None:
    with login(real, "t047_svc", A) as c:
        append(c, 1, A)
        append(c, 2, A)
    for stmt in ("UPDATE audit_log SET actor = 'x'", "DELETE FROM audit_log"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege, match="append-only"):
            with login(real, "t047_owner") as c:
                c.execute(stmt)  # type: ignore[call-overload]
    with login(real, "t047_owner") as c:  # the owner can disable the trigger: verify catches it
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_change")
        c.execute(
            "UPDATE audit_log SET detail = '{\"x\": 1}' "
            "WHERE audit_id = (SELECT min(audit_id) FROM audit_log)"
        )
    assert len(bad_rows(real)) >= 1


def test_owner_cannot_truncate_audit_log(real: str) -> None:
    with login(real, "t047_svc", A) as c:
        append(c, 1, A)
    with pytest.raises(psycopg.Error):
        with login(real, "t047_owner") as c:
            c.execute("TRUNCATE audit_log")


def test_advisory_lock_key_can_be_taken_by_any_session_documented(real: str) -> None:
    """Assessment: the chain lock is a plain advisory lock; any rag_app session can hold the same
    key and stall every append (DoS, no integrity loss). Mitigation: statement/lock timeouts on
    the app role."""
    with login(real, "t047_svc", A) as holder:
        holder.execute("SELECT pg_advisory_xact_lock(1096041805, 1633837924)")
        with login(real, "t047_svc", B) as c:
            c.execute("SET LOCAL lock_timeout = '300ms'")
            with pytest.raises(psycopg.errors.LockNotAvailable):
                append(c, 1, B)


# --- migration round trips with data ----------------------------------------------------------


def test_upgrade_onto_rows_in_every_table_and_round_trip_keep_data(real: str) -> None:
    with login(real, "t047_svc", A) as c:
        append(c, 1, A)
    with login(real, "t047_svc") as c:
        append(c, 2, None)
    command.downgrade(_cfg(), "0003")
    seed(real)  # rows in every tenant table while RLS is off
    with psycopg.connect(real) as c:
        before = {t: count(c, t) for t in NEW_TABLES}
    command.upgrade(_cfg(), "head")
    with psycopg.connect(real) as c:
        assert {t: count(c, t) for t in NEW_TABLES} == before
        nulls = c.execute("SELECT count(*) FROM audit_log WHERE tenant_id IS NULL")
        assert nulls.fetchone() == (1,)
    with login(real, "t047_svc", A) as c:
        for t in NEW_TABLES:
            assert count(c, t, B) == 0
        append(c, 9, A)
    # seed() writes raw audit rows (no hash); only those may fail verification, not 0004's rows
    with psycopg.connect(real) as c:
        raw = [r[0] for r in c.execute("SELECT audit_id FROM audit_log WHERE actor = 'a'")]
    assert len(raw) == 2
    assert set(bad_rows(real)) <= set(raw) | {r + 1 for r in raw}  # a raw row breaks its successor


# --- catalogue mutation ------------------------------------------------------------------------


def _offenders(c: psycopg.Connection) -> list[str]:
    return catalogue_offenders(catalogue(c))


def test_catalogue_flags_new_unprotected_tables(real: str) -> None:
    with psycopg.connect(real, autocommit=True) as c:
        assert _offenders(c) == []
        c.execute("CREATE TABLE zz_new (id int, tenant_id text)")
        assert _offenders(c) == ["public.zz_new"]
        c.execute("ALTER TABLE zz_new ENABLE ROW LEVEL SECURITY")  # RLS but no policy
        assert _offenders(c) == ["public.zz_new"]
        c.execute("CREATE POLICY p ON zz_new USING (true) WITH CHECK (true)")
        assert _offenders(c) == ["public.zz_new"]  # a permissive expression does not count
        c.execute("DROP POLICY p ON zz_new")
        c.execute(
            "CREATE POLICY p ON zz_new USING (tenant_id = current_setting('app.tenant_id', true)) "
            "WITH CHECK (tenant_id = current_setting('app.tenant_id', true))"
        )
        assert _offenders(c) == []
        c.execute("ALTER TABLE zz_new DISABLE ROW LEVEL SECURITY")  # policy but RLS off
        assert _offenders(c) == ["public.zz_new"]


# --- compose / bootstrap -----------------------------------------------------------------------


def test_every_ragapp_dsn_in_compose_uses_rag_svc_and_init_creates_it_unprivileged() -> None:
    text = (ROOT / "compose.yaml").read_text()
    pat = r"postgres(?:ql)?(?:\+\w+)?://([^:@/\s]+)[:@][^\s]*/ragapp\b"
    users: list[Any] = re.findall(pat, text)
    assert users
    assert set(users) == {"rag_svc"}
    init = (ROOT / "docker/postgres/init.sh").read_text()
    assert re.search(r"CREATE ROLE rag_svc LOGIN NOSUPERUSER NOBYPASSRLS", init)
    assert re.search(r"GRANT rag_app TO rag_svc", init)
