# ruff: noqa: SIM117, S608, F811, F401
"""Extra hunting tests for migration 0003 and the SQL policy adapters (T-045 part 2).

Skipped unless TEST_DATABASE_URL points at a reachable throwaway Postgres with pgvector >= 0.8.
Strict xfails mark real defects with a clean requirement; passing "pin" tests document behaviour
that is deliberate or a known gap (named in the test).
"""

import asyncio
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from typing import Any

import psycopg
import pytest
from alembic import command
from test_policy_tables import (
    HAPPY,
    INSERT_MIN,
    T0,
    S,
    _app,
    _audit_rows,
    _cfg,
    _event,
    _index_row,
    _insert,
    _norm,
    _walk,
    dsn,
    make_policy,
)

from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import (
    FakeEvalRuns,
    FakePolicyRepository,
    FakeRegistryMirror,
)
from rag_platform.adapters.sql_eval_runs import SqlEvalRuns, metrics_clean
from rag_platform.adapters.sql_policy_repository import SqlPolicyRepository
from rag_platform.domain.errors import Conflict, NotFound, ValidationFailed
from rag_platform.domain.policy import (
    ALLOWED,
    INDEX_STATUS,
    ChangeKind,
    IllegalTransition,
    PolicyDocument,
    PolicyDraft,
    PolicyRepository,
)
from rag_platform.services.policy_gate import (
    ChunkSummary,
    PolicyGate,
    PolicyRuleViolation,
    sync_pending,
)

ALL_DOCS = frozenset({("a", "d1"), ("b", "d2")})


pytestmark = pytest.mark.integration

VERIFY = """
SELECT audit_id FROM (
  SELECT audit_id, prev_hash, hash,
         lag(hash) OVER (ORDER BY audit_id) AS exp_prev,
         sha256(coalesce(lag(hash) OVER (ORDER BY audit_id), ''::bytea) ||
                convert_to(jsonb_build_array(
                  to_char(at AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'),
                  tenant_id, actor, action, target, detail)::text, 'UTF8')) AS recomputed
  FROM audit_log) s
WHERE prev_hash IS DISTINCT FROM exp_prev OR hash IS DISTINCT FROM recomputed
ORDER BY audit_id
"""
APPEND = "SELECT audit_append(%s, %s, %s, %s, %s, %s::jsonb)"


def bad_rows(d: str, tz: str = "UTC") -> list[int]:
    """Verify routine: recompute every hash from the stored rows; returns ids that disagree."""
    with psycopg.connect(d) as c:
        c.execute(f"SET TimeZone = '{tz}'")  # type: ignore[arg-type]
        return [r[0] for r in c.execute(VERIFY).fetchall()]  # type: ignore[call-overload]


def append(c: psycopg.Connection, n: int, actor: str = "a", action: str = "x") -> int:
    # audit_append (0004) accepts only the session's own tenant: the session says "t1" first
    c.execute("SELECT set_config('app.tenant_id', 't1', true)")
    row = c.execute(
        APPEND, (T0 + timedelta(seconds=n), "t1", actor, action, f"tg{n}", "{}")
    ).fetchone()
    assert row is not None
    return int(row[0])


def owner_tamper(d: str, sql: str) -> None:
    with psycopg.connect(d, autocommit=True) as c:
        c.execute("ALTER TABLE audit_log DISABLE TRIGGER audit_log_no_change")
        c.execute(sql)  # type: ignore[arg-type]
        c.execute("ALTER TABLE audit_log ENABLE TRIGGER audit_log_no_change")


# (1) migration probes -----------------------------------------------------------------------


def test_status_check_accepts_every_status_any_code_writes(dsn: str) -> None:
    """Literals found by grep: INDEX_STATUS values, 'building' (0001 default and upsert) and the
    'failed' value the check lists. The constraint is validated (not NOT VALID)."""
    with psycopg.connect(dsn) as c:
        for i, st in enumerate({*INDEX_STATUS.values(), "building", "failed"}):
            c.execute(
                "INSERT INTO index_versions (index_version, embedding_model, chunker_version, "
                "status) VALUES (%s, 'e', 'c', %s)",
                (f"v{i}", st),
            )
        assert c.execute(
            "SELECT convalidated FROM pg_constraint WHERE conname='index_versions_status_check'"
        ).fetchone() == (True,)
    with pytest.raises(psycopg.errors.CheckViolation):
        with psycopg.connect(dsn) as c:
            c.execute(
                "INSERT INTO index_versions (index_version, embedding_model, chunker_version, "
                "status) VALUES ('x', 'e', 'c', 'ready')"
            )


def test_upgrade_with_existing_rows_keeps_them_and_stays_atomic_on_legacy_status(
    dsn: str,
) -> None:
    command.downgrade(_cfg(), "0002")
    with psycopg.connect(dsn) as c:
        c.execute(
            "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
            "VALUES ('old', 'e', 'c')"
        )
    command.upgrade(_cfg(), "head")  # existing 'building' row validates and gains NULL policy cols
    assert _index_row(dsn, "old") == ("building", None, None)
    command.downgrade(_cfg(), "0002")
    with psycopg.connect(dsn) as c:
        c.execute("UPDATE index_versions SET status = 'legacy'")
    with pytest.raises(Exception):  # noqa: B017,PT011 - CheckViolation wrapped by alembic/psycopg
        command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn) as c:  # transactional DDL: still exactly at 0002, nothing leaked
        assert c.execute("SELECT version_num FROM alembic_version").fetchone() == ("0002",)
        assert c.execute("SELECT to_regclass('pipeline_policies')").fetchone() == (None,)
        assert c.execute("SELECT to_regproc('audit_append')").fetchone() == (None,)
        c.execute("UPDATE index_versions SET status = 'building'")
    command.upgrade(_cfg(), "head")


def test_downgrade_with_data_keeps_audit_rows_and_drops_policies(dsn: str) -> None:
    """Pin: downgrade is destructive for policies and documents (tables dropped, no archive; the
    ADR only says 'drops in reverse order') but never touches audit_log rows or the chain."""
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1")
    asyncio.run(repo.create(p, audit_event=_event(p)))
    with _app(dsn) as c:
        append(c, 1)
    before = _audit_rows(dsn)
    assert len(before) == 1
    command.downgrade(_cfg(), "0002")
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM audit_log").fetchone() == (2,)
        assert c.execute("SELECT status FROM index_versions").fetchone() == ("building",)
    assert bad_rows(dsn) == []
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM pipeline_policies").fetchone() == (0,)
    with _app(dsn) as c:  # the chain continues across down/up
        append(c, 2)
    assert bad_rows(dsn) == []


# (2) audit_append ---------------------------------------------------------------------------


def test_audit_chain_genesis_and_links(dsn: str) -> None:
    with _app(dsn) as c:
        ids = [append(c, n) for n in range(5)]
    assert ids == sorted(ids)
    with psycopg.connect(dsn) as c:
        rows = c.execute(
            "SELECT prev_hash, hash, sha256(convert_to(jsonb_build_array(to_char(at AT TIME ZONE "
            "'UTC', 'YYYY-MM-DD\"T\"HH24:MI:SS.US\"Z\"'), tenant_id, actor, action, target, "
            "detail)::text, 'UTF8')) FROM audit_log ORDER BY audit_id"
        ).fetchall()
    assert rows[0][0] is None and rows[0][1] == rows[0][2]  # genesis: no prev, hash of own data
    assert all(rows[i][0] == rows[i - 1][1] for i in range(1, 5))
    assert len({r[1] for r in rows}) == 5
    assert bad_rows(dsn) == []


def test_audit_concurrent_appends_stay_one_linear_chain(dsn: str) -> None:
    def work(n: int) -> int:
        with _app(dsn) as c:
            with c.transaction():
                i = append(c, n)
                c.execute("SELECT pg_sleep(0.01)")  # hold the lock so contenders queue up
            return i

    with ThreadPoolExecutor(20) as ex:
        ids = list(ex.map(work, range(20)))
    assert len(set(ids)) == 20
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*), count(DISTINCT hash) FROM audit_log").fetchone() == (
            20,
            20,
        )
        assert c.execute("SELECT count(*) FROM audit_log WHERE prev_hash IS NULL").fetchone() == (
            1,
        )
        assert c.execute(  # no fork: every hash is the prev_hash of at most one row
            "SELECT count(*) FROM (SELECT prev_hash FROM audit_log GROUP BY prev_hash "
            "HAVING count(*) > 1) x"
        ).fetchone() == (0,)
    assert bad_rows(dsn) == []


@pytest.mark.parametrize("who", ["rag_app", "owner"])
def test_audit_log_stays_append_only(dsn: str, who: str) -> None:
    with _app(dsn) as c:
        append(c, 1)
    for stmt in ("UPDATE audit_log SET actor='evil'", "DELETE FROM audit_log"):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with _app(dsn) if who == "rag_app" else psycopg.connect(dsn) as c:
                c.execute(stmt)  # type: ignore[call-overload]


def test_audit_tamper_and_deletion_are_detected(dsn: str) -> None:
    with _app(dsn) as c:
        ids = [append(c, n) for n in range(4)]
    owner_tamper(dsn, f"UPDATE audit_log SET detail='{{\"x\": 1}}' WHERE audit_id={ids[1]}")
    assert bad_rows(dsn) == [ids[1]]
    owner_tamper(dsn, f"DELETE FROM audit_log WHERE audit_id={ids[2]}")
    assert ids[3] in bad_rows(dsn)  # the row after the gap no longer links


def test_direct_insert_by_rag_app_is_denied(dsn: str) -> None:
    """0004 revoked INSERT on audit_log: the chain can only be extended through audit_append."""
    with _app(dsn) as c:
        append(c, 1)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with _app(dsn) as c:
            c.execute(
                "INSERT INTO audit_log (actor, action, prev_hash, hash) "
                "VALUES ('a','b','\\x00','\\x01')"
            )
    assert bad_rows(dsn) == []


def test_audit_function_privileges(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        row = c.execute(
            "SELECT prosecdef, proconfig FROM pg_proc WHERE proname='audit_append'"
        ).fetchone()
        assert row is not None
        definer, config = row
        # invoker rights are fine; a definer would need a pinned search_path
        assert not definer or any("search_path" in x for x in config or [])
        c.execute("CREATE ROLE rag_other NOLOGIN")
        c.execute("GRANT USAGE ON SCHEMA public TO rag_other")
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # no audit_log grant -> no append
        with psycopg.connect(dsn) as c:
            c.execute("SET ROLE rag_other")
            append(c, 1)
    with psycopg.connect(dsn) as c:
        c.execute("DROP OWNED BY rag_other")
        c.execute("DROP ROLE rag_other")


def test_audit_append_execute_not_public(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        row = c.execute(
            "SELECT has_function_privilege('public', 'audit_append(timestamptz,text,text,text,"
            "text,jsonb)', 'EXECUTE')"
        ).fetchone()
    assert row == (False,)


def test_audit_chain_verifies_in_any_session_timezone(dsn: str) -> None:
    with _app(dsn) as c:
        c.execute("SET TimeZone = 'Asia/Tokyo'")
        append(c, 1)
        append(c, 2)
    assert bad_rows(dsn, "UTC") == []


def test_audit_tamper_moving_field_boundary_is_detected(dsn: str) -> None:
    with _app(dsn) as c:
        i = append(c, 1, actor="a|b", action="c")
    owner_tamper(dsn, f"UPDATE audit_log SET actor='a', action='b|c' WHERE audit_id={i}")
    assert bad_rows(dsn) == [i]


def test_audit_rows_are_tenant_isolated(dsn: str) -> None:
    """0004: audit_log has RLS, so a tenant session sees only its own audit rows."""
    with _app(dsn, "tenant-b") as c:
        c.execute(APPEND, (T0, "tenant-b", "u", "login", None, "{}"))
    with _app(dsn, "tenant-a") as c:
        assert c.execute(
            "SELECT count(*) FROM audit_log WHERE tenant_id = 'tenant-b'"
        ).fetchone() == (0,)


# (3) declared-column trigger ----------------------------------------------------------------

HASHED = {
    "policy_id": f"'{uuid.uuid4()}'::uuid",
    "policy_hash": "repeat('f', 64)",
    "ingest_run_id": "'other'",
    "index_version": "'v2'",
    "parent_policy_id": "NULL",  # replaced below with a real id
    "change_kind": "'chunker'",
    "change_reason": "'edited'",
    "embedder_id": "'other'",
    "embedder_version": "'9'",
    "embedder_dim": "9",
    "chunker_version": "'other'",
    "chunk_tokens": "999",
    "chunk_overlap": "1",
    "parent_tokens": "5",  # NULL -> value must count as a change
    "redaction_rules_version": "'other'",
    "reranker_id": "'rr'",  # NULL -> value
    "prompt_version": "'other'",
    "generator_chain_ref": "'other'",
    "doc_tenants": "'{zz}'",
    "declared_by": "'other'",
    "declared_at": "now()",
}


@pytest.mark.parametrize("col", sorted(HASHED))
@pytest.mark.parametrize("who", ["rag_app", "owner"])
def test_every_hashed_column_immutable(dsn: str, col: str, who: str) -> None:
    with psycopg.connect(dsn) as c:
        for v in ("v1", "v2"):
            c.execute(
                "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
                "VALUES (%s, 'e', 'c')",
                (v,),
            )
        pid = uuid.uuid4()
        c.execute(INSERT_MIN, (str(pid), "a" * 64, "embedder", "x"))
    val = HASHED[col]
    if col == "parent_policy_id":
        val = f"'{pid}'::uuid"
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with _app(dsn) if who == "rag_app" else psycopg.connect(dsn) as c:
            c.execute(f"UPDATE pipeline_policies SET {col} = {val}")  # type: ignore[arg-type]


def test_noop_update_and_lifecycle_update_allowed_for_owner(v1: str) -> None:
    with _app(v1) as c:
        _insert(c)
    with psycopg.connect(v1) as c:
        c.execute("UPDATE pipeline_policies SET change_reason = change_reason")  # unchanged
        c.execute("UPDATE pipeline_policies SET status='rejected', rejected_at=now()")


@pytest.fixture
def v1(dsn: str) -> str:
    with psycopg.connect(dsn) as c:
        c.execute(
            "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
            "VALUES ('v1', 'e', 'c')"
        )
    return dsn


# (4) policy_documents RLS and exposure through pipeline_policies ------------------------------


def test_policy_documents_rls_write_paths_and_owner(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1")
    asyncio.run(repo.create(p))
    with _app(dsn, "a") as c:
        rows = c.execute("SELECT tenant_id FROM policy_documents").fetchall()
        assert rows == [("a",)]
    for stmt in (
        "UPDATE policy_documents SET tenant_id='a'",
        "DELETE FROM policy_documents",
    ):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with _app(dsn, "a") as c:
                c.execute(stmt)  # type: ignore[call-overload]
    with _app(dsn, "") as c:  # empty tenant string matches no real tenant
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (0,)
    with psycopg.connect(dsn) as c:  # pin: owner bypasses RLS (no FORCE ROW LEVEL SECURITY)
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (2,)


def test_pipeline_policies_row_and_doc_tenants_visible_across_tenants(dsn: str) -> None:
    """Pin (information exposure): pipeline_policies has no RLS, so a session of tenant 'a'
    reads the policy row incl. change_reason, declared_by and doc_tenants = ['a','b'] and so
    learns that tenant 'b' has documents in the run. Document rows themselves stay isolated."""
    p = make_policy("v1", reason="re-ingest for acme-corp contract")
    asyncio.run(SqlPolicyRepository(dsn).create(p))
    with _app(dsn, "a") as c:
        row = c.execute("SELECT doc_tenants, change_reason FROM pipeline_policies").fetchone()
        assert row == (["a", "b"], "re-ingest for acme-corp contract")
        assert c.execute(
            "SELECT count(*) FROM policy_documents WHERE tenant_id = 'b'"
        ).fetchone() == (0,)


def test_create_three_tenants_roundtrip_and_get_sees_all(dsn: str) -> None:
    docs = tuple(
        PolicyDocument(tenant_id=t, doc_id=f"d{n}", doc_version=1, doc_hash=f"h{t}{n}")
        for t in ("c", "a", "b")
        for n in range(2)
    )
    p = make_policy("v1", docs=docs)
    repo = SqlPolicyRepository(dsn)
    asyncio.run(repo.create(p))
    got = asyncio.run(repo.get(p.policy_id))
    assert got is not None and len(got.documents) == 6
    assert [d.tenant_id for d in got.documents] == ["a", "a", "b", "b", "c", "c"]


def test_create_with_no_documents_and_empty_tenant_id(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    empty = make_policy("v1", docs=())
    asyncio.run(repo.create(empty))
    got = asyncio.run(repo.get(empty.policy_id))
    assert got is not None and got.documents == ()


# (5) repository behaviour -------------------------------------------------------------------


def _force(d: str, pid: str, status: S) -> None:
    with psycopg.connect(d) as c:
        c.execute(
            "UPDATE pipeline_policies SET status=%s WHERE policy_id=%s::uuid", (status.value, pid)
        )
        c.execute(
            "UPDATE index_versions SET status=%s WHERE policy_id=%s::uuid",
            (INDEX_STATUS[status], pid),
        )


def _counts(d: str) -> tuple[Any, ...]:
    with psycopg.connect(d) as c:
        r = c.execute(
            "SELECT (SELECT count(*) FROM audit_log), (SELECT string_agg(status, ',' ORDER BY "
            "policy_id) FROM pipeline_policies), (SELECT string_agg(status, ',' ORDER BY "
            "index_version) FROM index_versions)"
        ).fetchone()
    assert r is not None
    return r


async def test_every_illegal_transition_is_refused_and_changes_nothing(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    for i, cur in enumerate(S):
        p = make_policy(f"v{i}")
        await repo.create(p)
        _force(dsn, p.policy_id, cur)
        before = _counts(dsn)
        for new in S:
            if new in ALLOWED[cur]:
                continue
            with pytest.raises(IllegalTransition):
                await repo.transition(
                    p.policy_id,
                    expected_status=cur,
                    expected_index_status=INDEX_STATUS[cur],
                    new_status=new,
                    at=T0,
                    audit_event=_event(p, "policy.bogus"),
                )
        assert _counts(dsn) == before
    with pytest.raises(NotFound):
        await repo.transition(
            str(uuid.uuid4()),
            expected_status=S.COMMITTED,
            expected_index_status="committed",
            new_status=S.DECLARED,
            at=T0,
        )


async def test_ten_contenders_exactly_one_winner_and_one_audit_row(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1")
    await repo.create(p)

    async def go(n: int) -> Any:
        return await repo.transition(
            p.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.VERIFIED,
            at=T0 + timedelta(seconds=n),
            audit_event=_event(p, f"policy.verify.{n}"),
        )

    res = await asyncio.gather(*(go(n) for n in range(10)), return_exceptions=True)
    assert sum(not isinstance(r, Exception) for r in res) == 1
    assert all(isinstance(r, IllegalTransition) for r in res if isinstance(r, Exception))
    with psycopg.connect(dsn) as c:
        assert c.execute(
            "SELECT count(*) FROM audit_log WHERE action LIKE 'policy.verify.%'"
        ).fetchone() == (1,)
    assert bad_rows(dsn) == []


async def test_redeclare_rules_sql_vs_fake(dsn: str) -> None:
    """Deliberate difference: SQL only reopens an index version that is building or rejected
    (Conflict otherwise, nothing left behind); the fake accepts any redeclare and overwrites
    the index status."""
    repo = SqlPolicyRepository(dsn)
    for i, st in enumerate(("building", "rejected")):
        first = make_policy(f"r{i}")
        await repo.create(first)
        _force(dsn, first.policy_id, S.REJECTED if st == "rejected" else S.DECLARED)
        await repo.create(make_policy(f"r{i}", reason="second"))
        assert _index_row(dsn, f"r{i}")[0] == "building"
    for i, st in enumerate(("staged", "committed", "rolled_back")):
        first = make_policy(f"c{i}")
        await repo.create(first)
        _force(dsn, first.policy_id, S.INDEXED if st == "staged" else S(st))
        before = _counts(dsn)
        with pytest.raises(Conflict):
            await repo.create(make_policy(f"c{i}", reason="again"), audit_event=_event(first))
        assert _counts(dsn) == before
    with psycopg.connect(dsn) as c:
        c.execute("UPDATE index_versions SET status='failed' WHERE index_version='r0'")
    with pytest.raises(Conflict):
        await repo.create(make_policy("r0", reason="third"))
    fake = FakePolicyRepository()
    first = make_policy("f")
    await fake.create(first)
    await fake.transition(
        first.policy_id, expected_status=S.DECLARED, expected_index_status="building",
        new_status=S.VERIFIED, at=T0,
    )  # fmt: skip
    await fake.create(make_policy("f", reason="again"))  # fake: no redeclare rule
    assert fake.index_status["f"] == "building"


async def test_concurrent_create_same_hash_one_conflict(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    a = make_policy("v1")
    b = a.model_copy(update={"policy_id": str(uuid.uuid4())})
    res = await asyncio.gather(repo.create(a), repo.create(b), return_exceptions=True)
    assert sorted(type(r).__name__ for r in res) == ["Conflict", "NoneType"]
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM pipeline_policies").fetchone() == (1,)


async def test_unknown_parent_policy_fails_and_leaves_nothing(dsn: str) -> None:
    """A dangling parent_policy_id is a clean domain error (ValidationFailed; the fake accepts
    it); the whole create (index row, documents, audit) rolls back."""
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1", parent=str(uuid.uuid4()))
    with pytest.raises(ValidationFailed):
        await repo.create(p, audit_event=_event(p))
    assert _counts(dsn) == (0, None, None)


async def test_malformed_policy_id_is_not_found(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    assert await repo.get("not-a-uuid") is None
    with pytest.raises(NotFound):
        await repo.transition(
            "not-a-uuid", expected_status=S.DECLARED, expected_index_status="building",
            new_status=S.VERIFIED, at=T0,
        )  # fmt: skip


async def test_list_unsynced_order_and_limit_matches_fake(dsn: str) -> None:
    async def run(repo: PolicyRepository) -> list[str]:
        ps = [make_policy(f"v{i}", declared_at=T0 + timedelta(hours=i)) for i in range(5)]
        for p in ps:
            await repo.create(p)
        # commit v3 first, then v1; v0, v2, v4 stay uncommitted (NULLS LAST by declared_at)
        await _walk(repo, ps[3], *HAPPY, base=10)
        await _walk(repo, ps[1], *HAPPY, base=30)
        out = await repo.list_unsynced(10)
        assert [p.index_version for p in out] == ["v3", "v1", "v0", "v2", "v4"]
        assert [p.index_version for p in await repo.list_unsynced(2)] == ["v3", "v1"]
        assert await repo.list_unsynced(0) == []
        await repo.mark_synced(ps[3].policy_id, "9")
        await repo.mark_synced(ps[3].policy_id, "9")  # idempotent
        await repo.record_mlflow_version(ps[1].policy_id, "5")
        await repo.record_mlflow_version(ps[1].policy_id, "5")
        got = await repo.get(ps[1].policy_id)
        assert got is not None and got.mlflow_model_version == "5" and not got.mlflow_synced
        return [p.index_version for p in await repo.list_unsynced(10)]

    assert await run(SqlPolicyRepository(dsn)) == await run(FakePolicyRepository())


async def _lineage(repo: PolicyRepository, mode: str) -> list[str | None]:
    ps = []
    parent = None
    for i in range(3):
        p = make_policy(f"v{i + 1}", parent=parent, declared_at=T0 + timedelta(minutes=i))
        await repo.create(p)
        ps.append(p)
        parent = p.policy_id
    names = {p.policy_id: p.index_version for p in ps}

    async def prev(i: int) -> str | None:
        cur = await repo.get(ps[i].policy_id)
        assert cur is not None
        got = await repo.get_previous_committed(cur)
        return names[got.policy_id] if got else None

    if mode == "gap":  # the middle policy was rejected, never committed
        await repo.transition(
            ps[1].policy_id, expected_status=S.DECLARED, expected_index_status="building",
            new_status=S.REJECTED, at=T0,
        )  # fmt: skip
        await _walk(repo, ps[0], *HAPPY, base=0)
        await _walk(repo, ps[2], *HAPPY, base=10)
        return [await prev(2), await prev(0)]
    await _walk(repo, ps[0], *HAPPY, base=0)
    await _walk(repo, ps[1], *HAPPY, base=10)
    await _walk(repo, ps[2], *HAPPY, base=20)
    out = [await prev(2), await prev(1), await prev(0)]
    if mode == "rolled_back_parent":
        await repo.transition(
            ps[1].policy_id, expected_status=S.COMMITTED, expected_index_status="committed",
            new_status=S.ROLLED_BACK, at=T0 + timedelta(days=1),
        )  # fmt: skip
        out.append(await prev(2))
    return out


@pytest.mark.parametrize("mode", ["plain", "gap", "rolled_back_parent"])
async def test_get_previous_committed_lineage(dsn: str, mode: str) -> None:
    sql_out = await _lineage(SqlPolicyRepository(dsn), mode)
    assert sql_out == await _lineage(FakePolicyRepository(), mode)
    expected = {
        "plain": ["v2", "v1", None],
        "gap": ["v1", None],
        "rolled_back_parent": ["v2", "v1", None, "v1"],
    }[mode]
    assert sql_out == expected


async def test_previous_ignores_ancestors_committed_later(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    a = make_policy("v1", declared_at=T0)
    b = make_policy("v2", parent=a.policy_id, declared_at=T0 + timedelta(minutes=1))
    await repo.create(a)
    await repo.create(b)
    await _walk(repo, b, *HAPPY, base=0)  # child commits first
    await _walk(repo, a, *HAPPY, base=50)
    cur = await repo.get(b.policy_id)
    assert cur is not None and await repo.get_previous_committed(cur) is None


# (6) injection, NUL, size, connections --------------------------------------------------------

NASTY = [
    "x'; DROP TABLE pipeline_policies; --",
    'quote " and \\ backslash and $$ dollar $$',
    "semi;colon /* c */ -- c",
    "unicode ‮​ \U0001f600 café",
    "%s %(name)s {0} {x}",
]


@pytest.mark.parametrize("reason", NASTY)
async def test_change_reason_is_data_not_sql(dsn: str, reason: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1", reason=reason)
    await repo.create(p, audit_event=_event(p))
    got = await repo.get(p.policy_id)
    assert got is not None and got.change_reason == p.change_reason
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT to_regclass('pipeline_policies')").fetchone() is not None
        assert c.execute("SELECT count(*) FROM pipeline_policies").fetchone() == (1,)


async def test_hostile_ids_and_hash_are_data(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    evil = "v1'; DROP TABLE audit_log; --"
    p = make_policy(evil)
    await repo.create(p)
    got = await repo.get_for_version(evil)
    assert got is not None and got.index_version == evil
    bad = p.model_copy(update={"policy_id": str(uuid.uuid4()), "policy_hash": "'; DROP TABLE x;--"})
    with pytest.raises(psycopg.errors.CheckViolation):  # CHECK regex refuses non-hex hashes
        await repo.create(bad)
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT to_regclass('audit_log')").fetchone() is not None
    await repo.mark_synced(p.policy_id, "1'; DROP TABLE audit_log;--")
    await repo.record_mlflow_version(p.policy_id, "\u0000"[:0] + "v;--")
    got = await repo.get(p.policy_id)
    assert got is not None and got.mlflow_model_version == "v;--"


async def test_nul_byte_is_refused_and_leaves_nothing(dsn: str) -> None:
    """NUL cannot be stored in text; the repository maps it to ValidationFailed and the whole
    create rolls back."""
    repo = SqlPolicyRepository(dsn)
    p = make_policy("v1", reason="bad\x00reason")
    with pytest.raises(ValidationFailed):
        await repo.create(p, audit_event=_event(p))
    assert _counts(dsn) == (0, None, None)


async def _open_conns(d: str) -> int:
    with psycopg.connect(d) as c:
        row = c.execute(
            "SELECT count(*) FROM pg_stat_activity WHERE datname = current_database() "
            "AND pid <> pg_backend_pid()"
        ).fetchone()
    assert row is not None
    return int(row[0])


async def test_no_connection_leak_after_200_failing_calls(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    broken_role = SqlPolicyRepository(dsn, role="no_such_role")
    p = make_policy("v1")
    await repo.create(p)
    before = await _open_conns(dsn)
    for _ in range(50):
        with pytest.raises(NotFound):
            await repo.transition(
                str(uuid.uuid4()), expected_status=S.DECLARED, expected_index_status="building",
                new_status=S.VERIFIED, at=T0,
            )  # fmt: skip
        with pytest.raises(IllegalTransition):
            await repo.transition(
                p.policy_id, expected_status=S.DECLARED, expected_index_status="staged",
                new_status=S.VERIFIED, at=T0,
            )  # fmt: skip
        with pytest.raises(Conflict):
            await repo.create(p.model_copy(update={"policy_id": str(uuid.uuid4())}))
        with pytest.raises(psycopg.Error):
            await broken_role.get(p.policy_id)
    await asyncio.sleep(0.3)
    assert await _open_conns(dsn) <= before


async def test_five_thousand_documents_roundtrip_is_fast_enough(dsn: str) -> None:
    docs = tuple(
        PolicyDocument(tenant_id=f"t{n % 5}", doc_id=f"d{n}", doc_version=1, doc_hash=f"h{n}")
        for n in range(5000)
    )
    p = make_policy("v1", docs=docs)
    repo = SqlPolicyRepository(dsn)
    t0 = time.monotonic()
    await repo.create(p)
    got = await repo.get(p.policy_id)
    elapsed = time.monotonic() - t0
    assert got is not None and len(got.documents) == 5000
    assert elapsed < 30, f"5000 docs took {elapsed:.1f}s"


# (7) SqlEvalRuns -----------------------------------------------------------------------------


def _gate_run(d: str, rid: str, passed: bool, metrics: str) -> None:
    with _app(d) as c:
        c.execute(
            "INSERT INTO gate_runs (gate_run_id, kind, passed, metrics) "
            "VALUES (%s,'eval',%s,%s::jsonb)",
            (rid, passed, metrics),
        )


@pytest.mark.parametrize(
    ("metrics", "ok"),
    [
        ('{"r": 0.9, "n": 3}', True),
        ('{"r": null}', False),
        ('{"r": "0.9"}', False),
        ('{"r": "NaN"}', False),
        ('{"r": true}', False),
        ('{"r": [1]}', False),
        ('{"r": {"x": 1}}', False),
        ("null", False),
        ('"str"', False),
        ("[]", False),
        ("1", False),
    ],
)
async def test_eval_reader_metric_payloads(dsn: str, metrics: str, ok: bool) -> None:
    _gate_run(dsn, "r", True, metrics)
    assert await SqlEvalRuns(dsn).passed("r") is ok


async def test_eval_reader_passed_flag_missing_run_and_nan_literal(dsn: str) -> None:
    _gate_run(dsn, "no", False, '{"r": 1}')
    reader = SqlEvalRuns(dsn)
    assert await reader.passed("no") is False
    assert await reader.passed("") is False
    assert await reader.passed("x'; DROP TABLE gate_runs;--") is False
    with pytest.raises(psycopg.errors.InvalidTextRepresentation):  # jsonb cannot hold NaN
        _gate_run(dsn, "nan", True, '{"r": NaN}')


async def test_eval_reader_empty_metrics_fail_closed(dsn: str) -> None:
    """A passed run with no metrics at all does not count as passed (gate_runs.metrics
    defaults to '{}')."""
    _gate_run(dsn, "empty", True, "{}")
    assert await SqlEvalRuns(dsn).passed("empty") is False


async def test_eval_reader_huge_integer_metric_fails_closed(dsn: str) -> None:
    for rid, payload in (
        ("big", '{"r": 1' + "0" * 400 + "}"),
        ("exp", '{"r": 1e999}'),  # jsonb normalises to a 1000-digit integer
        ("negexp", '{"r": -1e999}'),
    ):
        _gate_run(dsn, rid, True, payload)
        assert await SqlEvalRuns(dsn).passed(rid) is False
    assert metrics_clean({"r": 10**400}) is False


# (8) parity, three more scenarios through the gate -----------------------------------------


def _gate(
    repo: PolicyRepository, mirror: FakeRegistryMirror, audit: FakeAuditLog
) -> tuple[PolicyGate, FakeClock]:
    clock = FakeClock(T0)
    return PolicyGate(repo, audit, clock, mirror, evals=FakeEvalRuns({"ev"})), clock


def _draft(version: str, parent: str | None, reason: str = "refresh") -> PolicyDraft:
    p = make_policy(version, parent=parent, reason=reason)
    return PolicyDraft(**{k: getattr(p, k) for k in PolicyDraft.model_fields})


OK_SUMMARY = ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=8, documents_with_rows=ALL_DOCS)


async def _snapshot(repo: PolicyRepository, versions: list[str]) -> list[Any]:
    out = []
    for v in versions:
        p = await repo.get_for_version(v)
        assert p is not None
        out.append(
            (v, p.status.value, p.mlflow_synced, p.mlflow_model_version, p.rejected_at is not None)
        )
    return out


async def _rollback_lineage(repo: PolicyRepository, audit: FakeAuditLog) -> list[Any]:
    mirror = FakeRegistryMirror()
    gate, clock = _gate(repo, mirror, audit)
    ids: list[str] = []
    for i in range(3):
        d = await gate.declare(_draft(f"v{i + 1}", ids[-1] if ids else None), declared_by="op")
        ids.append(d.policy_id)
        await gate.verify(d.policy_id, OK_SUMMARY)
        await gate.mark_indexed(d.policy_id)
        clock.advance(60)
        await gate.mark_committed(d.policy_id)
    await gate.mark_rolled_back(ids[2])
    champion_after_first = dict(mirror.aliases)
    await gate.mark_rolled_back(ids[1])
    snap = await _snapshot(repo, ["v1", "v2", "v3"])
    return [snap, champion_after_first, dict(mirror.aliases)]


async def _failed_verify(repo: PolicyRepository, audit: FakeAuditLog) -> list[Any]:
    mirror = FakeRegistryMirror()
    gate, clock = _gate(repo, mirror, audit)
    d = await gate.declare(_draft("v1", None), declared_by="op")
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(
            d.policy_id,
            ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=99, documents_with_rows=ALL_DOCS),
        )
    out: list[Any] = [await _snapshot(repo, ["v1"])]
    with pytest.raises(IllegalTransition):  # rejected is terminal
        await gate.verify(d.policy_id, OK_SUMMARY)
    with pytest.raises(Conflict):  # same content again: unique (index_version, hash)
        await gate.declare(_draft("v1", None), declared_by="op")
    clock.advance(1)  # equal declared_at would make get_for_version order arbitrary
    d2 = await gate.declare(_draft("v1", None, reason="fixed dim"), declared_by="op")
    await gate.verify(d2.policy_id, OK_SUMMARY)
    out.append(await _snapshot(repo, ["v1"]))
    return out


async def _outbox_retry(repo: PolicyRepository, audit: FakeAuditLog) -> list[Any]:
    mirror = FakeRegistryMirror(down=True)
    gate, clock = _gate(repo, mirror, audit)
    ids = []
    for i in range(2):
        d = await gate.declare(_draft(f"v{i + 1}", None), declared_by="op")
        ids.append(d.policy_id)
        await gate.verify(d.policy_id, OK_SUMMARY)
        await gate.mark_indexed(d.policy_id)
        clock.advance(60)
        await gate.mark_committed(d.policy_id)
    out: list[Any] = [await sync_pending(repo, mirror), len(await repo.list_unsynced(10))]
    mirror.down = False
    out.append(await sync_pending(repo, mirror))
    out.append(await sync_pending(repo, mirror))  # idempotent: nothing left
    out.append(await _snapshot(repo, ["v1", "v2"]))
    out.append(dict(mirror.aliases))
    return out


@pytest.mark.parametrize("scenario", [_rollback_lineage, _failed_verify, _outbox_retry])
async def test_parity_gate_scenarios(dsn: str, scenario: Any) -> None:
    fake_audit = FakeAuditLog()
    fake_out = await scenario(FakePolicyRepository(), fake_audit)
    sql_out = await scenario(SqlPolicyRepository(dsn), FakeAuditLog())
    assert sql_out == fake_out
    with psycopg.connect(dsn) as c:
        sql_actions = [r[0] for r in c.execute("SELECT action FROM audit_log ORDER BY audit_id")]
    assert sql_actions == [e.action for e in fake_audit._events]
    if scenario is _rollback_lineage:  # champion walked back v3 -> v2 -> v1
        assert sql_out[1]["champion"] != sql_out[2]["champion"]
    assert bad_rows(dsn) == []
