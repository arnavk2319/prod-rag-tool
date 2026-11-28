# ruff: noqa: SIM117
"""Migration 0003 and the SQL policy repository (ADR-0006).

Skipped unless TEST_DATABASE_URL points at a reachable throwaway Postgres with pgvector >= 0.8.
"""

import asyncio
import os
import uuid
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import psycopg
import pytest
from alembic import command
from alembic.config import Config

from rag_platform.adapters.policy_fakes import FakePolicyRepository
from rag_platform.adapters.sql_eval_runs import SqlEvalRuns, metrics_clean
from rag_platform.adapters.sql_policy_repository import SqlPolicyRepository
from rag_platform.domain.errors import Conflict, NotFound
from rag_platform.domain.policy import (
    ChangeKind,
    IllegalTransition,
    PipelinePolicy,
    PolicyDocument,
    PolicyDraft,
    PolicyRepository,
    PolicyStatus,
    documents_sorted,
    policy_hash,
)
from rag_platform.domain.ports import AuditEvent

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")
T0 = datetime(2026, 1, 1, tzinfo=UTC)
S = PolicyStatus


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


def _app(d: str, tenant: str | None = None) -> psycopg.Connection:
    c = psycopg.connect(d)
    c.execute("SET ROLE rag_app")
    if tenant is not None:
        c.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    return c


def make_policy(
    version: str = "v1",
    *,
    kind: ChangeKind = ChangeKind.CORPUS_ONLY,
    reason: str = "monthly refresh",
    parent: str | None = None,
    docs: tuple[PolicyDocument, ...] | None = None,
    declared_at: datetime = T0,
) -> PipelinePolicy:
    if docs is None:
        docs = (
            PolicyDocument(tenant_id="a", doc_id="d1", doc_version=1, doc_hash="h1"),
            PolicyDocument(tenant_id="b", doc_id="d2", doc_version=1, doc_hash="h2"),
        )
    draft = PolicyDraft(
        change_kind=kind,
        change_reason=reason,
        parent_policy_id=parent,
        ingest_run_id="run-1",
        index_version=version,
        embedder_id="emb-1",
        embedder_version="1",
        embedder_dim=8,
        chunker_version="c1",
        chunk_tokens=200,
        chunk_overlap=20,
        parent_tokens=800,
        redaction_rules_version="r1",
        prompt_version="p1",
        generator_chain_ref="chain-1",
        documents=docs,
    )
    return PipelinePolicy(
        **draft.model_dump(exclude={"documents"}),
        documents=docs,
        policy_id=str(uuid.uuid4()),
        policy_hash=policy_hash(draft),
        declared_by="op",
        declared_at=declared_at,
    )


def _event(p: PipelinePolicy, action: str = "policy.declare") -> AuditEvent:
    return AuditEvent(
        at=T0, actor="system", action=action, tenant_id="", detail={"policy_id": p.policy_id}
    )


def _norm(p: PipelinePolicy | None) -> dict[str, Any] | None:
    if p is None:
        return None
    d = p.model_dump(exclude={"documents"})
    d["documents"] = [x.model_dump() for x in documents_sorted(p.documents)]
    return d


def _ids(p: PipelinePolicy) -> dict[str, Any]:
    d = _norm(p)
    assert d is not None
    return d


def _index_row(d: str, version: str) -> tuple[Any, ...]:
    with psycopg.connect(d) as c:
        row = c.execute(
            "SELECT status, policy_id::text, policy_hash FROM index_versions "
            "WHERE index_version=%s",
            (version,),
        ).fetchone()
    assert row is not None
    return row


def _audit_rows(d: str) -> list[tuple[Any, ...]]:
    with psycopg.connect(d) as c:
        return c.execute(
            "SELECT action, prev_hash, hash FROM audit_log WHERE action LIKE 'policy.%' "
            "ORDER BY audit_id"
        ).fetchall()


async def _walk(
    repo: PolicyRepository, p: PipelinePolicy, *steps: tuple[S, S, str], base: int = 0
) -> None:
    for n, (expected, new, idx) in enumerate(steps, start=1):
        await repo.transition(
            p.policy_id,
            expected_status=expected,
            expected_index_status=idx,
            new_status=new,
            at=T0 + timedelta(minutes=base + n),
            eval_run_id="run-ok" if new is S.COMMITTED else None,
        )


HAPPY = (
    (S.DECLARED, S.VERIFIED, "building"),
    (S.VERIFIED, S.INDEXED, "building"),
    (S.INDEXED, S.COMMITTED, "staged"),
)


# (1) migration ------------------------------------------------------------------------------


def test_upgrade_downgrade_upgrade(dsn: str) -> None:
    def tables() -> set[str]:
        with psycopg.connect(dsn) as c:
            return {
                r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'")
            }

    assert {"pipeline_policies", "policy_documents"} <= tables()
    command.downgrade(_cfg(), "0002")
    assert not {"pipeline_policies", "policy_documents"} & tables()
    with psycopg.connect(dsn) as c:
        cols = {
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name='index_versions'"
            )
        }
        assert not {"policy_id", "policy_hash"} & cols
        n = c.execute(
            "SELECT count(*) FROM pg_constraint WHERE conname='index_versions_status_check'"
        )
        assert n.fetchone() == (0,)
    command.upgrade(_cfg(), "head")
    assert {"pipeline_policies", "policy_documents"} <= tables()


def test_downgrade_to_base_and_back(dsn: str) -> None:
    command.downgrade(_cfg(), "base")
    command.upgrade(_cfg(), "head")


# (2) constraints, grants, RLS ---------------------------------------------------------------

INSERT_MIN = (
    "INSERT INTO pipeline_policies (policy_id, policy_hash, ingest_run_id, index_version, "
    "change_kind, change_reason, embedder_id, embedder_version, embedder_dim, chunker_version, "
    "chunk_tokens, chunk_overlap, redaction_rules_version, prompt_version, generator_chain_ref, "
    "declared_by) VALUES (%s::uuid, %s, 'r', 'v1', %s, %s, 'e', '1', 8, 'c', 200, 20, 'r1', "
    "'p', 'g', 'op')"
)


def _insert(
    c: psycopg.Connection, *, h: str = "a" * 64, kind: str = "embedder", why: str = "x"
) -> None:
    c.execute(INSERT_MIN, (str(uuid.uuid4()), h, kind, why))


@pytest.fixture
def v1(dsn: str) -> str:
    with psycopg.connect(dsn) as c:
        c.execute(
            "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
            "VALUES ('v1', 'e', 'c')"
        )
    return dsn


@pytest.mark.parametrize("why", ["", "   ", "\t\n"])
def test_blank_reason_rejected(v1: str, why: str) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        with _app(v1) as c:
            _insert(c, why=why)


def test_bad_enum_and_hash_rejected(v1: str) -> None:
    with pytest.raises(psycopg.errors.CheckViolation):
        with _app(v1) as c:
            _insert(c, kind="magic")
    with pytest.raises(psycopg.errors.CheckViolation):
        with _app(v1) as c:
            _insert(c, h="NOT-A-HASH")
    with pytest.raises(psycopg.errors.CheckViolation):
        with _app(v1) as c:
            _insert(c)
            c.execute("UPDATE pipeline_policies SET status='bogus'")


def test_duplicate_hash_per_version(v1: str) -> None:
    with _app(v1) as c:
        _insert(c)
    with pytest.raises(psycopg.errors.UniqueViolation):
        with _app(v1) as c:
            _insert(c)
    with _app(v1) as c:  # a different hash is fine
        _insert(c, h="b" * 64)


def test_index_versions_status_check(v1: str) -> None:
    with _app(v1) as c:
        for st in ("building", "staged", "committed", "rolled_back", "rejected"):
            c.execute("UPDATE index_versions SET status=%s", (st,))
    with pytest.raises(psycopg.errors.CheckViolation):
        with _app(v1) as c:
            c.execute("UPDATE index_versions SET status='nonsense'")


def test_grants_hashed_columns_immutable_and_no_delete(v1: str) -> None:
    with _app(v1) as c:
        _insert(c)
    for stmt in (
        "UPDATE pipeline_policies SET policy_hash = repeat('c', 64)",
        "UPDATE pipeline_policies SET change_reason = 'edited'",
        "UPDATE pipeline_policies SET embedder_id = 'other'",
        "UPDATE pipeline_policies SET doc_tenants = '{x}'",
        "DELETE FROM pipeline_policies",
        "DELETE FROM policy_documents",
        "UPDATE policy_documents SET doc_hash = 'z'",
    ):
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            with _app(v1) as c:
                c.execute(stmt)  # type: ignore[call-overload]
    with _app(v1) as c:  # lifecycle columns stay writable
        c.execute(
            "UPDATE pipeline_policies SET status='verified', verified_at=now(), eval_run_id='e', "
            "mlflow_model_version='3', mlflow_synced=true"
        )


def test_trigger_blocks_hashed_update_even_for_owner(v1: str) -> None:
    with _app(v1) as c:
        _insert(c)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with psycopg.connect(v1) as c:  # superuser: only the trigger stands in the way
            c.execute("UPDATE pipeline_policies SET policy_hash = repeat('c', 64)")
    with psycopg.connect(v1) as c:
        c.execute("UPDATE pipeline_policies SET status='verified'")  # lifecycle is fine


def test_policy_documents_rls(v1: str) -> None:
    pid = str(uuid.uuid4())
    with _app(v1) as c:
        c.execute(INSERT_MIN, (pid, "d" * 64, "embedder", "x"))
    with _app(v1, "a") as c:
        c.execute("INSERT INTO policy_documents VALUES (%s::uuid, 'a', 'd1', 1, 'h')", (pid,))
    with pytest.raises(psycopg.errors.InsufficientPrivilege):  # WITH CHECK: wrong tenant
        with _app(v1, "a") as c:
            c.execute("INSERT INTO policy_documents VALUES (%s::uuid, 'b', 'd1', 1, 'h')", (pid,))
    with _app(v1, "a") as c:
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (1,)
    with _app(v1, "b") as c:
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (0,)
    with _app(v1) as c:  # no tenant set: fail closed
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (0,)
    with pytest.raises(psycopg.errors.InsufficientPrivilege):
        with _app(v1) as c:
            c.execute("INSERT INTO policy_documents VALUES (%s::uuid, 'a', 'd9', 1, 'h')", (pid,))


def test_outbox_partial_index_exists(dsn: str) -> None:
    with psycopg.connect(dsn) as c:
        row = c.execute(
            "SELECT indexdef FROM pg_indexes WHERE indexname='pipeline_policies_unsynced_idx'"
        ).fetchone()
    assert row is not None
    assert "WHERE (NOT mlflow_synced)" in row[0]


# (3) repository -----------------------------------------------------------------------------


async def test_create_roundtrip_stamps_index_version(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p, audit_event=_event(p))
    got = await repo.get(p.policy_id)
    assert got is not None
    assert _norm(got) == _norm(p)
    assert (await repo.get_for_version("v1")) is not None
    assert _index_row(dsn, "v1") == ("building", p.policy_id, p.policy_hash)
    assert await repo.get(str(uuid.uuid4())) is None


async def test_create_duplicate_hash_conflicts_and_leaves_no_trace(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p)
    twin = p.model_copy(update={"policy_id": str(uuid.uuid4())})
    with pytest.raises(Conflict):
        await repo.create(twin, audit_event=_event(twin))
    assert _audit_rows(dsn) == []


async def test_documents_need_tenant_context_under_rls(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p)
    with _app(dsn) as c:  # a plain session sees no policy documents at all
        assert c.execute("SELECT count(*) FROM policy_documents").fetchone() == (0,)
    got = await repo.get(p.policy_id)
    assert got is not None
    assert {d.tenant_id for d in got.documents} == {"a", "b"}


async def test_happy_path_and_audit_chain_in_same_transaction(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p, audit_event=_event(p))
    for n, (exp, new, idx) in enumerate(HAPPY, start=1):
        await repo.transition(
            p.policy_id,
            expected_status=exp,
            expected_index_status=idx,
            new_status=new,
            at=T0 + timedelta(minutes=n),
            eval_run_id="run-ok" if new is S.COMMITTED else None,
            audit_event=_event(p, "policy.commit") if new is S.COMMITTED else None,
        )
    got = await repo.get(p.policy_id)
    assert got is not None
    assert got.status is S.COMMITTED
    assert got.eval_run_id == "run-ok"
    assert got.committed_at == T0 + timedelta(minutes=3)
    assert got.mlflow_synced is False
    assert _index_row(dsn, "v1")[0] == "committed"
    rows = _audit_rows(dsn)
    assert [r[0] for r in rows] == ["policy.declare", "policy.commit"]
    assert rows[0][1] is None and rows[0][2] is not None
    assert rows[1][1] == rows[0][2]  # chained: prev_hash is the previous row's hash
    assert rows[1][2] != rows[0][2]


async def test_illegal_expected_status_changes_nothing(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p)
    with pytest.raises(IllegalTransition):  # wrong expected policy status
        await repo.transition(
            p.policy_id,
            expected_status=S.VERIFIED,
            expected_index_status="building",
            new_status=S.INDEXED,
            at=T0,
            audit_event=_event(p, "policy.bogus"),
        )
    with pytest.raises(IllegalTransition):  # wrong expected index status: policy update rolls back
        await repo.transition(
            p.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="staged",
            new_status=S.VERIFIED,
            at=T0,
            audit_event=_event(p, "policy.bogus"),
        )
    with pytest.raises(IllegalTransition):  # not an allowed edge
        await repo.transition(
            p.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.COMMITTED,
            at=T0,
        )
    with pytest.raises(NotFound):
        await repo.transition(
            str(uuid.uuid4()),
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.VERIFIED,
            at=T0,
        )
    got = await repo.get(p.policy_id)
    assert got is not None and got.status is S.DECLARED and got.verified_at is None
    assert _index_row(dsn, "v1")[0] == "building"
    assert _audit_rows(dsn) == []


async def test_failing_audit_rolls_the_transition_back(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p = make_policy()
    await repo.create(p)
    bad = AuditEvent(at=T0, actor=None, action="policy.verify", tenant_id="")  # type: ignore[arg-type]
    with pytest.raises(psycopg.errors.NotNullViolation):
        await repo.transition(
            p.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.VERIFIED,
            at=T0,
            audit_event=bad,
        )
    got = await repo.get(p.policy_id)
    assert got is not None and got.status is S.DECLARED
    assert _index_row(dsn, "v1")[0] == "building"
    assert _audit_rows(dsn) == []


async def test_concurrent_transitions_exactly_one_wins(dsn: str) -> None:
    repo_a, repo_b = SqlPolicyRepository(dsn), SqlPolicyRepository(dsn)  # two connections
    p = make_policy()
    await repo_a.create(p)

    async def go(repo: SqlPolicyRepository, name: str) -> str:
        try:
            await repo.transition(
                p.policy_id,
                expected_status=S.DECLARED,
                expected_index_status="building",
                new_status=S.VERIFIED,
                at=T0,
                audit_event=_event(p, f"policy.verify.{name}"),
            )
        except IllegalTransition:
            return "lost"
        return "won"

    res = await asyncio.gather(*(go(repo_a, "a"), go(repo_b, "b"), go(repo_a, "c")))
    assert sorted(res) == ["lost", "lost", "won"]
    assert len([r for r in _audit_rows(dsn) if r[0].startswith("policy.verify")]) == 1


async def test_outbox_order_and_mark_synced(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    older = make_policy("v1", declared_at=T0)
    newer = make_policy("v2", declared_at=T0 + timedelta(hours=1))
    pending = make_policy("v3", declared_at=T0 + timedelta(hours=2))
    for p in (older, newer, pending):
        await repo.create(p)
    # commit the NEWER declared policy first: commit order, not declare order, drives the outbox
    await _walk(repo, newer, *HAPPY)
    await _walk(repo, older, *HAPPY, base=10)
    order = [p.policy_id for p in await repo.list_unsynced(10)]
    assert order == [newer.policy_id, older.policy_id, pending.policy_id]
    assert [p.policy_id for p in await repo.list_unsynced(1)] == [newer.policy_id]
    await repo.record_mlflow_version(newer.policy_id, "7")
    got = await repo.get(newer.policy_id)
    assert got is not None and got.mlflow_model_version == "7" and not got.mlflow_synced
    await repo.mark_synced(newer.policy_id, "7")
    assert [p.policy_id for p in await repo.list_unsynced(10)] == [
        older.policy_id,
        pending.policy_id,
    ]


async def test_get_previous_committed_follows_lineage(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    p1 = make_policy("v1")
    await repo.create(p1)
    await _walk(repo, p1, *HAPPY)
    p2 = make_policy("v2", parent=p1.policy_id, declared_at=T0 + timedelta(days=1))
    await repo.create(p2)
    await _walk(repo, p2, *HAPPY)
    unrelated = make_policy("v9", declared_at=T0 + timedelta(days=2))
    await repo.create(unrelated)
    await _walk(repo, unrelated, *HAPPY)
    p3 = make_policy("v3", parent=p2.policy_id, declared_at=T0 + timedelta(days=3))
    await repo.create(p3)
    await _walk(repo, p3, *HAPPY)
    c3, c2, c1 = [await repo.get(p.policy_id) for p in (p3, p2, p1)]
    assert c3 and c2 and c1
    prev3 = await repo.get_previous_committed(c3)
    assert prev3 is not None and prev3.policy_id == p2.policy_id
    prev2 = await repo.get_previous_committed(c2)
    assert prev2 is not None and prev2.policy_id == p1.policy_id
    assert await repo.get_previous_committed(c1) is None
    uncommitted = await repo.get(p3.policy_id)
    assert uncommitted is not None
    assert (
        await repo.get_previous_committed(uncommitted.model_copy(update={"committed_at": None}))
        is None
    )


async def test_reject_and_rollback_move_the_index_row(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    a, b = make_policy("v1"), make_policy("v2")
    await repo.create(a)
    await repo.create(b)
    await repo.transition(
        a.policy_id,
        expected_status=S.DECLARED,
        expected_index_status="building",
        new_status=S.REJECTED,
        at=T0,
    )
    assert _index_row(dsn, "v1")[0] == "rejected"
    await _walk(repo, b, *HAPPY, (S.COMMITTED, S.ROLLED_BACK, "committed"))
    assert _index_row(dsn, "v2")[0] == "rolled_back"
    got = await repo.get(b.policy_id)
    assert got is not None and got.rolled_back_at is not None and got.mlflow_synced is False


async def test_redeclare_after_reject_but_not_over_committed(dsn: str) -> None:
    repo = SqlPolicyRepository(dsn)
    a = make_policy("v1")
    await repo.create(a)
    await repo.transition(
        a.policy_id,
        expected_status=S.DECLARED,
        expected_index_status="building",
        new_status=S.REJECTED,
        at=T0,
    )
    b = make_policy("v1", reason="second try")
    await repo.create(b)
    assert _index_row(dsn, "v1")[:2] == ("building", b.policy_id)
    await _walk(repo, b, *HAPPY)
    with pytest.raises(Conflict):
        await repo.create(make_policy("v1", reason="third try"))


# (4) parity with the in-memory fake ---------------------------------------------------------


async def _scenario(repo: PolicyRepository) -> list[Any]:
    """Same steps, normalised outcomes (ids differ, so they are mapped to names)."""
    out: list[Any] = []
    p1 = make_policy("v1", declared_at=T0)
    p2 = make_policy("v2", declared_at=T0 + timedelta(hours=1), parent=p1.policy_id)
    ids = {p1.policy_id: "p1", p2.policy_id: "p2"}

    def shape(p: PipelinePolicy | None) -> Any:
        d = _norm(p)
        if d is None:
            return None
        d["policy_id"] = ids.get(d["policy_id"], d["policy_id"])
        d["policy_hash"] = "<hash>"  # parent_policy_id is hashed and the ids are random per run
        d["parent_policy_id"] = ids.get(d["parent_policy_id"])
        return d

    async def attempt(label: str, coro: Any) -> None:
        try:
            r = await coro
            out.append((label, "ok", shape(r) if isinstance(r, PipelinePolicy) else r))
        except (IllegalTransition, Conflict, NotFound) as e:
            out.append((label, type(e).__name__))

    await attempt("create1", repo.create(p1))
    await attempt("dup-hash", repo.create(p1.model_copy(update={"policy_id": str(uuid.uuid4())})))
    await attempt("create2", repo.create(p2))
    for n, (exp, new, idx) in enumerate(HAPPY, start=1):
        await attempt(
            f"p1-{new.value}",
            repo.transition(
                p1.policy_id,
                expected_status=exp,
                expected_index_status=idx,
                new_status=new,
                at=T0 + timedelta(minutes=n),
                eval_run_id="e1" if new is S.COMMITTED else None,
            ),
        )
    await attempt(
        "p1-double-commit",
        repo.transition(
            p1.policy_id,
            expected_status=S.INDEXED,
            expected_index_status="staged",
            new_status=S.COMMITTED,
            at=T0,
        ),
    )
    await attempt(
        "p2-skip",
        repo.transition(
            p2.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.INDEXED,
            at=T0,
        ),
    )
    await attempt(
        "p2-bad-index-status",
        repo.transition(
            p2.policy_id,
            expected_status=S.DECLARED,
            expected_index_status="staged",
            new_status=S.VERIFIED,
            at=T0,
        ),
    )
    await attempt(
        "missing",
        repo.transition(
            str(uuid.uuid4()),
            expected_status=S.DECLARED,
            expected_index_status="building",
            new_status=S.VERIFIED,
            at=T0,
        ),
    )
    await attempt("p2-walk", _walk(repo, p2, *HAPPY))
    await attempt(
        "p2-rollback",
        repo.transition(
            p2.policy_id,
            expected_status=S.COMMITTED,
            expected_index_status="committed",
            new_status=S.ROLLED_BACK,
            at=T0 + timedelta(days=1),
        ),
    )
    await repo.record_mlflow_version(p1.policy_id, "11")
    out.append(("unsynced", [shape(x)["policy_id"] for x in await repo.list_unsynced(10)]))
    await repo.mark_synced(p1.policy_id, "11")
    out.append(("unsynced-after", [shape(x)["policy_id"] for x in await repo.list_unsynced(10)]))
    cur2 = await repo.get(p2.policy_id)
    assert cur2 is not None
    out.append(("previous", shape(await repo.get_previous_committed(cur2))))
    out.append(("for-version", shape(await repo.get_for_version("v1"))))
    out.append(("get", shape(await repo.get(p2.policy_id))))
    return out


async def test_parity_fake_vs_sql(dsn: str) -> None:
    fake = await _scenario(FakePolicyRepository())
    sql_out = await _scenario(SqlPolicyRepository(dsn))
    assert sql_out == fake
    assert ("dup-hash", "Conflict") in sql_out and ("missing", "NotFound") in sql_out


# (5) eval runs, mirror down -----------------------------------------------------------------


async def test_sql_eval_runs_reader(dsn: str) -> None:
    with _app(dsn) as c:
        for rid, ok, m in (
            ("good", True, '{"recall": 0.9}'),
            ("failed", False, '{"recall": 0.2}'),
            ("badmetric", True, '{"recall": "high"}'),
            ("nested", True, "[1]"),
        ):
            c.execute(
                "INSERT INTO gate_runs (gate_run_id, kind, passed, metrics) "
                "VALUES (%s,'eval',%s,%s::jsonb)",
                (rid, ok, m),
            )
    reader = SqlEvalRuns(dsn)
    assert await reader.passed("good") is True
    for rid in ("failed", "badmetric", "nested", "missing"):
        assert await reader.passed(rid) is False


def test_metrics_clean_pure() -> None:
    assert metrics_clean({"a": 1, "b": 0.5}) and not metrics_clean({})  # empty fails closed
    assert not metrics_clean({"a": float("nan")}) and not metrics_clean({"a": True})
    assert not metrics_clean(None) and not metrics_clean({"a": float("inf")})


async def test_gate_flow_succeeds_with_mlflow_down(dsn: str) -> None:
    from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
    from rag_platform.adapters.policy_fakes import FakeRegistryMirror
    from rag_platform.services.policy_gate import ChunkSummary, PolicyGate

    repo = SqlPolicyRepository(dsn)
    with _app(dsn) as c:
        c.execute(
            "INSERT INTO gate_runs (gate_run_id, kind, passed, metrics) "
            "VALUES ('ev','eval',true,jsonb_build_object('recall', 0.9))"
        )
    gate = PolicyGate(
        repo,
        FakeAuditLog(),
        FakeClock(T0),
        mirror=FakeRegistryMirror(down=True),
        evals=SqlEvalRuns(dsn),
    )
    p = make_policy(kind=ChangeKind.EMBEDDER)
    draft = PolicyDraft(**{k: getattr(p, k) for k in PolicyDraft.model_fields})
    declared = await gate.declare(draft, declared_by="op")
    await gate.verify(declared.policy_id, ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=8))
    await gate.mark_indexed(declared.policy_id)
    done = await gate.mark_committed(declared.policy_id, eval_run_id="ev")
    assert done.status is S.COMMITTED
    assert _index_row(dsn, "v1")[0] == "committed"
    assert [r[0] for r in _audit_rows(dsn)][-1] == "policy.commit"
