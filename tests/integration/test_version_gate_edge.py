"""T-011 edge probes: gate inputs, policy states, concurrency, failure injection, rollback."""

# ruff: noqa: F811
import asyncio
import json
from typing import Any

import psycopg
import pytest
from index_version_helpers import make_service
from ingest_helpers import make_stack, q
from test_ingest_pipeline import _ingest, dsn  # noqa: F401  (fixture)

from rag_platform.adapters.sql_index_versions import SqlIndexVersionStore
from rag_platform.domain.errors import Conflict
from rag_platform.domain.policy import PolicyRuleViolation
from rag_platform.services.index_versions import GateFailed, IndexVersionService

pytestmark = pytest.mark.integration
ALIAS = "SELECT index_version, previous_version FROM index_alias"


def snap(dsn: str) -> dict[str, Any]:
    return {
        "alias": q(dsn, ALIAS + " ORDER BY 1"),
        "policy": q(dsn, "SELECT policy_id, status FROM pipeline_policies ORDER BY 1"),
        "versions": q(
            dsn, "SELECT index_version, status, committed_at FROM index_versions ORDER BY 1"
        ),  # noqa: E501
        "audit": q(dsn, "SELECT count(*) FROM audit_log"),
    }


def chunks(dsn: str, v: str) -> int:
    return int(q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v,))[0][0])  # type: ignore[call-overload]


def su(dsn: str, stmt: str, params: tuple[object, ...] = ()) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(stmt, params)  # type: ignore[arg-type]


async def chain(s, svc, n: int) -> list[str]:  # type: ignore[no-untyped-def]
    out = []
    for i in range(n):
        v = await _ingest(s, f"d{i}.txt", f"tag{i}")
        await svc.commit(v)
        out.append(v)
    return out


# ---- vector invariants the DB itself enforces -------------------------------------------------


def test_db_rejects_nan_inf_and_wrong_dim(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    asyncio.run(_ingest(s, "a.txt", "alpha"))
    for bad in ("[NaN," + ",".join("0" * 383) + "]", "[Infinity," + ",".join("0" * 383) + "]",
                "[" + ",".join("1" * 383) + "]", "[" + ",".join("1" * 385) + "]"):  # fmt: skip
        with pytest.raises(psycopg.Error):
            su(dsn, "UPDATE chunks SET embedding = %s::vector", (bad,))


# ---- gate: data defects -----------------------------------------------------------------------


async def test_all_chunks_wiped_fails_row_parity(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "DELETE FROM chunks WHERE index_version=%s", (v1,))
    with pytest.raises(GateFailed) as e:
        await svc.commit(v1)
    assert "row_parity" in str(e.value)
    assert q(dsn, ALIAS) == []


@pytest.mark.parametrize("delta", [1, -1])
async def test_recorded_row_count_off_by_one(dsn: str, delta: int) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "UPDATE index_versions SET row_count = row_count + %s", (delta,))
    with pytest.raises(GateFailed) as e:
        await make_service(s).commit(v1)
    assert "row_parity" in str(e.value)


async def test_zero_vector_among_many_fails_sanity(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "UPDATE chunks SET embedding = array_fill(0, ARRAY[384])::vector "
            "WHERE ctid = (SELECT ctid FROM chunks LIMIT 1)")  # fmt: skip
    with pytest.raises(GateFailed) as e:
        await make_service(s).commit(v1)
    assert "vector_sanity" in str(e.value)
    m = q(dsn, "SELECT metrics FROM gate_runs")[0][0]
    assert m["zero_vectors"] == 1 and m["vector_violations"] == 1  # type: ignore[index]


async def test_extra_unreferenced_parent_fails_parity_only(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "INSERT INTO parent_chunks SELECT index_version, tenant_id, 'extra-parent', doc_id, "
            "doc_version, 99, text, token_count, acl_labels, heading_path, doc_hash, span_start, "
            "span_end, effective_date, expires_date, chunker_version FROM parent_chunks "
            "WHERE index_version=%s LIMIT 1", (v1,))  # fmt: skip
    with pytest.raises(GateFailed) as e:
        await make_service(s).commit(v1)
    assert "parent_row_parity" in str(e.value) and "orphan_children" not in str(e.value)


async def test_swapped_parent_keeps_parity_but_orphans_fail(dsn: str) -> None:  # noqa: F811
    """Delete a referenced parent and add an unreferenced one: counts equal, orphan check fires."""
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "INSERT INTO parent_chunks SELECT index_version, tenant_id, 'extra-parent', doc_id, "
            "doc_version, 99, text, token_count, acl_labels, heading_path, doc_hash, span_start, "
            "span_end, effective_date, expires_date, chunker_version FROM parent_chunks "
            "WHERE index_version=%s LIMIT 1", (v1,))  # fmt: skip
    su(dsn, "DELETE FROM parent_chunks WHERE ctid=(SELECT ctid FROM parent_chunks "
            "WHERE parent_id <> 'extra-parent' LIMIT 1)")  # fmt: skip
    with pytest.raises(GateFailed) as e:
        await make_service(s).commit(v1)
    assert "orphan_children" in str(e.value) and "parent_row_parity" not in str(e.value)


async def test_parent_present_only_in_other_version_is_an_orphan(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    v2 = await _ingest(s, "b.txt", "bravo")  # copy-forward of a.txt: same parent ids as v1
    shared = q(dsn, "SELECT p.parent_id FROM parent_chunks p JOIN parent_chunks o USING "
                    "(parent_id) WHERE p.index_version=%s AND o.index_version=%s LIMIT 1",
               (v2, v1))  # fmt: skip
    assert shared, "copy-forward should share parent ids across versions"
    su(dsn, "DELETE FROM parent_chunks WHERE index_version=%s AND parent_id=%s",
       (v2, shared[0][0]))  # fmt: skip
    before = snap(dsn)["alias"]
    with pytest.raises(GateFailed) as e:
        await svc.commit(v2)
    assert "orphan_children" in str(e.value)
    assert snap(dsn)["alias"] == before


# ---- tenancy: measured tenants only -----------------------------------------------------------


async def test_injected_rows_of_unlisted_tenant_must_fail_gate(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    cols = (
        "doc_id, doc_version, ordinal, text, span_start, span_end, section, page, acl_labels, "
        "effective_date, expires_date, embedding_model, chunker_version, embedding, parent_id"
    )
    su(
        dsn,
        "INSERT INTO chunks (chunk_id, index_version, tenant_id, " + cols + ") "  # noqa: S608
        "SELECT chunk_id || '-x', index_version, 'evil', " + cols + " FROM chunks "
        "WHERE index_version=%s LIMIT 1",
        (v1,),
    )
    with pytest.raises(GateFailed):
        await make_service(s).commit(v1)


async def test_measure_is_tenant_scoped(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    store = SqlIndexVersionStore(dsn)
    mine = await store.measure(v1, ["t1"], dim=384, sample=500)
    other = await store.measure(v1, ["nobody"], dim=384, sample=500)
    assert mine.rows_actual > 0 and mine.self_hits == mine.self_total > 0
    assert other.rows_actual == 0 and other.self_total == 0 and other.parent_rows == 0


# ---- gate_runs content ------------------------------------------------------------------------


async def test_gate_runs_pass_and_fail_counts_only(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "secret-name.txt", "alpha")
    await svc.commit(v1)
    v2 = await _ingest(s, "b.txt", "bravo")
    su(dsn, "UPDATE index_versions SET row_count = row_count + 1 WHERE index_version=%s", (v2,))
    with pytest.raises(GateFailed):
        await svc.commit(v2)
    rows = q(dsn, "SELECT passed, metrics, index_version FROM gate_runs ORDER BY created_at")
    assert [r[0] for r in rows] == [True, False]
    blob = json.dumps([r[1] for r in rows])
    for forbidden in ("secret-name", "alpha", "bravo", "t1", "river", "staff"):
        assert forbidden not in blob
    for _, m, _v in rows:
        for k, val in m.items():  # type: ignore[union-attr]
            assert isinstance(val, (int, float, type(None))) or (
                k == "failures" and all(isinstance(f, str) for f in val)
            )
    assert rows[1][1]["failures"] == ["row_parity"]  # type: ignore[index]
    stamped = dict(q(dsn, "SELECT index_version, gate_run_id FROM index_versions"))
    assert stamped[v1] and stamped[v2] and stamped[v1] != stamped[v2]


# ---- policy states ----------------------------------------------------------------------------


@pytest.mark.parametrize("state", ["declared", "rejected", "committed", "rolled_back"])
async def test_commit_refused_unless_indexed(dsn: str, state: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    su(dsn, "UPDATE pipeline_policies SET status=%s", (state,))
    before = snap(dsn)
    with pytest.raises(PolicyRuleViolation):
        await make_service(s).commit(v1)
    assert snap(dsn) == before
    assert q(dsn, "SELECT count(*) FROM gate_runs") == [(0,)]


# ---- concurrency ------------------------------------------------------------------------------


async def test_concurrent_double_commit_of_same_version(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    res = await asyncio.gather(
        make_service(s).commit(v1), make_service(s).commit(v1), return_exceptions=True
    )
    ok = [r for r in res if not isinstance(r, BaseException)]
    bad = [r for r in res if isinstance(r, BaseException)]
    assert len(ok) == 1 and len(bad) == 1
    assert isinstance(bad[0], (Conflict, PolicyRuleViolation))
    assert q(dsn, ALIAS) == [(v1, None)]
    assert q(dsn, "SELECT status FROM pipeline_policies") == [("committed",)]
    assert q(dsn, "SELECT count(*) FROM audit_log WHERE action='policy.commit'") == [(1,)]


async def test_concurrent_sibling_commits_one_wins(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    v2 = await _ingest(s, "b.txt", "bravo")
    s.base[0] = v1
    v3 = await _ingest(s, "c.txt", "charlie")  # sibling of v2, also based on v1
    res = await asyncio.gather(
        make_service(s).commit(v2), make_service(s).commit(v3), return_exceptions=True
    )
    wins = [r for r in res if not isinstance(r, BaseException)]
    assert len(wins) == 1 and any(isinstance(r, Conflict) for r in res)
    live = q(dsn, ALIAS)
    assert len(live) == 1 and live[0][0] in (v2, v3) and live[0][1] == v1
    loser = v3 if live[0][0] == v2 else v2
    assert dict(q(dsn, "SELECT index_version, status FROM index_versions"))[loser] == "staged"
    assert ("indexed",) in q(dsn, "SELECT status FROM pipeline_policies")


async def test_commit_waits_for_alias_lock_held_by_other_tx(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    store = SqlIndexVersionStore(dsn)
    async with store.alias_tx():  # someone else holds the swap lock
        t = asyncio.create_task(make_service(s).commit(v1))
        await asyncio.sleep(1.0)
        assert not t.done() and q(dsn, ALIAS) == []
    await asyncio.wait_for(t, 10)
    assert q(dsn, ALIAS) == [(v1, None)]


class ReaderStore(SqlIndexVersionStore):
    """Runs a second-connection reader right after the alias write, before the tx commits."""

    seen: list[Any]

    def __init__(self, dsn: str) -> None:
        super().__init__(dsn)
        self.seen = []

    async def set_alias(self, c: Any, index_version: str, previous: str | None) -> None:
        await super().set_alias(c, index_version, previous)
        async with await psycopg.AsyncConnection.connect(self._dsn) as r:
            a = await (await r.execute("SELECT index_version FROM index_alias")).fetchall()
            p = await (
                await r.execute("SELECT status FROM pipeline_policies ORDER BY declared_at")
            ).fetchall()  # noqa: E501
            v = await (
                await r.execute("SELECT status FROM index_versions ORDER BY created_at")
            ).fetchall()  # noqa: E501
        self.seen.append((a, p, v))


def svc_with(s, store):  # type: ignore[no-untyped-def]
    base = make_service(s)
    return IndexVersionService(store, s.policies, base._gate)


async def test_reader_never_sees_half_state_commit_and_rollback(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    store = ReaderStore(dsn)
    svc = svc_with(s, store)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    v2 = await _ingest(s, "b.txt", "bravo")
    await svc.commit(v2)
    await svc.rollback()
    c1, c2, rb = store.seen
    assert c1 == ([], [("indexed",)], [("staged",)])
    assert c2 == ([(v1,)], [("committed",), ("indexed",)], [("committed",), ("staged",)])
    assert rb == ([(v2,)], [("committed",), ("committed",)], [("committed",), ("committed",)])


# ---- failure injection ------------------------------------------------------------------------


class Boom(RuntimeError):
    pass


class FailingStore(SqlIndexVersionStore):
    def __init__(self, dsn: str, where: str, after: bool) -> None:
        super().__init__(dsn)
        self.where, self.after = where, after

    async def _wrap(self, name: str, fn, *a):  # type: ignore[no-untyped-def]
        if self.where == name and not self.after:
            raise Boom(name)
        out = await fn(*a)
        if self.where == name and self.after:
            raise Boom(name)
        return out

    async def live(self, c, *a):  # type: ignore[no-untyped-def,override]
        return await self._wrap("live", super().live, c, *a)

    async def version_row(self, c, *a):  # type: ignore[no-untyped-def,override]
        return await self._wrap("version_row", super().version_row, c, *a)

    async def set_alias(self, c, *a):  # type: ignore[no-untyped-def,override]
        return await self._wrap("set_alias", super().set_alias, c, *a)


@pytest.mark.parametrize(
    ("where", "after"),
    [("version_row", False), ("live", False), ("live", True), ("set_alias", False),
     ("set_alias", True), ("mark_committed", True), ("on_swap", False)],
)  # fmt: skip
async def test_commit_failure_at_each_step_changes_nothing(
    dsn: str, where: str, after: bool
) -> None:  # noqa: F811,E501
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    await make_service(s).commit(v1)
    v2 = await _ingest(s, "b.txt", "bravo")
    before = snap(dsn)

    async def on_swap() -> None:
        raise Boom("on_swap")

    store = FailingStore(dsn, where, after)
    svc = IndexVersionService(store, s.policies, make_service(s)._gate,
                              on_swap=on_swap if where == "on_swap" else None)  # fmt: skip
    if where == "mark_committed":
        orig = svc._gate.mark_committed

        async def mc(*a: Any, **k: Any) -> Any:
            await orig(*a, **k)
            raise Boom("mark_committed")

        svc._gate.mark_committed = mc  # type: ignore[method-assign]
    with pytest.raises(Boom):
        await svc.commit(v2)
    assert snap(dsn) == before
    await make_service(s).commit(v2)  # state is clean: a retry succeeds
    assert q(dsn, ALIAS) == [(v2, v1)]


@pytest.mark.parametrize(
    ("where", "after"),
    [("version_row", False), ("set_alias", False), ("set_alias", True), ("mark_rolled_back", True)],
)
async def test_rollback_failure_at_each_step_changes_nothing(
    dsn: str, where: str, after: bool
) -> None:  # noqa: F811,E501
    s = make_stack(dsn)
    svc0 = make_service(s)
    await chain(s, svc0, 2)
    before = snap(dsn)
    svc = IndexVersionService(FailingStore(dsn, where, after), s.policies, svc0._gate)
    if where == "mark_rolled_back":
        orig = svc._gate.mark_rolled_back

        async def mr(*a: Any, **k: Any) -> Any:
            await orig(*a, **k)
            raise Boom("mr")

        svc._gate.mark_rolled_back = mr  # type: ignore[method-assign]
    with pytest.raises(Boom):
        await svc.rollback()
    assert snap(dsn) == before


async def test_mirror_down_does_not_undo_commit(dsn: str) -> None:  # noqa: F811
    from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
    from rag_platform.adapters.policy_fakes import FakeRegistryMirror
    from rag_platform.services.policy_gate import PolicyGate

    s = make_stack(dsn)
    gate = PolicyGate(s.policies, FakeAuditLog(), FakeClock(), FakeRegistryMirror(down=True))
    svc = IndexVersionService(SqlIndexVersionStore(dsn), s.policies, gate)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    assert q(dsn, ALIAS) == [(v1, None)]
    assert q(dsn, "SELECT status, mlflow_synced FROM pipeline_policies") == [("committed", False)]


# ---- rollback edge cases ----------------------------------------------------------------------


async def test_rollback_twice_walks_back_then_refuses(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1, v2, v3 = await chain(s, svc, 3)
    assert await svc.rollback() == v2
    assert await svc.rollback() == v1
    assert q(dsn, ALIAS) == [(v1, v2)]
    with pytest.raises(Conflict):
        await svc.rollback()
    st = dict(q(dsn, "SELECT index_version, status FROM index_versions"))
    assert st == {v1: "committed", v2: "rolled_back", v3: "rolled_back"}


async def test_rollback_to_dropped_partition_refused(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1, v2, v3 = await chain(s, svc, 3)
    await svc.rollback()  # live v2, previous v3
    assert set(await svc.retain(keep=1)) == {v1}
    before = snap(dsn)
    with pytest.raises(Conflict):
        await svc.rollback()  # v2's parent v1 has no partition
    assert snap(dsn) == before


async def test_rollback_with_null_parent_and_missing_parent_row(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1, v2 = await chain(s, svc, 2)
    su(dsn, "UPDATE index_versions SET parent_version=NULL WHERE index_version=%s", (v2,))
    before = snap(dsn)
    with pytest.raises(Conflict):
        await svc.rollback()
    assert snap(dsn) == before


async def test_rollback_target_not_committed_refused(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1, v2 = await chain(s, svc, 2)
    su(dsn, "UPDATE index_versions SET status='rejected' WHERE index_version=%s", (v1,))
    with pytest.raises(Conflict):
        await svc.rollback()


async def test_rolled_back_version_cannot_be_recommitted(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1, v2 = await chain(s, svc, 2)
    await svc.rollback()
    with pytest.raises(PolicyRuleViolation):
        await svc.commit(v2)
    assert q(dsn, ALIAS) == [(v1, v2)]


# ---- retention --------------------------------------------------------------------------------


@pytest.mark.parametrize(("n", "dropped"), [(3, 0), (4, 1), (5, 2)])
async def test_retention_counts(dsn: str, n: int, dropped: int) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    vs = await chain(s, svc, n)
    assert sorted(await svc.retain()) == sorted(vs[:dropped])
    assert [chunks(dsn, v) > 0 for v in vs] == [False] * dropped + [True] * (n - dropped)
    assert await svc.retain() == []
    assert len(q(dsn, "SELECT 1 FROM index_versions")) == n  # audit rows kept


async def test_retention_leaves_inflight_and_drops_dead(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    vs = await chain(s, svc, 4)
    staged = await _ingest(s, "s.txt", "staged")
    s.base[0] = vs[-1]
    building = await _ingest(s, "b.txt", "building")
    su(dsn, "UPDATE index_versions SET status='building' WHERE index_version=%s", (building,))
    s.base[0] = vs[-1]
    rej = await _ingest(s, "r.txt", "rejected")
    su(dsn, "UPDATE index_versions SET status='rejected' WHERE index_version=%s", (rej,))
    s.base[0] = vs[-1]
    failed = await _ingest(s, "f.txt", "failed")
    su(dsn, "UPDATE index_versions SET status='failed' WHERE index_version=%s", (failed,))
    removed = set(await svc.retain())
    assert removed == {vs[0], rej, failed}
    assert chunks(dsn, staged) > 0 and chunks(dsn, building) > 0
    assert chunks(dsn, vs[-1]) > 0
    for v in (rej, failed):
        assert q(dsn, "SELECT count(*) FROM parent_chunks WHERE index_version=%s", (v,)) == [(0,)]


async def test_retention_keep_zero_still_protects_alias_and_previous(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    vs = await chain(s, svc, 3)
    await svc.retain(keep=0)
    assert [chunks(dsn, v) > 0 for v in vs] == [False, True, True]


async def test_retention_concurrent_with_commit_is_serialised(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    vs = await chain(s, svc, 3)
    v4 = await _ingest(s, "n.txt", "next")
    res = await asyncio.gather(svc.retain(keep=3), make_service(s).commit(v4))
    assert q(dsn, ALIAS) == [(v4, vs[-1])]
    assert chunks(dsn, v4) > 0 and chunks(dsn, vs[-1]) > 0
    assert isinstance(res[0], list)


# ---- end to end -------------------------------------------------------------------------------


async def test_e2e_ingest_commit_copy_forward_commit_rollback(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    live_ids = (
        "SELECT chunk_id FROM chunks WHERE index_version=(SELECT index_version FROM index_alias) "
        "ORDER BY chunk_id"
    )
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    ids1 = q(dsn, live_ids)
    embedded = len(s.embedder.texts)
    v2 = await _ingest(s, "b.txt", "bravo")
    assert q(dsn, live_ids) == ids1  # staged v2 invisible until the swap
    assert len(s.embedder.texts) > embedded
    await svc.commit(v2)
    ids2 = q(dsn, live_ids)
    assert set(ids1) < set(ids2)  # copy-forward kept a.txt, added b.txt
    assert await svc.rollback() == v1
    assert q(dsn, live_ids) == ids1
    assert q(dsn, "SELECT count(*) FROM gate_runs WHERE passed") == [(2,)]
    assert q(dsn, "SELECT count(*) FROM audit_log WHERE action IN "
                  "('policy.commit','policy.rollback')") == [(3,)]  # fmt: skip
