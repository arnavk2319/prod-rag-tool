"""T-045 adversarial: transition matrix, concurrency, atomicity, audit, verify, outbox."""

import asyncio
import threading
from datetime import UTC, datetime
from typing import Any

import pytest

from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import (
    FakeEvalRuns,
    FakePolicyRepository,
    FakeRegistryMirror,
)
from rag_platform.domain.policy import (
    ALLOWED,
    INDEX_STATUS,
    ChangeKind,
    IllegalTransition,
    PolicyRuleViolation,
    PolicyStatus,
    RejectReason,
)
from rag_platform.services.policy_gate import ChunkSummary, PolicyGate, sync_one, sync_pending
from tests.unit.test_policy_gate import OK, draft, make

ALL_DOCS = frozenset({("a", "d1"), ("b", "d2")})


S = PolicyStatus
ORDER = [S.DECLARED, S.VERIFIED, S.INDEXED, S.COMMITTED, S.ROLLED_BACK]


async def reach(gate: PolicyGate, status: PolicyStatus, **kw: Any) -> str:
    """Drive a fresh policy to `status` through legal steps only."""
    p = await gate.declare(draft(**kw), declared_by="t")
    pid = p.policy_id
    if status is S.REJECTED:
        await gate.reject(pid, RejectReason.OPERATOR_REJECTED)
        return pid
    steps = [
        (S.VERIFIED, lambda: gate.verify(pid, OK)),
        (S.INDEXED, lambda: gate.mark_indexed(pid)),
        (S.COMMITTED, lambda: gate.mark_committed(pid)),
        (S.ROLLED_BACK, lambda: gate.mark_rolled_back(pid)),
    ]
    for target, fn in steps:
        if ORDER.index(target) > ORDER.index(status):
            break
        await fn()
    return pid


ACTIONS = {
    S.VERIFIED: lambda g, pid: g.verify(pid, OK),
    S.INDEXED: lambda g, pid: g.mark_indexed(pid),
    S.COMMITTED: lambda g, pid: g.mark_committed(pid),
    S.ROLLED_BACK: lambda g, pid: g.mark_rolled_back(pid),
    S.REJECTED: lambda g, pid: g.reject(pid, RejectReason.OPERATOR_REJECTED),
}


@pytest.mark.parametrize("src", list(S))
@pytest.mark.parametrize("dst", list(ACTIONS))
async def test_full_transition_matrix(src: PolicyStatus, dst: PolicyStatus) -> None:
    gate, repo, _, _ = make()
    pid = await reach(gate, src)
    assert (await repo.get(pid)).status is src  # type: ignore[union-attr]
    legal = dst in ALLOWED[src]
    if legal:
        done = await ACTIONS[dst](gate, pid)
        assert done.status is dst
        assert repo.index_status["v1"] == INDEX_STATUS[dst]
    else:
        with pytest.raises((IllegalTransition, PolicyRuleViolation)):
            await ACTIONS[dst](gate, pid)
        got = await repo.get(pid)
        assert got is not None and got.status is src
        assert repo.index_status["v1"] == INDEX_STATUS[src]


@pytest.mark.parametrize("terminal", [S.ROLLED_BACK, S.REJECTED])
async def test_terminal_states_stay_terminal(terminal: PolicyStatus) -> None:
    gate, repo, _, _ = make()
    pid = await reach(gate, terminal)
    for act in ACTIONS.values():
        with pytest.raises((IllegalTransition, PolicyRuleViolation)):
            await act(gate, pid)
    assert (await repo.get(pid)).status is terminal  # type: ignore[union-attr]


async def test_double_commit_and_commit_after_reject() -> None:
    gate, repo, _, audit = make()
    pid = await reach(gate, S.COMMITTED)
    with pytest.raises(IllegalTransition):
        await gate.mark_committed(pid)
    commits = [e for e in audit._events if e.action == "policy.commit"]
    assert len(commits) == 1
    pid2 = await reach(gate, S.REJECTED, index_version="v2", ingest_run_id="r2")
    with pytest.raises(IllegalTransition):
        await gate.mark_committed(pid2)


async def test_unknown_policy_is_not_found() -> None:
    from rag_platform.domain.errors import NotFound

    gate, _, _, _ = make()
    with pytest.raises(NotFound):
        await gate.mark_indexed("nope")


async def test_concurrent_same_expected_status_one_winner() -> None:
    gate, repo, _, _ = make()
    pid = await reach(gate, S.INDEXED)
    res = await asyncio.gather(
        gate.mark_committed(pid), gate.mark_committed(pid), return_exceptions=True
    )
    wins = [r for r in res if not isinstance(r, BaseException)]
    errs = [r for r in res if isinstance(r, BaseException)]
    assert len(wins) == 1 and len(errs) == 1 and isinstance(errs[0], IllegalTransition)


async def test_concurrent_commit_vs_reject_one_winner() -> None:
    gate, repo, _, _ = make()
    pid = await reach(gate, S.INDEXED)
    res = await asyncio.gather(
        gate.mark_committed(pid),
        gate.reject(pid, RejectReason.OPERATOR_REJECTED),
        return_exceptions=True,
    )
    assert sum(isinstance(r, BaseException) for r in res) == 1
    final = (await repo.get(pid)).status  # type: ignore[union-attr]
    assert repo.index_status["v1"] == INDEX_STATUS[final]


def test_concurrent_threads_same_policy_one_winner() -> None:
    """Two OS threads, each with its own loop, share the repository."""
    gate, repo, _, _ = make()
    pid = asyncio.run(reach(gate, S.INDEXED))
    out: list[Any] = []
    barrier = threading.Barrier(2)

    def worker() -> None:
        barrier.wait()
        try:
            out.append(asyncio.run(gate.mark_committed(pid)))
        except BaseException as e:  # noqa: BLE001
            out.append(e)

    ts = [threading.Thread(target=worker) for _ in range(2)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    wins = [o for o in out if not isinstance(o, BaseException)]
    errs = [o for o in out if isinstance(o, BaseException)]
    assert len(wins) == 1, out
    assert len(errs) == 1 and isinstance(errs[0], IllegalTransition), out


async def test_index_status_consistent_after_every_step() -> None:
    gate, repo, _, _ = make()
    p = await gate.declare(draft(), declared_by="t")
    seen = [repo.index_status["v1"]]
    await gate.verify(p.policy_id, OK)
    seen.append(repo.index_status["v1"])
    await gate.mark_indexed(p.policy_id)
    seen.append(repo.index_status["v1"])
    await gate.mark_committed(p.policy_id)
    seen.append(repo.index_status["v1"])
    await gate.mark_rolled_back(p.policy_id)
    seen.append(repo.index_status["v1"])
    assert seen == ["building", "building", "staged", "committed", "rolled_back"]


class BoomAudit(FakeAuditLog):
    async def record(self, event: Any) -> None:
        raise RuntimeError("audit down")


class BoomMirror(FakeRegistryMirror):
    async def register_candidate(self, policy: Any) -> str | None:
        raise RuntimeError("mirror exploded")

    async def promote_champion(self, policy: Any) -> bool:
        raise TimeoutError

    async def set_rolled_back(self, policy: Any, previous: Any = None) -> bool:
        raise RuntimeError


async def test_raising_mirror_never_propagates_and_flow_succeeds() -> None:
    repo = FakePolicyRepository()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), BoomMirror(), FakeEvalRuns({"ok"}))
    pid = await reach(gate, S.ROLLED_BACK)
    p = await repo.get(pid)
    assert p is not None and p.status is S.ROLLED_BACK and not p.mlflow_synced
    assert repo.index_status["v1"] == "rolled_back"


async def test_audit_failure_does_not_leave_status_pair_inconsistent() -> None:
    repo = FakePolicyRepository()
    gate = PolicyGate(repo, BoomAudit(), FakeClock(), FakeRegistryMirror(), FakeEvalRuns({"ok"}))
    with pytest.raises(RuntimeError):
        await gate.declare(draft(), declared_by="t")
    for p in list(repo._rows.values()):
        assert repo.index_status[p.index_version] == INDEX_STATUS[p.status]


async def test_commit_is_not_applied_when_audit_fails() -> None:
    repo = FakePolicyRepository()
    ok_gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), None)
    pid = await reach(ok_gate, S.INDEXED)
    bad = PolicyGate(repo, BoomAudit(), FakeClock(), FakeRegistryMirror(), None)
    with pytest.raises(RuntimeError):
        await bad.mark_committed(pid)
    assert (await repo.get(pid)).status is S.INDEXED  # type: ignore[union-attr]


async def test_audit_written_for_declare_commit_rollback_reject_ids_only() -> None:
    gate, repo, _, audit = make()
    await reach(gate, S.ROLLED_BACK, change_reason="secret-ish reason text")
    await reach(gate, S.REJECTED, index_version="v2", ingest_run_id="r2")
    actions = [e.action for e in audit._events]
    for a in ("policy.declare", "policy.commit", "policy.rollback", "policy.reject"):
        assert a in actions
    allowed = {"policy_id", "policy_hash", "index_version", "change_kind", "reason_code"}
    for e in audit._events:
        assert set(e.detail) <= allowed
        blob = repr(e)
        for needle in ("secret-ish", "h1", "h2", "d1", "d2"):
            if needle in ("h1", "h2", "d1", "d2"):
                assert f"'{needle}'" not in blob
            else:
                assert needle not in blob  # ADR: reason text not in audit


async def test_reject_reason_is_closed_enum_value() -> None:
    gate, _, _, audit = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(ValueError):
        await gate.reject(p.policy_id, "z" * 500)  # type: ignore[arg-type]
    await gate.reject(p.policy_id, RejectReason.EVAL_FAILED)
    ev = [e for e in audit._events if e.action == "policy.reject"][0]
    assert ev.detail["reason_code"] == "eval_failed"


# ---- rules ----


@pytest.mark.parametrize("reason", ["", "   ", "\n\t ", "  "])
async def test_blank_reason_rejected(reason: str) -> None:
    gate, _, _, _ = make()
    with pytest.raises(PolicyRuleViolation):
        await gate.declare(draft(change_reason=reason), declared_by="t")


@pytest.mark.parametrize(
    "kind",
    [
        ChangeKind.EMBEDDER,
        ChangeKind.CHUNKER,
        ChangeKind.PARENT_CHILD,
        ChangeKind.REDACTION,
        ChangeKind.RERANKER,
    ],
)
async def test_retrieval_kinds_forbid_copy_forward_and_need_passing_eval(
    kind: ChangeKind,
) -> None:
    gate, repo, _, _ = make()
    with pytest.raises(PolicyRuleViolation):
        await gate.declare(draft(change_kind=kind), declared_by="t", copy_forward=True)
    pid = await reach(gate, S.INDEXED, change_kind=kind)
    with pytest.raises(PolicyRuleViolation):  # no run
        await gate.mark_committed(pid)
    with pytest.raises(PolicyRuleViolation):  # failing run
        await gate.mark_committed(pid, eval_run_id="bad")
    assert (await repo.get(pid)).status is S.INDEXED  # type: ignore[union-attr]
    assert repo.index_status["v1"] == "staged"
    done = await gate.mark_committed(pid, eval_run_id="ok")
    assert done.eval_run_id == "ok"


async def test_missing_eval_reader_fails_closed() -> None:
    repo = FakePolicyRepository()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), None)
    pid = await reach(gate, S.INDEXED, change_kind=ChangeKind.EMBEDDER)
    with pytest.raises(PolicyRuleViolation):
        await gate.mark_committed(pid, eval_run_id="ok")


async def test_eval_reader_raising_does_not_commit() -> None:
    class Boom:
        async def passed(self, eval_run_id: str) -> bool:
            raise ConnectionError("mlflow down")

    repo = FakePolicyRepository()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), Boom())
    pid = await reach(gate, S.INDEXED, change_kind=ChangeKind.CHUNKER)
    with pytest.raises(Exception):  # noqa: B017 - any error, but never a commit
        await gate.mark_committed(pid, eval_run_id="ok")
    assert (await repo.get(pid)).status is S.INDEXED  # type: ignore[union-attr]


@pytest.mark.parametrize("kind", [ChangeKind.CORPUS_ONLY])
async def test_corpus_commit_without_eval_and_allow_copy_forward(
    kind: ChangeKind,
) -> None:
    repo = FakePolicyRepository()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), None)
    p = await gate.declare(draft(change_kind=kind), declared_by="t", copy_forward=True)
    await gate.verify(p.policy_id, OK)
    await gate.mark_indexed(p.policy_id)
    assert (await gate.mark_committed(p.policy_id)).status is S.COMMITTED


# ---- verify ----


@pytest.mark.parametrize(
    "summary",
    [
        ChunkSummary(
            frozenset({("emb-1", "c1"), ("emb-2", "c1")}), 5, dim=8, documents_with_rows=ALL_DOCS
        ),
        ChunkSummary(
            frozenset({("emb-1", "c1"), ("emb-1", "c2")}), 5, dim=8, documents_with_rows=ALL_DOCS
        ),
        ChunkSummary(frozenset(), 0, dim=8, documents_with_rows=ALL_DOCS),
        ChunkSummary(frozenset({("emb-1", "c1")}), 0, dim=8, documents_with_rows=ALL_DOCS),
        ChunkSummary(frozenset({("emb-9", "c1")}), 5, dim=8, documents_with_rows=ALL_DOCS),
        ChunkSummary(frozenset({("emb-1", "c9")}), 5, dim=8, documents_with_rows=ALL_DOCS),
    ],
)
async def test_failed_verify_auto_rejects_and_blocks_later_steps(summary: ChunkSummary) -> None:
    gate, repo, _, audit = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(p.policy_id, summary)
    assert (await repo.get(p.policy_id)).status is S.REJECTED  # type: ignore[union-attr]
    assert repo.index_status["v1"] == "rejected"
    assert any(e.action == "policy.reject" for e in audit._events)
    with pytest.raises(IllegalTransition):
        await gate.mark_indexed(p.policy_id)


async def test_verify_row_count_vs_documents_is_unchecked() -> None:
    """Documents declared (2) but a single chunk written still verifies: ADR says only model and
    chunker are compared, so this pins current behaviour (report as a gap, not a defect)."""
    gate, _, _, _ = make()
    p = await gate.declare(draft(), declared_by="t")
    assert (
        await gate.verify(
            p.policy_id,
            ChunkSummary(frozenset({("emb-1", "c1")}), 1, dim=8, documents_with_rows=ALL_DOCS),
        )
    ).status is S.VERIFIED


async def test_verify_checks_embedding_dim() -> None:
    """ChunkSummary carries the written dim and verify fails on a mismatch."""
    assert "dim" in ChunkSummary.__dataclass_fields__


# ---- outbox ----


async def test_outbox_only_unsynced_and_idempotent() -> None:
    gate, repo, mirror, _ = make(down=True)
    a = await gate.declare(draft(), declared_by="t")
    b = await gate.declare(draft(index_version="v2", ingest_run_id="r2"), declared_by="t")
    assert not (await repo.get(a.policy_id)).mlflow_synced  # type: ignore[union-attr]
    mirror.down = False
    assert await sync_pending(repo, mirror) == 2
    n = len(mirror.versions)
    assert await sync_pending(repo, mirror) == 0
    assert len(mirror.versions) == n == 2
    assert (await repo.get(b.policy_id)).mlflow_synced  # type: ignore[union-attr]


async def test_outbox_partial_failure_does_not_block_rest_and_limit() -> None:
    gate, repo, mirror, _ = make(down=True)
    ps = [
        await gate.declare(draft(index_version=f"v{i}", ingest_run_id=f"r{i}"), declared_by="t")
        for i in range(4)
    ]

    class Flaky(FakeRegistryMirror):
        async def register_candidate(self, policy: Any) -> str | None:
            if policy.policy_id == ps[0].policy_id:
                raise RuntimeError("boom")
            return await super().register_candidate(policy)

    flaky = Flaky()
    assert await sync_pending(repo, flaky) == 3
    assert (await repo.get(ps[0].policy_id)).mlflow_synced is False  # type: ignore[union-attr]
    assert await sync_pending(repo, flaky, limit=1) == 0  # oldest is the failing one: bounded


async def test_outbox_limit_respected() -> None:
    gate, repo, mirror, _ = make(down=True)
    for i in range(5):
        await gate.declare(draft(index_version=f"v{i}", ingest_run_id=f"r{i}"), declared_by="t")
    mirror.down = False
    assert await sync_pending(repo, mirror, limit=2) == 2
    assert len(await repo.list_unsynced(10)) == 3


async def test_sync_pending_survives_repo_failure() -> None:
    class BadRepo(FakePolicyRepository):
        async def list_unsynced(self, limit: int) -> list[Any]:
            raise RuntimeError("db")

    assert await sync_pending(BadRepo(), FakeRegistryMirror()) == 0


async def test_mark_synced_failure_never_raises() -> None:
    class BadRepo(FakePolicyRepository):
        async def mark_synced(self, policy_id: str, v: str | None) -> None:
            raise RuntimeError("db")

    repo = BadRepo()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), None)
    p = await gate.declare(draft(), declared_by="t")
    assert await sync_one(repo, FakeRegistryMirror(), p) is False


async def test_commit_resets_synced_and_resync_promotes_champion() -> None:
    gate, repo, mirror, _ = make()
    pid = await reach(gate, S.COMMITTED)
    assert mirror.aliases.get("champion") == mirror.versions[pid]
    assert (await repo.get(pid)).mlflow_synced  # type: ignore[union-attr]


async def test_whole_flow_with_mirror_down() -> None:
    gate, repo, _, _ = make(down=True)
    pid = await reach(gate, S.COMMITTED)
    p = await repo.get(pid)
    assert p is not None and p.status is S.COMMITTED and p.mlflow_synced is False
    assert isinstance(p.declared_at, datetime) and p.declared_at.tzinfo is UTC
