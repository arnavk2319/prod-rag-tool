"""One test per ADR-0006 rule, against in-memory fakes."""

import asyncio
import subprocess
import sys
from typing import Any

import pytest
from pydantic import ValidationError

from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import (
    FakeEvalRuns,
    FakePolicyRepository,
    FakeRegistryMirror,
)
from rag_platform.domain.policy import (
    ChangeKind,
    IllegalTransition,
    PolicyDocument,
    PolicyDraft,
    PolicyRuleViolation,
    PolicyStatus,
    RejectReason,
    policy_hash,
)
from rag_platform.services.policy_gate import ChunkSummary, PolicyGate, sync_pending

ALL_DOCS = frozenset({("a", "d1"), ("b", "d2")})


OK = ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=8, documents_with_rows=ALL_DOCS)


def draft(**kw: Any) -> PolicyDraft:
    base: dict[str, Any] = {
        "change_kind": ChangeKind.CORPUS_ONLY,
        "change_reason": "monthly refresh",
        "ingest_run_id": "run-1",
        "index_version": "v1",
        "embedder_id": "emb-1",
        "embedder_version": "1",
        "embedder_dim": 8,
        "chunker_version": "c1",
        "chunk_tokens": 200,
        "chunk_overlap": 20,
        "redaction_rules_version": "r1",
        "prompt_version": "p1",
        "generator_chain_ref": "chain-1",
        "documents": (
            PolicyDocument(tenant_id="b", doc_id="d2", doc_version=1, doc_hash="h2"),
            PolicyDocument(tenant_id="a", doc_id="d1", doc_version=1, doc_hash="h1"),
        ),
    }
    base.update(kw)
    return PolicyDraft(**base)


def make(
    down: bool = False,
) -> tuple[PolicyGate, FakePolicyRepository, FakeRegistryMirror, FakeAuditLog]:
    repo, mirror, audit = FakePolicyRepository(), FakeRegistryMirror(down=down), FakeAuditLog()
    gate = PolicyGate(repo, audit, FakeClock(), mirror, FakeEvalRuns({"ok"}))
    return gate, repo, mirror, audit


async def to_indexed(gate: PolicyGate, d: PolicyDraft) -> str:
    p = await gate.declare(d, declared_by="t")
    await gate.verify(p.policy_id, OK)
    await gate.mark_indexed(p.policy_id)
    return p.policy_id


async def test_reason_and_kind_mandatory() -> None:
    gate, *_ = make()
    with pytest.raises(PolicyRuleViolation):
        await gate.declare(draft(change_reason="   "), declared_by="t")
    with pytest.raises(ValidationError):
        PolicyDraft(**{k: v for k, v in draft().model_dump().items() if k != "change_kind"})


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
async def test_rebuild_kinds_forbid_copy_forward(kind: ChangeKind) -> None:
    gate, *_ = make()
    with pytest.raises(PolicyRuleViolation):
        await gate.declare(draft(change_kind=kind), declared_by="t", copy_forward=True)


@pytest.mark.parametrize("kind", [ChangeKind.CORPUS_ONLY, ChangeKind.PROMPT])
async def test_corpus_and_prompt_may_copy_forward(kind: ChangeKind) -> None:
    gate, *_ = make()
    p = await gate.declare(draft(change_kind=kind), declared_by="t", copy_forward=True)
    assert p.status is PolicyStatus.DECLARED


@pytest.mark.parametrize(
    "summary",
    [
        ChunkSummary(frozenset(), 0, dim=8, documents_with_rows=ALL_DOCS),
        ChunkSummary(
            frozenset({("emb-1", "c1"), ("emb-2", "c1")}), 4, dim=8, documents_with_rows=ALL_DOCS
        ),
        ChunkSummary(
            frozenset({("emb-1", "c1"), ("emb-1", "c0")}), 4, dim=8, documents_with_rows=ALL_DOCS
        ),
        ChunkSummary(frozenset({("other", "c1")}), 4, dim=8, documents_with_rows=ALL_DOCS),
        ChunkSummary(frozenset({("emb-1", "c0")}), 4, dim=8, documents_with_rows=ALL_DOCS),
    ],
)
async def test_verify_fails_and_rejects(summary: ChunkSummary) -> None:
    gate, repo, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(p.policy_id, summary)
    got = await repo.get(p.policy_id)
    assert got is not None and got.status is PolicyStatus.REJECTED
    assert repo.index_status["v1"] == "rejected"


async def test_commit_refused_unless_indexed() -> None:
    gate, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(IllegalTransition):
        await gate.mark_committed(p.policy_id)
    await gate.verify(p.policy_id, OK)
    with pytest.raises(IllegalTransition):
        await gate.mark_committed(p.policy_id)


async def test_retrieval_change_needs_passing_eval() -> None:
    gate, repo, *_ = make()
    pid = await to_indexed(gate, draft(change_kind=ChangeKind.EMBEDDER))
    with pytest.raises(PolicyRuleViolation):
        await gate.mark_committed(pid)  # no eval run
    with pytest.raises(PolicyRuleViolation):
        await gate.mark_committed(pid, eval_run_id="failed")
    done = await gate.mark_committed(pid, eval_run_id="ok")
    assert done.status is PolicyStatus.COMMITTED and done.eval_run_id == "ok"
    assert repo.index_status["v1"] == "committed"


async def test_corpus_only_commits_without_eval() -> None:
    gate, *_ = make()
    pid = await to_indexed(gate, draft())
    assert (await gate.mark_committed(pid)).status is PolicyStatus.COMMITTED


async def test_illegal_transitions_raise() -> None:
    gate, *_ = make()
    pid = await to_indexed(gate, draft())
    with pytest.raises(IllegalTransition):
        await gate.mark_rolled_back(pid)  # indexed -> rolled_back
    await gate.mark_committed(pid)
    with pytest.raises(IllegalTransition):
        await gate.reject(pid, RejectReason.OPERATOR_REJECTED)
    await gate.mark_rolled_back(pid)
    with pytest.raises(IllegalTransition):
        await gate.mark_indexed(pid)


async def test_atomic_transition_rejects_stale_expectation() -> None:
    gate, repo, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(IllegalTransition):
        await repo.transition(
            p.policy_id,
            expected_status=PolicyStatus.DECLARED,
            expected_index_status="staged",  # index_versions row disagrees
            new_status=PolicyStatus.VERIFIED,
            at=FakeClock().now(),
        )
    again = await repo.get(p.policy_id)
    assert again is not None and again.status is PolicyStatus.DECLARED


async def test_concurrent_transitions_only_one_wins() -> None:
    gate, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    res = await asyncio.gather(
        gate.verify(p.policy_id, OK), gate.verify(p.policy_id, OK), return_exceptions=True
    )
    assert sum(isinstance(r, IllegalTransition) for r in res) == 1


async def test_audit_has_ids_and_hashes_only() -> None:
    gate, _, _, audit = make()
    pid = await to_indexed(gate, draft(change_reason="secret-looking reason text"))
    await gate.mark_committed(pid)
    await gate.mark_rolled_back(pid)
    evs = await audit.list_events("_system")
    assert [e.action for e in evs] == ["policy.declare", "policy.commit", "policy.rollback"]
    assert "secret-looking" not in repr(evs)
    assert all(e.detail["policy_hash"] for e in evs)


def test_hash_changes_with_each_field_and_is_stable() -> None:
    base = draft()
    assert policy_hash(base) == policy_hash(draft())
    assert policy_hash(base) == policy_hash(draft(documents=tuple(reversed(base.documents))))
    changes: dict[str, Any] = {
        "change_kind": ChangeKind.PROMPT,
        "change_reason": "other",
        "parent_policy_id": "p0",
        "ingest_run_id": "run-2",
        "index_version": "v2",
        "embedder_id": "e2",
        "embedder_version": "2",
        "embedder_dim": 9,
        "chunker_version": "c2",
        "chunk_tokens": 201,
        "chunk_overlap": 21,
        "parent_tokens": 800,
        "redaction_rules_version": "r2",
        "reranker_id": "rr",
        "prompt_version": "p2",
        "generator_chain_ref": "chain-2",
        "documents": (PolicyDocument(tenant_id="a", doc_id="d1", doc_version=2, doc_hash="h1"),),
    }
    assert set(changes) == set(PolicyDraft.model_fields)
    for k, v in changes.items():
        assert policy_hash(draft(**{k: v})) != policy_hash(base), k


async def test_hash_ignores_lifecycle_fields() -> None:
    gate, repo, *_ = make()
    pid = await to_indexed(gate, draft(change_kind=ChangeKind.EMBEDDER))
    before = await repo.get(pid)
    done = await gate.mark_committed(pid, eval_run_id="ok")
    assert before is not None and done.policy_hash == before.policy_hash == policy_hash(
        draft(change_kind=ChangeKind.EMBEDDER)
    )


def test_hash_stable_across_hashseed() -> None:
    code = (
        "from tests.unit.test_policy_gate import draft;"
        "from rag_platform.domain.policy import policy_hash;"
        "print(policy_hash(draft()))"
    )
    outs = {
        subprocess.run(  # noqa: S603
            [sys.executable, "-c", code],
            env={"PYTHONHASHSEED": s, "PATH": ""},
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        for s in ("0", "1", "12345")
    }
    assert len(outs) == 1


async def test_mirror_down_does_not_block_lifecycle() -> None:
    gate, repo, mirror, _ = make(down=True)
    pid = await to_indexed(gate, draft())
    done = await gate.mark_committed(pid)
    assert done.status is PolicyStatus.COMMITTED
    got = await repo.get(pid)
    assert got is not None and not got.mlflow_synced
    mirror.down = False
    assert await sync_pending(repo, mirror, 10) == 1
    got = await repo.get(pid)
    assert (
        got is not None
        and got.mlflow_synced
        and mirror.aliases["champion"] == got.mlflow_model_version
    )


async def test_mirror_up_sets_candidate_then_champion() -> None:
    gate, repo, mirror, _ = make()
    p = await gate.declare(draft(), declared_by="t")
    assert mirror.aliases["candidate"] and (await repo.get(p.policy_id)).mlflow_synced  # type: ignore[union-attr]
    await gate.verify(p.policy_id, OK)
    await gate.mark_indexed(p.policy_id)
    await gate.mark_committed(p.policy_id)
    assert "champion" in mirror.aliases


async def test_verify_fails_on_dimension_mismatch() -> None:
    gate, repo, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(
            p.policy_id,
            ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=9, documents_with_rows=ALL_DOCS),
        )
    got = await repo.get(p.policy_id)
    assert got is not None and got.status is PolicyStatus.REJECTED


async def test_verify_passes_matching_dim_and_documents() -> None:
    gate, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    docs = frozenset({("a", "d1"), ("b", "d2")})
    ok = ChunkSummary(frozenset({("emb-1", "c1")}), 5, dim=8, documents_with_rows=docs)
    assert (await gate.verify(p.policy_id, ok)).status is PolicyStatus.VERIFIED


async def test_verify_fails_when_declared_document_has_no_rows() -> None:
    gate, *_ = make()
    p = await gate.declare(draft(), declared_by="t")
    s = ChunkSummary(
        frozenset({("emb-1", "c1")}), 5, dim=8, documents_with_rows=frozenset({("a", "d1")})
    )
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(p.policy_id, s)


@pytest.mark.parametrize("step", ["indexed", "commit", "rollback", "reject"])
async def test_failing_audit_leaves_status_unchanged(step: str) -> None:
    from tests.unit.test_policy_gate_matrix import BoomAudit

    repo = FakePolicyRepository()
    good = PolicyGate(repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), None)
    bad = PolicyGate(repo, BoomAudit(), FakeClock(), FakeRegistryMirror(), None)
    p = await good.declare(draft(), declared_by="t")
    pid = p.policy_id
    await good.verify(pid, OK)
    if step in ("commit", "rollback"):
        await good.mark_indexed(pid)
    if step == "rollback":
        await good.mark_committed(pid)
    before = await repo.get(pid)
    acts = {
        "indexed": lambda: bad.reject(pid, RejectReason.OPERATOR_REJECTED),
        "commit": lambda: bad.mark_committed(pid),
        "rollback": lambda: bad.mark_rolled_back(pid),
        "reject": lambda: bad.reject(pid, RejectReason.OPERATOR_REJECTED),
    }
    if True:
        with pytest.raises(RuntimeError):
            await acts[step]()
    assert (await repo.get(pid)) == before


async def test_prompt_change_needs_passing_eval_to_commit() -> None:
    gate, *_ = make()
    pid = await to_indexed(gate, draft(change_kind=ChangeKind.PROMPT))
    with pytest.raises(PolicyRuleViolation):
        await gate.mark_committed(pid)
    assert (await gate.mark_committed(pid, eval_run_id="ok")).status is PolicyStatus.COMMITTED


async def test_commit_audit_carries_eval_run_id_only() -> None:
    gate, _, _, audit = make()
    pid = await to_indexed(gate, draft(change_kind=ChangeKind.EMBEDDER))
    await gate.mark_committed(pid, eval_run_id="ok")
    ev = [e for e in await audit.list_events("_system") if e.action == "policy.commit"][0]
    assert ev.detail["eval_run_id"] == "ok"


async def test_retry_reuses_registered_version_and_registers_once() -> None:
    class FlakyPromote(FakeRegistryMirror):
        fails = 1

        async def promote_champion(self, policy: Any) -> bool:
            if self.fails:
                self.fails -= 1
                return False
            return await super().promote_champion(policy)

    repo, mirror = FakePolicyRepository(), FlakyPromote()
    gate = PolicyGate(repo, FakeAuditLog(), FakeClock(), mirror, None)
    pid = await to_indexed(gate, draft())
    await gate.mark_committed(pid)
    got = await repo.get(pid)
    assert got is not None and not got.mlflow_synced and got.mlflow_model_version
    assert await sync_pending(repo, mirror, 10) == 1
    assert mirror.register_calls == 1
    assert mirror.aliases["champion"] == mirror.versions[pid]


async def test_rollback_without_known_previous_stays_unsynced() -> None:
    gate, repo, mirror, _ = make()
    pid = await to_indexed(gate, draft())
    await gate.mark_committed(pid)
    await gate.mark_rolled_back(pid)
    got = await repo.get(pid)
    assert got is not None and not got.mlflow_synced
    assert "champion" in mirror.aliases  # never deleted on unknown history


async def test_unsynced_promote_in_committed_order() -> None:
    clock = FakeClock()
    repo, mirror = FakePolicyRepository(), FakeRegistryMirror(down=True)
    gate = PolicyGate(repo, FakeAuditLog(), clock, mirror, None)
    ids = []
    for i in (1, 2):  # declared in order 1, 2 but committed 2 then 1
        p = await gate.declare(draft(index_version=f"v{i}", ingest_run_id=f"r{i}"), declared_by="t")
        await gate.verify(p.policy_id, OK)
        await gate.mark_indexed(p.policy_id)
        ids.append(p.policy_id)
    for pid in (ids[1], ids[0]):
        clock.advance(1)
        await gate.mark_committed(pid)
    assert [p.policy_id for p in await repo.list_unsynced(10)] == [ids[1], ids[0]]
    mirror.down = False
    await sync_pending(repo, mirror, 10)
    assert mirror.aliases["champion"] == mirror.versions[ids[0]]  # last committed wins
