"""policy_violations (hard zero): committed index versions without a verified, intact policy.

Runs an in-memory scenario through the real PolicyGate, then audits the resulting store.
"""

import asyncio

from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
from rag_platform.adapters.policy_fakes import (
    FakeEvalRuns,
    FakePolicyRepository,
    FakeRegistryMirror,
)
from rag_platform.domain.policy import (
    ChangeKind,
    PolicyDraft,
    PolicyStatus,
    RejectReason,
    hash_matches,
)
from rag_platform.services.policy_gate import ChunkSummary, PolicyGate


def _draft(version: str, kind: ChangeKind) -> PolicyDraft:
    return PolicyDraft(
        change_kind=kind,
        change_reason="scenario",
        ingest_run_id=f"run-{version}",
        index_version=version,
        embedder_id="hash-64-v1",
        embedder_version="1",
        embedder_dim=64,
        chunker_version="c1",
        chunk_tokens=200,
        chunk_overlap=20,
        redaction_rules_version="r1",
        prompt_version="p1",
        generator_chain_ref="fake",
    )


async def audit_store(repo: FakePolicyRepository) -> int:
    """Count committed versions whose policy is missing, not verified or stale, and policies
    marked committed whose index version row is not committed."""
    bad = 0
    for version, status in sorted(repo.index_status.items()):
        if status != "committed":
            continue
        p = await repo.get_for_version(version)
        if (
            p is None
            or p.status is not PolicyStatus.COMMITTED
            or p.verified_at is None
            or not hash_matches(p)
        ):
            bad += 1
    for p in repo.all():  # the reverse direction: policy committed, index row disagrees
        if (
            p.status is PolicyStatus.COMMITTED
            and repo.index_status.get(p.index_version) != "committed"
        ):
            bad += 1
    return bad


async def _scenario(inject_violation: bool) -> int:
    repo = FakePolicyRepository()
    gate = PolicyGate(
        repo, FakeAuditLog(), FakeClock(), FakeRegistryMirror(), FakeEvalRuns({"eval-ok"})
    )
    ok = ChunkSummary(frozenset({("hash-64-v1", "c1")}), 3, dim=64, documents_with_rows=frozenset())
    for version, kind in (("v1", ChangeKind.EMBEDDER), ("v2", ChangeKind.CORPUS_ONLY)):
        p = await gate.declare(_draft(version, kind), declared_by="harness")
        await gate.verify(p.policy_id, ok)
        await gate.mark_indexed(p.policy_id)
        await gate.mark_committed(p.policy_id, eval_run_id="eval-ok")
    rej = await gate.declare(_draft("v3", ChangeKind.PROMPT), declared_by="harness")
    await gate.reject(rej.policy_id, RejectReason.OPERATOR_REJECTED)
    if inject_violation:
        repo.index_status["v-injected"] = "committed"  # committed with no policy at all
    return await audit_store(repo)


def evaluate(*, inject_violation: bool = False) -> dict[str, int]:
    return {"policy_violations": asyncio.run(_scenario(inject_violation))}
