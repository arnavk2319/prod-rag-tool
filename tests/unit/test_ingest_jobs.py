"""T-010 unit checks that need no database."""

import pytest
from pydantic import ValidationError

from rag_platform.adapters.fake_job_queue import FakeJobQueue
from rag_platform.services.ingest.jobs import IngestJob, doc_id_for, job_id_for
from rag_platform.services.policy_gate import ChunkSummary

SHA = "a" * 64


def _job(**kw: object) -> IngestJob:
    base: dict[str, object] = {
        "job_id": job_id_for("t1", SHA),
        "tenant_id": "t1",
        "raw_sha256": SHA,
        "filename": "a.txt",
        "acl_labels": ("staff",),
    }
    return IngestJob(**{**base, **kw})  # type: ignore[arg-type]


def test_job_id_is_stable_per_tenant_and_bytes() -> None:
    assert job_id_for("t1", SHA) == job_id_for("t1", SHA)
    assert job_id_for("t1", SHA) != job_id_for("t2", SHA)
    assert doc_id_for(SHA) == "doc-" + "a" * 16


@pytest.mark.parametrize(
    "bad", [{"raw_sha256": "xyz"}, {"acl_labels": ()}, {"tenant_id": ""}, {"job_id": "../x"}]
)
def test_job_message_is_validated(bad: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        _job(**bad)


def test_chunk_summary_dim_and_documents_are_mandatory() -> None:
    with pytest.raises(TypeError):
        ChunkSummary(frozenset({("m", "c")}), 1)  # type: ignore[call-arg]


async def test_fake_queue_redelivers_unacked_and_not_acked() -> None:
    q = FakeJobQueue()
    await q.publish(_job())
    first = await q.claim("w1", min_idle_ms=60_000)
    assert first is not None
    assert await q.claim("w2", min_idle_ms=60_000) is None  # in flight, not stale
    again = await q.claim("w2", min_idle_ms=0)  # crashed consumer's message is reclaimed
    assert again is not None and again.message_id == first.message_id
    await q.ack(again.message_id)
    assert await q.claim("w2", min_idle_ms=0) is None


async def test_attempt_numbers_and_version_names() -> None:
    from rag_platform.services.ingest.jobs import index_version_for

    assert index_version_for("j" * 8) == "iv-" + "j" * 8
    assert index_version_for("j" * 8, 2) == "iv-" + "j" * 8 + "-a2"


async def test_verify_allows_empty_version_only_when_no_documents_expected() -> None:
    from rag_platform.adapters.fakes import FakeAuditLog, FakeClock
    from rag_platform.adapters.policy_fakes import FakePolicyRepository, FakeRegistryMirror
    from rag_platform.domain.policy import (
        ChangeKind,
        PolicyDocument,
        PolicyDraft,
        PolicyRuleViolation,
    )
    from rag_platform.services.policy_gate import PolicyGate

    def draft(docs: tuple[PolicyDocument, ...], v: str) -> PolicyDraft:
        return PolicyDraft(
            change_kind=ChangeKind.CORPUS_ONLY, change_reason="r", ingest_run_id=v,
            index_version=v, embedder_id="e", embedder_version="1", embedder_dim=8,
            chunker_version="c", chunk_tokens=1, chunk_overlap=0, redaction_rules_version="r",
            prompt_version="p", generator_chain_ref="g", documents=docs,
        )  # fmt: skip

    gate = PolicyGate(FakePolicyRepository(), FakeAuditLog(), FakeClock(), FakeRegistryMirror())
    empty = ChunkSummary(frozenset(), 0, dim=-1, documents_with_rows=frozenset())
    ok = await gate.declare(draft((), "v-empty"), declared_by="t")
    await gate.verify(ok.policy_id, empty)
    doc = PolicyDocument(tenant_id="t", doc_id="d", doc_version=1, doc_hash="h")
    bad = await gate.declare(draft((doc,), "v-docs"), declared_by="t")
    with pytest.raises(PolicyRuleViolation):
        await gate.verify(bad.policy_id, empty)
