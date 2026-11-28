"""T-010: object store contract, register, ingest, idempotency, copy-forward, kill, verify, rebuild.

Skipped unless TEST_DATABASE_URL points at a throwaway Postgres with pgvector >= 0.8.
"""

import os
import sys
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from ingest_helpers import (
    TENANT,
    CountingEmbedder,
    SimulatedKill,
    Stack,
    make_stack,
    q,
    text_for,
)

from rag_platform.adapters.fakes import FakeDocumentRepository
from rag_platform.adapters.sql_document_repository import SqlDocumentRepository
from rag_platform.adapters.sql_object_store import PostgresObjectStore
from rag_platform.domain.errors import ValidationFailed
from rag_platform.domain.policy import ChangeKind, PolicyStatus
from rag_platform.services.ingest.pipeline import PipelineConfig
from rag_platform.services.ingest.submit import submit_upload

pytestmark = pytest.mark.integration
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # tests/integration is not a package; the contract helper is
from tests.contract.ports_contract import check_object_store  # noqa: E402

WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")


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
    cfg = Config(str(ROOT / "alembic.ini"))
    cfg.set_main_option("script_location", str(ROOT / "migrations"))
    command.upgrade(cfg, "head")
    yield d
    _reset(d)


async def _ingest(s: Stack, name: str, tag: str) -> str:
    """Upload, run the worker, point the base at the new staged version; returns its name."""
    await s.upload(name, text_for(tag))
    assert await s.run_all() >= 1
    version = q(s.dsn, "SELECT index_version FROM index_versions ORDER BY created_at DESC LIMIT 1")
    s.base[0] = str(version[0][0])
    return s.base[0]


async def test_object_store_contract_and_quarantine(dsn: str) -> None:
    store = PostgresObjectStore(dsn)
    await check_object_store(store)
    sha = await store.put(TENANT, b"quarantine me", "text/plain")
    assert await store.status(TENANT, sha) == "raw"
    assert await store.mark_quarantined(TENANT, sha)
    assert await store.status(TENANT, sha) == "quarantined"
    assert await store.status("other", sha) is None


@pytest.mark.parametrize("backend", ["fake", "sql"])
async def test_register_outcomes_match_fake(dsn: str, backend: str) -> None:
    repo = FakeDocumentRepository() if backend == "fake" else SqlDocumentRepository(dsn)
    doc, outcome = await repo.register("t1", "h1", "d1", "a.txt")
    assert (doc.doc_id, outcome) == ("d1", "created")
    doc, outcome = await repo.register("t1", "h1", "d-other", "b.txt")
    assert (doc.doc_id, outcome) == ("d1", "duplicate")
    await repo.set_status("t1", "d1", "retired")
    doc, outcome = await repo.register("t1", "h1", "d-third", "c.txt")
    assert (doc.doc_id, doc.status, outcome) == ("d1", "active", "reactivated")
    _, outcome = await repo.register("t2", "h1", "d1", "a.txt")  # other tenant: its own row
    assert outcome == "created"


async def test_ingest_stages_version_with_lineage_columns(dsn: str) -> None:
    s = make_stack(dsn)
    version = await _ingest(s, "a.txt", "alpha")
    assert (
        q(dsn, "SELECT status, row_count FROM index_versions WHERE index_version=%s", (version,))[
            0
        ][0]
        == "staged"
    )
    pol = await s.policies.get_for_version(version)
    assert pol is not None and pol.status is PolicyStatus.INDEXED
    rows = q(
        dsn,
        "SELECT DISTINCT embedding_model, chunker_version, index_version FROM chunks "
        "WHERE index_version=%s",
        (version,),
    )
    assert rows == [(s.embedder.model_id, pol.chunker_version, version)]
    orphans = q(
        dsn,
        "SELECT count(*) FROM chunks c LEFT JOIN parent_chunks p "
        "ON p.index_version=c.index_version AND p.tenant_id=c.tenant_id "
        "AND p.parent_id=c.parent_id WHERE p.parent_id IS NULL",
    )
    assert orphans[0][0] == 0
    # children are embedded, parents are not: embedder saw one text per child chunk
    n_children = q(dsn, "SELECT count(*) FROM chunks")[0][0]
    assert len(s.embedder.texts) == n_children and n_children > 1
    assert s.embedder.calls == -(-n_children // 2)  # batches of 2
    assert q(dsn, "SELECT count(*) FROM index_alias")[0][0] == 0  # alias is T-011's job


async def test_resubmit_is_idempotent(dsn: str) -> None:
    s = make_stack(dsn)
    data = text_for("alpha")
    first = await s.upload("a.txt", data)
    again = await s.upload("a.txt", data)  # while still queued
    assert again.job_id == first.job_id
    await s.run_all()
    after = await s.upload("a.txt", data)  # after done
    assert after.duplicate and after.job_id == first.job_id
    assert await s.run_all() == 0  # nothing was published for the duplicate
    assert q(dsn, "SELECT count(*) FROM documents")[0][0] == 1
    assert q(dsn, "SELECT count(*) FROM ingest_jobs")[0][0] == 1
    assert q(dsn, "SELECT count(*) FROM raw_objects")[0][0] == 1
    assert q(dsn, "SELECT count(*) FROM index_versions")[0][0] == 1
    n = q(dsn, "SELECT count(*) FROM chunks")[0][0]
    assert n == len(s.embedder.texts)  # the duplicate message embedded nothing more
    assert q(dsn, "SELECT status FROM ingest_jobs")[0][0] == "done"


async def test_copy_forward_no_embedder_no_update_omits_retired(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "CREATE FUNCTION no_upd() RETURNS trigger LANGUAGE plpgsql AS "
            "$$ BEGIN RAISE EXCEPTION 'chunks are immutable'; END $$"
        )
        c.execute("CREATE TRIGGER chunks_no_update BEFORE UPDATE ON chunks "
                  "FOR EACH ROW EXECUTE FUNCTION no_upd()")  # fmt: skip
        c.execute("CREATE TRIGGER parents_no_update BEFORE UPDATE ON parent_chunks "
                  "FOR EACH ROW EXECUTE FUNCTION no_upd()")  # fmt: skip
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    embedded_a = len(s.embedder.texts)
    v2 = await _ingest(s, "b.txt", "bravo")
    embedded_b = len(s.embedder.texts) - embedded_a
    pol2 = await s.policies.get_for_version(v2)
    assert pol2 is not None and pol2.parent_policy_id is not None
    assert {d.doc_id for d in pol2.documents} and len(pol2.documents) == 2
    a_ids = lambda v: sorted(  # noqa: E731
        r[0] for r in q(dsn, "SELECT chunk_id FROM chunks WHERE index_version=%s", (v,))
    )
    doc_a = q(dsn, "SELECT doc_id FROM documents WHERE title='a.txt'")[0][0]
    from_v2 = sorted(
        r[0]
        for r in q(
            dsn, "SELECT chunk_id FROM chunks WHERE index_version=%s AND doc_id=%s", (v2, doc_a)
        )  # fmt: skip
    )
    assert from_v2 == a_ids(v1)  # same chunk ids and no embedder call for them
    assert embedded_b == q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s AND doc_id<>%s",
                           (v2, doc_a))[0][0]  # fmt: skip
    assert q(dsn, "SELECT count(*) FROM parent_chunks WHERE index_version=%s", (v2,))[0][0] > 1
    # previous version untouched
    assert len(a_ids(v1)) == embedded_a
    src = (ROOT / "src").rglob("*.py")
    assert not any("UPDATE chunks" in p.read_text() for p in src)


async def test_kill_mid_job_then_restart_completes_exactly_once(dsn: str) -> None:
    emb = CountingEmbedder()
    emb.kill_on_call = 2  # dies after the first batch is embedded, before anything is acked
    s = make_stack(dsn, emb)
    await s.upload("a.txt", text_for("alpha"))
    with pytest.raises(SimulatedKill):
        await s.worker.run_once()
    assert q(dsn, "SELECT status FROM ingest_jobs")[0][0] == "running"
    assert len(s.queue.pending_ids()) == 1 and not s.queue.acked
    emb.kill_on_call = None
    assert await s.worker.run_once() is True  # a restarted worker reclaims the pending message
    assert await s.worker.run_once() is False
    assert q(dsn, "SELECT status FROM ingest_jobs")[0][0] == "done"
    assert q(dsn, "SELECT count(*) FROM documents")[0][0] == 1
    assert q(dsn, "SELECT count(*) FROM index_versions")[0][0] == 1
    assert q(dsn, "SELECT count(*), count(DISTINCT chunk_id) FROM chunks")[0][0] > 0
    assert q(dsn, "SELECT status FROM pipeline_policies")[0][0] == "indexed"
    assert len(q(dsn, "SELECT 1 FROM pipeline_policies")) == 1
    assert s.queue.acked == ["1-0"] and not s.queue.pending_ids()


async def test_failed_verify_rejects_and_leaves_alias_and_previous_untouched(dsn: str) -> None:
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("INSERT INTO index_alias (alias, index_version) VALUES ('live', %s)", (v1,))
    before = q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v1,))[0][0]

    class Flaky(CountingEmbedder):
        reads = 0

        @property
        def model_id(self) -> str:  # declared id differs from the id written on the chunks
            Flaky.reads += 1
            return "model-declared" if Flaky.reads <= 2 else "model-written"

    bad = make_stack(dsn, Flaky())
    bad.base[0] = v1
    await bad.upload("b.txt", text_for("bravo"))
    await bad.run_all()
    assert [r[0] for r in q(dsn, "SELECT status FROM ingest_jobs ORDER BY created_at")] == [
        "done",
        "rejected",
    ]
    assert q(dsn, "SELECT error FROM ingest_jobs WHERE status='rejected'") == [("verify_failed",)]
    pol = [p for p in q(dsn, "SELECT index_version, status FROM pipeline_policies") if p[0] != v1]
    assert pol and pol[0][1] == "rejected"
    assert q(dsn, "SELECT status FROM index_versions WHERE index_version=%s", (pol[0][0],)) == [
        ("rejected",)
    ]
    assert q(dsn, "SELECT index_version, previous_version FROM index_alias") == [(v1, None)]
    assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v1,))[0][0] == before
    assert q(dsn, "SELECT status FROM index_versions WHERE index_version=%s", (v1,)) == [
        ("staged",)
    ]


async def test_embedder_change_forces_rebuild_without_copy_forward(dsn: str) -> None:
    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    emb2 = CountingEmbedder()
    s2 = make_stack(dsn, emb2, PipelineConfig(change_kind=ChangeKind.EMBEDDER))
    s2.base[0] = v1
    await s2.upload("b.txt", text_for("bravo"))
    await s2.run_all()
    v2 = q(dsn, "SELECT index_version FROM index_versions ORDER BY created_at DESC LIMIT 1")[0][0]
    pol = await s2.policies.get_for_version(str(v2))
    assert pol is not None and pol.change_kind is ChangeKind.EMBEDDER
    assert len(pol.documents) == 2
    total = q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v2,))[0][0]
    assert len(emb2.texts) == total  # every child re-embedded, none copied


async def test_pii_dense_upload_is_quarantined_in_the_object_store(dsn: str) -> None:
    s = make_stack(dsn)
    pii = " ".join(f"user{n}@example.com" for n in range(40)).encode()
    sub = await s.upload("pii.txt", pii)
    await s.run_all()
    assert q(dsn, "SELECT status, error FROM ingest_jobs") == [("quarantined", "pii_density_high")]
    assert q(dsn, "SELECT status FROM raw_objects") == [("quarantined",)]
    assert q(dsn, "SELECT reason FROM quarantine_items") == [("pii_density_high",)]
    assert q(dsn, "SELECT count(*) FROM documents")[0][0] == 0
    assert sub.raw_sha256


async def test_oversized_upload_is_refused_before_storing(dsn: str) -> None:
    s = make_stack(dsn)
    with pytest.raises(ValidationFailed):
        await submit_upload(
            store=s.store, jobs=s.jobs, queue=s.queue, docs=s.docs, tenant_id=TENANT,
            filename="big.txt", data=b"x" * 11, acl_labels=("staff",), max_bytes=10,
        )  # fmt: skip
    assert q(dsn, "SELECT count(*) FROM raw_objects")[0][0] == 0
