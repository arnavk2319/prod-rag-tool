"""T-011: rollback swaps the alias to parent_version; retention keeps 3 and the alias target."""

import pytest
from index_version_helpers import make_service
from ingest_helpers import make_stack, q
from test_ingest_pipeline import _ingest, dsn  # noqa: F401  (fixture)

from rag_platform.domain.errors import Conflict

pytestmark = pytest.mark.integration
ALIAS = "SELECT index_version, previous_version FROM index_alias"
PARTS = "SELECT count(*) FROM pg_inherits WHERE inhparent = %s::regclass"


async def test_rollback_restores_previous_results(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    old = q(dsn, "SELECT chunk_id FROM chunks WHERE index_version=%s ORDER BY chunk_id", (v1,))
    v2 = await _ingest(s, "b.txt", "bravo")
    await svc.commit(v2)
    assert q(dsn, ALIAS) == [(v2, v1)]
    assert await svc.rollback() == v1
    assert q(dsn, ALIAS) == [(v1, v2)]
    live = q(
        dsn,
        "SELECT chunk_id FROM chunks WHERE index_version=(SELECT index_version FROM index_alias) "
        "ORDER BY chunk_id",
    )
    assert live == old
    st = dict(q(dsn, "SELECT index_version, status FROM index_versions"))
    assert st == {v1: "committed", v2: "rolled_back"}
    pol = q(dsn, "SELECT status FROM pipeline_policies ORDER BY declared_at")
    assert pol == [("committed",), ("rolled_back",)]
    actions = [r[0] for r in q(dsn, "SELECT action FROM audit_log")]
    assert "policy.rollback" in actions and "policy.commit" in actions


async def test_rollback_without_parent_is_refused(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    with pytest.raises(Conflict):  # nothing live
        await svc.rollback()
    await svc.commit(await _ingest(s, "a.txt", "alpha"))
    with pytest.raises(Conflict):  # first version has no parent
        await svc.rollback()


async def test_retention_keeps_last_three_and_alias_target(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    versions = []
    for i in range(5):
        v = await _ingest(s, f"d{i}.txt", f"tag{i}")
        await svc.commit(v)
        versions.append(v)
    removed = await svc.retain()
    assert sorted(removed) == sorted(versions[:2])
    for v in versions[:2]:
        assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v,))[0][0] == 0
    for v in versions[2:]:
        assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v,))[0][0] > 0
    assert q(dsn, PARTS, ("chunks",))[0][0] == 3 and q(dsn, PARTS, ("parent_chunks",))[0][0] == 3
    assert q(dsn, ALIAS)[0][0] == versions[4]
    assert await svc.retain() == []  # idempotent


async def test_retention_never_drops_alias_target_after_rollbacks(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    vs = []
    for i in range(4):
        v = await _ingest(s, f"d{i}.txt", f"tag{i}")
        await svc.commit(v)
        vs.append(v)
    await svc.rollback()  # live = vs[2]
    await svc.retain(keep=1)
    live = q(dsn, ALIAS)[0][0]
    assert live == vs[2]
    assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (live,))[0][0] > 0
    assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (vs[3],))[0][0] > 0  # prev
    assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (vs[0],))[0][0] == 0
