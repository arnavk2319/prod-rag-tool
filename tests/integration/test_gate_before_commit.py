"""T-011: the data gate runs before the alias moves; a failing gate leaves the alias alone."""

import psycopg
import pytest
from index_version_helpers import make_service
from ingest_helpers import make_stack, q
from test_ingest_pipeline import _ingest, dsn  # noqa: F401  (fixture)

from rag_platform.domain.errors import Conflict
from rag_platform.domain.policy import PolicyRuleViolation
from rag_platform.services.index_versions import GateFailed

pytestmark = pytest.mark.integration
ALIAS = "SELECT index_version, previous_version FROM index_alias"
LIVE_ROWS = (
    "SELECT count(*) FROM chunks WHERE index_version = (SELECT index_version FROM index_alias)"  # noqa: E501
)


async def test_passing_gate_commits_and_records_gate_run(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    res = await svc.commit(v1)
    assert res.passed
    assert res.metrics["self_retrieval"] >= 0.98 and res.metrics["vector_violations"] == 0
    assert q(dsn, ALIAS) == [(v1, None)]
    assert q(dsn, "SELECT passed, kind FROM gate_runs") == [(True, "index_data_gate")]
    assert q(dsn, "SELECT gate_run_id FROM index_versions WHERE index_version=%s", (v1,))[0][0]
    assert q(dsn, "SELECT status FROM pipeline_policies") == [("committed",)]
    assert q(dsn, "SELECT status FROM index_versions") == [("committed",)]
    with pytest.raises(PolicyRuleViolation):  # no longer indexed: refused, no new gate run
        await svc.commit(v1)
    assert q(dsn, "SELECT count(*) FROM gate_runs")[0][0] == 1


async def test_failed_gate_leaves_alias_and_old_version_serving(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    before = q(dsn, ALIAS)
    rows_before = q(dsn, LIVE_ROWS)
    v2 = await _ingest(s, "b.txt", "bravo")
    # a parent vanishes from the staged version (superuser, as a corrupted build would)
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute(
            "DELETE FROM parent_chunks WHERE ctid = (SELECT ctid FROM parent_chunks "
            "WHERE index_version=%s LIMIT 1)",
            (v2,),
        )
    with pytest.raises(GateFailed) as e:
        await svc.commit(v2)
    assert "orphan_children" in str(e.value) and "parent_row_parity" in str(e.value)
    assert q(dsn, ALIAS) == before  # alias before == after
    assert q(dsn, LIVE_ROWS) == rows_before  # queries still hit the old version
    assert q(dsn, "SELECT passed FROM gate_runs WHERE index_version=%s", (v2,)) == [(False,)]
    assert q(dsn, "SELECT status FROM index_versions WHERE index_version=%s", (v2,)) == [
        ("rejected",)
    ]


@pytest.mark.parametrize(
    "bad",
    ["zero", "tamper_rowcount"],
)
async def test_vector_and_row_parity_failures(dsn: str, bad: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    with psycopg.connect(dsn, autocommit=True) as c:
        if bad == "zero":
            c.execute(
                "UPDATE chunks SET embedding = array_fill(0, ARRAY[384])::vector "
                "WHERE ctid = (SELECT ctid FROM chunks WHERE index_version=%s LIMIT 1)",
                (v1,),
            )
        else:
            c.execute("UPDATE index_versions SET row_count = row_count + 1")
    with pytest.raises(GateFailed):
        await svc.commit(v1)
    assert q(dsn, "SELECT count(*) FROM index_alias")[0][0] == 0


async def test_concurrent_reader_never_sees_half_state(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    seen: list[tuple[object, object]] = []

    async def reader() -> None:  # second connection, while the swap transaction is open
        async with await psycopg.AsyncConnection.connect(dsn) as c:
            a = await (await c.execute("SELECT index_version FROM index_alias")).fetchall()
            p = await (await c.execute("SELECT status FROM pipeline_policies")).fetchall()
            seen.append((a, p))

    svc = make_service(s, on_swap=reader)
    v1 = await _ingest(s, "a.txt", "alpha")
    await svc.commit(v1)
    assert seen == [([], [("indexed",)])]  # old alias and old policy state together
    assert q(dsn, ALIAS) == [(v1, None)]
    assert q(dsn, "SELECT status FROM pipeline_policies") == [("committed",)]


async def test_swap_error_rolls_back_policy_and_alias(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)

    async def boom() -> None:
        raise RuntimeError("crash mid swap")

    svc = make_service(s, on_swap=boom)
    v1 = await _ingest(s, "a.txt", "alpha")
    with pytest.raises(RuntimeError):
        await svc.commit(v1)
    assert q(dsn, "SELECT count(*) FROM index_alias")[0][0] == 0
    assert q(dsn, "SELECT status FROM pipeline_policies") == [("indexed",)]
    assert q(dsn, "SELECT status FROM index_versions") == [("staged",)]


async def test_stale_base_is_refused(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    svc = make_service(s)
    v1 = await _ingest(s, "a.txt", "alpha")
    s.base[0] = None
    v2 = await _ingest(s, "b.txt", "bravo")  # built without a base
    await svc.commit(v1)
    with pytest.raises(Conflict):
        await svc.commit(v2)
    assert q(dsn, ALIAS) == [(v1, None)]


async def test_commit_fails_closed_without_rls_bypassing_connection(dsn: str) -> None:  # noqa: F811
    from psycopg.conninfo import conninfo_to_dict, make_conninfo

    from rag_platform.adapters.sql_index_versions import SqlIndexVersionStore

    s = make_stack(dsn)
    v1 = await _ingest(s, "a.txt", "alpha")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("CREATE ROLE gate_login LOGIN PASSWORD 'pw' IN ROLE rag_app")
    login = make_conninfo(**{**conninfo_to_dict(dsn), "user": "gate_login", "password": "pw"})
    svc = make_service(s)
    svc._store = SqlIndexVersionStore(login)  # type: ignore[attr-defined]  # no admin_dsn
    with pytest.raises(RuntimeError, match="RLS-bypassing"):
        await svc.commit(v1)
    assert q(dsn, "SELECT count(*) FROM index_alias")[0][0] == 0
    assert q(dsn, "SELECT status FROM pipeline_policies") == [("indexed",)]
    assert q(dsn, "SELECT count(*) FROM gate_runs")[0][0] == 0
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("DROP ROLE gate_login")
