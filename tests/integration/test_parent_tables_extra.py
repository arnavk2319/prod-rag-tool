# ruff: noqa: SIM117
"""Extra probes for migration 0002 and parent tenancy (T-044, test-writer).

Skipped unless TEST_DATABASE_URL points at a reachable throwaway Postgres with pgvector >= 0.8.
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pg_helpers import add_chunk, add_parent, vec

from rag_platform.services.ingest.chunker import chunk_document
from rag_platform.services.ingest.copy_forward import COPY_FORWARD_STATEMENTS

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")
DENIED = (psycopg.errors.InsufficientPrivilege, psycopg.errors.InternalError)


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
    yield d
    _reset(d)


def _app(d: str, tenant: str | None) -> psycopg.Connection:
    c = psycopg.connect(d)
    c.execute("SET ROLE rag_app")
    if tenant is not None:
        c.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    return c


def _fn(d: str) -> str:
    with psycopg.connect(d) as c:
        row = c.execute(
            "SELECT pg_get_functiondef('create_chunks_partition(text)'::regprocedure)"
        ).fetchone()
    assert row
    return str(row[0])


def _names(d: str, like: str) -> set[str]:
    with psycopg.connect(d) as c:
        rows = c.execute(
            "SELECT tablename FROM pg_tables WHERE schemaname='public' AND tablename LIKE %s",
            (like,),
        ).fetchall()
    return {r[0] for r in rows}


def test_upgrade_on_empty_chunks_and_roundtrip_is_idempotent(dsn: str) -> None:
    cfg = _cfg()
    command.upgrade(cfg, "0001")
    fn_0001 = _fn(dsn)
    command.upgrade(cfg, "head")
    fn_0002 = _fn(dsn)
    assert "parent_chunks" in fn_0002 and "parent_chunks" not in fn_0001
    command.downgrade(cfg, "0001")
    assert _fn(dsn) == fn_0001  # 0001 body restored byte for byte
    assert not _names(dsn, "parent_chunks%")
    command.upgrade(cfg, "head")
    assert _fn(dsn) == fn_0002
    command.downgrade(cfg, "0001")
    command.upgrade(cfg, "head")
    with psycopg.connect(dsn) as c:
        assert c.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='parent_id'"
        ).fetchone() == (1,)


def test_downgrade_with_data_and_partitions(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        add_chunk(c, "a", "c1")
    command.downgrade(_cfg(), "0001")
    assert not _names(dsn, "parent_chunks%")
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == (1,)  # children survive
        cols = [
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='chunks'"
            )
        ]
    assert "parent_id" not in cols


def test_existing_partitions_with_odd_names_get_parent_partitions(dsn: str) -> None:
    command.upgrade(_cfg(), "0001")
    with psycopg.connect(dsn, autocommit=True) as c:
        for v in ("v1", "idx 2026-01", "o'k"):
            c.execute("SELECT create_chunks_partition(%s)", (v,))
    command.upgrade(_cfg(), "head")
    assert {"parent_chunks_v1", "parent_chunks_idx_2026_01", "parent_chunks_o_k"} <= _names(
        dsn, "parent_chunks%"
    )
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v1')")


def test_parent_id_not_null_and_pk_scoped_by_version_and_tenant(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v2')")
        add_parent(c, "a", "p1", "v1")
        add_parent(c, "a", "p1", "v2")  # same id in another version
        add_parent(c, "b", "p1", "v1")  # same id in another tenant
        with pytest.raises(psycopg.errors.UniqueViolation):
            add_parent(c, "a", "p1", "v1")
        with pytest.raises(psycopg.errors.NotNullViolation):
            c.execute(
                "INSERT INTO chunks (chunk_id,index_version,tenant_id,doc_id,ordinal,text,"
                "span_start,span_end,acl_labels,effective_date,embedding_model,chunker_version,"
                "embedding) VALUES "
                "('x','v1','a','d',0,'t',0,1,'{a}',current_date,'m','c',%s::vector)",
                (vec(),),
            )


def test_destructive_and_partition_access_denied_for_app_role(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        add_parent(c, "a", "p1")
    for sql in (
        "TRUNCATE parent_chunks",
        "TRUNCATE parent_chunks_v1",
        "SELECT * FROM parent_chunks_v1",
        "DELETE FROM parent_chunks_v1",
        "INSERT INTO parent_chunks_v1 SELECT * FROM parent_chunks",
        "UPDATE parent_chunks SET text='x'",
    ):
        with _app(dsn, "a") as c:
            with pytest.raises(DENIED):
                c.execute(sql)  # type: ignore[arg-type]
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (1,)


def test_rls_cross_tenant_read_delete_insert_and_fail_closed(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        add_parent(c, "a", "pa")
        add_parent(c, "b", "pb")
    with _app(dsn, "a") as c:
        assert c.execute("SELECT parent_id FROM parent_chunks").fetchall() == [("pa",)]
        assert c.execute("SELECT 1 FROM parent_chunks WHERE tenant_id='b'").fetchall() == []
        cur = c.execute("DELETE FROM parent_chunks WHERE tenant_id='b'")
        assert cur.rowcount == 0  # invisible rows cannot be deleted
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            add_parent(c, "b", "forged")
    with _app(dsn, None) as c:  # no app.tenant_id at all: fail closed
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (0,)
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == (0,)
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            add_parent(c, "a", "pz")
    with _app(dsn, "") as c:  # empty tenant matches nothing
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (0,)
    with psycopg.connect(dsn) as c:
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (2,)


def test_forged_parent_in_other_version_joins_to_nothing(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v2')")
        add_parent(c, "a", "p-only-v1", "v1")
        add_chunk(c, "a", "ca", "v2", parent_id="p-only-v1")  # right tenant, wrong version
    join = (
        "SELECT count(*) FROM chunks c JOIN parent_chunks p ON p.index_version=c.index_version "
        "AND p.tenant_id=c.tenant_id AND p.parent_id=c.parent_id"
    )
    with _app(dsn, "a") as c:
        assert c.execute(join).fetchone() == (0,)


def _insert_doc(c: psycopg.Connection, tenant: str, doc: str, version: str, text: str) -> int:
    d = chunk_document(f"h-{doc}", text)
    for p in d.parents:
        c.execute(
            "INSERT INTO parent_chunks (index_version,tenant_id,parent_id,doc_id,ordinal,text,"
            "token_count,acl_labels,doc_hash,span_start,span_end,effective_date,chunker_version)"
            " VALUES (%s,%s,%s,%s,%s,%s,%s,'{a}',%s,%s,%s,current_date,%s)",
            (version, tenant, p.parent_id, doc, p.ordinal, p.text, p.tokens, f"h-{doc}",
             p.span[0], p.span[1], d.chunker_version),
        )  # fmt: skip
    for ch in d.children:
        c.execute(
            "INSERT INTO chunks (chunk_id,index_version,tenant_id,doc_id,ordinal,text,span_start,"
            "span_end,acl_labels,effective_date,embedding_model,chunker_version,embedding,"
            "parent_id) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'{a}',current_date,'m',%s,%s::vector,%s)",
            (ch.chunk_id, version, tenant, doc, ch.ordinal, ch.text, ch.span[0], ch.span[1],
             d.chunker_version, vec(), ch.parent_id),
        )  # fmt: skip
    return len(d.children)


def test_copy_forward_kept_docs_only_no_orphans_and_double_run(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    long_text = " ".join(f"w{i}" for i in range(3000))
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v2')")
        kept = _insert_doc(c, "a", "kept", "v1", long_text)
        _insert_doc(c, "a", "deleted", "v1", long_text + " extra tail")
        _insert_doc(c, "b", "kept", "v1", long_text)  # other tenant, same doc id
    params = {"from_version": "v1", "to_version": "v2", "tenant_id": "a", "doc_ids": ["kept"]}
    with _app(dsn, "a") as c:
        for _ in range(2):  # second run must not fail or duplicate
            for stmt in COPY_FORWARD_STATEMENTS:
                c.execute(stmt, params)  # type: ignore[arg-type]
        assert c.execute(
            "SELECT doc_id, tenant_id, count(*) FROM chunks WHERE index_version='v2' GROUP BY 1,2"
        ).fetchall() == [("kept", "a", kept)]
        assert c.execute(
            "SELECT DISTINCT doc_id, tenant_id FROM parent_chunks WHERE index_version='v2'"
        ).fetchall() == [("kept", "a")]
        n_parents = len(chunk_document("h-kept", long_text).parents)
        assert c.execute(
            "SELECT count(*) FROM parent_chunks WHERE index_version='v2'"
        ).fetchone() == (n_parents,)
        orphans = c.execute(
            "SELECT count(*) FROM chunks c LEFT JOIN parent_chunks p ON "
            "p.index_version=c.index_version AND p.tenant_id=c.tenant_id AND "
            "p.parent_id=c.parent_id WHERE c.index_version='v2' AND p.parent_id IS NULL"
        ).fetchone()
        assert orphans == (0,)
        for stmt in COPY_FORWARD_STATEMENTS:  # unknown doc ids copy nothing
            cur = c.execute(stmt, {**params, "doc_ids": ["nope"]})  # type: ignore[arg-type]
            assert cur.rowcount == 0
