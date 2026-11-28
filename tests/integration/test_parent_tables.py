# ruff: noqa: SIM117
"""Migration 0002 (ADR-0005): parent_chunks, chunks.parent_id, RLS, grants, partitions.

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

from rag_platform.services.ingest.copy_forward import COPY_FORWARD_STATEMENTS

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")


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


def _app(d: str, tenant: str) -> psycopg.Connection:
    c = psycopg.connect(d)
    c.execute("SET ROLE rag_app")
    c.execute("SELECT set_config('app.tenant_id', %s, true)", (tenant,))
    return c


def _tables(d: str) -> set[str]:
    with psycopg.connect(d) as c:
        rows = c.execute("SELECT tablename FROM pg_tables WHERE schemaname='public'").fetchall()
    return {r[0] for r in rows}


def test_partition_function_creates_parent_partition_idempotently(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v1')")  # idempotent
        c.execute("SELECT create_chunks_partition('v-2.x')")
    assert {"chunks_v1", "parent_chunks_v1", "parent_chunks_v_2_x"} <= _tables(dsn)
    with psycopg.connect(dsn) as c:
        parent_idx = {
            r[0]
            for r in c.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename='parent_chunks_v1'"
            ).fetchall()
        }
        assert any("(tenant_id)" in d for d in parent_idx)
        assert any("gin" in d and "acl_labels" in d for d in parent_idx)
        assert not any("hnsw" in d or "tsv" in d for d in parent_idx)
        n = c.execute(
            "SELECT count(*) FROM pg_indexes WHERE tablename='chunks_v1' AND indexdef LIKE '%hnsw%'"
        ).fetchone()
        assert n == (1,)
        assert c.execute(
            "SELECT is_nullable FROM information_schema.columns "
            "WHERE table_name='chunks' AND column_name='parent_id'"
        ).fetchone() == ("NO",)


def test_existing_partitions_gain_parent_partition_on_upgrade(dsn: str) -> None:
    command.upgrade(_cfg(), "0001")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v7')")
    command.upgrade(_cfg(), "head")
    assert "parent_chunks_v7" in _tables(dsn)


def test_upgrade_refuses_when_chunks_not_empty(dsn: str) -> None:
    command.upgrade(_cfg(), "0001")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute(
            "INSERT INTO chunks (chunk_id,index_version,tenant_id,doc_id,ordinal,text,span_start,"
            "span_end,acl_labels,effective_date,embedding_model,chunker_version,embedding) "
            "VALUES ('c','v1','t','d',0,'x',0,1,'{a}',current_date,'m','c1',%s::vector)",
            (vec(),),
        )
    with pytest.raises(Exception, match="re-index required"):
        command.upgrade(_cfg(), "head")


def test_rls_enabled_and_policy_text_matches_chunks(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn) as c:
        assert c.execute(
            "SELECT relrowsecurity FROM pg_class WHERE relname='parent_chunks'"
        ).fetchone() == (True,)
        pol = dict(
            c.execute(
                "SELECT tablename, qual || ' / ' || with_check FROM pg_policies "
                "WHERE policyname='tenant_isolation' AND tablename IN ('chunks','parent_chunks')"
            ).fetchall()
        )
    assert pol["chunks"] == pol["parent_chunks"]


def test_app_role_privileges_on_parents(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
    with _app(dsn, "bank-a") as c:
        add_parent(c, "bank-a", "p1")
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (1,)
    with _app(dsn, "bank-a") as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("UPDATE parent_chunks SET text='x'")
    with _app(dsn, "bank-a") as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            c.execute("SELECT count(*) FROM parent_chunks_v1")  # no grants on partitions
    with _app(dsn, "bank-a") as c:
        with pytest.raises(psycopg.errors.InsufficientPrivilege):
            add_parent(c, "bank-b", "px")  # RLS WITH CHECK (SQLSTATE 42501)
    with _app(dsn, "bank-a") as c:
        c.execute("DELETE FROM parent_chunks")  # delete is granted (delete and re-index)
        assert c.execute("SELECT count(*) FROM parent_chunks").fetchone() == (0,)


def test_forged_parent_id_resolves_to_nothing_across_tenants(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        add_chunk(c, "bank-b", "cb")  # tenant B child + its parent "p-cb"
        add_chunk(c, "bank-a", "ca", parent_id="p-cb")  # A's child forged to point at B's parent
        add_chunk(c, "bank-a", "ok")  # honest child of A
    join = (
        "SELECT c.chunk_id, p.parent_id FROM chunks c JOIN parent_chunks p "
        "ON p.index_version=c.index_version AND p.tenant_id=c.tenant_id "
        "AND p.parent_id=c.parent_id "
        "ORDER BY 1"
    )
    with _app(dsn, "bank-a") as c:
        assert c.execute(join).fetchall() == [("ok", "p-ok")]  # forged child resolves to nothing
        assert c.execute("SELECT 1 FROM parent_chunks WHERE parent_id='p-cb'").fetchall() == []
    with _app(dsn, "bank-b") as c:
        assert c.execute(join).fetchall() == [("cb", "p-cb")]


def test_copy_forward_copies_parents_with_children(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute("SELECT create_chunks_partition('v2')")
        add_chunk(c, "bank-a", "ca")
        add_chunk(c, "bank-b", "cb")
    params = {"from_version": "v1", "to_version": "v2", "tenant_id": "bank-a", "doc_ids": ["d"]}
    with _app(dsn, "bank-a") as c:
        for stmt in COPY_FORWARD_STATEMENTS:
            c.execute(stmt, params)  # type: ignore[arg-type]
            c.execute(stmt, params)  # type: ignore[arg-type]  # idempotent (ON CONFLICT)
        assert c.execute(
            "SELECT count(*) FROM parent_chunks WHERE index_version='v2'"
        ).fetchone() == (1,)
        assert c.execute("SELECT parent_id FROM chunks WHERE index_version='v2'").fetchall() == [
            ("p-ca",)
        ]


def test_downgrade_to_0001_and_back(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
    command.downgrade(_cfg(), "0001")
    assert not {t for t in _tables(dsn) if t.startswith("parent_chunks")}
    with psycopg.connect(dsn, autocommit=True) as c:
        cols = {
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='chunks'"
            )
        }
        assert "parent_id" not in cols
        c.execute("SELECT create_chunks_partition('v5')")  # 0001 function body is back
        assert "parent_chunks_v5" not in _tables(dsn)
    command.upgrade(_cfg(), "head")
    assert "parent_chunks" in _tables(dsn)
    command.downgrade(_cfg(), "base")
    assert not {t for t in _tables(dsn) if t.startswith(("parent_chunks", "chunks"))}
