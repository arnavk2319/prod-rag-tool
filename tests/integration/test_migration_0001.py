"""Migration 0001: schema contract, RLS, trace privacy, pgvector floor.

Skipped unless TEST_DATABASE_URL points at a reachable Postgres with pgvector >= 0.8
(a throwaway database; the test wipes and recreates the public schema).
"""

import os
from collections.abc import Iterator
from pathlib import Path

import psycopg
import pytest
from alembic import command
from alembic.config import Config
from pg_helpers import add_chunk

pytestmark = pytest.mark.integration

ROOT = Path(__file__).resolve().parents[2]
EXPECTED = {
    "documents", "ingest_jobs", "quarantine_items", "index_versions", "chunks",
    "index_alias", "chat_sessions", "chat_messages", "control_flags",
    "registry_pointers", "golden_sets", "gate_runs", "audit_log", "role_labels",
    "raw_objects", "request_traces",
}  # fmt: skip

WIPE = ("DROP " + "SCHEMA public CASCADE", "CREATE SCHEMA public")


def _dsn() -> str:
    url = os.environ.get("TEST_DATABASE_URL")
    if not url:
        pytest.skip("TEST_DATABASE_URL not set")
    return url.replace("postgresql+psycopg://", "postgresql://")


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
    d = _dsn()
    try:
        _reset(d)
    except psycopg.OperationalError:
        pytest.skip("TEST_DATABASE_URL unreachable")
    monkeypatch.setenv("DATABASE_URL", d.replace("postgresql://", "postgresql+psycopg://"))
    yield d
    _reset(d)


def _tables(d: str) -> set[str]:
    with psycopg.connect(d) as c:
        rows = c.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_schema='public'"
        ).fetchall()
    return {r[0] for r in rows}


def _add_chunk(c: psycopg.Connection, tenant: str, cid: str) -> None:
    add_chunk(c, tenant, cid)


def test_upgrade_downgrade_upgrade(dsn: str) -> None:
    cfg = _cfg()
    command.upgrade(cfg, "head")
    tables = _tables(dsn)
    assert tables >= EXPECTED
    assert "alert_events" not in tables
    command.downgrade(cfg, "base")
    assert not (EXPECTED & _tables(dsn))
    command.upgrade(cfg, "head")
    assert _tables(dsn) >= EXPECTED


def test_chunk_columns_partition_and_defaults(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn) as c:
        cols = {
            r[0]
            for r in c.execute(
                "SELECT column_name FROM information_schema.columns WHERE table_name='chunks'"
            )
        }
        assert {"embedding_model", "chunker_version", "index_version", "effective_date",
                "expires_date", "acl_labels", "tsv"} <= cols  # fmt: skip
        c.execute("SELECT create_chunks_partition('v1')")
        idx = {
            r[0] for r in c.execute("SELECT indexname FROM pg_indexes WHERE tablename='chunks_v1'")
        }
        assert "chunks_v1_hnsw_idx" in idx
        default = c.execute(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name='registry_pointers' AND column_name='canary_percent'"
        ).fetchone()
        assert default is not None and default[0] == "0"


def test_rls_blocks_other_tenant(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn, autocommit=True) as c:
        c.execute("SELECT create_chunks_partition('v1')")
        c.execute(
            "INSERT INTO index_versions (index_version, embedding_model, chunker_version) "
            "VALUES ('v1','m','c1')"
        )
        for t in ("bank-a", "bank-b"):
            _add_chunk(c, t, f"c-{t}")
            c.execute(
                "INSERT INTO documents (doc_id,tenant_id,title,doc_hash) VALUES ('d',%s,'t','h')",
                (t,),
            )
    with psycopg.connect(dsn) as c:
        c.execute("SET ROLE rag_app")
        c.execute("SET LOCAL app.tenant_id = 'bank-a'")
        for table in ("chunks", "documents"):
            rows = c.execute(f"SELECT tenant_id FROM {table}").fetchall()  # type: ignore[arg-type]  # noqa: S608
            assert rows == [("bank-a",)]
    with psycopg.connect(dsn) as c:
        c.execute("SET ROLE rag_app")
        assert c.execute("SELECT count(*) FROM chunks").fetchone() == (0,)  # no tenant: deny


def test_request_traces_has_no_text_columns(dsn: str) -> None:
    command.upgrade(_cfg(), "head")
    with psycopg.connect(dsn) as c:
        rows = c.execute(
            "SELECT column_name, data_type FROM information_schema.columns "
            "WHERE table_name='request_traces'"
        ).fetchall()
    names = {r[0] for r in rows}
    banned = ("question", "answer", "query", "prompt", "content")
    assert not {n for n in names if any(k in n for k in banned)}
    id_cols = {"trace_id", "message_id", "tenant_id", "outcome", "cache"}
    assert all(r[1] != "text" for r in rows if r[0] not in id_cols)


def test_refuses_old_pgvector(dsn: str) -> None:
    with psycopg.connect(dsn, autocommit=True) as c:
        avail = c.execute(
            "SELECT 1 FROM pg_available_extension_versions WHERE name='vector' AND version='0.7.4'"
        ).fetchone()
        if avail is None:
            pytest.skip("pgvector 0.7.x not installable here")
        c.execute("CREATE EXTENSION vector VERSION '0.7.4'")
    with pytest.raises(Exception, match="pgvector >= 0.8"):
        command.upgrade(_cfg(), "head")
