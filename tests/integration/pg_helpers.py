"""Shared inserts for integration tests: a chunk always gets its parent first (ADR-0005)."""

import psycopg


def vec(n: int = 384) -> str:
    return "[" + ",".join(["0.1"] * n) + "]"


def add_parent(
    c: psycopg.Connection, tenant: str, parent_id: str, version: str = "v1", doc: str = "d"
) -> None:
    c.execute(
        "INSERT INTO parent_chunks (index_version,tenant_id,parent_id,doc_id,ordinal,text,"
        "token_count,acl_labels,doc_hash,span_start,span_end,effective_date,chunker_version) "
        "VALUES (%s,%s,%s,%s,0,'hello world parent',3,'{a}','h',0,18,current_date,'c1')",
        (version, tenant, parent_id, doc),
    )


def add_chunk(
    c: psycopg.Connection,
    tenant: str,
    cid: str,
    version: str = "v1",
    parent_id: str | None = None,
    dim: int = 384,
) -> str:
    """Insert a parent (unless `parent_id` is given) and then the child; returns the parent id."""
    pid = parent_id or f"p-{cid}"
    if parent_id is None:
        add_parent(c, tenant, pid, version)
    c.execute(
        "INSERT INTO chunks (chunk_id,index_version,tenant_id,doc_id,ordinal,text,span_start,"
        "span_end,acl_labels,effective_date,embedding_model,chunker_version,embedding,parent_id) "
        "VALUES (%s,%s,%s,'d',0,'hello world',0,11,'{a}',current_date,'m','c1',%s::vector,%s)",
        (cid, version, tenant, vec(dim), pid),
    )
    return pid
