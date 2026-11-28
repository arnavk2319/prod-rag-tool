"""Copy-forward SQL: unchanged documents move to a new index version without re-embedding.

Parents are copied together with their children, in this order, in ONE transaction, so a child
never points at a parent that is missing in the target version. Both statements are constants
(no string formatting); values are bound parameters. They run under the app role, so RLS limits
them to the current tenant (`app.tenant_id`) on top of the explicit tenant predicate.

Parameters: `from_version`, `to_version`, `tenant_id`, `doc_ids` (list of text).
"""

COPY_FORWARD_PARENTS_SQL = """
INSERT INTO parent_chunks (
    index_version, tenant_id, parent_id, doc_id, doc_version, ordinal, text, token_count,
    acl_labels, heading_path, doc_hash, span_start, span_end, effective_date, expires_date,
    chunker_version)
SELECT %(to_version)s, tenant_id, parent_id, doc_id, doc_version, ordinal, text, token_count,
    acl_labels, heading_path, doc_hash, span_start, span_end, effective_date, expires_date,
    chunker_version
FROM parent_chunks
WHERE index_version = %(from_version)s AND tenant_id = %(tenant_id)s
    AND doc_id = ANY(%(doc_ids)s)
ON CONFLICT DO NOTHING
"""

COPY_FORWARD_CHUNKS_SQL = """
INSERT INTO chunks (
    chunk_id, index_version, tenant_id, doc_id, doc_version, ordinal, text, span_start, span_end,
    section, page, acl_labels, effective_date, expires_date, embedding_model, chunker_version,
    embedding, parent_id)
SELECT chunk_id, %(to_version)s, tenant_id, doc_id, doc_version, ordinal, text, span_start,
    span_end, section, page, acl_labels, effective_date, expires_date, embedding_model,
    chunker_version, embedding, parent_id
FROM chunks
WHERE index_version = %(from_version)s AND tenant_id = %(tenant_id)s
    AND doc_id = ANY(%(doc_ids)s)
ON CONFLICT DO NOTHING
"""

COPY_FORWARD_STATEMENTS = (COPY_FORWARD_PARENTS_SQL, COPY_FORWARD_CHUNKS_SQL)
