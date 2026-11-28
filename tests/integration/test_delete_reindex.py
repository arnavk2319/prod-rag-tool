"""T-010: delete a document, re-index; the next version holds none of its chunks."""

import pytest
from ingest_helpers import TENANT, make_stack, q, text_for
from test_ingest_pipeline import _ingest, dsn  # noqa: F401  (fixture)

pytestmark = pytest.mark.integration


async def test_deleted_doc_has_zero_chunks_in_next_version_and_raw_removed(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    await _ingest(s, "a.txt", "alpha")
    v1 = await _ingest(s, "b.txt", "bravo")
    doc_a = str(q(dsn, "SELECT doc_id FROM documents WHERE title='a.txt'")[0][0])
    sha_a = str(q(dsn, "SELECT doc_hash FROM documents WHERE doc_id=%s", (doc_a,))[0][0])
    embedded_before = len(s.embedder.texts)

    await s.pipeline.delete_document(TENANT, doc_a)
    assert await s.store.get(TENANT, sha_a) is None  # raw object removed
    v2 = await s.pipeline.reindex("rebuild-1")
    assert v2 != v1
    assert (
        q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s AND doc_id=%s", (v2, doc_a))[0][
            0
        ]
        == 0
    )
    assert q(dsn, "SELECT count(*) FROM parent_chunks WHERE index_version=%s AND doc_id=%s",
             (v2, doc_a))[0][0] == 0  # fmt: skip
    assert q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s", (v2,))[0][0] > 0
    assert len(s.embedder.texts) == embedded_before  # reindex copied forward, embedded nothing
    assert (
        q(dsn, "SELECT count(*) FROM chunks WHERE index_version=%s AND doc_id=%s", (v1, doc_a))[0][
            0
        ]
        > 0
    )  # the previous version is untouched until the alias moves (T-011)


async def test_reupload_after_delete_reactivates_and_reingests(dsn: str) -> None:  # noqa: F811
    s = make_stack(dsn)
    await _ingest(s, "a.txt", "alpha")
    doc_a = str(q(dsn, "SELECT doc_id FROM documents")[0][0])
    await s.pipeline.delete_document(TENANT, doc_a)
    again = await s.upload("a.txt", text_for("alpha"))
    assert not again.duplicate  # job re-queued: the document was deleted
    await s.run_all()
    assert q(dsn, "SELECT status FROM documents WHERE doc_id=%s", (doc_a,)) == [("active",)]
    assert q(dsn, "SELECT count(*) FROM documents")[0][0] == 1
    assert q(dsn, "SELECT status FROM ingest_jobs") == [("done",)]
