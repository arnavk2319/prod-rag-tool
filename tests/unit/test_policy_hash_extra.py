"""T-045 adversarial: canonical policy hash."""

import os
import subprocess
import sys
import unicodedata
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_platform.domain.policy import (
    ChangeKind,
    PipelinePolicy,
    PolicyDocument,
    PolicyDraft,
    canonical_json,
    hash_matches,
    policy_hash,
)
from tests.unit.test_policy_gate import draft

GOLDEN = "9c6a862df0363a29bc961c21678c37fabcd7cee04ae3665e61c22ca3b1a2a5c4"
ROOT = Path(__file__).resolve().parents[2]

CHANGES: dict[str, Any] = {
    "change_kind": ChangeKind.EMBEDDER,
    "change_reason": "other",
    "parent_policy_id": "pp",
    "ingest_run_id": "run-2",
    "index_version": "v2",
    "embedder_id": "emb-2",
    "embedder_version": "2",
    "embedder_dim": 16,
    "chunker_version": "c2",
    "chunk_tokens": 201,
    "chunk_overlap": 21,
    "parent_tokens": 800,
    "redaction_rules_version": "r2",
    "reranker_id": "rr",
    "prompt_version": "p2",
    "generator_chain_ref": "chain-2",
    "documents": (PolicyDocument(tenant_id="a", doc_id="d1", doc_version=1, doc_hash="h1"),),
}


def test_every_hashed_field_is_covered_by_a_case() -> None:
    assert set(CHANGES) == set(PolicyDraft.model_fields)


@pytest.mark.parametrize("field", sorted(CHANGES))
def test_hash_changes_with_each_included_field(field: str) -> None:
    assert policy_hash(draft(**{field: CHANGES[field]})) != policy_hash(draft())


@pytest.mark.parametrize(
    "doc_change",
    [
        {"tenant_id": "z"},
        {"doc_id": "dX"},
        {"doc_version": 9},
        {"doc_hash": "hX"},
    ],
)
def test_hash_changes_with_each_document_subfield(doc_change: dict[str, Any]) -> None:
    base = PolicyDocument(tenant_id="a", doc_id="d1", doc_version=1, doc_hash="h1")
    d1 = draft(documents=(base,))
    d2 = draft(documents=(base.model_copy(update=doc_change),))
    assert policy_hash(d1) != policy_hash(d2)


def _full(**kw: Any) -> PipelinePolicy:
    d = draft()
    base: dict[str, Any] = {
        "policy_id": "p",
        "policy_hash": policy_hash(d),
        "declared_by": "u",
        "declared_at": datetime(2026, 1, 1, tzinfo=UTC),
    }
    base.update(kw)
    return PipelinePolicy(**{**d.model_dump(), **base})


@pytest.mark.parametrize(
    "kw",
    [
        {"policy_id": "other"},
        {"declared_by": "someone else"},
        {"declared_at": datetime(2030, 1, 1, tzinfo=UTC)},
        {"verified_at": datetime(2030, 1, 1, tzinfo=UTC)},
        {"committed_at": datetime(2030, 1, 1, tzinfo=UTC)},
        {"eval_run_id": "e1"},
        {"mlflow_model_version": "7"},
        {"mlflow_synced": True},
    ],
)
def test_excluded_fields_do_not_change_hash(kw: dict[str, Any]) -> None:
    assert hash_matches(_full(**kw))


def test_excluded_status_does_not_change_hash() -> None:
    from rag_platform.domain.policy import PolicyStatus

    for s in PolicyStatus:
        assert hash_matches(_full(status=s))


def test_document_order_independent_and_duplicates_matter() -> None:
    docs = draft().documents
    assert policy_hash(draft(documents=docs)) == policy_hash(draft(documents=docs[::-1]))
    assert policy_hash(draft(documents=docs + docs[:1])) != policy_hash(draft(documents=docs))


def test_golden_hash_pins_canonicalisation() -> None:
    assert policy_hash(draft()) == GOLDEN


def test_canonical_json_shape() -> None:
    raw = canonical_json(draft())
    assert b" " not in raw.replace(b"monthly refresh", b"")
    assert raw.startswith(b'{"change_kind"') and b"\n" not in raw
    assert b'"status"' not in raw and b'"policy_id"' not in raw and b'"eval_run_id"' not in raw


def test_reason_unicode_nfc_canonicalised() -> None:
    """The reason is NFC-normalised before hashing: NFC and NFD forms hash the same."""
    nfc = unicodedata.normalize("NFC", "café")
    nfd = unicodedata.normalize("NFD", "café")
    assert nfc != nfd
    assert policy_hash(draft(change_reason=nfc)) == policy_hash(draft(change_reason=nfd))
    assert policy_hash(draft(change_reason=nfc)) == policy_hash(draft(change_reason=nfc))


def test_non_ascii_emitted_as_utf8_not_escaped() -> None:
    raw = canonical_json(draft(change_reason="café 中"))
    assert "café 中".encode() in raw and b"\\u" not in raw


def test_whitespace_edges_stripped_inner_preserved() -> None:
    assert policy_hash(draft(change_reason="  x\n")) == policy_hash(draft(change_reason="x"))
    assert policy_hash(draft(change_reason="a  b")) != policy_hash(draft(change_reason="a b"))


def test_floats_rejected_in_sizes() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        draft(chunk_tokens=200.5)
    with pytest.raises(ValidationError):
        draft(embedder_dim=8.5)


def test_hash_stable_across_hashseeds() -> None:
    code = (
        "from tests.unit.test_policy_gate import draft;"
        "from rag_platform.domain.policy import policy_hash;print(policy_hash(draft()))"
    )
    outs = set()
    for seed in ("0", "1", "12345"):
        env = {**os.environ, "PYTHONHASHSEED": seed, "PYTHONPATH": f"{ROOT}:{ROOT / 'src'}"}
        r = subprocess.run(  # noqa: S603
            [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=ROOT
        )
        assert r.returncode == 0, r.stderr
        outs.add(r.stdout.strip())
    assert outs == {GOLDEN}
