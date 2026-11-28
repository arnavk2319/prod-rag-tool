"""T-045 adversarial: policy_violations metric detects each fault; report is deterministic."""

import asyncio
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from evals import check, metrics_policy, run
from evals.lib import REPO_ROOT

from rag_platform.adapters.policy_fakes import FakePolicyRepository
from rag_platform.domain.policy import PipelinePolicy, PolicyStatus, policy_hash
from tests.unit.test_policy_gate import draft

NOW = datetime(2026, 1, 1, tzinfo=UTC)


async def _store(**kw: Any) -> FakePolicyRepository:
    d = draft()
    base: dict[str, Any] = {
        "policy_id": "p",
        "policy_hash": policy_hash(d),
        "declared_by": "t",
        "declared_at": NOW,
        "status": PolicyStatus.COMMITTED,
        "verified_at": NOW,
        "committed_at": NOW,
    }
    base.update(kw)
    repo = FakePolicyRepository()
    await repo.create(PipelinePolicy(**{**d.model_dump(), **base}))
    return repo


def audit(**kw: Any) -> int:
    return asyncio.run(metrics_policy.audit_store(asyncio.run(_store(**kw))))


def test_healthy_store_is_zero() -> None:
    assert audit() == 0


def test_committed_without_verified_at() -> None:
    assert audit(verified_at=None) == 1


def test_hash_mismatch() -> None:
    assert audit(policy_hash="0" * 64) == 1


def test_committed_version_with_no_policy() -> None:
    async def go() -> int:
        repo = await _store()
        repo.index_status["ghost"] = "committed"
        return await metrics_policy.audit_store(repo)

    assert asyncio.run(go()) == 1


@pytest.mark.parametrize("st", [PolicyStatus.REJECTED, PolicyStatus.ROLLED_BACK])
def test_index_committed_but_policy_rejected_or_rolled_back(st: PolicyStatus) -> None:
    async def go() -> int:
        repo = await _store(status=st)
        repo.index_status["v1"] = "committed"  # index says committed, policy disagrees
        return await metrics_policy.audit_store(repo)

    assert asyncio.run(go()) == 1


def test_policy_committed_but_index_not_committed_is_counted() -> None:
    async def go() -> int:
        repo = await _store()
        repo.index_status["v1"] = "staged"
        return await metrics_policy.audit_store(repo)

    assert asyncio.run(go()) == 1


def test_evaluate_is_deterministic() -> None:
    assert metrics_policy.evaluate() == metrics_policy.evaluate() == {"policy_violations": 0}
    assert metrics_policy.evaluate(inject_violation=True) == {"policy_violations": 1}


def _root(tmp_path: Path) -> Path:
    (tmp_path / "evals" / "suites").mkdir(parents=True)
    (tmp_path / "evals" / "results").mkdir()
    shutil.copy(REPO_ROOT / "evals/suites/smoke.jsonl", tmp_path / "evals/suites/smoke.jsonl")
    shutil.copy(REPO_ROOT / "evals/thresholds.json", tmp_path / "evals/thresholds.json")
    (tmp_path / "src").mkdir()
    return tmp_path


def test_report_byte_identical_across_runs(tmp_path: Path) -> None:
    root = _root(tmp_path)
    out = []
    for _ in range(2):
        assert run.main(["--suite", "smoke", "--root", str(root)]) == 0
        out.append((root / "evals/results/latest.json").read_bytes())
    assert out[0] == out[1]
    assert json.loads(out[0])["hard_zero"]["policy_violations"] == 0


@pytest.mark.parametrize("n", [1, 2])
def test_check_fails_on_any_nonzero_violation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, n: int
) -> None:
    root = _root(tmp_path)
    monkeypatch.setattr(run, "policy_evaluate", lambda: {"policy_violations": n})
    assert run.main(["--suite", "smoke", "--root", str(root)]) == 0
    assert check.main(["--root", str(root)]) == 1


def test_check_fails_when_metric_missing_or_not_int(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _root(tmp_path)
    assert run.main(["--suite", "smoke", "--root", str(root)]) == 0
    p = root / "evals/results/latest.json"
    rep = json.loads(p.read_text())
    rep["hard_zero"]["policy_violations"] = None
    p.write_text(json.dumps(rep))
    assert check.main(["--root", str(root)]) == 1
