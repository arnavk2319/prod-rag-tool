"""policy_violations is a hard-zero metric in the smoke report; injecting one fails evals.check."""

import json
import shutil
from pathlib import Path

import pytest
from evals import check, metrics_policy, run
from evals.lib import REPO_ROOT


def test_clean_scenario_has_zero_violations() -> None:
    assert metrics_policy.evaluate() == {"policy_violations": 0}
    assert metrics_policy.evaluate(inject_violation=True) == {"policy_violations": 1}


def test_mutation_unverified_commit_fails_check(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "evals" / "suites").mkdir(parents=True)
    (tmp_path / "evals" / "results").mkdir()
    shutil.copy(REPO_ROOT / "evals/suites/smoke.jsonl", tmp_path / "evals/suites/smoke.jsonl")
    shutil.copy(REPO_ROOT / "evals/thresholds.json", tmp_path / "evals/thresholds.json")
    (tmp_path / "src").mkdir()
    assert run.main(["--suite", "smoke", "--root", str(tmp_path)]) == 0
    assert check.main(["--root", str(tmp_path)]) == 0
    monkeypatch.setattr(
        run, "policy_evaluate", lambda: metrics_policy.evaluate(inject_violation=True)
    )
    assert run.main(["--suite", "smoke", "--root", str(tmp_path)]) == 0
    report = json.loads((tmp_path / "evals/results/latest.json").read_text())
    assert report["hard_zero"]["policy_violations"] == 1
    assert check.main(["--root", str(tmp_path)]) == 1
