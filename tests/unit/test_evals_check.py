"""One test per evals.check rule."""

import json
import shutil
from pathlib import Path

import pytest
from evals import check, run
from evals.lib import REPO_ROOT


@pytest.fixture()
def env(tmp_path: Path) -> Path:
    """Throwaway root with real thresholds, a src file and a fresh report."""
    (tmp_path / "evals" / "suites").mkdir(parents=True)
    (tmp_path / "evals" / "results").mkdir()
    shutil.copy(REPO_ROOT / "evals/suites/smoke.jsonl", tmp_path / "evals/suites/smoke.jsonl")
    shutil.copy(REPO_ROOT / "evals/thresholds.json", tmp_path / "evals/thresholds.json")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    assert run.main(["--suite", "smoke", "--root", str(tmp_path)]) == 0
    return tmp_path


def _load(root: Path) -> dict[str, object]:
    return json.loads((root / "evals/results/latest.json").read_text())  # type: ignore[no-any-return]


def _save(root: Path, report: dict[str, object], name: str = "latest.json") -> None:
    (root / "evals/results" / name).write_text(json.dumps(report))


def test_clean_run_passes(env: Path) -> None:
    assert check.main(["--root", str(env)]) == 0


def test_threshold_failure(env: Path) -> None:
    r = _load(env)
    r["metrics"]["recall_at_5"] = 0.1  # type: ignore[index]
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1


def test_hard_zero_fails_regardless_of_thresholds(env: Path) -> None:
    r = _load(env)
    r["hard_zero"]["cross_tenant_leakage"] = 1  # type: ignore[index]
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1


def test_stale_inputs_hash(env: Path) -> None:
    (env / "src" / "a.py").write_text("x = 2\n")
    assert check.main(["--root", str(env)]) == 1


def test_baseline_regression_fails_and_known_flip_allows(env: Path) -> None:
    r = _load(env)
    _save(env, r, "baseline.json")
    r["cases"]["ans-refund"] = False  # type: ignore[index]
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1
    (env / "evals/known_flips.yaml").write_text("- id: ans-refund  # reason: fixture\n")
    assert check.main(["--root", str(env)]) == 0


def test_known_flips_empty_list(tmp_path: Path) -> None:
    p = tmp_path / "k.yaml"
    p.write_text("[]\n")
    assert check.load_known_flips(p) == set()
