"""Round 2: metric validation, required thresholds, merge-base policy diff, hash coverage."""

import json
import math
import subprocess
from pathlib import Path

import pytest
from evals import check, policy_check
from evals.lib import inputs_hash

THR = {"retrieval": {"recall_at_5": 0.8, "mrr": 0.6}, "safety": {"injection_block_rate": 0.9}}
GOOD = {"recall_at_5": 1.0, "mrr": 1.0, "injection_block_rate": 1.0}


@pytest.mark.parametrize("bad", [math.nan, math.inf, "1.0", True, None])
def test_bad_metric_values_fail(bad: object) -> None:
    errs = check.check_thresholds({**GOOD, "mrr": bad}, THR, "smoke")  # type: ignore[dict-item]
    assert errs


def test_empty_thresholds_do_not_pass_smoke() -> None:
    assert check.check_thresholds(GOOD, {}, "smoke")
    assert not check.check_thresholds(GOOD, THR, "smoke")


def test_no_baseline_message(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    from evals import run

    (tmp_path / "evals/suites").mkdir(parents=True)
    suite = Path(__file__).resolve().parents[2] / "evals/suites/smoke.jsonl"
    (tmp_path / "evals/suites/smoke.jsonl").write_bytes(suite.read_bytes())
    (tmp_path / "evals/thresholds.json").write_text(json.dumps(THR))
    assert run.main(["--suite", "smoke", "--root", str(tmp_path)]) == 0
    assert check.main(["--root", str(tmp_path)]) == 0
    assert "no baseline: regression rule skipped" in capsys.readouterr().err


def test_inputs_hash_covers_suites_and_harness(tmp_path: Path) -> None:
    (tmp_path / "evals/suites").mkdir(parents=True)
    (tmp_path / "evals/suites/s.jsonl").write_text("a")
    (tmp_path / "evals/run.py").write_text("x")
    h = inputs_hash(tmp_path)
    (tmp_path / "evals/suites/s.jsonl").write_text("b")
    h2 = inputs_hash(tmp_path)
    (tmp_path / "evals/run.py").write_text("y")
    h3 = inputs_hash(tmp_path)
    assert len({h, h2, h3}) == 3


def _git(repo: Path, *a: str) -> None:
    subprocess.run(  # noqa: S603
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", *a],  # noqa: S607
        cwd=repo,
        check=True,
        capture_output=True,
    )


def _repo(tmp_path: Path) -> Path:
    _git(tmp_path, "init", "-q", "-b", "main")
    (tmp_path / "evals/golden").mkdir(parents=True)
    (tmp_path / "evals/golden/a.jsonl").write_text("1\n")
    (tmp_path / "src").mkdir()
    (tmp_path / "src/a.py").write_text("x\n")
    (tmp_path / "docs").mkdir()
    _git(tmp_path, "add", "-A")
    _git(tmp_path, "commit", "-q", "-m", "init")
    _git(tmp_path, "checkout", "-q", "-b", "feat")
    return tmp_path


def test_policy_git_rename_out_of_golden_plus_src_flagged(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    _git(repo, "mv", "evals/golden/a.jsonl", "docs/a.jsonl")
    (repo / "src/a.py").write_text("y\n")
    assert policy_check.main(["--base", "main", "--repo", str(repo)]) == 1


def test_policy_base_advancing_is_not_flagged(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src/a.py").write_text("y\n")
    _git(repo, "commit", "-q", "-am", "feat src")
    _git(repo, "checkout", "-q", "main")
    (repo / "evals/golden/a.jsonl").write_text("2\n")
    _git(repo, "commit", "-q", "-am", "golden edit on base")
    _git(repo, "checkout", "-q", "feat")
    assert policy_check.main(["--base", "main", "--repo", str(repo)]) == 0


def test_policy_sees_untracked_and_uncommitted(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "src/a.py").write_text("z\n")  # uncommitted
    (repo / "evals/golden/new.jsonl").write_text("n\n")  # untracked
    assert policy_check.main(["--base", "main", "--repo", str(repo)]) == 1
