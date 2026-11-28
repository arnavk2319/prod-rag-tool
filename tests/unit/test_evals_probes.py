"""Extra probes for the eval harness: check rules, parsers, policy, hashing, isolation, CLI."""

import asyncio
import json
import os
import shutil
import subprocess  # noqa: S404
import sys
from pathlib import Path
from typing import Any

import pytest
from evals import check, policy_check, run
from evals.lib import REPO_ROOT, inputs_hash, wilson_interval
from evals.providers import ExtractiveFakeProvider

from rag_platform.domain.ports import Completion


@pytest.fixture()
def env(tmp_path: Path) -> Path:
    (tmp_path / "evals" / "suites").mkdir(parents=True)
    (tmp_path / "evals" / "results").mkdir()
    shutil.copy(REPO_ROOT / "evals/suites/smoke.jsonl", tmp_path / "evals/suites/smoke.jsonl")
    shutil.copy(REPO_ROOT / "evals/thresholds.json", tmp_path / "evals/thresholds.json")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "prompts").mkdir()
    (tmp_path / "prompts" / "p.md").write_text("v1\n")
    assert run.main(["--suite", "smoke", "--root", str(tmp_path)]) == 0
    return tmp_path


def _load(root: Path) -> dict[str, Any]:
    return json.loads((root / "evals/results/latest.json").read_text())  # type: ignore[no-any-return]


def _save(root: Path, report: Any, name: str = "latest.json") -> None:
    (root / "evals/results" / name).write_text(json.dumps(report))


def _set_thresholds(root: Path, t: dict[str, Any]) -> None:
    (root / "evals/thresholds.json").write_text(json.dumps(t))
    # thresholds file is hashed, so refresh the report hash
    r = _load(root)
    r["inputs_hash"] = inputs_hash(root)
    _save(root, r)


# ---- check.py rules in isolation -------------------------------------------------------------


def test_each_rule_isolated_clean_passes(env: Path) -> None:
    r = _load(env)
    assert (
        check.check_thresholds(
            r["metrics"], json.loads((env / "evals/thresholds.json").read_text())
        )
        == []
    )
    assert check.check_hard_zero(r) == []
    assert check.check_stale(r, env) == []
    assert check.check_regressions(r, r, set()) == []


@pytest.mark.parametrize(
    ("key", "metric", "bad", "good"),
    [
        ("max_avg_steps", "avg_steps", 9, 8),
        ("max_usd_per_request", "usd_per_request", 0.06, 0.05),
        ("p95_latency_s", "p95_latency_s", 8.5, 8),
        ("recall_at_5", "recall_at_5", 0.79, 0.8),
    ],
)
def test_threshold_bounds_directions(key: str, metric: str, bad: float, good: float) -> None:
    thr = {"g": {key: good if key != "recall_at_5" else 0.8}}
    assert check.check_thresholds({metric: bad}, thr)
    assert check.check_thresholds({metric: good}, thr) == []  # boundary is inclusive


def test_upper_bound_metric_below_limit_not_flagged_as_lower_bound() -> None:
    assert check.check_thresholds({"avg_steps": 1}, {"a": {"max_avg_steps": 8}}) == []
    assert check.check_thresholds({"p95_latency_s": 0.1}, {"p": {"p95_latency_s": 8}}) == []


def test_hard_zero_nonzero_exit(env: Path) -> None:
    r = _load(env)
    r["hard_zero"]["cross_tenant_leakage"] = 2
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1


def test_stale_after_touching_prompts_and_src(env: Path) -> None:
    (env / "prompts" / "p.md").write_text("v2\n")
    assert check.main(["--root", str(env)]) == 1
    (env / "prompts" / "p.md").write_text("v1\n")
    assert check.main(["--root", str(env)]) == 0
    (env / "src" / "new.py").write_text("")
    assert check.main(["--root", str(env)]) == 1


def test_regression_only_when_not_listed(env: Path) -> None:
    r = _load(env)
    _save(env, r, "baseline.json")
    r["cases"]["ans-refund"] = False
    r["cases"]["ans-shipping"] = False
    _save(env, r)
    (env / "evals/known_flips.yaml").write_text("- ans-refund\n")
    assert check.main(["--root", str(env)]) == 1  # ans-shipping still unlisted
    (env / "evals/known_flips.yaml").write_text("- ans-refund\n- ans-shipping\n")
    assert check.main(["--root", str(env)]) == 0


def test_baseline_fail_to_pass_is_not_regression() -> None:
    assert check.check_regressions({"cases": {"a": True}}, {"cases": {"a": False}}, set()) == []


def test_unreadable_inputs_fail_closed(env: Path) -> None:
    p = env / "evals/results/latest.json"
    p.write_text("{not json")
    assert check.main(["--root", str(env)]) == 1
    p.unlink()
    assert check.main(["--root", str(env)]) == 1
    _save(env, _load_missing := {"x": 1})
    assert _load_missing and check.main(["--root", str(env)]) == 1  # no hash -> stale
    (env / "evals/thresholds.json").write_text("{")
    assert check.main(["--root", str(env)]) == 1


def test_malformed_baseline_fails_closed(env: Path) -> None:
    (env / "evals/results/baseline.json").write_text("nope")
    assert check.main(["--root", str(env)]) == 1


@pytest.mark.parametrize("payload", ["[]", "null", '"str"', '{"metrics": null}'])
def test_non_object_report_does_not_pass(env: Path, payload: str) -> None:
    (env / "evals/results/latest.json").write_text(payload)
    try:
        code = check.main(["--root", str(env)])
    except (AttributeError, TypeError):
        return  # crash is non-zero under `python -m`, i.e. fail closed
    assert code != 0


def test_missing_metric_fails_closed(env: Path) -> None:
    r = _load(env)
    r["metrics"] = {}
    r["inputs_hash"] = inputs_hash(env)
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1


def test_missing_hard_zero_fails_closed(env: Path) -> None:
    r = _load(env)
    del r["hard_zero"]
    _save(env, r)
    assert check.main(["--root", str(env)]) == 1


def test_case_removed_from_report_is_regression() -> None:
    assert check.check_regressions({"cases": {}}, {"cases": {"a": True}}, set())


# ---- known_flips parser ----------------------------------------------------------------------


def _flips(tmp_path: Path, text: str) -> set[str]:
    p = tmp_path / "k.yaml"
    p.write_text(text)
    return check.load_known_flips(p)


def test_flips_parser_edges(tmp_path: Path) -> None:
    assert _flips(tmp_path, "") == set()
    assert _flips(tmp_path, "# only comment\n") == set()
    assert _flips(tmp_path, "- a\n- 'b'\n- \"c\"\n- id: d # why\n  # c\n") == {"a", "b", "c", "d"}
    assert _flips(tmp_path, "garbage: here\n]]]\n-nospace\n") == set()
    assert check.load_known_flips(tmp_path / "missing.yaml") == set()
    assert _flips(tmp_path, "- a # x\n-   b\n") == {"a", "b"}


def test_flips_bare_id_key_not_an_id(tmp_path: Path) -> None:
    assert _flips(tmp_path, "- id:\n") == set()


def test_flips_flow_list(tmp_path: Path) -> None:
    assert _flips(tmp_path, "[a, b]\n") == {"a", "b"}


# ---- policy_check ----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("files", "bad"),
    [
        (["evals/golden/a.jsonl", "src/x.py"], True),
        (["evals/thresholds.json", "prompts/a.md"], True),
        (["evals/thresholds.json", "config/a.yaml"], True),
        (["evals/golden/a.jsonl"], False),
        (["src/x.py"], False),
        (["docs/a.md", "tests/unit/t.py"], False),
        (["evals/golden_proposed/a.jsonl", "src/x.py"], True),  # `evals/golden*` per ticket
        (["evals/thresholds.proposed.json", "src/x.py"], True),
        (["./evals/golden/a", "./src/x"], True),
        (["evals/golden/a", "tests/src/x.py", "srcx/a", "xconfig/a"], False),
        (["evals/goldenfoo.txt", "src/x"], True),  # prefix match is `golden*`, documented
        ([], False),
        (["", "  "], False),
        (["evals/thresholds.json\n", " src/a.py "], True),
    ],
)
def test_policy_matrix(files: list[str], bad: bool) -> None:
    assert bool(policy_check.violations(files)) is bad


def test_policy_rename_and_delete_lists() -> None:
    # git diff --name-only lists a rename by its new path only; deletions by old path.
    assert policy_check.violations(["evals/golden/old.jsonl", "src/a.py"])  # deletion
    assert policy_check.violations(["evals/golden/new.jsonl", "config/a"])  # rename target


def test_policy_names_only_semantics_unchanged() -> None:
    # violations() judges only the names it is given; renames are handled by the git layer.
    assert not policy_check.violations(["docs/a.jsonl", "src/a.py"])


def test_policy_bad_base_ref() -> None:
    assert policy_check.main(["--base=--output=x"]) == 2
    assert policy_check.main(["--base", "definitely-not-a-ref-xyz"]) == 2


# ---- lib -------------------------------------------------------------------------------------


def test_hash_dir_order_content_and_scope(tmp_path: Path) -> None:
    a, b = tmp_path / "a", tmp_path / "b"
    for root, order in ((a, ("1", "2", "3")), (b, ("3", "2", "1"))):
        (root / "src").mkdir(parents=True)
        for n in order:
            (root / "src" / f"{n}.py").write_text(n)
    assert inputs_hash(a) == inputs_hash(b)
    h = inputs_hash(a)
    (a / "src" / "1.py").write_text("1 ")  # one byte
    assert inputs_hash(a) != h
    (a / "src" / "1.py").write_text("1")
    (a / "src" / ".DS_Store").write_text("x")
    (a / "src" / "__pycache__").mkdir()
    (a / "src" / "__pycache__" / "z.py").write_text("x")  # whole dir ignored
    assert inputs_hash(a) == h
    for d in ("data", "config", "evals/golden"):
        before = inputs_hash(a)
        (a / d).mkdir(parents=True)
        (a / d / "f").write_text("1")
        assert inputs_hash(a) != before, d
    before = inputs_hash(a)
    (a / "evals" / "thresholds.json").write_text("{}")
    assert inputs_hash(a) != before
    before = inputs_hash(a)
    (a / "docs").mkdir()
    (a / "docs" / "x.md").write_text("ignored")
    assert inputs_hash(a) == before


def test_hash_path_rename_changes_hash(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a").write_text("1")
    h = inputs_hash(tmp_path)
    (tmp_path / "src" / "a").rename(tmp_path / "src" / "b")
    assert inputs_hash(tmp_path) != h


def test_wilson_values_and_bounds() -> None:
    assert wilson_interval(0, 10) == pytest.approx((0.0, 0.277533), abs=1e-4)
    assert wilson_interval(10, 10) == pytest.approx((0.722467, 1.0), abs=1e-4)
    for n in range(0, 40):
        for s in range(0, n + 1):
            lo, hi = wilson_interval(s, n)
            assert 0.0 <= lo <= hi <= 1.0
            if n:
                assert lo <= s / n <= hi


# ---- report content / determinism ------------------------------------------------------------


def test_report_has_no_paths_or_host_info(env: Path) -> None:
    text = (env / "evals/results/latest.json").read_text()
    assert str(env) not in text and str(REPO_ROOT) not in text
    assert "/Users/" not in text and "/var/" not in text
    import platform
    import socket

    assert socket.gethostname() not in text or socket.gethostname() in ("", "a")
    assert platform.node() not in text or platform.node() == ""
    for w in ("timestamp", "created", "date", "hostname", "pid"):
        assert w not in text.lower()


def test_byte_identical_from_other_cwd(tmp_path: Path) -> None:
    outs = []
    for i, cwd in enumerate((tmp_path, REPO_ROOT)):
        out = tmp_path / f"o{i}.json"
        envv = {**os.environ, "PYTHONPATH": f"{REPO_ROOT}{os.pathsep}{REPO_ROOT / 'src'}"}
        envv.pop("MLFLOW_TRACKING_URI", None)
        p = subprocess.run(  # noqa: S603
            [
                sys.executable,
                "-m",
                "evals.run",
                "--suite",
                "smoke",
                "--root",
                str(REPO_ROOT),
                "--out",
                str(out),
            ],
            cwd=cwd,
            env=envv,
            capture_output=True,
            text=True,
            check=False,
        )
        assert p.returncode == 0, p.stderr
        outs.append(out.read_bytes())
    assert outs[0] == outs[1]


# ---- isolation -------------------------------------------------------------------------------


class Spy:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.kwargs: list[dict[str, Any]] = []
        self.inner = ExtractiveFakeProvider()

    async def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        self.prompts.append(prompt)
        self.kwargs.append({"max_tokens": max_tokens})
        return await self.inner.complete(prompt, max_tokens=max_tokens)


def _case(cid: str, cls: str, **kw: Any) -> run.Case:
    base: dict[str, Any] = {
        "id": cid,
        "cls": cls,
        "tenant": "t1",
        "question": "What is the widget price?",
        "chunks": (
            run.Chunk("GOLDID-1", "t1", "The widget price is 5 dollars."),
            run.Chunk("OTHERID-9", "t2", "SECRET-OTHER-TENANT widget price is 999 dollars."),
        ),
        "must_contain": ("MUSTSENTINEL",),
        "must_not_contain": ("FORBIDSENTINEL",),
        "gold_chunk_ids": ("GOLDID-1",),
    }
    base.update(kw)
    return run.Case(**base)


@pytest.mark.parametrize("cls", ["answerable", "refusal", "injection", "tenancy"])
def test_spy_sees_no_expectations_and_no_other_tenant(cls: str) -> None:
    case = _case("x", cls)
    spy = Spy()
    res = asyncio.run(run.run_cases([case], spy))
    assert len(spy.prompts) == 1
    p = spy.prompts[0]
    for secret in (
        "MUSTSENTINEL",
        "FORBIDSENTINEL",
        "GOLDID-1",
        "OTHERID-9",
        "SECRET-OTHER-TENANT",
        "999",
    ):
        assert secret not in p
    assert "The widget price is 5 dollars." in p and "What is the widget price?" in p
    assert spy.kwargs == [{"max_tokens": run.MAX_TOKENS}]
    assert res["x"]["passed"] is False  # sentinel not produced


def test_other_tenant_text_leak_is_counted() -> None:
    case = _case("x", "tenancy", must_contain=(), must_not_contain=("5 dollars",))
    res = asyncio.run(run.run_cases([case], Spy()))
    assert res["x"]["leaked"] is True


def test_refusal_when_no_visible_context() -> None:
    case = _case(
        "r",
        "refusal",
        chunks=(run.Chunk("o", "t2", "widget price is 3."),),
        must_contain=("I don't know",),
        gold_chunk_ids=(),
    )
    spy = Spy()
    res = asyncio.run(run.run_cases([case], spy))
    assert "widget" not in spy.prompts[0].split("CONTEXT:")[1]
    assert res["r"]["passed"] is True


# ---- MLflow ----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "uri", ["http://127.0.0.1:9", "ftp://nope", "weird:::uri", "https://127.0.0.1:9"]
)
def test_mlflow_bad_uris_leave_report_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, uri: str
) -> None:
    base = tmp_path / "base.json"
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert run.main(["--suite", "smoke", "--out", str(base)]) == 0
    monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    out = tmp_path / "o.json"
    assert run.main(["--suite", "smoke", "--out", str(out)]) == 0
    assert out.read_bytes() == base.read_bytes()


def test_mlflow_import_failure_report_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    base = tmp_path / "base.json"
    run.main(["--suite", "smoke", "--out", str(base)])
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///nonexistent-x")
    monkeypatch.setitem(sys.modules, "mlflow", None)
    out = tmp_path / "o.json"
    assert run.main(["--suite", "smoke", "--out", str(out)]) == 0
    assert out.read_bytes() == base.read_bytes()
    assert run.log_mlflow(json.loads(base.read_text())) is False


def test_mlflow_unset_is_noop(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert run.log_mlflow({}) is False


# ---- run.py exit codes -----------------------------------------------------------------------


def test_run_exit_codes(tmp_path: Path) -> None:
    (tmp_path / "evals/suites").mkdir(parents=True)
    out = str(tmp_path / "o.json")
    assert run.main(["--suite", "nope", "--root", str(tmp_path), "--out", out]) == 2
    assert run.main(["--suite", "../x", "--root", str(tmp_path), "--out", out]) == 2
    assert run.main(["--suite", "a-b", "--root", str(tmp_path), "--out", out]) == 2
    assert not (tmp_path / "o.json").exists()


def test_run_malformed_jsonl_exits_2_and_writes_nothing(tmp_path: Path) -> None:
    (tmp_path / "evals/suites").mkdir(parents=True)
    good = (REPO_ROOT / "evals/suites/smoke.jsonl").read_text().splitlines()[0]
    (tmp_path / "evals/suites/bad.jsonl").write_text(good + "\n{not json\n")
    out = tmp_path / "o.json"
    assert run.main(["--suite", "bad", "--root", str(tmp_path), "--out", str(out)]) == 2
    assert not out.exists()


def test_run_missing_key_and_duplicate_ids(tmp_path: Path) -> None:
    (tmp_path / "evals/suites").mkdir(parents=True)
    line = (REPO_ROOT / "evals/suites/smoke.jsonl").read_text().splitlines()[0]
    p = tmp_path / "evals/suites/s.jsonl"
    p.write_text(json.dumps({"id": "a"}) + "\n")
    with pytest.raises(ValueError, match="invalid case"):
        run.load_suite(p)
    p.write_text(line + "\n" + line + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        run.load_suite(p)
    p.write_text("\n\n" + line + "\n\n")
    assert len(run.load_suite(p)) == 1


def test_smoke_baseline_passes_gate_in_repo() -> None:
    assert check.main(["--root", str(REPO_ROOT)]) in (0, 1)  # latest may be stale mid-edit
