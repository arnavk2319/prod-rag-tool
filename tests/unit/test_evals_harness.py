"""Harness basics: report shape, Wilson, isolation, MLflow best-effort, policy_check."""

import asyncio
import json
from pathlib import Path

import pytest
from evals import policy_check, run
from evals.lib import REPO_ROOT, dump_report, inputs_hash, wilson_interval
from evals.providers import ExtractiveFakeProvider, render_prompt

from rag_platform.domain.ports import Completion

SUITE = REPO_ROOT / "evals" / "suites" / "smoke.jsonl"


def _report(tmp_path: Path) -> dict[str, object]:
    out = tmp_path / "latest.json"
    assert run.main(["--suite", "smoke", "--out", str(out)]) == 0
    return json.loads(out.read_text())  # type: ignore[no-any-return]


def test_report_shape_and_label(tmp_path: Path) -> None:
    r = _report(tmp_path)
    for key in ("metrics", "cases", "served_model", "inputs_hash", "versions", "classes"):
        assert key in r
    assert "mechanism proof, not model quality" in str(r["header"])
    for cls in r["classes"].values():  # type: ignore[attr-defined]
        assert cls["wilson95_low"] <= cls["rate"] <= cls["wilson95_high"]
    text = (tmp_path / "latest.json").read_text()
    assert "timestamp" not in text and text == dump_report(r)  # type: ignore[arg-type]


def test_wilson_known_values() -> None:
    assert wilson_interval(0, 0) == (0.0, 0.0)
    lo, hi = wilson_interval(5, 10)
    assert (lo, hi) == pytest.approx((0.236593, 0.763407), abs=1e-4)
    assert wilson_interval(10, 10)[1] == 1.0


def test_inputs_hash_skips_missing_and_ignored(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "a.py").write_text("x")
    h1 = inputs_hash(tmp_path)
    (tmp_path / "src" / "__pycache__").mkdir()
    (tmp_path / "src" / "__pycache__" / "a.pyc").write_text("junk")
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / ".gitkeep").write_text("")
    assert inputs_hash(tmp_path) == h1
    (tmp_path / "src" / "a.py").write_text("y")
    assert inputs_hash(tmp_path) != h1


class SpyProvider:
    def __init__(self) -> None:
        self.prompts: list[str] = []
        self.inner = ExtractiveFakeProvider()

    async def complete(self, prompt: str, *, max_tokens: int) -> Completion:
        self.prompts.append(prompt)
        return await self.inner.complete(prompt, max_tokens=max_tokens)


def test_provider_gets_only_question_and_context() -> None:
    cases = run.load_suite(SUITE)
    spy = SpyProvider()
    asyncio.run(run.run_cases(cases, spy))
    assert len(spy.prompts) == len(cases)
    for case, prompt in zip(cases, spy.prompts, strict=True):
        ctx = [c.text for c in run.visible_context(case)]
        assert prompt == render_prompt(case.question, ctx)
        for forbidden in case.must_not_contain:  # other-tenant / injected secrets
            if case.cls == "tenancy":
                assert forbidden not in prompt
        for cid in case.gold_chunk_ids:  # chunk ids never rendered
            assert f"{cid}:" not in prompt and f"id={cid}" not in prompt


def test_mlflow_unreachable_does_not_fail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:9")
    assert run.main(["--suite", "smoke", "--out", str(tmp_path / "l.json")]) == 0


def test_mlflow_import_failure_does_not_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setenv("MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    monkeypatch.setitem(sys.modules, "mlflow", None)
    assert run.main(["--suite", "smoke", "--out", str(tmp_path / "l.json")]) == 0


def test_policy_check_fixture_diffs(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = ["evals/thresholds.json", "src/rag_platform/api/app.py"]
    assert policy_check.violations(bad)
    assert policy_check.violations(["evals/golden/a.jsonl", "prompts/answer.md"])
    assert policy_check.violations(["evals/golden_proposed/x.jsonl", "config/a.yaml"])
    assert not policy_check.violations(["evals/thresholds.json", "docs/x.md"])
    assert not policy_check.violations(["src/a.py", "prompts/b.md"])

    import io
    import sys

    monkeypatch.setattr(sys, "stdin", io.StringIO("\n".join(bad)))
    assert policy_check.main([]) == 1
    monkeypatch.setattr(sys, "stdin", io.StringIO("src/a.py\n"))
    assert policy_check.main([]) == 0
