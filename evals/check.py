"""Gate on a report: `python -m evals.check [--baseline evals/results/baseline.json]`.

Fails (exit 1) when: a threshold fails, any hard-zero metric is non-zero, inputs_hash is
stale, or a case that passed in the baseline now fails and is not in evals/known_flips.yaml.
Also fails on non-finite/non-numeric/bool metrics and on required metrics with no threshold.
If baseline.json is absent the case-level regression rule is skipped (one stderr line says so).
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

from evals.lib import REPO_ROOT, inputs_hash


def flatten_thresholds(t: dict[str, Any]) -> dict[str, float]:
    return {
        k: float(v) for group in t.values() if isinstance(group, dict) for k, v in group.items()
    }


# Metrics each suite must report (fail closed). Unknown suites must report every thresholded metric.
SUITE_REQUIRED: dict[str, frozenset[str]] = {
    "smoke": frozenset({"recall_at_5", "mrr", "injection_block_rate"}),
}


def _is_valid(v: Any) -> bool:
    return isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)


def invalid_metrics(metrics: dict[str, Any], label: str = "metric") -> list[str]:
    return [
        f"{label} {k} is not a finite number: {v!r}"
        for k, v in sorted(metrics.items())
        if not _is_valid(v)
    ]


def check_thresholds(
    metrics: dict[str, float], thresholds: dict[str, Any], suite: str | None = None
) -> list[str]:
    """Thresholded metrics absent from the report fail if the suite must report them.

    `max_X` and `*_latency_s` thresholds are upper bounds on metric X; the rest are lower bounds.
    """
    errs: list[str] = []
    required = frozenset() if suite is None else SUITE_REQUIRED.get(suite)
    flat = flatten_thresholds(thresholds)
    have = {k.removeprefix("max_") for k in flat}
    for name in sorted(required or ()):
        if name not in have:
            errs.append(f"required metric {name} has no entry in thresholds")
    errs += invalid_metrics(metrics)
    for key, limit in flat.items():
        upper = key.startswith("max_") or key.endswith("_latency_s")
        name = key.removeprefix("max_")
        metric = metrics.get(name)
        if metric is not None and not _is_valid(metric):
            continue  # already reported by invalid_metrics
        if metric is None:
            if required is None or name in required:
                errs.append(f"threshold {key}: metric {name} missing from report")
            continue
        if upper and metric > limit:
            errs.append(f"threshold {key}: {metric} > {limit}")
        elif not upper and metric < limit:
            errs.append(f"threshold {key}: {metric} < {limit}")
    return errs


def check_hard_zero(report: dict[str, Any]) -> list[str]:
    hz = report.get("hard_zero")
    if not isinstance(hz, dict) or "cross_tenant_leakage" not in hz:
        return ["hard-zero section missing (cross_tenant_leakage must be reported)"]
    return [f"hard-zero {k} = {v}" for k, v in sorted(hz.items()) if v != 0]


def check_stale(report: dict[str, Any], root: Path) -> list[str]:
    cur = inputs_hash(root)
    if report.get("inputs_hash") != cur:
        return ["stale: inputs_hash does not match current inputs (re-run make eval)"]
    return []


def load_known_flips(path: Path) -> set[str]:
    """Minimal YAML subset: `[]`, or lines like `- case_id` / `- id: case_id  # reason`."""
    if not path.is_file():
        return set()
    ids: set[str] = set()
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line.startswith("["):
            inner = line.strip("[]")
            ids.update(t.strip().strip("'\"") for t in inner.split(",") if t.strip())
            continue
        m = re.match(r"^-\s+(?:id:\s+)?([^\s:]\S*)", line)
        if m and not m.group(1).endswith(":"):
            ids.add(m.group(1).strip("'\""))
    return ids


def check_regressions(
    report: dict[str, Any], baseline: dict[str, Any], known: set[str]
) -> list[str]:
    cur = report.get("cases", {})
    return [
        f"regression: case {cid} passed in baseline, fails now"
        for cid, ok in sorted(baseline.get("cases", {}).items())
        if ok and cur.get(cid) is not True and cid not in known
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.check")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--report", type=Path, default=None)
    ap.add_argument("--baseline", type=Path, default=None)
    ap.add_argument("--thresholds", type=Path, default=None)
    ap.add_argument("--known-flips", type=Path, default=None)
    args = ap.parse_args(argv)
    root: Path = args.root
    report_p = args.report or root / "evals/results/latest.json"
    base_p = args.baseline or root / "evals/results/baseline.json"
    thr_p = args.thresholds or root / "evals/thresholds.json"
    flips_p = args.known_flips or root / "evals/known_flips.yaml"
    try:
        report = json.loads(report_p.read_text(encoding="utf-8"))
        thresholds = json.loads(thr_p.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        print(f"FAIL cannot read inputs: {e}", file=sys.stderr)
        return 1
    if not isinstance(report, dict) or not isinstance(thresholds, dict):
        print("FAIL report and thresholds must be JSON objects", file=sys.stderr)
        return 1
    metrics = report.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    errs = check_thresholds(metrics, thresholds, str(report.get("suite", "")))
    errs += check_hard_zero(report)
    errs += check_stale(report, root)
    if base_p.is_file():
        try:
            baseline = json.loads(base_p.read_text(encoding="utf-8"))
        except ValueError as e:
            print(f"FAIL unreadable baseline: {e}", file=sys.stderr)
            return 1
        if not isinstance(baseline, dict):
            print("FAIL baseline must be a JSON object", file=sys.stderr)
            return 1
        bm = baseline.get("metrics")
        if isinstance(bm, dict):
            errs += invalid_metrics(bm, "baseline metric")
        errs += check_regressions(report, baseline, load_known_flips(flips_p))
    else:
        print("no baseline: regression rule skipped", file=sys.stderr)
    for msg in errs:
        print(f"FAIL {msg}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
