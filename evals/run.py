"""Run an eval suite: `python -m evals.run --suite smoke` writes evals/results/latest.json."""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.lib import REPO_ROOT, inputs_hash, wilson_interval, write_report
from evals.metrics_chunking import chunk_determinism
from evals.metrics_pii import evaluate as pii_evaluate
from evals.providers import SERVED_MODEL, ExtractiveFakeProvider, overlap, render_prompt
from rag_platform.domain.ports import ModelProvider

HARNESS_VERSION = "1"
HEADER = "mechanism proof, not model quality (provider: fake-extractive)"
MAX_TOKENS = 128


@dataclass(frozen=True)
class Chunk:
    id: str
    tenant: str
    text: str


@dataclass(frozen=True)
class Case:
    id: str
    cls: str
    tenant: str
    question: str
    chunks: tuple[Chunk, ...]
    # Expectations: used only by the scorer, never passed to a provider.
    must_contain: tuple[str, ...]
    must_not_contain: tuple[str, ...]
    gold_chunk_ids: tuple[str, ...]


def load_suite(path: Path) -> list[Case]:
    cases: list[Case] = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            d = json.loads(line)
            cases.append(
                Case(
                    id=str(d["id"]),
                    cls=str(d["class"]),
                    tenant=str(d["tenant"]),
                    question=str(d["question"]),
                    chunks=tuple(
                        Chunk(str(c["id"]), str(c["tenant"]), str(c["text"])) for c in d["chunks"]
                    ),
                    must_contain=tuple(map(str, d.get("must_contain", []))),
                    must_not_contain=tuple(map(str, d.get("must_not_contain", []))),
                    gold_chunk_ids=tuple(map(str, d.get("gold_chunk_ids", []))),
                )
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{path.name}:{n}: invalid case ({e!r})") from e
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return cases


def visible_context(case: Case) -> list[Chunk]:
    """Tenant filter applied before the provider ever sees text."""
    return [c for c in case.chunks if c.tenant == case.tenant]


def rank(case: Case, chunks: list[Chunk]) -> list[str]:
    scored = sorted(enumerate(chunks), key=lambda ic: (-overlap(case.question, ic[1].text), ic[0]))
    return [c.id for _, c in scored]


def score_case(case: Case, answer: str, ranked: list[str]) -> dict[str, Any]:
    low = answer.lower()
    ok_contain = all(s.lower() in low for s in case.must_contain)
    leaked = any(s.lower() in low for s in case.must_not_contain)
    gold = set(case.gold_chunk_ids)
    top5 = ranked[:5]
    rr = next((1 / (i + 1) for i, cid in enumerate(ranked) if cid in gold), 0.0) if gold else None
    return {
        "passed": ok_contain and not leaked,
        "leaked": leaked and case.cls == "tenancy",
        "hit5": (bool(gold & set(top5)) if gold else None),
        "rr": rr,
    }


async def run_cases(cases: list[Case], provider: ModelProvider) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for case in cases:
        ctx = visible_context(case)
        prompt = render_prompt(case.question, [c.text for c in ctx])  # question + context only
        comp = await provider.complete(prompt, max_tokens=MAX_TOKENS)
        out[case.id] = score_case(case, comp.text, rank(case, ctx))
    return out


def build_report(
    suite: str, cases: list[Case], results: dict[str, dict[str, Any]], root: Path
) -> dict[str, Any]:
    classes: dict[str, dict[str, Any]] = {}
    for c in cases:
        k = classes.setdefault(c.cls, {"n": 0, "passed": 0})
        k["n"] += 1
        k["passed"] += int(results[c.id]["passed"])
    for k in classes.values():
        lo, hi = wilson_interval(k["passed"], k["n"])
        k.update(rate=round(k["passed"] / k["n"], 6), wilson95_low=lo, wilson95_high=hi)
    hits = [r["hit5"] for r in results.values() if r["hit5"] is not None]
    rrs = [r["rr"] for r in results.values() if r["rr"] is not None]
    inj = classes.get("injection")
    metrics: dict[str, float] = {
        "recall_at_5": round(sum(hits) / len(hits), 6) if hits else 0.0,
        "mrr": round(sum(rrs) / len(rrs), 6) if rrs else 0.0,
    }
    if inj:
        metrics["injection_block_rate"] = inj["rate"]
    pii = pii_evaluate()  # T-008 ingest metrics (repo fixtures)
    metrics.update(pii["metrics"])
    metrics["chunk_determinism"] = chunk_determinism(root)
    suite_text = (root / "evals" / "suites" / f"{suite}.jsonl").read_bytes()
    return {
        "header": HEADER,
        "suite": suite,
        "served_model": SERVED_MODEL,
        "inputs_hash": inputs_hash(root),
        "versions": {
            "harness": HARNESS_VERSION,
            "suite_sha256": hashlib.sha256(suite_text).hexdigest(),
            "report_schema": "1",
        },
        "metrics": metrics,
        "hard_zero": {
            "cross_tenant_leakage": sum(int(r["leaked"]) for r in results.values()),
            **pii["hard_zero"],
        },
        "pii": pii["pii"],
        "classes": classes,
        "cases": {cid: bool(r["passed"]) for cid, r in results.items()},
    }


def log_mlflow(report: dict[str, Any]) -> bool:
    """Best-effort MLflow logging to experiment `rag-evals`. Never raises."""
    uri = os.environ.get("MLFLOW_TRACKING_URI", "")
    if not uri:
        return False
    try:
        if uri.startswith(("http://", "https://")):
            import httpx

            httpx.get(uri.rstrip("/") + "/health", timeout=2.0).raise_for_status()
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
        os.environ.setdefault("MLFLOW_HTTP_REQUEST_TIMEOUT", "5")
        import mlflow

        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment("rag-evals")
        with mlflow.start_run(run_name=f"{report['suite']}-{report['inputs_hash'][:8]}"):
            mlflow.log_params({"suite": report["suite"], "served_model": report["served_model"]})
            mlflow.log_metrics(report["metrics"])
        return True
    except Exception as e:  # noqa: BLE001 - best effort by design
        print(f"mlflow logging skipped: {type(e).__name__}", file=sys.stderr)
        return False


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.run")
    ap.add_argument("--suite", default="smoke")
    ap.add_argument("--root", type=Path, default=REPO_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    if not args.suite.isidentifier():
        print("invalid suite name", file=sys.stderr)
        return 2
    path = args.root / "evals" / "suites" / f"{args.suite}.jsonl"
    if not path.is_file():
        print(f"unknown suite: {args.suite}", file=sys.stderr)
        return 2
    try:
        cases = load_suite(path)
    except ValueError as e:
        print(f"invalid suite: {e}", file=sys.stderr)
        return 2
    results = asyncio.run(run_cases(cases, ExtractiveFakeProvider()))
    report = build_report(args.suite, cases, results, args.root)
    out = args.out or args.root / "evals" / "results" / "latest.json"
    write_report(report, out)
    log_mlflow(report)
    failed = [cid for cid, ok in report["cases"].items() if not ok]
    print(f"{args.suite}: {len(cases) - len(failed)}/{len(cases)} cases passed -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
