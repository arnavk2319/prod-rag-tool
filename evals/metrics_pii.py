"""pii_recall and pii_leak over evals/fixtures/pii_seed.yaml, with Wilson 95% intervals.

Scope: kinds with a typed placeholder (email, phone, ssn, sin, card). Personal names have no
regex-safe placeholder and are reported separately as out of scope (not counted in recall).
"""

import json
import re
from pathlib import Path
from typing import Any

from evals.lib import REPO_ROOT, wilson_interval
from rag_platform.services.ingest.redaction import redact

IN_SCOPE = frozenset({"email", "phone", "ssn", "sin", "card"})
_TEMPLATES = ("{v}", "Contact on file: {v}.", "Ref {v} was noted by the customer.")


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


def evaluate(root: Path = REPO_ROOT) -> dict[str, Any]:
    items = json.loads((root / "evals/fixtures/pii_seed.yaml").read_text(encoding="utf-8"))["items"]
    scoped = [i for i in items if i["kind"] in IN_SCOPE]
    redacted = leaked = 0
    for it in scoped:
        v = str(it["value"])
        outs = [redact(t.format(v=v)).text for t in _TEMPLATES]
        has_leak = any(v in o or (_digits(v) and _digits(v) in _digits(o)) for o in outs)
        leaked += int(has_leak)
        redacted += int(not has_leak and all("[" in o for o in outs))
    n = len(scoped)
    lo, hi = wilson_interval(redacted, n)
    return {
        "metrics": {"pii_recall": round(redacted / n, 6) if n else 0.0},
        "hard_zero": {"pii_leak": leaked},
        "pii": {
            "n_in_scope": n,
            "n_out_of_scope": len(items) - n,
            "out_of_scope_kinds": sorted({i["kind"] for i in items} - IN_SCOPE),
            "note": "recall is over structured seed kinds only (85 of 100); names out of scope",
            "redacted": redacted,
            "leaked": leaked,
            "recall_wilson95_low": lo,
            "recall_wilson95_high": hi,
        },
    }
