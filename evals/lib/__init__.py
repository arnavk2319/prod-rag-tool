"""Small stdlib helpers for the eval harness (hashing, Wilson interval, report writer)."""

import hashlib
import json
import math
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
HASHED_PATHS = (
    "src",
    "prompts",
    "config",
    "evals/golden",
    "evals/thresholds.json",
    "data",
    # harness itself: editing the suite or the harness makes latest.json stale
    "evals/suites",
    "evals/lib",
    "evals/*.py",
)
_IGNORED_DIRS = {"__pycache__"}
_IGNORED_FILES = {".gitkeep", ".DS_Store"}


def _iter_files(root: Path, rel: str) -> list[Path]:
    """Files under one hashed path, sorted; a missing path yields nothing (deterministic skip)."""
    if "*" in rel:
        return sorted(
            (f for f in root.glob(rel) if f.is_file() and f.suffix != ".pyc"),
            key=lambda f: f.relative_to(root).as_posix(),
        )
    p = root / rel
    if p.is_file():
        return [p]
    if not p.is_dir():
        return []
    out = [
        f
        for f in p.rglob("*")
        if f.is_file()
        and f.name not in _IGNORED_FILES
        and f.suffix != ".pyc"
        and not (_IGNORED_DIRS & set(f.relative_to(p).parts))
    ]
    return sorted(out, key=lambda f: f.relative_to(root).as_posix())


def inputs_hash(root: Path = REPO_ROOT, paths: tuple[str, ...] = HASHED_PATHS) -> str:
    """sha256 over (relative path, content hash) of every input file.

    Ignored: __pycache__/, *.pyc, .gitkeep, .DS_Store. Missing paths are skipped.
    """
    h = hashlib.sha256()
    for rel in paths:
        for f in _iter_files(root, rel):
            h.update(f.relative_to(root).as_posix().encode())
            h.update(b"\0")
            h.update(hashlib.sha256(f.read_bytes()).digest())
    return h.hexdigest()


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval (95% by default); (0, 0) when n == 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (round(max(0.0, centre - half), 6), round(min(1.0, centre + half), 6))


def dump_report(report: dict[str, Any]) -> str:
    """Canonical serialisation: sorted keys, fixed indent, trailing newline, no timestamps."""
    return json.dumps(report, sort_keys=True, indent=2, ensure_ascii=True) + "\n"


def write_report(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_report(report), encoding="utf-8")
