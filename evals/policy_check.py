"""Fail a change that edits thresholds/golden cases AND src/, prompts/ or config/.

Input: `--base <ref>` (merge-base diff `<ref>...HEAD`
plus staged/unstaged changes and untracked files; renames split into delete+add)
or a file list on stdin.
"""

import argparse
import subprocess  # noqa: S404
import sys

# Prefix match on purpose: golden/, golden_proposed/, thresholds.json, thresholds.proposed.json.
EVAL_PREFIXES = ("evals/golden", "evals/thresholds")
CODE_PREFIXES = ("src/", "prompts/", "config/")


def violations(files: list[str]) -> list[str]:
    files = [f.strip().removeprefix("./") for f in files if f.strip()]
    evals_touched = [f for f in files if f.startswith(EVAL_PREFIXES)]
    code_touched = [f for f in files if f.startswith(CODE_PREFIXES)]
    if evals_touched and code_touched:
        return [
            "edits thresholds/golden (" + ", ".join(sorted(evals_touched)) + ") together with "
            "code/prompts/config (" + ", ".join(sorted(code_touched)) + "); split into two PRs"
        ]
    return []


def _git(args: list[str], cwd: str | None) -> list[str] | None:
    proc = subprocess.run(  # noqa: S603
        ["git", *args],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
        cwd=cwd,
    )
    if proc.returncode != 0:
        print(f"git {args[0]} failed: {proc.stderr.strip()}", file=sys.stderr)
        return None
    return proc.stdout.splitlines()


def git_changed_files(base: str, cwd: str | None = None) -> list[str] | None:
    """Files changed on this branch since the merge-base, plus local uncommitted/untracked."""
    parts = [
        ["diff", "--name-only", "--no-renames", f"{base}...HEAD"],
        ["diff", "--name-only", "--no-renames", "HEAD"],
        ["ls-files", "-o", "--exclude-standard"],
    ]
    out: list[str] = []
    for a in parts:
        r = _git(a, cwd)
        if r is None:
            return None
        out += r
    return sorted(set(out))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.policy_check")
    ap.add_argument("--base", default=None, help="git ref to diff against; else read stdin")
    ap.add_argument("--repo", default=None, help="repository directory (default: cwd)")
    args = ap.parse_args(argv)
    if args.base:
        if args.base.startswith("-"):
            print("invalid base ref", file=sys.stderr)
            return 2
        changed = git_changed_files(args.base, args.repo)
        if changed is None:
            return 2
        files = changed
    else:
        files = sys.stdin.read().splitlines()
    errs = violations(files)
    for e in errs:
        print(f"FAIL policy: {e}", file=sys.stderr)
    return 1 if errs else 0


if __name__ == "__main__":
    raise SystemExit(main())
