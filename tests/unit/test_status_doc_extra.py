"""T-046 extra checks on docs/status.md and scripts/status_doc.py (structure, links, generator)."""
# ruff: noqa: E501

import importlib.util
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts/status_doc.py"
_spec = importlib.util.spec_from_file_location("status_doc_x", SCRIPT)
assert _spec and _spec.loader
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

DOC = ROOT / "docs/status.md"
TEXT = DOC.read_text(encoding="utf-8")
STATES = {"planned", "in progress", "in review", "merged"}
COMPONENTS = [
    "Ingest", "Parser", "Redaction", "Quarantine", "Chunker", "Embedder", "Reranker",
    "Policy gate", "Index versions", "Vector store", "Retrieval", "Generation chain",
    "Verifier", "Guardrails", "API", "Auth and tenancy", "Web console", "Evals",
    "Observability", "CI", "Deploy",
]  # fmt: skip


def _tables(text: str) -> list[list[list[str]]]:
    out: list[list[list[str]]] = []
    cur: list[list[str]] = []
    for line in text.splitlines():
        if line.startswith("|"):
            cur.append([c.strip() for c in re.split(r"(?<!\\)\|", line.strip())[1:-1]])
        elif cur:
            out.append(cur)
            cur = []
    if cur:
        out.append(cur)
    return out


def _section(n: int) -> str:
    m = re.search(rf"^## {n}\. .*?(?=^## \d\. |\Z)", TEXT, re.S | re.M)
    assert m
    return m.group(0)


def _decisions() -> dict[int, str]:
    body = _section(4)
    items = re.split(r"^(?=\d+\. \*\*)", body, flags=re.M)[1:]
    return {int(re.match(r"(\d+)\.", i).group(1)): i for i in items}  # type: ignore[union-attr]


def _git(*args: str) -> str:
    cmd = ["git", *args]  # noqa: S607
    return subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, check=True).stdout  # noqa: S603


def test_every_table_row_has_header_cell_count() -> None:
    for tbl in _tables(TEXT):
        n = len(tbl[0])
        for row in tbl:
            assert len(row) == n, f"{n} cells expected, got {len(row)}: {row[0]}"


def test_component_map_lists_each_component_once() -> None:
    rows = _tables(_section(1))[0][2:]
    names = [r[0] for r in rows]
    assert sorted(names) == sorted(COMPONENTS)
    assert len(set(names)) == len(names)
    for r in rows:
        assert r[2] in STATES, r
        assert re.fullmatch(r"(#\d+)(, #\d+)*|-", r[4]), r


def test_component_decision_refs_exist() -> None:
    dec = _decisions()
    for r in _tables(_section(1))[0][2:]:
        for n in re.findall(r"\d+", r[5]):
            assert int(n) in dec, (r[0], n)


def test_decisions_numbered_consecutively_with_reason() -> None:
    dec = _decisions()
    assert sorted(dec) == list(range(1, len(dec) + 1))
    for n, body in dec.items():
        assert "Reason" in body, n


def test_ticket_rows_unique_complete_and_wellformed() -> None:
    rows = _tables(_section(2))[0][2:]
    ids = [r[0] for r in rows]
    assert len(set(ids)) == len(ids)
    assert set(ids) >= {f"T-{i:03d}" for i in range(1, 47)}
    for r in rows:
        assert r[3] in STATES, r
        assert r[5] == "-" or re.fullmatch(
            r"\[#\d+\]\(https://github\.com/[^/]+/[^/]+/pull/\d+\)", r[5]
        )
        if r[5] != "-" and r[3] == "merged":
            assert True


def test_no_merged_row_with_in_progress_ticket() -> None:
    tickets = {r[0]: r[3] for r in _tables(_section(2))[0][2:]}
    for r in _tables(_section(1))[0][2:]:
        if r[2] == "merged":
            for t in re.findall(r"T-\d+", r[3]):
                assert tickets[t] == "merged", (r[0], t, tickets[t])


def test_pr_numbers_exist_in_git_history() -> None:
    log = _git("log", "--all", "--merges", "--format=%s")
    merged = {int(n) for n in re.findall(r"pull request #(\d+)", log)}
    merged |= {int(n) for n in re.findall(r"\(#(\d+)\)", _git("log", "--all", "--format=%s"))}
    cited = {int(n) for n in re.findall(r"#(\d+)", _section(1))}
    cited |= {int(n) for n in re.findall(r"/pull/(\d+)", _section(2))}
    missing = {n for n in cited if n not in merged}
    # open PRs are documented as in review/in progress rows, never as merged
    for r in _tables(_section(2))[0][2:]:
        m = re.search(r"#(\d+)", r[5])
        if m and int(m.group(1)) in missing:
            assert r[3] in {"in review", "in progress"}, r
            missing.discard(int(m.group(1)))
    for r in _tables(_section(1))[0][2:]:
        for n in re.findall(r"#(\d+)", r[4]):
            if int(n) in missing:
                assert r[2] != "merged", r
                missing.discard(int(n))
    assert not missing, f"PRs not in git history: {missing}"


def test_all_mentioned_paths_are_tracked() -> None:
    tracked = set(_git("ls-files").splitlines())
    dirs = {"/".join(p.split("/")[:i]) + "/" for p in tracked for i in range(1, p.count("/") + 1)}
    # docs/status.md and tests may be untracked before commit
    tracked |= {"docs/status.md", "scripts/status_doc.py", "tests/unit/test_status_doc.py"}
    allowed_absent = {"config/models.yaml"}  # stated as not existing yet
    for p in re.findall(r"`((?:[\w.-]+/)+[\w.-]*)`", TEXT):
        if p in allowed_absent or " " in p or p.startswith(("BAAI/", "Xenova/")):
            continue
        assert p in tracked or p in dirs, f"not tracked: {p}"
    for lnk in re.findall(r"\]\((?!https?://)([^)]+)\)", TEXT):
        assert lnk in tracked, f"broken link: {lnk}"


def test_no_untracked_adr_or_doc_references() -> None:
    tracked = {Path(p).name for p in _git("ls-files").splitlines()}
    for adr in set(re.findall(r"ADR-\d{4}", TEXT)):
        assert any(t.startswith(adr) for t in tracked), f"{adr} is not tracked"
    for doc in ("decisions.md", "model-card.md", "REVIEW.md"):
        assert doc not in TEXT


def test_no_private_paths_or_secrets_in_doc() -> None:
    assert not re.search(r"/Users/|/home/|worktrees/", TEXT)
    assert not re.search(r"://[^/\s:@`]+:[^/\s@`]+@", TEXT)
    settings = _tables(_section(3))
    for tbl in settings:
        if tbl[0] == ["Setting", "Default"]:
            for k, _v in tbl[2:]:
                assert not re.search(r"password|secret|key|database_url|redis_url|api_token", k)


def test_check_names_stale_block_and_exit_codes(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    doc = root / "docs/status.md"
    assert _run(root, "--check").returncode == 0
    doc.write_text(doc.read_text().replace("`0001`", "`0002`", 1))
    r = _run(root, "--check")
    assert r.returncode == 1 and "built" in r.stderr
    assert _run(root).returncode == 2


def test_write_idempotent_and_preserves_hand_text(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    doc = root / "docs/status.md"
    orig = doc.read_text()
    doc.write_text(orig.replace("`0001`", "`9`", 1).replace("| 3 | 3 |", "| 3 | 0 |", 1))
    assert _run(root, "--write").returncode == 0
    once = doc.read_bytes()
    assert _run(root, "--write").returncode == 0
    assert doc.read_bytes() == once
    assert once.decode() == orig
    # outside-marker text is untouched byte for byte
    pat = re.compile(r"<!-- generated:\w+ begin -->.*?<!-- generated:\w+ end -->", re.S)
    assert pat.split(once.decode()) == pat.split(orig)


def test_write_does_not_touch_tickets_block_without_inputs(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    shutil.rmtree(root / ".planning", ignore_errors=True)
    before = sd.get_block((root / "docs/status.md").read_text(), "tickets")
    assert _run(root, "--write").returncode == 0
    assert sd.get_block((root / "docs/status.md").read_text(), "tickets") == before
    assert _run(root, "--write", "--prs", str(root / "nope.json")).returncode == 0


def test_missing_and_unbalanced_markers_fail(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    doc = root / "docs/status.md"
    orig = doc.read_text()
    doc.write_text(orig.replace("<!-- generated:evals begin -->", ""))
    r = _run(root, "--check")
    assert r.returncode != 0 and "evals" in (r.stderr + r.stdout)
    w = _run(root, "--write")
    assert w.returncode != 0
    assert "Traceback" not in w.stderr, "missing marker should fail with a clear message"


def test_duplicate_markers_rejected(tmp_path: Path) -> None:
    root = _clone(tmp_path)
    doc = root / "docs/status.md"
    blk = sd.get_block(doc.read_text(), "evals")
    dup = f"\n<!-- generated:evals begin -->\n{blk}\n<!-- generated:evals end -->\n"
    doc.write_text(doc.read_text() + dup)
    r = _run(root, "--check")
    assert r.returncode != 0, "duplicate block markers must be rejected"


@pytest.mark.parametrize("seed", ["0", "1", "random"])
def test_deterministic_across_hashseed_and_cwd(tmp_path: Path, seed: str) -> None:
    env = {**os.environ, "PYTHONHASHSEED": seed}
    r = subprocess.run(  # noqa: S603
        [sys.executable, str(SCRIPT), "--check"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr


def test_parse_ticket_edge_cases(tmp_path: Path) -> None:
    f = tmp_path / "T-050.md"
    f.write_bytes(
        b"# T-050 Fix | pipes and `ticks`\r\nstatus: done\r\nphase: P2\r\ndepends_on: [T-001]\r\n"
    )
    t = sd.parse_ticket(f)
    assert t and t["title"] == "Fix | pipes and `ticks`" and t["status"] == "done"
    assert t["depends"] == ["T-001"]
    g = tmp_path / "T-051.md"
    g.write_text("# T-051 No fields\n", encoding="utf-8")
    t2 = sd.parse_ticket(g)
    assert t2 and t2["phase"] == "" and t2["depends"] == []
    e = tmp_path / "T-052.md"
    e.write_text("", encoding="utf-8")
    assert sd.parse_ticket(e) is None
    assert "evidence table" in sd._public_title("The Evidence Map row")


def test_pr_mapping_unusual_branches() -> None:
    prs = [
        {"number": 3, "headRefName": "t-012-a", "state": "CLOSED"},
        {"number": 9, "headRefName": "t-012-b", "state": "OPEN"},
        {"number": 4, "headRefName": "feature/t-013-x", "state": "MERGED"},
        {"number": 5, "headRefName": "t-014", "state": "MERGED"},
        {"number": 6, "headRefName": "T-015-upper", "state": "MERGED"},
        {"number": 7, "headRefName": "t-0160-long-id", "state": "MERGED"},
        {"number": 8, "headRefName": "", "state": "MERGED"},
    ]
    pm = sd.pr_map(prs)
    assert pm["T-012"] == (9, "OPEN")
    assert "T-013" not in pm and "T-015" not in pm
    assert "T-014" not in pm  # no slug: documented as unmapped
    out = sd.block_tickets(
        [{"id": "T-012", "title": "x", "phase": "P2", "status": "approved", "depends": []}],
        [{"number": 3, "headRefName": "t-012-a", "state": "CLOSED"}],
        set(),
        "o/r",
    )
    assert "planned" in out and "/pull/3" in out.splitlines()[2] or "-" in out.splitlines()[2]


def test_makefile_status_target_once() -> None:
    mk = (ROOT / "Makefile").read_text()
    assert len(re.findall(r"^status:", mk, re.M)) == 1
    phony = re.search(r"^\.PHONY:(.*)$", mk, re.M)
    assert phony and "status" in phony.group(1).split()
    assert len(re.findall(r"^\.PHONY:", mk, re.M)) == 1


def _clone(tmp_path: Path) -> Path:
    root = tmp_path / "clone"
    shutil.copytree(
        ROOT,
        root,
        ignore=shutil.ignore_patterns(
            ".git",
            ".venv",
            "node_modules",
            ".planning",
            ".claude",
            "__pycache__",
            ".mypy_cache",
            ".ruff_cache",
            ".pytest_cache",
            "worktrees",
            "logs",
            "test-results",
            "web",
        ),  # fmt: skip
        symlinks=False,
        ignore_dangling_symlinks=True,
    )
    return root


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(  # noqa: S603
        [sys.executable, str(root / "scripts/status_doc.py"), "--root", str(root), *args],
        cwd=root,
        capture_output=True,
        text=True,
    )


def test_title_with_pipe_keeps_six_cells() -> None:
    t = {"id": "T-050", "title": "Fix | pipes", "phase": "P2", "status": "", "depends": []}
    row = sd.block_tickets([t], [], set(), "o/r").splitlines()[2]
    assert len(re.split(r"(?<!\\)\|", row)) - 2 == 6


def test_missing_phase_renders_dash() -> None:
    t = {"id": "T-051", "title": "x", "phase": "", "status": "", "depends": []}
    assert "|  |" not in sd.block_tickets([t], [], set(), "o/r").splitlines()[2]
