"""docs/status.md generator: freshness, determinism, forbidden strings, parsers."""

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("status_doc", ROOT / "scripts/status_doc.py")
assert _spec and _spec.loader
sd = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sd)

DOC = ROOT / "docs/status.md"
FORBIDDEN = [
    ".planning",
    ".claude",
    "CLAUDE.md",
    "spec.md",
    "implementation_plan",
    "evidence-map",
    "quality-rubric",
    "backlog",
    "Acceptance",
]


def test_doc_has_no_forbidden_strings() -> None:
    text = DOC.read_text(encoding="utf-8").lower()
    for bad in FORBIDDEN:
        assert bad.lower() not in text, bad


def test_doc_has_required_sections() -> None:
    text = DOC.read_text(encoding="utf-8")
    for head in (
        "## 1. Component map",
        "## 2. Tickets",
        "## 3. Currently built",
        "## 4. Decisions",
        "## 5. Evals",
        "## 6. Known gaps",
    ):
        assert head in text
    assert "Contributor rule" in text


@pytest.mark.parametrize("name", sd.TRACKED_BLOCKS)
def test_tracked_blocks_are_fresh(name: str) -> None:
    blocks = sd.compute_blocks(ROOT, None, None, set(), sd.DEFAULT_REPO)
    cur = sd.get_block(DOC.read_text(encoding="utf-8"), name)
    assert cur is not None
    assert cur.strip() == blocks[name].strip(), f"stale block: {name} (run make status)"


def test_check_cli_passes_and_names_stale_block(tmp_path: Path) -> None:
    assert sd.main(["--check"]) == 0
    text = DOC.read_text(encoding="utf-8")
    assert (
        sd.check(
            text.replace("`hash`", "`HASH`", 1), sd.compute_blocks(ROOT, None, None, set(), "r")
        )
        == "built"
    )


def test_deterministic_and_no_timestamps() -> None:
    a = sd.compute_blocks(ROOT, None, None, set(), "r")
    b = sd.compute_blocks(ROOT, None, None, set(), "r")
    assert a == b
    for body in a.values():
        assert not re.search(r"\d{4}-\d{2}-\d{2}", body)


def test_replace_block_leaves_other_text() -> None:
    text = "intro\n<!-- generated:x begin -->\nold\n<!-- generated:x end -->\noutro\n"
    out = sd.replace_block(text, "x", "new")
    assert out == "intro\n<!-- generated:x begin -->\nnew\n<!-- generated:x end -->\noutro\n"
    assert sd.get_block(out, "x") == "new"
    assert sd.replace_block(out, "x", "new") == out
    with pytest.raises(KeyError):
        sd.replace_block(text, "missing", "z")


def test_parse_ticket_reads_only_allowed_fields(tmp_path: Path) -> None:
    f = tmp_path / "T-099.md"
    f.write_text(
        "# T-099 Some title\nstatus: done\nphase: P2\nspec: secret\ndepends_on: [T-001, T-002]\n"
        "size: M\n## Acceptance\n- [ ] hidden\n",
        encoding="utf-8",
    )
    t = sd.parse_ticket(f)
    assert t == {
        "id": "T-099",
        "title": "Some title",
        "phase": "P2",
        "status": "done",
        "depends": ["T-001", "T-002"],
    }
    bad = tmp_path / "x.md"
    bad.write_text("no heading\n", encoding="utf-8")
    assert sd.parse_ticket(bad) is None


def test_tickets_block_and_pr_mapping(tmp_path: Path) -> None:
    d = tmp_path / "tickets"
    d.mkdir()
    for n, status in (("1", "done"), ("2", "approved"), ("3", "approved"), ("4", "approved")):
        (d / f"T-00{n}.md").write_text(
            f"# T-00{n} Title {n}\nstatus: {status}\nphase: P1\ndepends_on: []\n", encoding="utf-8"
        )
    prs = [
        {"number": 5, "headRefName": "t-001-scaffold", "state": "MERGED"},
        {"number": 6, "headRefName": "t-002-x", "state": "OPEN"},
        {"number": 7, "headRefName": "chore/other", "state": "MERGED"},
    ]
    pf = tmp_path / "prs.json"
    pf.write_text(json.dumps(prs), encoding="utf-8")
    blocks = sd.compute_blocks(ROOT, d, pf, {"T-003"}, "o/r")
    rows = {line.split("|")[1].strip(): line for line in blocks["tickets"].splitlines()[2:]}
    assert "merged" in rows["T-001"] and "/pull/5" in rows["T-001"]
    assert "in review" in rows["T-002"]
    assert "in progress" in rows["T-003"]
    assert "planned" in rows["T-004"]
    # missing inputs leave the block out
    assert "tickets" not in sd.compute_blocks(ROOT, tmp_path / "none", pf, set(), "o/r")
    assert "tickets" not in sd.compute_blocks(ROOT, d, None, set(), "o/r")
