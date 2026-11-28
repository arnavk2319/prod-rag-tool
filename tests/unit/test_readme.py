"""README and CHANGELOG hygiene: forbidden strings, links, make targets, status."""

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = ["README.md", "CHANGELOG.md"]
FORBIDDEN = [".planning", ".claude", "CLAUDE.md", "autopilot", "implementer", "subagent"]


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


def _tracked() -> set[str]:
    out = subprocess.run(  # noqa: S603
        [shutil.which("git") or "git", "ls-files"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return set(out.splitlines())


@pytest.mark.parametrize("name", FILES)
def test_no_forbidden_strings(name: str) -> None:
    text = _text(name)
    for s in FORBIDDEN:
        assert s not in text, f"{name} contains {s!r}"


@pytest.mark.parametrize("name", FILES)
def test_relative_links_are_tracked(name: str) -> None:
    tracked = _tracked()
    for target in re.findall(r"\]\(([^)\s]+)\)", _text(name)):
        if re.match(r"^(https?:|mailto:|#)", target):
            continue
        path = target.split("#")[0]
        assert path in tracked or any(t.startswith(path.rstrip("/") + "/") for t in tracked), (
            f"{name}: link {target} is not a tracked file"
        )


def test_make_targets_exist() -> None:
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    defined = set(re.findall(r"^([A-Za-z0-9_-]+):", makefile, flags=re.M))
    mentioned = set(re.findall(r"`?make ([a-z][a-z0-9-]*)", _text("README.md")))
    assert mentioned
    assert mentioned <= defined, mentioned - defined


def test_readme_states_foundation_status() -> None:
    text = _text("README.md")
    assert "Status: v0.1 Foundation" in text
    assert "docs/status.md" in text
    assert "GPL-3.0-or-later" in text and "[LICENSE](LICENSE)" in text
    assert "LICENSE" in _tracked()
