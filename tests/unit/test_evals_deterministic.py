from pathlib import Path

from evals import run


def test_two_runs_are_byte_identical(tmp_path: Path) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    assert run.main(["--suite", "smoke", "--out", str(a)]) == 0
    assert run.main(["--suite", "smoke", "--out", str(b)]) == 0
    assert a.read_bytes() == b.read_bytes()
