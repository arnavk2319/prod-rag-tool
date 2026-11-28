"""T-011: pure gate decision at its boundaries, and no composed SQL in the new adapter."""

import ast
from pathlib import Path

import pytest

from rag_platform.services.index_versions import GateMeasures, evaluate

SRC = Path(__file__).resolve().parents[2] / "src" / "rag_platform"


def _m(**kw: int | None) -> GateMeasures:
    base = {"rows_actual": 100, "rows_recorded": 100, "self_total": 100, "self_hits": 100,
            "parent_rows": 10, "parents_referenced": 10}  # fmt: skip
    base.update(kw)
    return GateMeasures(**base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("hits", "total", "ok"),
    [(98, 100, True), (49, 50, True), (97, 100, False), (4899, 5000, False), (4900, 5000, True),
     (0, 0, True), (1, 1, True), (0, 1, False)],
)  # fmt: skip
def test_self_retrieval_boundary(hits: int, total: int, ok: bool) -> None:
    r = evaluate(_m(self_hits=hits, self_total=total))
    assert r.passed is ok
    assert ("self_retrieval" in r.failures) is (not ok)


@pytest.mark.parametrize(
    ("field", "name"),
    [
        ("bad_dim", "vector_sanity"),
        ("nonfinite", "vector_sanity"),
        ("zero_vectors", "vector_sanity"),
        ("orphan_children", "orphan_children"),
    ],
)
def test_single_violation_fails(field: str, name: str) -> None:
    r = evaluate(_m(**{field: 1}))
    assert not r.passed and name in r.failures


@pytest.mark.parametrize(("rec", "act"), [(100, 101), (100, 99), (None, 100), (None, 0)])
def test_row_parity_both_directions_and_unknown(rec: int | None, act: int) -> None:
    r = evaluate(_m(rows_recorded=rec, rows_actual=act))
    assert r.failures == ("row_parity",)


@pytest.mark.parametrize(("parents", "refd"), [(11, 10), (9, 10)])
def test_parent_parity_off_by_one_both_ways(parents: int, refd: int) -> None:
    r = evaluate(_m(parent_rows=parents, parents_referenced=refd))
    assert r.failures == ("parent_row_parity",)


def test_clean_measures_pass_and_metrics_are_counts_only() -> None:
    r = evaluate(_m())
    assert r.passed and r.failures == ()
    assert all(isinstance(v, (int, float, type(None))) for v in r.metrics.values())


def test_no_composed_sql_in_new_adapter() -> None:
    """Stand-in for semgrep (not installed here): execute() gets a constant/Name/sql.SQL only."""
    tree = ast.parse((SRC / "adapters" / "sql_index_versions.py").read_text())
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "execute" and n.args:
            a = n.args[0]
            assert not isinstance(a, (ast.JoinedStr, ast.BinOp)), ast.unparse(n)
            if isinstance(a, ast.Call):  # only sql.SQL(...).format(...) allowed
                assert "SQL" in ast.unparse(a), ast.unparse(n)
            if isinstance(a, ast.Constant):
                assert isinstance(a.value, str)
