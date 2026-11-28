"""EvalRunReader over `gate_runs` (ADR-0006): conservative, Postgres only.

A run passes only when the row exists, its stored `passed` flag is true and every stored metric
is a finite number. Anything else (missing row, bad metric payload) counts as not passed.
"""

import math
from typing import Any, LiteralString

import psycopg
from psycopg import sql
from psycopg.rows import dict_row

_SELECT: LiteralString = "SELECT passed, metrics FROM gate_runs WHERE gate_run_id = %s"


def metrics_clean(metrics: Any) -> bool:
    """Pure check: a dict whose values are all finite, non-bool numbers."""
    if not isinstance(metrics, dict):
        return False
    try:
        return all(
            isinstance(v, int | float) and not isinstance(v, bool) and math.isfinite(v)
            for v in metrics.values()
        )
    except OverflowError:  # absurdly large integers: fail closed
        return False


class SqlEvalRuns:
    def __init__(self, dsn: str, *, role: str | None = "rag_app") -> None:
        self._dsn = dsn.replace("postgresql+psycopg://", "postgresql://")
        self._role = role

    async def passed(self, eval_run_id: str) -> bool:
        async with await psycopg.AsyncConnection.connect(self._dsn, row_factory=dict_row) as conn:
            if self._role:
                await conn.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(self._role)))
            cur = await conn.execute(_SELECT, (eval_run_id,))
            row = await cur.fetchone()
        return bool(row and row["passed"] is True and metrics_clean(row["metrics"]))
