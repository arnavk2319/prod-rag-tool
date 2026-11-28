"""Source-level checks of migration 0001 (no database needed)."""

import importlib.util
from pathlib import Path

PATH = Path(__file__).resolve().parents[2] / "migrations" / "versions" / "0001_initial_schema.py"


def _load():  # type: ignore[no-untyped-def]
    spec = importlib.util.spec_from_file_location("mig0001", PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_revision_and_downgrade_defined() -> None:
    mod = _load()
    assert mod.revision == "0001"
    assert mod.down_revision is None
    assert callable(mod.upgrade)
    assert callable(mod.downgrade)


def test_alert_events_absent_from_ddl_source() -> None:
    assert "alert_events" not in PATH.read_text()


def test_pgvector_floor_and_hnsw_params_in_source() -> None:
    src = PATH.read_text()
    assert "pgvector >= 0.8" in src
    assert "vector_cosine_ops" in src
    assert "m = 16" in src
    assert "ef_construction = 64" in src
