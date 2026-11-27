"""T-045 adversarial: MlflowRegistryMirror against a stub mlflow module (no network)."""

import asyncio
import subprocess
import sys
import time
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from rag_platform.adapters.mlflow_mirror import MODEL_NAME, MlflowRegistryMirror
from rag_platform.domain.policy import PipelinePolicy, policy_hash
from tests.unit.test_policy_gate import draft

ROOT = Path(__file__).resolve().parents[2]


class StubClient:
    log: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
    aliases: dict[str, str] = {}
    n = 0
    fail: str | None = None
    sleep = 0.0

    def __init__(self, tracking_uri: str | None = None) -> None:
        pass

    def _go(self, name: str, *a: Any, **k: Any) -> None:
        StubClient.log.append((name, a, k))
        if StubClient.sleep:
            time.sleep(StubClient.sleep)
        if StubClient.fail == name or StubClient.fail == "*":
            raise RuntimeError("stub failure with secret")

    def create_registered_model(self, name: str) -> None:
        self._go("create_registered_model", name)

    def create_model_version(self, name: str, source: str, tags: dict[str, str]) -> Any:
        self._go("create_model_version", name, source=source, tags=tags)
        StubClient.n += 1
        return types.SimpleNamespace(version=str(StubClient.n))

    def set_registered_model_alias(self, name: str, alias: str, version: str) -> None:
        self._go("set_registered_model_alias", name, alias, version)
        StubClient.aliases[alias] = version

    def delete_registered_model_alias(self, name: str, alias: str) -> None:
        self._go("delete_registered_model_alias", name, alias)
        StubClient.aliases.pop(alias, None)


@pytest.fixture
def stub(monkeypatch: pytest.MonkeyPatch) -> type[StubClient]:
    StubClient.log, StubClient.aliases = [], {}
    StubClient.n, StubClient.fail, StubClient.sleep = 0, None, 0.0
    ml = types.ModuleType("mlflow")
    ml.set_tracking_uri = lambda uri: None  # type: ignore[attr-defined]
    tr = types.ModuleType("mlflow.tracking")
    tr.MlflowClient = StubClient  # type: ignore[attr-defined]
    ml.tracking = tr  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlflow", ml)
    monkeypatch.setitem(sys.modules, "mlflow.tracking", tr)
    return StubClient


def pol(pid: str = "p1", version: str | None = None, **kw: Any) -> PipelinePolicy:
    d = draft(**kw)
    return PipelinePolicy(
        **d.model_dump(),
        policy_id=pid,
        policy_hash=policy_hash(d),
        declared_by="t",
        declared_at=datetime(2026, 1, 1, tzinfo=UTC),
        mlflow_model_version=version,
    )


def test_importing_adapter_does_not_import_mlflow() -> None:
    code = (
        "import sys;import rag_platform.adapters.mlflow_mirror;"
        "assert 'mlflow' not in sys.modules, 'mlflow imported eagerly'"
    )
    r = subprocess.run(  # noqa: S603
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=ROOT,
        env={"PYTHONPATH": str(ROOT / "src"), "PATH": "/usr/bin:/bin"},
    )
    assert r.returncode == 0, r.stderr


async def test_register_candidate_model_name_tags_alias(stub: type[StubClient]) -> None:
    p = pol(change_reason="why")
    v = await MlflowRegistryMirror("http://x").register_candidate(p)
    assert v == "1"
    cmv = [e for e in stub.log if e[0] == "create_model_version"][0]
    assert cmv[1][0] == MODEL_NAME == "rag-index-policy"
    assert cmv[2]["tags"] == {
        "policy_hash": p.policy_hash,
        "change_kind": "corpus_only",
        "embedder": "emb-1@1",
        "chunker_version": "c1",
        "index_version": "v1",
        "reason": "why",
    }
    assert stub.aliases == {"candidate": "1"}


async def test_promote_champion_sets_alias(stub: type[StubClient]) -> None:
    assert await MlflowRegistryMirror("u").promote_champion(pol(version="3")) is True
    assert stub.aliases == {"champion": "3"}
    assert stub.log[-1] == ("set_registered_model_alias", (MODEL_NAME, "champion", "3"), {})


async def test_promote_without_version_is_false_and_no_calls(stub: type[StubClient]) -> None:
    assert await MlflowRegistryMirror("u").promote_champion(pol()) is False
    assert stub.log == []


@pytest.mark.parametrize("op", ["register", "promote", "rollback"])
@pytest.mark.parametrize("fail", ["*", "create_model_version", "set_registered_model_alias"])
async def test_exceptions_in_calls_never_propagate(
    stub: type[StubClient], op: str, fail: str
) -> None:
    if op != "register" and fail == "create_model_version":
        pytest.skip("not on this call path")
    stub.fail = fail
    m = MlflowRegistryMirror("u")
    p = pol(version="2")
    if op == "register":
        assert await m.register_candidate(p) is None
    elif op == "promote":
        assert await m.promote_champion(p) is False
    else:
        stub.fail = "*"
        assert await m.set_rolled_back(p, None) is False


async def test_register_existing_model_error_is_tolerated(stub: type[StubClient]) -> None:
    stub.fail = "create_registered_model"
    assert await MlflowRegistryMirror("u").register_candidate(pol()) == "1"


@pytest.mark.parametrize("exc", [ImportError, ModuleNotFoundError, OSError])
async def test_import_failure_never_propagates(
    monkeypatch: pytest.MonkeyPatch, exc: type[Exception]
) -> None:
    import builtins

    real = builtins.__import__

    def fake(name: str, *a: Any, **k: Any) -> Any:
        if name.startswith("mlflow"):
            raise exc("nope")
        return real(name, *a, **k)

    monkeypatch.delitem(sys.modules, "mlflow", raising=False)
    monkeypatch.delitem(sys.modules, "mlflow.tracking", raising=False)
    monkeypatch.setattr(builtins, "__import__", fake)
    m = MlflowRegistryMirror("u")
    assert await m.register_candidate(pol()) is None
    assert await m.promote_champion(pol(version="1")) is False
    assert await m.set_rolled_back(pol(version="1"), None) is False


async def test_timeout_returns_quickly_and_does_not_raise(stub: type[StubClient]) -> None:
    stub.sleep = 0.5
    m = MlflowRegistryMirror("u", timeout_s=0.05)
    t0 = time.monotonic()
    assert await m.register_candidate(pol()) is None
    assert await m.promote_champion(pol(version="1")) is False
    assert time.monotonic() - t0 < 0.4


async def test_rollback_without_version_is_false(stub: type[StubClient]) -> None:
    assert await MlflowRegistryMirror("u").set_rolled_back(pol(), None) is False


async def test_rollback_restores_previous_champion_mlflow(stub: type[StubClient]) -> None:
    m = MlflowRegistryMirror("u")
    p1, p2 = pol("p1", "1"), pol("p2", "2", index_version="v2", parent_policy_id="p1")
    await m.promote_champion(p1)
    await m.promote_champion(p2)
    await m.set_rolled_back(p2, p1)
    assert stub.aliases.get("champion") == "1"


async def test_rollback_restores_previous_champion_via_gate() -> None:
    from tests.unit.test_policy_gate import OK, make

    gate, _, mirror, _ = make()
    ids = []
    for i in (1, 2):
        parent = ids[0] if ids else None
        p = await gate.declare(
            draft(index_version=f"v{i}", ingest_run_id=f"r{i}", parent_policy_id=parent),
            declared_by="t",
        )
        await gate.verify(p.policy_id, OK)
        await gate.mark_indexed(p.policy_id)
        await gate.mark_committed(p.policy_id)
        ids.append(p.policy_id)
    await gate.mark_rolled_back(ids[1])
    assert mirror.aliases.get("champion") == mirror.versions[ids[0]]


async def test_concurrent_register_calls_do_not_deadlock(stub: type[StubClient]) -> None:
    m = MlflowRegistryMirror("u")
    res = await asyncio.gather(*[m.register_candidate(pol(f"p{i}")) for i in range(5)])
    assert sorted(res) == ["1", "2", "3", "4", "5"]
