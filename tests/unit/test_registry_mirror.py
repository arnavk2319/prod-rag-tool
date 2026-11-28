"""MlflowRegistryMirror and the outbox runner never raise into callers."""

import builtins
import time
from typing import Any

import pytest

from rag_platform.adapters import mlflow_mirror
from rag_platform.adapters.mlflow_mirror import MlflowRegistryMirror, tags_for
from rag_platform.adapters.policy_fakes import FakePolicyRepository, FakeRegistryMirror
from rag_platform.domain.policy import PipelinePolicy, policy_hash
from rag_platform.services.policy_gate import sync_pending
from tests.unit.test_policy_gate import draft


def policy() -> PipelinePolicy:
    d = draft(change_reason="r" * 300)
    return PipelinePolicy(
        **d.model_dump(),
        policy_id="p1",
        policy_hash=policy_hash(d),
        declared_by="t",
        declared_at=__import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").UTC),
    )


def test_tags_ids_only_and_truncated() -> None:
    t = tags_for(policy())
    assert set(t) == {
        "policy_hash",
        "change_kind",
        "embedder",
        "chunker_version",
        "index_version",
        "reason",
    }
    assert len(t["reason"]) == mlflow_mirror.TAG_REASON_MAX


async def test_import_failure_returns_none_and_false(monkeypatch: pytest.MonkeyPatch) -> None:
    real = builtins.__import__

    def fake(name: str, *a: Any, **k: Any) -> Any:
        if name.startswith("mlflow"):
            raise ImportError("no mlflow")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    m = MlflowRegistryMirror("http://127.0.0.1:1")
    p = policy().model_copy(update={"mlflow_model_version": "1"})
    assert await m.register_candidate(p) is None
    assert await m.promote_champion(p) is False
    assert await m.set_rolled_back(p, None) is False


async def test_timeout_returns_none(monkeypatch: pytest.MonkeyPatch) -> None:
    m = MlflowRegistryMirror("http://127.0.0.1:1", timeout_s=0.05)

    def slow() -> Any:
        time.sleep(0.3)

    monkeypatch.setattr(m, "_client", slow)
    assert await m.register_candidate(policy()) is None


async def test_outbox_retries_and_never_raises() -> None:
    repo = FakePolicyRepository()
    await repo.create(policy())
    down = FakeRegistryMirror(down=True)
    assert await sync_pending(repo, down, 10) == 0
    got = await repo.get("p1")
    assert got is not None and not got.mlflow_synced
    up = FakeRegistryMirror()
    assert await sync_pending(repo, up, 10) == 1
    assert await sync_pending(repo, up, 10) == 0


async def test_outbox_survives_raising_mirror() -> None:
    class Boom:
        async def register_candidate(self, p: PipelinePolicy) -> str | None:
            raise RuntimeError("boom")

        async def promote_champion(self, p: PipelinePolicy) -> bool:
            raise RuntimeError("boom")

        async def set_rolled_back(
            self, p: PipelinePolicy, previous: PipelinePolicy | None = None
        ) -> bool:
            raise RuntimeError("boom")

    repo = FakePolicyRepository()
    await repo.create(policy())
    assert await sync_pending(repo, Boom(), 10) == 0
