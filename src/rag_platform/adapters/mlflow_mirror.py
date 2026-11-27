"""MLflow registry mirror for pipeline policies (best effort; never raises). ADR-0006."""

import asyncio
import contextlib
import logging
from typing import Any

from rag_platform.domain.policy import PipelinePolicy

log = logging.getLogger(__name__)
MODEL_NAME = "rag-index-policy"
TAG_REASON_MAX = 100


def tags_for(p: PipelinePolicy) -> dict[str, str]:
    """Ids, versions and a truncated operator reason only; never document text."""
    return {
        "policy_hash": p.policy_hash,
        "change_kind": p.change_kind.value,
        "embedder": f"{p.embedder_id}@{p.embedder_version}",
        "chunker_version": p.chunker_version,
        "index_version": p.index_version,
        "reason": p.change_reason[:TAG_REASON_MAX],
    }


class MlflowRegistryMirror:
    def __init__(self, tracking_uri: str, timeout_s: float = 5.0) -> None:
        self._uri = tracking_uri
        self._timeout = timeout_s

    def _client(self) -> Any:
        import mlflow  # lazy: a missing or broken install must not break ingest
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(self._uri)
        return MlflowClient(tracking_uri=self._uri)

    async def _run(self, fn: Any) -> Any:
        return await asyncio.wait_for(asyncio.to_thread(fn), timeout=self._timeout)

    async def register_candidate(self, policy: PipelinePolicy) -> str | None:
        def work() -> str:
            c = self._client()
            with contextlib.suppress(Exception):  # already exists
                c.create_registered_model(MODEL_NAME)
            mv = c.create_model_version(
                MODEL_NAME, source=f"policy:{policy.policy_id}", tags=tags_for(policy)
            )
            c.set_registered_model_alias(MODEL_NAME, "candidate", mv.version)
            return str(mv.version)

        try:
            return str(await self._run(work))
        except Exception as e:  # noqa: BLE001
            log.warning("mlflow register failed: %s", type(e).__name__)
            return None

    async def _set_alias(self, alias: str, version: str | None) -> bool:
        """Set `alias` to `version`, or remove it when version is None. Never raises."""

        def work() -> None:
            c = self._client()
            if version is None:
                c.delete_registered_model_alias(MODEL_NAME, alias)
            else:
                c.set_registered_model_alias(MODEL_NAME, alias, version)

        try:
            await self._run(work)
            return True
        except Exception as e:  # noqa: BLE001
            log.warning("mlflow alias failed: %s", type(e).__name__)
            return False

    async def promote_champion(self, policy: PipelinePolicy) -> bool:
        version = policy.mlflow_model_version
        return version is not None and await self._set_alias("champion", version)

    async def set_rolled_back(
        self, policy: PipelinePolicy, previous: PipelinePolicy | None
    ) -> bool:
        """Point `champion` at the previous committed version.

        With no previous policy, or one without an MLflow version, this returns False
        (not synced; the outbox keeps retrying). The alias is never deleted on unknown history.
        """
        if previous is None or previous.mlflow_model_version is None:
            return False
        return await self._set_alias("champion", previous.mlflow_model_version)
