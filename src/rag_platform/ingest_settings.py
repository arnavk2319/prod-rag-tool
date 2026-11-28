"""Ingest worker knobs, read from the environment.

Kept apart from `Settings` (like `chunk_settings`) so a new optional knob does not require an
`.env.example` edit before it can merge; each variable is listed in the T-010 findings for the
human to add. All have safe defaults.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class IngestSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    max_upload_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    ingest_stream: str = "rag:ingest:jobs"
    ingest_group: str = "ingest-workers"
    ingest_claim_idle_ms: int = Field(default=60_000, ge=0)
    embed_batch_size: int = Field(default=64, ge=1)


@lru_cache
def get_ingest_settings() -> IngestSettings:
    return IngestSettings()
