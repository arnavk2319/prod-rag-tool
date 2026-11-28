"""Parent-child chunk sizes and retrieval collapse knobs, read from the environment.

Kept apart from `Settings` on purpose: these are optional tuning knobs with safe defaults and are
read from CHILD_TOKENS, PARENT_TOKENS and PARENT_TOP_N (CONTEXT_TOKEN_BUDGET lives in `Settings`).
Sizes are part of `chunker_version`, so changing one means a re-index.
"""

from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkSettings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    child_tokens: int = Field(default=300, ge=30)
    parent_tokens: int = Field(default=1200, ge=90)
    parent_top_n: int = Field(default=3, ge=1)

    @model_validator(mode="after")
    def _parent_larger_than_child(self) -> "ChunkSettings":
        # parent minimum is 2/3 of target and child maximum is 7/6 of target; keep them ordered
        if self.parent_tokens < 2 * self.child_tokens:
            raise ValueError("PARENT_TOKENS must be at least 2 x CHILD_TOKENS")
        return self


@lru_cache
def get_chunk_settings() -> ChunkSettings:
    return ChunkSettings()
