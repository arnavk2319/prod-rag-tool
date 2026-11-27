"""Typed application settings loaded from the environment (see .env.example)."""

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Every field maps to an upper-case env var of the same name."""

    model_config = SettingsConfigDict(extra="ignore", case_sensitive=False)

    app_env: Literal["local", "ci", "prod"] = "local"
    database_url: str = "postgresql+psycopg://rag:rag@postgres:5432/rag"
    redis_url: str = "redis://redis:6379/0"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    provider: Literal["fake", "real"] = "fake"
    models_file: str = "config/models.yaml"
    ollama_base_url: str = "http://ollama:11434/v1"
    user_daily_token_cap: int = Field(default=200_000, ge=0)
    embedding_backend: Literal["fake", "fastembed"] = "fake"
    jwt_signing_key: str = "change-me-dev-only"
    cache_ttl_s: int = Field(default=300, ge=0)
    rate_limit_user_rpm: int = Field(default=30, ge=1)
    rate_limit_tenant_rpm: int = Field(default=300, ge=1)
    context_token_budget: int = Field(default=3000, ge=1)
    rerank_min_score: float = 0.0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    public_base_url: str = "http://localhost:8000"
    allowed_origins: str = "http://localhost:5173"


@lru_cache
def get_settings() -> Settings:
    return Settings()
