"""Typed application settings loaded from the environment (see .env.example)."""

from functools import lru_cache
from typing import Literal
from urllib.parse import urlsplit

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

_DEV_JWT_KEY = "change-me-dev-only"
_MIN_JWT_KEY_LEN = 32
_DEV_PASSWORDS = frozenset({"rag", "change-me", "change-me-dev-only", "password", "postgres"})


class Settings(BaseSettings):
    """Every field maps to an upper-case env var of the same name."""

    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=False, hide_input_in_errors=True
    )

    app_env: Literal["local", "ci", "prod"] = "local"
    database_url: str = "postgresql+psycopg://rag:rag@postgres:5432/rag"
    redis_url: str = "redis://redis:6379/0"
    mlflow_tracking_uri: str = "http://mlflow:5000"
    provider: Literal["fake", "real"] = "fake"
    models_file: str = "config/models.yaml"
    ollama_base_url: str = "http://ollama:11434/v1"
    user_daily_token_cap: int = Field(default=200_000, ge=0)
    embedding_backend: Literal["hash", "fastembed"] = "hash"
    jwt_signing_key: str = "change-me-dev-only"
    cache_ttl_s: int = Field(default=300, ge=0)
    rate_limit_user_rpm: int = Field(default=30, ge=1)
    rate_limit_tenant_rpm: int = Field(default=300, ge=1)
    context_token_budget: int = Field(default=4000, ge=1)
    rerank_min_score: float = 0.0
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    public_base_url: str = "http://localhost:8000"
    allowed_origins: str = "http://localhost:5173"

    @model_validator(mode="after")
    def _refuse_dev_secrets_in_prod(self) -> "Settings":
        """Refuse to boot in prod with default or weak secrets (never echo values)."""
        if self.app_env != "prod":
            return self
        if self.jwt_signing_key == _DEV_JWT_KEY:
            raise ValueError("jwt_signing_key must not be the dev default when app_env=prod")
        if len(self.jwt_signing_key) < _MIN_JWT_KEY_LEN:
            raise ValueError(
                f"jwt_signing_key must be at least {_MIN_JWT_KEY_LEN} characters when app_env=prod"
            )
        for name in ("database_url", "redis_url"):
            password = urlsplit(getattr(self, name)).password
            if password is not None and password.lower() in _DEV_PASSWORDS:
                raise ValueError(f"{name} embeds a dev placeholder password when app_env=prod")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
