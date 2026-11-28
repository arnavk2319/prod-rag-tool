"""Prod boot guard: refuse default or weak secrets outside local/ci."""

import pytest
from pydantic import ValidationError

from rag_platform.settings import Settings

GOOD_KEY = "k" * 40
GOOD_DB = "postgresql+psycopg://rag:s3cure-Xy9zQ@db:5432/rag"


def _prod(**kw: str) -> Settings:
    base = {"app_env": "prod", "jwt_signing_key": GOOD_KEY, "database_url": GOOD_DB}
    return Settings(_env_file=None, **{**base, **kw})  # type: ignore[arg-type]


def test_prod_default_key_raises() -> None:
    with pytest.raises(ValidationError, match="jwt_signing_key"):
        Settings(_env_file=None, app_env="prod", database_url=GOOD_DB)  # type: ignore[call-arg]


def test_prod_short_key_raises() -> None:
    with pytest.raises(ValidationError, match="jwt_signing_key"):
        _prod(jwt_signing_key="short-key")


def test_prod_good_secrets_pass() -> None:
    assert _prod().app_env == "prod"


def test_prod_placeholder_db_password_raises() -> None:
    with pytest.raises(ValidationError, match="database_url"):
        _prod(database_url="postgresql+psycopg://rag:rag@db:5432/rag")


def test_prod_placeholder_redis_password_raises() -> None:
    with pytest.raises(ValidationError, match="redis_url"):
        _prod(redis_url="redis://:change-me@redis:6379/0")


@pytest.mark.parametrize("env", ["local", "ci"])
def test_local_and_ci_defaults_pass(env: str) -> None:
    assert Settings(_env_file=None, app_env=env).app_env == env  # type: ignore[arg-type,call-arg]


def test_error_never_contains_secret() -> None:
    secret = "tiny-secret-value"  # noqa: S105
    with pytest.raises(ValidationError) as exc:
        _prod(jwt_signing_key=secret)
    assert secret not in str(exc.value)
    with pytest.raises(ValidationError) as exc2:
        _prod(database_url="postgresql+psycopg://rag:rag@db:5432/rag")
    assert ":rag@" not in str(exc2.value).replace("input_value", "")


def test_guard_runs_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_ENV", "prod")
    monkeypatch.setenv("JWT_SIGNING_KEY", "change-me-dev-only")
    with pytest.raises(ValidationError, match="jwt_signing_key"):
        Settings()
    monkeypatch.setenv("JWT_SIGNING_KEY", GOOD_KEY)
    monkeypatch.setenv("DATABASE_URL", GOOD_DB)
    assert Settings().app_env == "prod"
