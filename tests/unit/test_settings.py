import pytest
from pydantic import ValidationError

from rag_platform.settings import Settings

ENV_VARS = [name.upper() for name in Settings.model_fields]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_defaults() -> None:
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.app_env == "local"
    assert s.provider == "fake"
    assert s.embedding_backend == "hash"
    assert s.user_daily_token_cap == 200_000
    assert s.cache_ttl_s == 300
    assert s.rate_limit_user_rpm == 30
    assert s.rate_limit_tenant_rpm == 300
    assert s.context_token_budget == 3000
    assert s.log_level == "INFO"


def test_field_names_match_spec_env_vars() -> None:
    expected = {
        "APP_ENV", "DATABASE_URL", "REDIS_URL", "MLFLOW_TRACKING_URI", "PROVIDER",
        "MODELS_FILE", "OLLAMA_BASE_URL", "USER_DAILY_TOKEN_CAP", "EMBEDDING_BACKEND",
        "JWT_SIGNING_KEY", "CACHE_TTL_S", "RATE_LIMIT_USER_RPM", "RATE_LIMIT_TENANT_RPM",
        "CONTEXT_TOKEN_BUDGET", "RERANK_MIN_SCORE", "LOG_LEVEL", "PUBLIC_BASE_URL",
        "ALLOWED_ORIGINS",
    }  # fmt: skip
    assert set(ENV_VARS) == expected


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RATE_LIMIT_USER_RPM", "7")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("RERANK_MIN_SCORE", "0.25")
    s = Settings(_env_file=None)  # type: ignore[call-arg]
    assert s.rate_limit_user_rpm == 7
    assert s.log_level == "DEBUG"
    assert s.rerank_min_score == 0.25


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("APP_ENV", "staging"),
        ("PROVIDER", "openai"),
        ("EMBEDDING_BACKEND", "bogus"),
        ("LOG_LEVEL", "TRACE"),
        ("USER_DAILY_TOKEN_CAP", "-1"),
        ("RATE_LIMIT_USER_RPM", "0"),
        ("RATE_LIMIT_TENANT_RPM", "0"),
        ("CONTEXT_TOKEN_BUDGET", "0"),
        ("CACHE_TTL_S", "abc"),
    ],
)
def test_bad_values_rejected(monkeypatch: pytest.MonkeyPatch, var: str, value: str) -> None:
    monkeypatch.setenv(var, value)
    with pytest.raises(ValidationError):
        Settings(_env_file=None)  # type: ignore[call-arg]
