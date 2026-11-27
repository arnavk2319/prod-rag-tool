import re
from pathlib import Path

from rag_platform.settings import Settings

ENV_EXAMPLE = Path(__file__).resolve().parents[2] / ".env.example"


def test_env_example_matches_settings() -> None:
    text = ENV_EXAMPLE.read_text()
    documented = set(re.findall(r"^#?\s*([A-Z][A-Z0-9_]+)=", text, flags=re.M))
    expected = {name.upper() for name in Settings.model_fields}
    assert expected <= documented, f"missing from .env.example: {expected - documented}"
