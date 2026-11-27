"""Static tests of scripts/urls.sh, scripts/doctor.sh, scripts/ollama-pull.sh and make urls."""

from __future__ import annotations

# ruff: noqa: S603, S607, S104
import os
import shutil
import socket
import stat
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
VARS = {
    "API_PORT": "8000",
    "MLFLOW_PORT": "5000",
    "GRAFANA_PORT": "3000",
    "PROMETHEUS_PORT": "9090",
    "OLLAMA_PORT": "11434",
    "POSTGRES_PORT": "5432",
}


def _clean_env() -> dict[str, str]:
    return {k: v for k, v in os.environ.items() if k not in VARS and k != "BIND_ADDR"}


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


@pytest.mark.parametrize("name", ["doctor.sh", "urls.sh", "ollama-pull.sh"])
def test_script_syntax(name: str) -> None:
    p = ROOT / "scripts" / name
    assert subprocess.run(["bash", "-n", str(p)]).returncode == 0
    if name != "ollama-pull.sh":  # run via /bin/sh in the container
        assert p.stat().st_mode & stat.S_IXUSR


def test_urls_defaults() -> None:
    out = subprocess.run(
        [str(ROOT / "scripts/urls.sh")],
        env=_clean_env(),
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for var, port in VARS.items():
        assert var in out
        assert f"127.0.0.1:{port}" in out
    for prof in ("core", "obs", "llm"):
        assert prof in out


def test_urls_from_env_via_make() -> None:
    if shutil.which("make") is None:
        pytest.skip("make missing")
    env = _clean_env() | {v: str(30000 + i) for i, v in enumerate(VARS)} | {"BIND_ADDR": "0.0.0.0"}
    out = subprocess.run(
        ["make", "-s", "urls"], cwd=ROOT, env=env, capture_output=True, text=True, check=True
    ).stdout
    for i in range(len(VARS)):
        assert f"http://0.0.0.0:{30000 + i}" in out or f"0.0.0.0:{30000 + i}" in out
    assert "127.0.0.1" not in out


def _stub_bin(tmp_path: Path) -> Path:
    """Fake docker so doctor can run without a daemon."""
    b = tmp_path / "bin"
    b.mkdir()
    d = b / "docker"
    d.write_text(
        "#!/bin/sh\n"
        'case "$1" in\n'
        '  info) case "$*" in *MemTotal*) echo 17179869184;; esac; exit 0;;\n'
        "  compose) echo v2.40.0; exit 0;;\n"
        "esac\nexit 0\n"
    )
    d.chmod(0o755)
    return b


def _doctor(tmp_path: Path, extra: dict[str, str]) -> subprocess.CompletedProcess[str]:
    env = (
        _clean_env()
        | {
            "PATH": f"{_stub_bin(tmp_path)}:{os.environ['PATH']}",
            "DOCTOR_MIN_DISK_GB": "0",
        }
        | extra
    )
    return subprocess.run(
        ["bash", str(ROOT / "scripts/doctor.sh")],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


def test_doctor_names_occupied_port(tmp_path: Path) -> None:
    free = {v: str(_free_port()) for v in VARS}
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = str(s.getsockname()[1])
        r = _doctor(tmp_path, free | {"GRAFANA_PORT": port})
    assert r.returncode != 0
    assert "GRAFANA_PORT" in r.stderr and port in r.stderr


def test_doctor_passes_when_ports_free(tmp_path: Path) -> None:
    r = _doctor(tmp_path, {v: str(_free_port()) for v in VARS})
    assert r.returncode == 0, r.stderr
    assert "all checks passed" in r.stdout


def test_doctor_fails_without_docker(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    bash = shutil.which("bash")
    assert bash
    r = subprocess.run(
        [bash, str(ROOT / "scripts/doctor.sh")],
        env={"PATH": str(empty)},
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "docker" in r.stderr


def test_doctor_low_memory_named(tmp_path: Path) -> None:
    r = _doctor(tmp_path, {v: str(_free_port()) for v in VARS} | {"DOCTOR_MIN_MEM_GB": "999"})
    assert r.returncode != 0 and "memory" in r.stderr.lower()


def test_ollama_pull_missing_models_file(tmp_path: Path) -> None:
    r = subprocess.run(
        ["sh", str(ROOT / "scripts/ollama-pull.sh")],
        env={"PATH": os.environ["PATH"], "MODELS_FILE": str(tmp_path / "nope.yaml")},
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "not found" in r.stderr


def test_ollama_pull_no_tags(tmp_path: Path) -> None:
    f = tmp_path / "m.yaml"
    f.write_text("models: {}\n")
    r = subprocess.run(
        ["sh", str(ROOT / "scripts/ollama-pull.sh")],
        env={"PATH": os.environ["PATH"], "MODELS_FILE": str(f)},
        capture_output=True,
        text=True,
    )
    assert r.returncode != 0 and "no ollama_tag" in r.stderr


def test_urls_reads_env_file(tmp_path: Path) -> None:
    f = tmp_path / "envfile"
    f.write_text(
        'API_PORT=18123\nBIND_ADDR="0.0.0.0"  # lan\nJWT_SIGNING_KEY=secret-do-not-print\n'
    )
    out = subprocess.run(
        [str(ROOT / "scripts/urls.sh")],
        env=_clean_env() | {"ENV_FILE": str(f)},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert "0.0.0.0:18123" in out
    assert "secret-do-not-print" not in out


def test_shell_env_wins_over_env_file(tmp_path: Path) -> None:
    f = tmp_path / "envfile"
    f.write_text("API_PORT=18123\n")
    out = subprocess.run(
        [str(ROOT / "scripts/urls.sh")],
        env=_clean_env() | {"ENV_FILE": str(f), "API_PORT": "19000"},
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    assert ":19000" in out and "18123" not in out


def test_doctor_checks_port_from_env_file(tmp_path: Path) -> None:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        port = str(s.getsockname()[1])
        f = tmp_path / "envfile"
        f.write_text(f"MLFLOW_PORT={port}\n")
        free = {v: str(_free_port()) for v in VARS if v != "MLFLOW_PORT"}
        r = _doctor(tmp_path, free | {"ENV_FILE": str(f)})
    assert r.returncode != 0 and "MLFLOW_PORT" in r.stderr and port in r.stderr


def test_build_context_and_image_hygiene() -> None:
    ignore = (ROOT / ".dockerignore").read_text().split()
    for entry in (".planning", ".claude", "CLAUDE.md", ".env", "logs", "worktrees"):
        assert entry in ignore
    assert "/mlflow/artifacts" in (ROOT / "Dockerfile").read_text()
    assert (ROOT / "config/.gitkeep").exists() and (ROOT / "prompts/.gitkeep").exists()
    compose = (ROOT / "compose.yaml").read_text()
    assert "--allowed-hosts" in compose
