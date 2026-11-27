"""Static checks of compose.yaml, postgres init, observability configs (no containers started)."""
# ruff: noqa: S603, S607, S104

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
CORE = {"api", "worker", "postgres", "redis", "mlflow"}
LLM = {"ollama", "ollama-pull"}
OBS = {"prometheus", "grafana"}
ONE_SHOT = {"ollama-pull"}
PORT_VARS = {
    "api": ("API_PORT", "8000"),
    "mlflow": ("MLFLOW_PORT", "5000"),
    "postgres": ("POSTGRES_PORT", "5432"),
    "ollama": ("OLLAMA_PORT", "11434"),
    "prometheus": ("PROMETHEUS_PORT", "9090"),
    "grafana": ("GRAFANA_PORT", "3000"),
}

needs_docker = pytest.mark.skipif(shutil.which("docker") is None, reason="docker not installed")


def _compose(*args: str, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    e = {k: v for k, v in os.environ.items() if not k.endswith("_PORT") and k != "BIND_ADDR"}
    e.update(env or {})
    return subprocess.run(
        ["docker", "compose", *args], cwd=ROOT, env=e, capture_output=True, text=True, timeout=60
    )


def _config(profiles: bool, env: dict[str, str] | None = None) -> dict[str, Any]:
    flags = ["--profile", "llm", "--profile", "obs"] if profiles else []
    r = _compose(*flags, "config", "--format", "json", env=env)
    if r.returncode != 0 and "compose" in r.stderr.lower() and "not a docker command" in r.stderr:
        pytest.skip("docker compose plugin missing")
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)  # type: ignore[no-any-return]


@pytest.fixture(scope="module")
def full() -> dict[str, Any]:
    if shutil.which("docker") is None:
        pytest.skip("docker not installed")
    return _config(True)


@needs_docker
@pytest.mark.parametrize("profiles", [False, True])
def test_config_valid(profiles: bool) -> None:
    flags = ["--profile", "llm", "--profile", "obs"] if profiles else []
    r = _compose(*flags, "config", "-q")
    assert r.returncode == 0, r.stderr


@needs_docker
def test_core_has_only_core_services() -> None:
    assert set(_config(False)["services"]) == CORE


def test_profiles_services(full: dict[str, Any]) -> None:
    assert set(full["services"]) == CORE | LLM | OBS
    for n in LLM:
        assert full["services"][n]["profiles"] == ["llm"]
    for n in OBS:
        assert full["services"][n]["profiles"] == ["obs"]
    for n in CORE:
        assert "profiles" not in full["services"][n]


def test_no_parked_containers() -> None:
    text = (ROOT / "compose.yaml").read_text()
    for parked in (
        "qdrant",
        "neo4j",
        "weaviate",
        "elasticsearch",
        "opensearch",
        "minio",
        "langfuse",
        "kafka",
        "rabbitmq",
        "celery",
        "flower",
        "jaeger",
        "loki",
    ):
        assert not re.search(rf"^\s*{parked}\s*:", text, re.M | re.I), parked


def test_mem_limit_and_healthcheck(full: dict[str, Any]) -> None:
    for name, svc in full["services"].items():
        assert int(svc.get("mem_limit", 0)) > 0, f"{name}: no mem_limit"
        if name not in ONE_SHOT:
            assert "healthcheck" in svc, f"{name}: no healthcheck"


def test_ollama_pull_is_one_shot(full: dict[str, Any]) -> None:
    assert full["services"]["ollama-pull"]["restart"] == "no"


def test_ports_published_from_env(full: dict[str, Any]) -> None:
    text = (ROOT / "compose.yaml").read_text()
    for name, svc in full["services"].items():
        if name not in PORT_VARS:
            assert not svc.get("ports"), f"{name} must not publish ports"
            continue
        var, default = PORT_VARS[name]
        assert f"${{{var}:-{default}}}" in text
        (p,) = svc["ports"]
        assert p["host_ip"] == "127.0.0.1", f"{name} BIND_ADDR default"
        assert p["published"] == default


def test_no_hardcoded_published_port() -> None:
    for line in (ROOT / "compose.yaml").read_text().splitlines():
        if re.match(r'\s*-\s*"[^"]*:\d+:\d+"', line):
            assert "${BIND_ADDR" in line and "_PORT" in line, line


def test_default_ports_do_not_collide(full: dict[str, Any]) -> None:
    pubs = [p["published"] for s in full["services"].values() for p in s.get("ports", [])]
    assert len(pubs) == len(set(pubs)) == len(PORT_VARS)


@needs_docker
def test_api_port_override_and_bind_addr() -> None:
    cfg = _config(False, {"API_PORT": "8123", "BIND_ADDR": "0.0.0.0"})
    (p,) = cfg["services"]["api"]["ports"]
    assert p["published"] == "8123" and p["target"] == 8000 and p["host_ip"] == "0.0.0.0"
    assert "8123" in cfg["services"]["api"]["environment"]["PUBLIC_BASE_URL"]


@needs_docker
def test_other_port_overrides() -> None:
    env = {v: str(20000 + i) for i, (v, _) in enumerate(PORT_VARS.values())}
    cfg = _config(True, env)
    for name, (var, _) in PORT_VARS.items():
        assert cfg["services"][name]["ports"][0]["published"] == env[var]


def test_api_depends_only_on_postgres_redis(full: dict[str, Any]) -> None:
    deps = set(full["services"]["api"]["depends_on"])
    assert deps == {"postgres", "redis"}
    assert not deps & ({"mlflow"} | OBS | LLM)


def test_images_pinned(full: dict[str, Any]) -> None:
    for name, svc in full["services"].items():
        img = svc["image"]
        assert ":" in img and not img.endswith(":latest"), f"{name}: {img}"
    text = (ROOT / "Dockerfile").read_text()
    for m in re.finditer(r"^FROM\s+(\S+)", text, re.M):
        assert ":" in m.group(1) and ":latest" not in m.group(1), m.group(1)


def test_no_real_looking_secrets() -> None:
    text = (ROOT / "compose.yaml").read_text()
    assert not re.search(r"sk-[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|ghp_[A-Za-z0-9]{20,}", text)
    for m in re.finditer(r"(PASSWORD|KEY|SECRET|TOKEN)\w*:\s*\$\{[A-Z_]+:-([^}]*)\}", text):
        assert m.group(2).startswith(("dev-only", "change-me")) or m.group(2) == "", m.group(0)
    for m in re.finditer(r"^\s*\w*(PASSWORD|SECRET|TOKEN)\w*:\s*([^$\s#][^\s#]*)", text, re.M):
        pytest.fail(f"literal secret: {m.group(0)}")


def test_postgres_init_script() -> None:
    sql = (ROOT / "docker/postgres/init.sh").read_text()
    assert re.search(r"CREATE DATABASE ragapp\b", sql)
    assert re.search(r"CREATE DATABASE mlflow\b", sql)
    assert re.search(r"GRANT rag_app TO rag_svc", sql)
    assert re.search(r"CREATE ROLE rag_app NOLOGIN", sql)
    for m in re.finditer(r"CREATE ROLE (\w+)([^;]*);", sql):
        attrs = m.group(2)
        assert not re.search(r"(?<!NO)SUPERUSER", attrs), m.group(1)
        assert not re.search(r"(?<!NO)BYPASSRLS", attrs), m.group(1)
        assert "NOBYPASSRLS" in attrs, m.group(1)
    assert "rag_svc" in sql and "OWNER rag_svc" not in sql


def test_init_script_mounted_and_executable_syntax() -> None:
    cfg = (ROOT / "compose.yaml").read_text()
    assert "./docker/postgres/init.sh:/docker-entrypoint-initdb.d/" in cfg
    r = subprocess.run(["bash", "-n", str(ROOT / "docker/postgres/init.sh")], capture_output=True)
    assert r.returncode == 0


def test_database_url_uses_rag_svc(full: dict[str, Any]) -> None:
    url = full["services"]["api"]["environment"]["DATABASE_URL"]
    assert url.startswith("postgresql+psycopg://rag_svc:") and url.endswith("/ragapp")
    assert "postgres:" not in url.split("@")[0]


@pytest.mark.xfail(
    reason="known gap (T-004 Findings): config/models.yaml not created yet; ollama-pull exits 1",
    strict=True,
)
def test_models_file_exists() -> None:
    """compose mounts ./config and sets MODELS_FILE; ollama-pull reads it."""
    assert (ROOT / "config" / "models.yaml").is_file()


def test_prometheus_targets_match_services(full: dict[str, Any]) -> None:
    text = (ROOT / "observability/prometheus/prometheus.yml").read_text()
    targets = re.findall(r'"([\w-]+):(\d+)"', text)
    assert targets
    for host, _port in targets:
        assert host in full["services"], host
        assert host in CORE, f"{host} scraped but not a core service"
    api = dict(targets)["api"]
    assert f'--port", "{api}"' in (ROOT / "compose.yaml").read_text().replace("'", '"')


def test_grafana_datasource_points_at_prometheus(full: dict[str, Any]) -> None:
    text = (ROOT / "observability/grafana/provisioning/datasources/prometheus.yml").read_text()
    m = re.search(r"url:\s*http://([\w-]+):(\d+)", text)
    assert m and m.group(1) in full["services"] and m.group(1) in OBS
    assert m.group(2) == PORT_VARS["prometheus"][1]


def test_dashboard_json_valid_and_provisioned() -> None:
    d = json.loads((ROOT / "observability/grafana/dashboards/overview.json").read_text())
    assert d["uid"] and d["title"] and isinstance(d["panels"], list)
    prov = (ROOT / "observability/grafana/provisioning/dashboards/dashboards.yml").read_text()
    assert "/var/lib/grafana/dashboards" in prov
    assert (
        "./observability/grafana/dashboards:/var/lib/grafana/dashboards"
        in (ROOT / "compose.yaml").read_text()
    )
    for p in d["panels"]:
        ds = p.get("datasource")
        if isinstance(ds, dict) and ds.get("uid"):
            assert ds["uid"] == "prometheus", p.get("title")


def test_compose_mounts_exist() -> None:
    text = (ROOT / "compose.yaml").read_text()
    for m in re.finditer(r"^\s*-\s*(\./[^:\s]+):", text, re.M):
        path = ROOT / m.group(1)
        if path.name == "config":
            continue  # covered by test_models_file_exists
        assert path.exists(), m.group(1)


def test_only_one_service_builds_the_shared_image(full: dict[str, Any]) -> None:
    """Regression (first real run): parallel builds of one tag fail with 'image already exists'."""
    builders = [n for n, s in full["services"].items() if "build" in s]
    assert builders == ["api"]
    for name in ("worker", "mlflow"):
        assert full["services"][name]["image"] == full["services"]["api"]["image"]


def test_mlflow_job_execution_disabled(full: dict[str, Any]) -> None:
    """Regression (first real run): MLflow job runner (~8 consumers) OOMs the 768m cap."""
    env = full["services"]["mlflow"]["environment"]
    assert env["MLFLOW_SERVER_ENABLE_JOB_EXECUTION"] == "false"
    assert env["MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS"] == "false"
    assert "--workers=1" in full["services"]["mlflow"]["command"]
