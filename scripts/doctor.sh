#!/usr/bin/env bash
# Preflight: docker, daemon, compose v2, free disk and memory, configured ports free.
# Prints one line per failure naming the cause (and the env var for ports); exits non-zero on any failure.
set -u

# shellcheck source=scripts/_envfile.sh
. "$(dirname "${BASH_SOURCE[0]}")/_envfile.sh"

MIN_DISK_GB="${DOCTOR_MIN_DISK_GB:-10}"
MIN_MEM_GB="${DOCTOR_MIN_MEM_GB:-4}"
fail=0
bad() { echo "FAIL: $*" >&2; fail=1; }

command -v docker >/dev/null 2>&1 || { bad "docker not found on PATH (install Docker)"; exit 1; }
docker info >/dev/null 2>&1 || bad "docker daemon not reachable (start Docker Desktop / dockerd)"
docker compose version --short 2>/dev/null | grep -qE '^v?2\.' \
    || bad "docker compose v2 plugin not found (need 'docker compose', not docker-compose v1)"

disk_gb=$(df -Pk . | awk 'NR==2 {printf "%d", $4/1024/1024}')
[ "${disk_gb:-0}" -ge "$MIN_DISK_GB" ] || bad "free disk ${disk_gb} GB < ${MIN_DISK_GB} GB in $(pwd)"

mem_bytes=$(docker info --format '{{.MemTotal}}' 2>/dev/null || echo 0)
mem_gb=$(( ${mem_bytes:-0} / 1024 / 1024 / 1024 ))
[ "$mem_gb" -ge "$MIN_MEM_GB" ] || bad "Docker memory ${mem_gb} GB < ${MIN_MEM_GB} GB (raise in Docker settings; llm profile needs ~10 GB)"

port_in_use() {
    local p=$1
    if command -v lsof >/dev/null 2>&1; then
        lsof -nP -iTCP:"$p" -sTCP:LISTEN >/dev/null 2>&1
    elif command -v nc >/dev/null 2>&1; then
        nc -z 127.0.0.1 "$p" >/dev/null 2>&1
    else
        (exec 3<>"/dev/tcp/127.0.0.1/$p") 2>/dev/null
    fi
}

check_port() {
    local var=$1 def=$2
    local port="${!var:-$def}"
    if port_in_use "$port"; then bad "port $port (env $var) is already in use; free it or set $var"; fi
}
check_port API_PORT 8000
check_port GRAFANA_PORT 3000
check_port PROMETHEUS_PORT 9090
check_port MLFLOW_PORT 5000
check_port OLLAMA_PORT 11434
check_port POSTGRES_PORT 5432

[ "$fail" -eq 0 ] && echo "doctor: all checks passed"
exit "$fail"
