#!/usr/bin/env bash
# Sourced helper: load port/bind vars from .env (or $ENV_FILE) the way compose does.
# The shell environment wins over .env. Only the listed non-secret vars are read; values are never printed.
load_env_ports() {
    local f="${ENV_FILE:-.env}" var line val
    [ -f "$f" ] || return 0
    for var in BIND_ADDR API_PORT GRAFANA_PORT PROMETHEUS_PORT MLFLOW_PORT OLLAMA_PORT POSTGRES_PORT; do
        [ -n "${!var:-}" ] && continue
        line=$(grep -E "^[[:space:]]*(export[[:space:]]+)?${var}=" "$f" | tail -n1 || true)
        [ -n "$line" ] || continue
        val=${line#*=}
        val=${val%%#*}
        val=$(printf '%s' "$val" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/')
        [ -n "$val" ] && export "$var=$val"
    done
    return 0
}
load_env_ports
