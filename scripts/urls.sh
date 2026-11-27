#!/usr/bin/env bash
# Print the service URL table (same list as implementation plan section 2).
set -u
# shellcheck source=scripts/_envfile.sh
. "$(dirname "${BASH_SOURCE[0]}")/_envfile.sh"

h="${BIND_ADDR:-127.0.0.1}"
row() { printf '%-20s %-34s %-18s %s\n' "$1" "$2" "$3" "$4"; }
row "SERVICE" "URL" "ENV VAR" "PROFILE"
row "Web UI + API" "http://$h:${API_PORT:-8000}" API_PORT core
row "MLflow" "http://$h:${MLFLOW_PORT:-5000}" MLFLOW_PORT core
row "Grafana" "http://$h:${GRAFANA_PORT:-3000}" GRAFANA_PORT obs
row "Prometheus" "http://$h:${PROMETHEUS_PORT:-9090}" PROMETHEUS_PORT obs
row "Ollama API" "http://$h:${OLLAMA_PORT:-11434}" OLLAMA_PORT llm
row "Postgres" "$h:${POSTGRES_PORT:-5432}" POSTGRES_PORT "dev compose only"
row "Redis, worker" "(internal to ragnet)" - core
