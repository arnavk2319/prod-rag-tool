#!/usr/bin/env bash
# Apply Alembic migrations to the compose Postgres as the OWNER role (rag_owner). Local addresses only.
# The app itself connects as rag_svc; only the owner may create tables. The password is never printed.
# MIGRATE_DRY_RUN=1 prints the target with the password masked and exits without running alembic.
set -eu
# shellcheck source=scripts/_envfile.sh
. "$(dirname "${BASH_SOURCE[0]}")/_envfile.sh"

bind="${BIND_ADDR:-127.0.0.1}"
case "$bind" in
    "" | 0.0.0.0 | "::") host="127.0.0.1" ;; # wildcard bind: the published port is still reachable on loopback
    *) host="$bind" ;;
esac
case "$host" in
    127.0.0.1 | localhost | ::1) ;;
    *)
        echo "make migrate: refusing non-local database host '$host' (allowed: 127.0.0.1, localhost, ::1)." >&2
        exit 1
        ;;
esac

port="${POSTGRES_PORT:-5432}"
case "$port" in
    "" | *[!0-9]*)
        echo "make migrate: POSTGRES_PORT must be a number." >&2
        exit 1
        ;;
esac

# Same default as compose.yaml; the shell wins over .env, as in compose.
pw="${RAG_OWNER_PASSWORD:-}"
if [ -z "$pw" ] && [ -f "${ENV_FILE:-.env}" ]; then
    line=$(grep -E '^[[:space:]]*(export[[:space:]]+)?RAG_OWNER_PASSWORD=' "${ENV_FILE:-.env}" | tail -n1 || true)
    pw=${line#*=}
    pw=$(printf '%s' "$pw" | sed -E 's/^[[:space:]]+//; s/[[:space:]]+$//; s/^"(.*)"$/\1/; s/^'"'"'(.*)'"'"'$/\1/')
fi
pw="${pw:-dev-only-owner-password}"

uhost="$host"
[ "$host" = "::1" ] && uhost="[::1]"

if [ "${MIGRATE_DRY_RUN:-}" = "1" ]; then
    echo "make migrate: would apply migrations as rag_owner to ${uhost}:${port}/ragapp (password hidden)"
    exit 0
fi

echo "make migrate: applying migrations as rag_owner to ${uhost}:${port}/ragapp"
DATABASE_URL="postgresql+psycopg://rag_owner:${pw}@${uhost}:${port}/ragapp" exec uv run alembic upgrade head
