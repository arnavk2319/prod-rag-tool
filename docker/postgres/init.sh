#!/usr/bin/env bash
# Runs once on first start of an empty data volume (as the postgres superuser).
# Role model (migration 0001): rag_app is a NOLOGIN group role holding the table grants and RLS
# policies. rag_owner owns the schema and runs migrations. rag_svc is the api/worker login and is a
# MEMBER of rag_app only: it is not the table owner, not a superuser, and cannot bypass RLS.
set -euo pipefail

: "${RAG_OWNER_PASSWORD:?}" "${RAG_SVC_PASSWORD:?}"

psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname postgres \
    -v owner_pw="$RAG_OWNER_PASSWORD" -v svc_pw="$RAG_SVC_PASSWORD" <<'SQL'
CREATE ROLE rag_app NOLOGIN NOBYPASSRLS;
CREATE ROLE rag_owner LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD :'owner_pw';
CREATE ROLE rag_svc LOGIN NOSUPERUSER NOBYPASSRLS PASSWORD :'svc_pw';
GRANT rag_app TO rag_svc;
CREATE DATABASE ragapp OWNER rag_owner;
CREATE DATABASE mlflow OWNER rag_owner;
SQL

# pgvector needs superuser (or trusted extension); create it here so migrations by rag_owner pass.
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname ragapp <<'SQL'
CREATE EXTENSION IF NOT EXISTS vector;
GRANT CONNECT ON DATABASE ragapp TO rag_svc;
SQL
