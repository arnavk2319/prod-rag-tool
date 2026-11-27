# Running locally

- `make doctor` then `make up`; `make urls` lists every service URL. Ports and `BIND_ADDR` come from the
  shell environment or `.env` (the shell wins), the same as Compose.
- Passwords embedded in database URLs (`RAG_OWNER_PASSWORD`, `RAG_SVC_PASSWORD`) must be URL-safe
  (letters, digits, `- . _ ~`). For mlflow you may instead set `MLFLOW_BACKEND_STORE_URI` with an encoded password.
- Grafana runs with anonymous Viewer access for local convenience. That is only safe with the default
  `BIND_ADDR=127.0.0.1`; if you set `BIND_ADDR=0.0.0.0` anyone on the network can view dashboards.
  Also set real passwords and `JWT_SIGNING_KEY` before sharing the stack (the compose defaults are dev-only placeholders).
