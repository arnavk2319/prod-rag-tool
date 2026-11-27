# P1 memory measurement

Status: measured once on the machine below. Idle readings only (no load, no model request except where stated); Ollama measured with no model loaded because `config/models.yaml` does not exist yet, so `make up-llm` model pulls and the two-model `ollama ps` check are still pending.

## Procedure

1. Record hardware (machine, CPU, RAM), Docker runtime and its configured memory limit, date, git SHA.
2. `make clean && make up`; wait until `docker compose ps` shows all healthy; idle 60 s.
3. `docker stats --no-stream --format 'table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}'`; record per service.
4. Repeat for `make up-obs` (core + obs) and `make up-llm` (core + llm, one model loaded after one request).
5. llm profile: confirm `OLLAMA_MAX_LOADED_MODELS=1` (`docker compose exec ollama env | grep MAX_LOADED`), send one
   request to each of two models in turn, and check `ollama ps` lists a single loaded model.
6. Compare with the spec estimates (Architecture, Containers) and mark each confirmed or corrected.
7. Record also `docker compose config | grep mem_limit` to show the configured caps.

## Environment

| Item | Value |
| --- | --- |
| Hardware | MacBook Pro (Mac14,9), Apple M2 Pro, 16 GB RAM |
| Docker runtime / VM memory | Docker 28.5.1, VM limit 7.65 GiB |
| Date | 2026-11-27 |
| Git SHA | branch t-004-docker-compose (see PR) |

## Results

Core and obs readings were taken about 60 s after `--wait` reported healthy. `make up-obs` (core + obs, built from scratch) became healthy in about 70 s after images were pulled.

| Profile | Service | Idle MiB | Configured mem_limit | Verdict |
| --- | --- | --- | --- | --- |
| core | api | 40 | 512m | cap generous; revisit once the real pipeline loads models |
| core | worker (placeholder) | 5 | 512m | placeholder only |
| core | postgres | 79 | 768m | fine at idle |
| core | redis | 9 | 192m | fine |
| core | mlflow | 82-371 (settles near 370 after start) | 768m | fine only with job execution disabled, see below |
| obs | prometheus | 26 | 384m | fine |
| obs | grafana | 133 | 384m | fine |
| llm | ollama (no model loaded) | 42 | 5g | model memory not yet measured |
| llm | `OLLAMA_MAX_LOADED_MODELS=1` | confirmed present in the container env (`OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_NUM_PARALLEL=1`) | - | the two-model `ollama ps` check is pending |

Core plus obs idle total is about 0.7 GB, well under the 8.5 GB of configured limits; the spec's "about 8.5 GB" figure counts limits, not use, and stays an upper bound until models are loaded.

## Findings from the first real run

- MLflow 3.x starts a background job runner (about eight huey consumers at about 240 MiB each) that pushed the container past 1.5 GiB and into OOM restarts. The tracking server does not need it, so `MLFLOW_SERVER_ENABLE_JOB_EXECUTION=false` and `MLFLOW_SERVER_JOB_ENABLE_PERIODIC_TASKS=false` are set, which brought it to about 370 MiB.
- Three services built the same image tag in parallel and failed with "image already exists"; only `api` builds now.
- On macOS, port 5000 is taken by AirPlay Receiver and 11434 by a native Ollama if one is installed. `make doctor` reports both by variable name; set `MLFLOW_PORT` and `OLLAMA_PORT` in `.env`.
