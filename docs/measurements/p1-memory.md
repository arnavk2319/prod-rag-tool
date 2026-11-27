# P1 memory measurement

Status: NOT YET MEASURED. The table below holds no results; nothing here is an estimate presented as fact.

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
| Hardware | NOT YET MEASURED |
| Docker runtime / VM memory | NOT YET MEASURED |
| Date | NOT YET MEASURED |
| Git SHA | NOT YET MEASURED |

## Results

| Profile | Service | Peak/idle MiB | Configured mem_limit | Spec estimate | Verdict |
| --- | --- | --- | --- | --- | --- |
| core | api | NOT YET MEASURED | 512m | NOT YET MEASURED | - |
| core | worker (placeholder) | NOT YET MEASURED | 512m | NOT YET MEASURED | - |
| core | postgres | NOT YET MEASURED | 768m | NOT YET MEASURED | - |
| core | redis | NOT YET MEASURED | 192m | NOT YET MEASURED | - |
| core | mlflow | NOT YET MEASURED | 768m | NOT YET MEASURED | - |
| obs | prometheus | NOT YET MEASURED | 384m | NOT YET MEASURED | - |
| obs | grafana | NOT YET MEASURED | 384m | NOT YET MEASURED | - |
| llm | ollama (1 model loaded) | NOT YET MEASURED | 5g | NOT YET MEASURED | - |
| llm | `OLLAMA_MAX_LOADED_MODELS=1` check | NOT YET MEASURED | - | - | - |

The mem_limit values are starting caps chosen before any measurement; they may be corrected by the results.
