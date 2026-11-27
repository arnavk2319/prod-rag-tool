.PHONY: lint test typecheck verify contract-update up up-obs up-llm down clean urls doctor

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy --strict src

test:
	uv run pytest -q

verify:
	uv run ruff check . && uv run pytest -q

contract-update:
	uv run python scripts/contract_update.py

# --- local stack (Compose v2). Reads .env if present; ports and BIND_ADDR come from env.
up:
	PROVIDER=$${PROVIDER:-fake} docker compose up -d --build --wait

up-obs:
	PROVIDER=$${PROVIDER:-fake} docker compose --profile obs up -d --build --wait

up-llm:
	PROVIDER=$${PROVIDER:-fake} docker compose --profile llm up -d --build --wait ollama
	docker compose --profile llm run --rm ollama-pull

down:
	docker compose --profile llm --profile obs down

# removes containers AND named volumes (database, models, dashboards state)
clean:
	docker compose --profile llm --profile obs down --volumes --remove-orphans

urls:
	@./scripts/urls.sh

doctor:
	@./scripts/doctor.sh
