.PHONY: lint test typecheck verify contract-update

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
