.PHONY: lint test typecheck verify

lint:
	uv run ruff check . && uv run ruff format --check .

typecheck:
	uv run mypy --strict src

test:
	uv run pytest -q

verify:
	uv run ruff check . && uv run pytest -q
