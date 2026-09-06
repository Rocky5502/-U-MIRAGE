.PHONY: install test lint smoke
install:
	python -m pip install -e .[dev]
test:
	pytest -q
lint:
	ruff check src tests scripts
smoke:
	python scripts/run_smoke.py
