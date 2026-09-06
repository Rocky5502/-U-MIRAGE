.PHONY: install test lint smoke demo
install:
	python -m pip install -e .[dev]
test:
	pytest -q
lint:
	ruff check src tests scripts
demo:
	python scripts/make_demo_data.py
smoke: demo
	python scripts/run_smoke.py
