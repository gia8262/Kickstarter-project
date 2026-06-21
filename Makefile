.PHONY: install test lint format typecheck check pipeline clean

install:
	pip install -e ".[dev,analysis]"
	pre-commit install

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy

check: lint typecheck test

pipeline:
	python make_synthetic_data.py
	python run_01_frame.py
	python run_02_sample.py

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache *.egg-info build htmlcov .coverage
	find . -name __pycache__ -type d -exec rm -rf {} +
