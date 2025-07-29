.PHONY: help install test lint format type-check clean docker-build docker-run benchmark docs

help:
	@echo "Available commands:"
	@echo "  install     Install package and dependencies"
	@echo "  test        Run tests with coverage"
	@echo "  lint        Run flake8 linting"
	@echo "  format      Format code with black"
	@echo "  type-check  Run mypy type checking"
	@echo "  clean       Clean build artifacts"
	@echo "  docker-build Build Docker image"
	@echo "  docker-run  Run in Docker container"
	@echo "  benchmark   Run performance benchmarks"
	@echo "  docs        Build documentation"

install:
	pip install -r requirements.txt -r requirements-dev.txt
	pip install -e .
	pre-commit install

test:
	pytest tests/ --cov=optimizers --cov=test_functions --cov=visualization --cov-report=html --cov-report=term

lint:
	flake8 optimizers test_functions visualization tests

format:
	black optimizers test_functions visualization tests benchmarks

type-check:
	mypy optimizers test_functions visualization --ignore-missing-imports

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

docker-build:
	docker-compose build

docker-run:
	docker-compose up

benchmark:
	python benchmarks/scipy_comparison.py

docs:
	cd docs && make html

all-checks: format lint type-check test
