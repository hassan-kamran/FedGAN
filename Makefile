.PHONY: help install install-dev test test-unit test-integration test-e2e coverage lint format type-check clean docs

help:
	@echo "FedGAN Development Commands"
	@echo "==========================="
	@echo "install          - Install package in production mode"
	@echo "install-dev      - Install package with dev dependencies"
	@echo "test             - Run all tests"
	@echo "test-unit        - Run unit tests only"
	@echo "test-integration - Run integration tests only"
	@echo "test-e2e         - Run end-to-end tests only"
	@echo "coverage         - Run tests with coverage report"
	@echo "lint             - Run flake8 linter"
	@echo "format           - Format code with black and isort"
	@echo "type-check       - Run mypy type checker"
	@echo "clean            - Remove build artifacts and cache"
	@echo "docs             - Build documentation"
	@echo "pre-commit       - Install pre-commit hooks"

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"

pre-commit:
	pre-commit install

test:
	pytest tests/ -v

test-unit:
	pytest tests/unit/ -v -m "not slow"

test-integration:
	pytest tests/integration/ -v -m integration

test-e2e:
	pytest tests/e2e/ -v -m e2e

coverage:
	pytest tests/ --cov=fedgan --cov-report=html --cov-report=term
	@echo "Coverage report generated in htmlcov/index.html"

lint:
	flake8 fedgan/ --max-line-length=100 --ignore=E203,W503,E501

format:
	black fedgan/ tests/ scripts/
	isort fedgan/ tests/ scripts/

type-check:
	mypy fedgan/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ .pytest_cache/ .mypy_cache/ .coverage htmlcov/

docs:
	cd docs && make html
	@echo "Documentation built in docs/_build/html/index.html"

# Development workflow shortcuts
dev-setup: install-dev pre-commit
	@echo "Development environment ready!"

check: format lint type-check test-unit
	@echo "All checks passed!"
