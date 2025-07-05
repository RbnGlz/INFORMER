.PHONY: install test lint format docs clean security help

# Default target
help:
	@echo "Available commands:"
	@echo "  install       Install package in development mode"
	@echo "  install-dev   Install with development dependencies"
	@echo "  test          Run tests with coverage"
	@echo "  test-parallel Run tests in parallel"
	@echo "  lint          Run linting (flake8, mypy)"
	@echo "  format        Format code (black, isort)"
	@echo "  security      Run security checks (bandit, safety)"
	@echo "  docs          Build documentation"
	@echo "  clean         Clean build artifacts"
	@echo "  all           Run format, lint, test, security"

# Installation
install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install

# Testing
test:
	pytest tests/ -v --cov=src/informer --cov-report=html --cov-report=term-missing

test-parallel:
	pytest tests/ -v --cov=src/informer --cov-report=html --cov-report=term-missing -n auto

# Code Quality
lint:
	flake8 src/ tests/
	mypy src/ --ignore-missing-imports

format:
	black src/ tests/
	isort src/ tests/

# Security
security:
	bandit -r src/ -ll
	safety check

# Documentation
docs:
	cd docs && make html

# Cleanup
clean:
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*~" -delete
	find . -type f -name "*.orig" -delete
	rm -rf build/ dist/ *.egg-info/
	rm -rf .coverage htmlcov/ .pytest_cache/
	rm -rf .mypy_cache/ .tox/

# All-in-one quality check
all: format lint test security
	@echo "All quality checks passed!"