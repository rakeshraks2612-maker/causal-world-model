.PHONY: help install install-dev test test-fast cli dashboard web lint clean docker-build docker-up docker-down

SHELL := /bin/bash
VENV ?= $(shell if [ -d ".venv" ]; then echo ".venv/bin/"; fi)
PYTHON ?= $(VENV)python3
PYTEST ?= $(VENV)pytest
STREAMLIT ?= $(VENV)streamlit
PRISM ?= $(VENV)prism

help:  ## Display this help screen
	@echo "================================================================================"
	@echo "PRISM: Causal World Model & Safe Decision Intelligence Engine"
	@echo "================================================================================"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install PRISM package in editable mode
	$(PYTHON) -m pip install -e .

install-dev:  ## Install dependencies and editable package with development tools
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m pip install -e ".[dev,notebooks]"

test:  ## Run full test suite (421 invariant & physics proofs)
	$(PYTEST) -v tests/ --durations=10

test-fast:  ## Run test suite in quiet fast mode
	$(PYTEST) -q tests/

cli:  ## Run PRISM decision engine across all 6 benchmark scenarios
	$(PRISM) --all

cli-s1:  ## Run PRISM decision engine on Scenario 1 (Normal Operations)
	$(PRISM) --scenario s1

dashboard:  ## Launch the interactive Streamlit Operator Console on port 8501
	$(STREAMLIT) run scripts/run_dashboard.py --server.port 8501 --server.headless true

web:  ## Serve the lightweight PRISM Web SPA on port 8000
	$(PYTHON) -m http.server 8000 --directory prism/dashboard/web

lint:  ## Run ruff code linter
	ruff check .

format:  ## Format codebase using black and ruff
	black prism/ tests/ scripts/
	ruff check --fix .

docker-build:  ## Build PRISM production Docker image
	docker build -t prism-cwm:latest .

docker-up:  ## Start PRISM container services with docker-compose
	docker-compose up -d

docker-down:  ## Stop PRISM container services
	docker-compose down

clean:  ## Clean temporary build artifacts, bytecode, caches, and test runs
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type f -name ".coverage" -delete
	rm -rf build/ dist/ .ruff_cache/
