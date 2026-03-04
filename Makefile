PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
FMEH := $(VENV)/bin/fmeh
STREAMLIT := $(VENV)/bin/streamlit
EXP ?= baseline_models

.PHONY: setup lint test data run report serve docker-build docker-run space-sync space-sync-all

setup:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel
	$(PIP) install -e ".[dev]"
	$(VENV)/bin/pre-commit install || true

lint:
	$(VENV)/bin/ruff check .
	$(VENV)/bin/black --check .
	$(VENV)/bin/isort --check-only .

test:
	$(VENV)/bin/pytest

data:
	$(FMEH) data build --experiment $(EXP)

run:
	$(FMEH) run --experiment $(EXP)

report:
	$(FMEH) report --run-dir runs/$(EXP)

serve:
	$(STREAMLIT) run app.py

docker-build:
	docker build -f docker/Dockerfile -t fmeh:latest .

docker-run:
	docker run --rm -p 8501:8501 -v $(PWD)/runs:/app/runs -v $(PWD)/data:/app/data fmeh:latest

space-sync:
	$(MAKE) data EXP=$(EXP)
	$(MAKE) run EXP=$(EXP)
	$(MAKE) report EXP=$(EXP)
	$(PY) scripts/prepare_space_assets.py --experiments $(EXP)
	$(PY) scripts/push_space_assets.py

space-sync-all:
	$(MAKE) data EXP=baseline_models
	$(MAKE) run EXP=baseline_models
	$(MAKE) report EXP=baseline_models
	$(MAKE) data EXP=rag_baseline
	$(MAKE) run EXP=rag_baseline
	$(MAKE) report EXP=rag_baseline
	$(PY) scripts/prepare_space_assets.py --experiments baseline_models rag_baseline smoke_ci
	$(PY) scripts/push_space_assets.py
