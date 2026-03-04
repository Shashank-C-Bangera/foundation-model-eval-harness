PYTHON ?= python3.11
VENV ?= .venv
PIP := $(VENV)/bin/pip
PY := $(VENV)/bin/python
FMEH := $(VENV)/bin/fmeh
STREAMLIT := $(VENV)/bin/streamlit
EXP ?= baseline_models

.PHONY: setup lint test data run report serve docker-build docker-run

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
	$(STREAMLIT) run app/streamlit_app.py

docker-build:
	docker build -f docker/Dockerfile -t fmeh:latest .

docker-run:
	docker run --rm -p 8501:8501 -v $(PWD)/runs:/app/runs -v $(PWD)/data:/app/data fmeh:latest
