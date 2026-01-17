PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.PHONY: install dev test lint fmt run docker-build docker-run clean

install:
	$(PIP) install -r requirements.txt

dev:
	$(PIP) install -r requirements-dev.txt

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check .

fmt:
	$(PYTHON) -m ruff format .

run:
	$(PYTHON) app.py

docker-build:
	docker build -t deepface-age-server:latest .

docker-run:
	docker run --rm -p 5000:5000 deepface-age-server:latest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage build dist
	find . -name "__pycache__" -type d -exec rm -rf {} +
