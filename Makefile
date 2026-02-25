PYTHON ?= python

install:
	$(PYTHON) -m pip install -r requirements-dev.txt

test:
	pytest

coverage:
	pytest --cov=src --cov-report=term-missing

lint:
	$(PYTHON) -m py_compile src/*.py

run:
	uvicorn src.api:app --host 0.0.0.0 --port 8000

docker-build:
	docker compose build

docker-test:
	docker compose run --rm tests
