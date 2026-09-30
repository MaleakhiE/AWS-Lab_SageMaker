PYTHON ?= python3

.PHONY: install test validate

install:
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

test:
	$(PYTHON) -m pytest -q

validate:
	$(PYTHON) -m src.validation.environment
