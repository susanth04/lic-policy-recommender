PYTHON ?= python

.PHONY: reproduce test

reproduce:
	$(PYTHON) tools/run_notebooks.py

test:
	$(PYTHON) -m pytest -q
