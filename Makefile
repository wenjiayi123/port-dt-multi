PYTHON ?= .venv312/bin/python

.PHONY: test business-benchmark business-verify release-check run

test:
	$(PYTHON) -m unittest discover -s tests -v

business-benchmark:
	$(PYTHON) -m scripts.business_kpi_benchmark

business-verify:
	$(PYTHON) -m scripts.business_kpi_benchmark --verify

release-check:
	$(PYTHON) -m scripts.release_check

run:
	$(PYTHON) -m app.server
