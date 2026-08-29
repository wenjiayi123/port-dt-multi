PYTHON ?= .venv312/bin/python

.PHONY: test business-benchmark business-verify privacy-scan release-check run

test:
	$(PYTHON) -m unittest discover -s tests -v

business-benchmark:
	$(PYTHON) -m scripts.business_kpi_benchmark

business-verify:
	$(PYTHON) -m scripts.business_kpi_benchmark --verify

privacy-scan:
	$(PYTHON) scripts/public_privacy_scan.py

release-check: privacy-scan
	$(PYTHON) -m scripts.release_check

run:
	$(PYTHON) -m app.server
