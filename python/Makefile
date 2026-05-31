# Convenience targets for the criticality pipeline on WSL / Linux.
# Most targets assume the `criticality` conda env exists (run `make setup`).

ENV ?= criticality
EXT_DIR ?= extdata
RUN = conda run -n $(ENV)

.PHONY: help setup data test verify simulate run clean

help:
	@echo "Targets:"
	@echo "  setup     Install Miniforge (if needed), create the conda env, install the package, fetch data"
	@echo "  data      (Re)download the cross-section library"
	@echo "  test      Run the pytest suite"
	@echo "  verify    Run a live OpenMC k-eigenvalue smoke test"
	@echo "  simulate  Generate data with OpenMC in parallel (N=50, generation only)"
	@echo "  run       Run the full CLI pipeline against EXT_DIR=$(EXT_DIR)"
	@echo "  clean     Remove caches and generated training artifacts"

setup:
	bash scripts/setup_wsl.sh

data:
	$(RUN) bash scripts/download_data.sh

test:
	$(RUN) pytest -q tests

verify:
	$(RUN) python scripts/verify_openmc.py

# Parallel OpenMC generation only (no training). Override N and WORKERS, e.g.:
#   make simulate N=16752 WORKERS=16
N ?= 50
WORKERS ?=
simulate:
	$(RUN) python -m criticality --ext-dir $(EXT_DIR) --simulate $(N) --simulate-only \
		$(if $(WORKERS),--sim-workers $(WORKERS),) --verbose

run:
	$(RUN) python -m criticality --ext-dir $(EXT_DIR)

clean:
	rm -rf .pytest_cache **/__pycache__ $(EXT_DIR)/training $(EXT_DIR)/*-dataset.pkl
