# Thin wrappers over the CLI entry points. Spec §1.
#
# PYTHONHASHSEED must be set before the interpreter starts, so it is exported
# here rather than inside Python: `seeding.py` can detect its absence but
# cannot fix it retroactively (spec §2).

PYTHON ?= .venv/bin/python
export PYTHONHASHSEED = 0

.DEFAULT_GOAL := help
.PHONY: help venv install test test-fast test-causality lint snapshot snapshot-dry \
        features hmm encoder states tier1 backtest report clean-reports phase-a

help:  ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-18s\033[0m %s\n", $$1, $$2}'

venv:  ## Create the virtualenv (Python 3.11)
	python3.11 -m venv .venv
	.venv/bin/python -m pip install --upgrade pip setuptools wheel

install: venv  ## Install the project and dev dependencies
	.venv/bin/pip install -e ".[dev]"

# --- tests ---------------------------------------------------------------- #
test:  ## Run the full test suite
	$(PYTHON) -m pytest tests/ -ra

test-fast:  ## Run the suite, skipping tests marked slow
	$(PYTHON) -m pytest tests/ -ra -m "not slow"

test-causality:  ## Run only the causality suite (the most important file)
	$(PYTHON) -m pytest tests/test_causality.py -v

coverage:  ## Run the suite with a coverage report
	$(PYTHON) -m pytest tests/ --cov=prism --cov-report=term-missing

# --- Phase A pipeline (spec §16) ------------------------------------------ #
snapshot-dry:  ## Step 0: show what a snapshot would fetch, download nothing
	$(PYTHON) scripts/00_snapshot.py --dry-run

snapshot:  ## Step 0: create the immutable raw snapshot (the ONLY network call)
	$(PYTHON) scripts/00_snapshot.py

features:  ## Step 1: build both universes' features and the QA report
	$(PYTHON) scripts/01_build_features.py

hmm:  ## Step 2: fit the HMM on Universe A, emit filtered posteriors
	$(PYTHON) scripts/02_fit_hmm.py

encoder:  ## Step 3: train the LSTM encoder on Universe A
	$(PYTHON) scripts/03_train_encoder.py

states:  ## Step 3b: assemble all nine state variants
	$(PYTHON) scripts/03b_build_states.py

tier1:  ## Step 4a: run the Tier 1 representation ablation
	$(PYTHON) scripts/04_tier1_ablation.py

backtest:  ## Step 4a: backtest the allocator variants
	$(PYTHON) scripts/06_backtest.py

report:  ## Step 4a: regenerate every figure and table
	$(PYTHON) scripts/07_report.py

phase-a: test features hmm encoder states tier1 report  ## Steps 1-4a end to end
	@echo "Phase A complete. Review against the §0.3 exit criteria before Phase B."

clean-reports:  ## Remove generated figures, tables and logs
	find reports -type f ! -name '.gitkeep' ! -name 'holdout_access.jsonl' -delete

# --- Phase B (spec §0.3) -------------------------------------------------- #
# Deliberately absent: there is no `agents` or `holdout` target. Phase B must
# not be reachable by a single command until Phase A's exit criteria are met
# and reviewed, and the holdout is evaluated once, by hand, with both gates.
