# Thin wrappers over the CLI entry points. Spec §1.
#
# PYTHONHASHSEED must be set before the interpreter starts, so it is exported
# here rather than inside Python: `seeding.py` can detect its absence but
# cannot fix it retroactively (spec §2).

PYTHON ?= .venv/bin/python
# A ~12 hour run must not be interrupted by the laptop sleeping (macOS); empty elsewhere.
CAFFEINATE ?= $(shell command -v caffeinate 2>/dev/null && echo -i)
export PYTHONHASHSEED = 0

.DEFAULT_GOAL := help
.PHONY: help venv install test test-fast test-causality lint snapshot snapshot-dry \
        features hmm encoder states tier1 backtest report clean-reports phase-a env-check \
        tier2 tier2-sanity tier2-smoke final-report holdout-rehearsal \
        dashboard-install dashboard-verify-frozen

help:  ## Show this help
	@grep -hE '^[a-zA-Z0-9_-]+:.*?## ' $(MAKEFILE_LIST) \
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

env-check:  ## Step 4b: acceptance check of the environment on the train split
	$(PYTHON) scripts/04b_env_check.py

backtest:  ## Step 4a: backtest the allocator variants
	$(PYTHON) scripts/06_backtest.py

report:  ## Step 4a: regenerate every figure and table
	$(PYTHON) scripts/07_report.py

phase-a: test features hmm encoder states tier1 report  ## Steps 1-4a end to end
	@echo "Phase A complete. Review against the §0.3 exit criteria before Phase B."

clean-reports:  ## Remove generated figures, tables and logs
	find reports -type f ! -name '.gitkeep' ! -name 'holdout_access.jsonl' ! -name 'preregistration*.md' -delete

# --- Phase B, step 4c: Tier 2 (SAC) ------------------------------------- #
tier2:  ## Step 4c: sanity -> tune -> freeze -> final -> single test evaluation -> report (resumable, ~12 h)
	$(CAFFEINATE) $(PYTHON) scripts/05_train_agents.py

tier2-sanity:  ## Step 4c: only the SAC sanity gates (train/validation only)
	$(PYTHON) scripts/05_train_agents.py --stage sanity

tier2-smoke:  ## Step 4c: whole pipeline at toy size on the VALIDATION split (never reads test)
	$(PYTHON) scripts/05_train_agents.py --smoke

final-report:  ## Step 5: regenerate the consolidated final report from stored results (never opens the holdout)
	$(PYTHON) scripts/08_final_report.py

holdout-rehearsal:  ## Step 5: exercise the whole holdout path on the validation split (never opens the holdout)
	$(PYTHON) scripts/99_final_holdout.py --rehearse

# --- Dashboard (DASHBOARD.md) --------------------------------------------- #
dashboard-install:  ## Dashboard: add the pinned streamlit/plotly group to the existing venv
	.venv/bin/pip install -e ".[dev,dashboard]"

dashboard-verify-frozen:  ## Dashboard: check every frozen data file against the D0 SHA-256 baseline
	shasum -a 256 -c --quiet dashboard/frozen_sources.sha256 && echo "frozen files match the baseline"

# --- Phase B (spec §0.3) -------------------------------------------------- #
# Deliberately absent: there is no `holdout` target; the one real run is typed by hand:
#   PRISM_ALLOW_HOLDOUT=1 python scripts/99_final_holdout.py --i-am-sure Phase B must
# not be reachable by a single command until Phase A's exit criteria are met
# and reviewed, and the holdout is evaluated once, by hand, with both gates.
