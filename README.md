# PRISM

**Probabilistic Regime-Informed Systematic Management** — a regime-aware
framework for dynamic portfolio allocation.

> **Research question.** Does explicit probabilistic regime conditioning (HMM)
> improve portfolio allocation performance beyond what a temporal
> representation model (LSTM autoencoder) and raw features already capture?

The question is answered by a two-tier ablation. **Tier 1** (no RL) asks
whether each component carries information the raw features lack, using
supervised probes and a classical allocator. **Tier 2** (with RL) asks whether
a policy can exploit it. Tier 1 exists because RL on ~2,500 daily steps has
enormous seed variance: without it, a null result cannot distinguish "the HMM
adds nothing" from "SAC failed to exploit it".

`PRISM_BUILD_SPEC.md` is the authoritative specification. This README says
what is built and how to run it.

---

## Current status — Phase A, Step 1 complete

Only **Phase A** is in scope (spec §0.3). `src/prism/env/` and
`src/prism/agents/` are docstring stubs; `gymnasium` and
`stable-baselines3` are **not installed**, and `prism.config` raises at load
time if either appears while `phase: A`.

| Step | Scope | Status |
|---|---|---|
| 0 | Repo, config, snapshot, splits, **tests** | complete |
| **1** | Dataset expansion: two universes, macro, cross-sectional, QA | **complete — awaiting review** |
| 2 | HMM rebuild (fit on Universe A) | not started |
| 3 | Encoder rebuild (fit on Universe A) | not started |
| 3b | `state.py` — all nine variants | not started |
| 4a | Tier 1 ablation + Phase A report | not started |
| — | **Phase A ends — review** | — |
| 4b–5 | Environment, SAC, Tier 2, holdout | **Phase B, deferred** |

```
173 passed, 10 skipped, 25 xfailed
```

Skips are Phase B (`test_costs.py`, `test_env.py`). The xfails are the
contracts steps 2, 3 and 3b must satisfy; they are `strict=True`, so when a
step lands its tests turn XPASS — a failure — and the marker must be removed.

**Step 1 deliverables:** the real snapshot (2026-10-01, 23 tickers,
1990-01-02 .. 2026-09-30); both universes built, truncated to the non-holdout
span, correlation-pruned (A: 130/131 columns kept, B: 184/185) and persisted
to `data/processed/{A,B}_features.parquet`; `reports/tables/data_quality_{A,B}.md`
— **0 hard failures on either universe**; causality and the strong-form
Universe-A-never-touches-B-only-tickers proof re-verified against the real
data (not just the synthetic fixture the automated suite runs on). Two real
data issues were found and fixed along the way — see DECISIONS.md D-012 and
D-014.

---

## Quick start

```bash
make install        # Python 3.11 venv + pinned dependencies
make test           # the full suite
make test-causality # just the causality suite
make snapshot-dry   # show what a snapshot would fetch, download nothing
```

`make help` lists every target.

---

## The four non-negotiables

These are enforced by tests, not by inspection (spec §0.4).

**1 · Causality.** Anything computed for day `t` uses only data available at
the close of `t`. `tests/test_causality.py` proves it two ways — rebuilding on
truncated data must reproduce the past exactly, and randomising the future
must leave the past byte-identical — and is itself validated against **five
injected leaks** (a full-sample scaler, a `bfill`, a `shift(-1)`, a centred
window, and full-sample winsorisation quantiles). Each half of the harness is
separately proven to catch each leak, so neither can quietly become dead code.

**2 · Fit scope.** Anything fitted sees only training data. Every fitted
object records its fit range and hashes its parameters, so
`tests/test_fit_scope.py` can mutate data after `train_end` and assert the
parameters did not move. There is no `fit_transform` anywhere, and re-fitting
an instance requires an explicit `refit=True`.

**3 · Reproducibility.** One frozen, hashed data snapshot; one config file as
the single source of truth for every date and ticker; fixed seeds; one command
per experiment. Every script writes a run manifest with the config hash,
snapshot hash, git commit, seeds and full `pip freeze`.

**4 · Falsifiability.** Baselines and controls are first-class. The LSTM is
compared against PCA, a random encoder, rolling statistics and the raw
window; the HMM against a volatility-threshold rule and against *shuffled*
posteriors. Config refuses to load a Tier 1 experiment missing any control.

**And the holdout is touched exactly once**, at the end of Phase B. It is
locked behind two independent gates (`final=True` **and**
`PRISM_ALLOW_HOLDOUT=1`) plus a mandatory reason, and every access appends to
`reports/logs/holdout_access.jsonl`.

---

## Design: two universes

Spec §3.1. This is settled, and it is the fix for design gaps D4 and D5.

| | Universe A — model fitting | Universe B — allocation |
|---|---|---|
| Range | 1999-01-04 → present | 2007-04-04 → present |
| Assets | 9 sector ETFs + SPY | A + SHY, IEF, TLT, GLD |
| Macro | VIX, rates | + HYG/LQD credit, dollar, oil, VIX3M |
| Used for | HMM, encoder | state assembly, probes, allocator |
| Crises | dot-com, GFC, COVID, 2022 | GFC, COVID, 2022 |

A regime detector benefits enormously from seeing three crises instead of one;
it does not need to trade bonds to learn what a crisis looks like. An
allocator needs the defensive sleeve but tolerates a shorter history.

Because the HMM and encoder are fitted on A's 1999–2006 and first applied to B
from 2007-04, the representation models have **never seen B's training data at
fit time** — which is what removes the train/test distribution mismatch that
the previous iteration had. Universe isolation is enforced structurally: the
raw panel is filtered to the universe's own tickers before any feature code
runs, and `test_fit_scope.py` corrupts every B-only series and asserts
Universe A's features come out byte-identical.

### Splits (Universe B)

The embargo is applied by **purging the end of the earlier split**, which
preserves the reported evaluation windows and pays the cost out of training
data. See DECISIONS.md D-002.

| Split | Declared | Effective | Sessions |
|---|---|---|---|
| train | 2007-04-04 .. 2017-12-31 | .. 2017-11-22 | 2681 |
| val | 2018-01-01 .. 2018-12-31 | .. 2018-11-21 | 226 |
| test | 2019-01-01 .. 2023-12-31 | .. 2023-11-22 | 1233 |
| holdout | 2024-01-01 .. 2026-09-30 | **LOCKED** | 689 |

Universe A supplies `fit_early` (effectively 1999-12-15 .. 2006-12-31, once
the 252-session warm-up is paid), entirely before B's training window.

---

## The nine state variants

Spec §10. The controls are what make the claims falsifiable rather than
suggestive.

| Variant | Contents | Isolates |
|---|---|---|
| V1 | current-day features | baseline |
| V1′ | same features over the window | "more history" vs "the LSTM" |
| V2 | V1 + LSTM latent | — |
| V3 | V1 + HMM filtered posteriors | — |
| V4 | V1 + latent + posteriors | — |
| C1 | V1 + random-encoder latent | "more dimensions" vs "learned representation" |
| C2 | V1 + threshold-regime one-hot | "the HMM" vs "any regime signal" |
| C3 | V1 + *shuffled* posteriors | permutation control |
| O1 | V1 + **smoothed** posteriors — leaky | diagnostic only, never reported |

- LSTM adds value ⟺ **V2 > V1′ and V2 > C1**
- HMM adds value ⟺ **V3 > C2 and V3 > C3**
- Regime conditioning adds beyond temporal ⟺ **V4 > V2** ← *the research question*

---

## What this replaces

Three exploratory notebooks, kept in `reference/` as read-only artefacts. They
are **not** ported — several of their design choices are wrong, and Appendix A
of the spec catalogues them. The ones that shaped this build most:

- **B1** — smoothed (`predict_proba` over the full sample) posteriors fed the
  RL state vector. A look-ahead leak straight into the state; any V3/V4
  advantage would have been an artefact. Filtered posteriors are now mandatory
  and the identity `filtered[t] == predict_proba(X[:t+1])[-1]` is a test.
- **B3** — `spy_vol20`, a 20-day rolling statistic, used as an HMM
  observation. Violates conditional independence, saturates the posteriors,
  and collapsed the states into a volatility ladder. Config now *rejects* any
  observation matching a rolling-statistic blocklist.
- **C1/C2/C3** — the encoder's off-by-one window labelling, early stopping on
  *training* loss, and ReLU on the latent (permanently dead units). All three
  are now either config-rejected or covered by an alignment test.
- **A1/A2** — data re-downloaded and overwritten each run; a `bfill` sitting
  in the QA cell. The snapshot is write-once and hash-verified; there is no
  `bfill` anywhere and forward-fill is capped at 3 sessions inside an
  explicit availability window.

---

## Expected outcomes

Stated before running, so the result is interpretable either way (spec §15.1).
The single most probable outcome is **modest, statistically fragile
improvements visible in risk metrics during drawdown episodes and invisible in
full-period Sharpe.** That is a normal and publishable result for this class of
research, and the apparatus in §14 — block-bootstrap CIs, permutation tests,
deflated Sharpe — exists so that a modest true effect can be distinguished
from noise, and so that a null result is a *conclusion* rather than a failure.

Signals that something is **wrong** rather than genuinely null: out-of-sample
R² above ~0.15 on 20-day forward *returns*; net Sharpe above ~2.0 on the test
period; an advantage that survives the C3 shuffle control; posterior entropy
near zero.

---

## Reproducibility statement

- **Snapshot:** `data/raw/snapshot_20261001/`, taken 2026-10-01. Hash
  `9de525958b076f93d8355029c2f489f1aae9ac93428a26a23e4e2285f2d585f1`
  (`MANIFEST.json`, committed; the parquet data is not — see DECISIONS.md
  D-013 for why re-running the download does not reproduce it).
- **Config hash:** recorded per run in `reports/logs/<run_id>.json`
- **Git commit:** recorded per run, including whether the tree was dirty
- **Seeds:** master `20260101`; runs `0..9` (`configs/data.yaml`)
- **Python:** 3.11; every dependency pinned in `pyproject.toml`
- **Command sequence:** `make snapshot && make phase-a` (the real snapshot
  already exists at the date above; a fresh `make snapshot` takes a *new*,
  differently-hashed one rather than reproducing this one)

Determinism caveats are logged, never silently ignored: `PYTHONHASHSEED` must
be set before interpreter start (the `Makefile` exports it), and cuDNN LSTM
kernels may remain non-deterministic even with deterministic algorithms
requested. `seeding.py` reports both.

---

## Layout

```
configs/      single source of truth for every date, ticker, hyperparameter
src/prism/
  config.py   typed loader that enforces the spec's non-negotiables at load time
  splits.py   splits, embargo purging, expanding walk-forward folds
  fitting.py  FitRecord / FittedArtifact — fit scope made testable
  data/       write-once snapshot, calendar cleaning, QA, holdout lock
  features/   causal builders (asset, cross, macro) + fitted scaling
  models/     hmm/, encoder/, baselines/          — steps 2 and 3
  probes/     Tier 1 probes and allocator          — step 4a
  backtest/   metrics, timing, benchmarks
  env/        PHASE B — docstring stubs only
  agents/     PHASE B — docstring stubs only
scripts/      thin CLI entry points, one per build step
tests/        test_causality.py is the most important file in the repo
reference/    the three previous notebooks — read-only, never imported
```

`notebooks/` may not define logic; it imports from `src/prism/` and displays
results.

See **DECISIONS.md** for every non-obvious choice and its rationale.
