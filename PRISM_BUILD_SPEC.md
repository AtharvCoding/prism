# PRISM — Build Specification

**Probabilistic Regime-Informed Systematic Management**
A regime-aware reinforcement learning framework for dynamic portfolio allocation.

> **Purpose of this document.** This is a complete, self-contained build spec for an agentic coding assistant (Claude Code). It replaces three exploratory Colab notebooks (`pipeline.ipynb`, `prismhmm.ipynb`, `LSTM.ipynb`) with a local, tested, reproducible research codebase. The notebooks are **reference only** — do not port their code. Several of their design choices are wrong and are documented in Appendix A so they are not repeated.

---

## 0. Executive summary

### 0.1 Research question

> Does explicit probabilistic regime conditioning (HMM) improve portfolio allocation performance beyond what a temporal representation model (LSTM autoencoder) and raw features already capture?

### 0.2 How the question is answered

A **two-tier ablation**:

- **Tier 1 — Representation ablation (no RL).** Cheap, deterministic, fast. Does each component carry information the raw features lack? Measured by (a) supervised probes predicting forward risk/return quantities, and (b) a classical allocator (mean-variance / vol-target) driven by each feature set. Acts as a **gate**: a component that fails Tier 1 does not proceed to Tier 2 unchanged.
- **Tier 2 — Policy ablation (with RL).** The headline result. SAC agents trained on four state variants plus controls, multi-seed, with confidence intervals and cost-aware metrics.

**Why Tier 1 exists.** RL on ~2,500 daily steps with a single crisis in training has enormous seed variance. If you go straight to Tier 2, a null result is uninterpretable — you cannot distinguish "the HMM adds nothing" from "SAC failed to exploit it." Tier 1 isolates the information question from the optimisation question. Tier 1 is not a replacement for Tier 2; the project's claim must be made at the policy level.

### 0.3 Current build scope — PHASE A (pre-RL) only

**This is the most important instruction in the document.**

The build is split into two phases. **Only Phase A is in scope right now.**

| | Phase A — **IN SCOPE NOW** | Phase B — **DEFERRED** |
|---|---|---|
| Sections | §1–§10, §13.1, §14, §15 | §11, §12, §13.2 |
| Build steps (§16) | 0, 1, 2, 3, 4a | 4b, 4c, 5 |
| Deliverable | A tested, reproducible representation pipeline + Tier 1 ablation results | SAC agents + Tier 2 policy ablation |

**Do not implement `src/prism/env/`, `src/prism/agents/`, or any SAC code in Phase A.** Do not install `stable-baselines3` or `gymnasium` yet. Those sections remain in this document so the architecture anticipates them — directory stubs with docstrings are acceptable, implementations are not.

**Why the project stops here for now.** Every RL result is downstream of the representation. If the HMM posteriors leak the future, or the encoder is misaligned, or a component carries no information, then every agent trained on them is wasted compute and every conclusion is void. Phase A produces a set of inputs that are *known-good* and a Tier 1 answer to the research question that is *already publishable in its own right*. Phase B then tests whether a policy can exploit what Tier 1 showed is there.

**Phase A exit criteria** (all must hold before Phase B is discussed):
1. Test suite green; causality tests validated against injected leaks.
2. Dataset expanded, QA clean, all features causal.
3. HMM producing filtered posteriors, `K` selected on validation, canonically labelled, characterised against NBER/drawdown labels and the threshold baseline.
4. Encoder alignment-tested, validation-driven, no dead units, walk-forward, compared against all four baselines.
5. Tier 1 ablation table complete with confidence intervals, and gate decisions recorded in `DECISIONS.md`.

### 0.4 Non-negotiable principles

1. **Causality.** Anything computed for day `t` uses only data available at the close of day `t`. Enforced by automated tests, not by inspection.
2. **Fit scope.** Anything *fitted* (scalers, HMM parameters, encoder weights, policies) sees only training data. Walk-forward refits are permitted; look-ahead refits are not.
3. **Reproducibility.** One frozen data snapshot, one config file, fixed seeds, one command per experiment.
4. **Falsifiability.** Baselines and controls (random latent, shuffled regimes, threshold rules) are first-class citizens. Negative results are publishable results.
5. **The final holdout is touched exactly once**, at the end, after all decisions are frozen.

---

## 1. Repository structure

```
prism/
├── README.md
├── pyproject.toml                 # or requirements.txt + setup.cfg
├── Makefile                       # thin wrappers over CLI entry points
├── configs/
│   ├── base.yaml                  # master config (single source of truth)
│   ├── data.yaml                  # universe, dates, splits, embargo
│   ├── hmm.yaml
│   ├── encoder.yaml
│   ├── env.yaml                   # RL environment + cost model (PHASE B)
│   └── experiments/
│       ├── tier1_probes.yaml
│       ├── tier2_v1.yaml          # PHASE B
│       ├── tier2_v1p.yaml         # PHASE B
│       ├── tier2_v2.yaml          # PHASE B
│       ├── tier2_v3.yaml          # PHASE B
│       ├── tier2_v4.yaml          # PHASE B
│       └── tier2_controls.yaml    # PHASE B
├── data/
│   ├── raw/                       # IMMUTABLE snapshot (git-ignored, hashed)
│   │   ├── snapshot_YYYYMMDD/
│   │   │   ├── ohlcv.parquet
│   │   │   ├── macro.parquet
│   │   │   └── MANIFEST.json      # hashes, download date, lib versions
│   ├── interim/                   # cleaned, calendar-aligned
│   ├── processed/                 # features, targets, states
│   └── holdout/                   # LOCKED. Loader refuses without final=True
├── src/prism/
│   ├── __init__.py
│   ├── config.py                  # dataclass config loader + validation
│   ├── utils/
│   │   ├── seeding.py
│   │   ├── logging.py
│   │   ├── hashing.py
│   │   └── calendar.py            # NYSE trading calendar helpers│   ├── data/
│   │   ├── download.py            # snapshot creation (write-once)
│   │   ├── quality.py             # QA report
│   │   ├── clean.py               # calendar alignment, no bfill ever
│   │   └── loaders.py             # snapshot + holdout lock
│   ├── features/
│   │   ├── asset.py               # per-asset features
│   │   ├── cross.py               # cross-sectional features
│   │   ├── macro.py               # VIX, curve, credit, dollar
│   │   ├── targets.py             # forward-looking targets (probes only)
│   │   └── build.py               # orchestrator -> features.parquet
│   ├── splits.py                  # split objects, embargo, walk-forward folds
│   ├── models/
│   │   ├── hmm/
│   │   │   ├── fit.py             # Baum-Welch, restarts, selection
│   │   │   ├── filtered.py        # CAUSAL forward-pass posteriors
│   │   │   ├── labeling.py        # deterministic state relabeling
│   │   │   ├── walkforward.py     # expanding-window refit w/ carried state
│   │   │   └── evaluate.py        # duration, delay, entropy, vs. labels
│   │   ├── encoder/
│   │   │   ├── dataset.py         # windowing (alignment-tested)
│   │   │   ├── models.py          # AE / DAE / VAE / predictive heads
│   │   │   ├── train.py           # val-based early stopping
│   │   │   ├── walkforward.py     # periodic refit
│   │   │   └── evaluate.py        # per-feature MSE, dead units, drift
│   │   └── baselines/
│   │       ├── pca_encoder.py
│   │       ├── random_encoder.py
│   │       ├── rolling_stats.py
│   │       └── threshold_regime.py
│   ├── state.py                   # assembles state vectors per variant
│   ├── probes/
│   │   ├── probe.py               # ridge / logistic walk-forward probes
│   │   └── allocator.py           # MVO / vol-target / risk-parity driver
│   ├── env/                       # PHASE B — stubs only
│   │   ├── portfolio_env.py       # Gymnasium env
│   │   ├── costs.py               # transaction cost + slippage model
│   │   └── rewards.py             # reward variants
│   ├── agents/                    # PHASE B — stubs only
│   │   ├── sac.py
│   │   └── train.py
│   ├── backtest/
│   │   ├── engine.py
│   │   ├── metrics.py
│   │   └── benchmarks.py          # EW, 60/40, MinVar, RP, vol-target
│   ├── analysis/
│   │   ├── bootstrap.py           # CIs, block bootstrap
│   │   ├── significance.py        # permutation tests, deflated Sharpe
│   │   └── episodes.py            # drawdown-episode segmentation
│   └── reporting/
│       ├── tables.py
│       ├── figures.py
│       └── report.py
├── scripts/                       # CLI entry points (thin)
│   ├── 00_snapshot.py
│   ├── 01_build_features.py
│   ├── 02_fit_hmm.py
│   ├── 03_train_encoder.py
│   ├── 03b_build_states.py
│   ├── 04_tier1_ablation.py
│   ├── 05_train_agents.py         # PHASE B
│   ├── 06_backtest.py
│   ├── 07_report.py
│   └── 99_final_holdout.py        # PHASE B — requires explicit --i-am-sure
├── tests/
│   ├── conftest.py
│   ├── test_causality.py          # ★ most important file in the repo
│   ├── test_fit_scope.py
│   ├── test_alignment.py
│   ├── test_splits.py
│   ├── test_costs.py              # PHASE B
│   ├── test_env.py                # PHASE B
│   ├── test_hmm.py
│   ├── test_encoder.py
│   ├── test_state.py
│   └── test_metrics.py
├── notebooks/                     # EXPLORATION ONLY — import from src/
└── reports/
    ├── figures/
    ├── tables/
    └── logs/
```

**Rules.**
- Notebooks may not define logic. They import from `src/prism/` and display results.
- Every script writes a run manifest to `reports/logs/<run_id>.json` containing the config hash, the data snapshot hash, the git commit, the seeds, and the library versions.
- No global mutable state. No `warnings.filterwarnings('ignore')`.

---

## 2. Environment and dependencies

**Python 3.11.**

Core (Phase A):
```
numpy, pandas, pyarrow, scipy, scikit-learn
hmmlearn
torch
yfinance
pandas-market-calendars     # NYSE calendar
matplotlib, seaborn
pyyaml, pydantic            # config validation
pytest, pytest-cov
tqdm, rich
statsmodels                 # Ljung-Box, Markov-switching cross-check
```

Deferred to Phase B — **do not install yet**:
```
gymnasium
stable-baselines3           # SAC (or implement SAC; see §12)
```

Pin every version in `pyproject.toml`. Record `pip freeze` in the snapshot manifest.

**Determinism.** Set `PYTHONHASHSEED`, `numpy` generator seeds, `torch.manual_seed`, `torch.use_deterministic_algorithms(True)` where feasible. Log when full determinism is not achievable (cuDNN paths) rather than silently ignoring it.

---

## 3. Configuration

### 3.1 Two-universe design

The project uses **two nested universes** over different date ranges. This is a deliberate design decision, not a workaround, and it must be implemented explicitly rather than by silently dropping NaN rows.

**Verified inception dates** (issuer-confirmed; re-verify at snapshot time):

| Asset | Inception | Binding constraint |
|---|---|---|
| XLK, XLE, XLV, XLF, XLU, XLI, XLP, XLY, XLB | 1998-12-16 | — |
| SPY | 1993-01 | — |
| SHY, IEF, TLT, LQD | 2002-07-22 | Universe B start |
| GLD | 2004-11-18 | Universe B start |
| HYG | 2007-04-04 | **Universe B start** |
| XLRE | 2015-10 | excluded — insufficient history |
| XLC | 2018-06 | excluded — insufficient history |
| ^VIX | 1990-01 | — |

**Universe A — "core equity", 1999-01-04 → present.**
Nine sector ETFs plus SPY. Macro signals limited to those with full history (VIX, rates). No defensive sleeve, no credit proxy.
*Purpose:* a long history containing **three** distinct crises — dot-com (2000–02), GFC (2007–09), COVID (2020) — plus the 2022 rate shock. Used to fit the **regime and representation models** (HMM, encoder), which need crisis diversity more than they need a tradable defensive sleeve.

**Universe B — "full allocation", 2007-04-04 → present.**
Universe A plus SHY, IEF, TLT, GLD, and the full macro set including the HYG/LQD credit proxy.
*Purpose:* the **allocation** universe. This is what the Tier 1 allocator (and later the RL agent) actually trades, and what the Tier 1 ablation is evaluated on.

**The key insight:** the two universes serve different stages. A regime detector benefits enormously from seeing three crises instead of one; it does not need to trade bonds to learn what a crisis looks like. An allocator needs the defensive sleeve but can tolerate a shorter history. Splitting them buys the strengths of both.

### 3.2 How this works with training

The fitting stages run on different ranges, and this **must** be made explicit in config and enforced in `test_fit_scope.py`:

| Stage | Universe | Fit range | Rationale |
|---|---|---|---|
| Feature construction | A and B separately | full respective range | B features exist only from 2007 |
| HMM | **A** | 1999-01 → 2006-12 (fit), extended walk-forward thereafter | 3 crises available; observations (SPY returns, VIX) have full history |
| Encoder | **A** | 1999-01 → 2006-12 (fit), walk-forward thereafter | same reason; more windows, more regime diversity |
| Scalers for A-features | A | as above | — |
| Scalers for B-features | B | 2007-04 → 2017-12 | — |
| State assembly | **B** | 2007-04 onward | B is the allocation universe |
| Tier 1 probes + allocator | **B** | walk-forward from 2007-04 | evaluated where the full sleeve exists |

**Revised split calendar:**

```
UNIVERSE A (model fitting)
  fit_early    1999-01-04 .. 2006-12-31    # dot-com crisis + recovery
  [continues via expanding-window walk-forward across all later dates]

UNIVERSE B (allocation + evaluation)
  train        2007-04-04 .. 2017-12-31    # GFC + recovery
  [embargo 25 trading days]
  val          2018-01-01 .. 2018-12-31    # vol spike, no crisis
  [embargo 25 trading days]
  test         2019-01-01 .. 2023-12-31    # COVID + rate shock
  [embargo 25 trading days]
  holdout      2024-01-01 .. 2026-09-30    # LOCKED
```

**Three properties this buys you.**

1. **The HMM and encoder are already fitted before Universe B's training period begins.** Fitting on A's 1999–2006 and first applying to B from 2007-04 means the representation models have *never seen* B's training data at fit time. This removes the train/test distribution mismatch that Appendix A (D4) flags in the reference design — the agent now trains on genuinely out-of-sample encodings, exactly the kind it meets at test.
2. **Crisis diversity.** The HMM sees dot-com in its fit window and GFC/COVID/rate-shock via walk-forward. The reference design saw one crisis.
3. **The dot-com crisis becomes a held-out sanity check of a rare kind:** a crisis with a completely different character (equity-concentrated, no credit event, slow-burn over 2.5 years) from the GFC. If the HMM's crisis state generalises across both, that is real evidence.

**Three hazards, to be handled explicitly:**

- **Universe A features must not leak B-only information.** `build_features(universe="A")` and `build_features(universe="B")` are separate calls producing separate frames with separate scalers. `test_fit_scope.py` asserts the A-pipeline never touches a B-only ticker.
- **Regime shift across the boundary.** Market microstructure changed materially between 1999 and 2007 (decimalisation 2001, RegNMS 2007, the rise of HFT). Volatility levels are not directly comparable across the whole span. Mitigate by using stationary feature forms (changes in log-vol, z-scores computed on expanding windows) rather than raw levels, and report an explicit robustness check: refit the HMM on 2007+ only and compare state characterisations.
- **Alignment.** The two universes share an index from 2007-04 onward. `state.py` asserts index equality on the overlap and fails loudly otherwise.

**Robustness check (required, not optional):** run the entire Tier 1 ablation a second time with the HMM and encoder fitted on B's training window only (2007–2017), and report both. If conclusions differ between the two, that is a finding about data requirements and belongs in the report.

### 3.3 Config sketch

`configs/data.yaml` (illustrative; the assistant should finalise field names):

```yaml
snapshot:
  date: null                 # set at snapshot time, then frozen
  allow_overwrite: false

universes:
  A:                         # core equity — model fitting
    start: "1999-01-04"
    equity_sectors: [XLK, XLE, XLV, XLF, XLU, XLI, XLP, XLY, XLB]
    benchmark: SPY
    macro: [^VIX, ^TNX, ^FVX, ^IRX]
    fit_early: ["1999-01-04", "2006-12-31"]
  B:                         # full allocation universe
    start: "2007-04-04"      # bound by HYG inception 2007-04-04
    inherits: A
    defensive: [SHY, IEF, TLT, GLD]
    macro_extra: [HYG, LQD, "DX-Y.NYB", "CL=F"]

excluded:
  XLRE: "inception 2015-10 — insufficient history"
  XLC:  "inception 2018-06 — insufficient history"

splits:                      # applied to Universe B
  train:   ["2007-04-04", "2017-12-31"]
  val:     ["2018-01-01", "2018-12-31"]
  test:    ["2019-01-01", "2023-12-31"]
  holdout: ["2024-01-01", "2026-09-30"]     # LOCKED
  embargo_days: 25           # >= max forward target horizon (20) + margin

decision:
  frequency: weekly          # rebalance weekly; features daily
  rebalance_day: "friday_close"
  execution_lag_days: 1      # decide at close t, execute at close t+1

warmup_days: 60              # longest lookback; series aligned after this

seeds:
  master: 20260101
  runs: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
```

**Open decisions the assistant must surface (do not silently choose):**
1. Weekly vs daily rebalancing — spec defaults to weekly. Daily is a robustness check.
2. Long-only vs long-only-with-cash vs limited shorting. Spec defaults to **long-only with cash**, weights summing to 1, each weight in [0, 0.35].
3. Whether the encoder is fitted on Universe A features only (recommended, for the longer history) or refitted on B — spec defaults to A, with the §3.2 robustness check covering the alternative.

---

## 4. Data layer

### 4.1 Snapshotting (write-once)

`scripts/00_snapshot.py`:
- Downloads OHLCV for the full universe with `auto_adjust=True` (dividend- and split-adjusted → total returns) plus macro series.
- Writes to `data/raw/snapshot_YYYYMMDD/`.
- Writes `MANIFEST.json`: per-file SHA-256, row counts, first/last dates, download timestamp (UTC), `yfinance` version, full `pip freeze`.
- **Refuses to overwrite an existing snapshot** unless `--force` plus a new snapshot directory.
- Everything downstream reads the snapshot path from config, never the network.

**Why this matters:** adjusted prices are restated whenever a new dividend is paid, so re-downloading silently changes history. Every result must trace to one hash.

### 4.2 Cleaning

`src/prism/data/clean.py`:
- Reindex to the **NYSE trading calendar** (`pandas-market-calendars`) for the full date range.
- Identify genuinely missing sessions vs. pre-inception dates (assets have different inception dates — do not forward-fill across inception).
- **Forward-fill only, never backward-fill.** A `bfill` writes the future into the past. Cap forward-fills at 3 consecutive days and flag longer gaps for manual review.
- Never interpolate prices.
- Truncate the panel to dates where all required assets exist, or use an explicit availability mask.

### 4.3 Quality assurance

`src/prism/data/quality.py` produces `reports/tables/data_quality.md`, with hard failures (abort) and soft warnings (log):

| Check | Threshold | Severity |
|---|---|---|
| Missing sessions vs NYSE calendar | any | hard |
| Zero/negative prices | any | hard |
| Duplicate index entries | any | hard |
| Non-monotonic index | any | hard |
| Extreme daily move | \|r\| > 15% | soft (inspect) |
| Extreme daily move | \|r\| > 30% | hard |
| Stale price run | ≥ 3 identical closes | soft |
| Zero volume | any session | soft |
| Return distribution drift | KS test train vs test | soft |
| Constant feature | std == 0 | hard |
| Feature NaN/inf after warm-up | any | hard |
| Index coverage vs SPY | > 1 day divergence | soft |

Known artefacts to document, not silently fix: the 2018 GICS reclassification (telecom → communication services, affecting XLK/XLY constituents) and ETF constituent drift generally.

---

## 5. Feature engineering

All features must be **causal** and, preferably, **stationary**.

### 5.1 Per-asset features (`features/asset.py`)

For each asset in the universe:
- `return_1d` — log return
- `vol_20` — 20d realised vol (annualised); include `vol_60` only if it survives a correlation-pruning step
- `dvol` — change in `log(vol_20)` (stationary form)
- `mom_20`, `mom_60` — momentum, **ex-most-recent-week** (skip-1w) to avoid short-term reversal contamination
- `downside_vol_20`, `skew_60`, `kurt_60`
- `volume_ratio` — volume / 20d mean volume (note: same-day volume is only final at the close; permitted under the timing rule)
- `beta_60_spy` — rolling beta to SPY (replaces `spycorr`, which is degenerate for SPY itself)
- `dist_from_52w_high`

### 5.2 Cross-sectional features (`features/cross.py`)

Computed across the equity sleeve:
- `avg_pairwise_corr_60` — average off-diagonal correlation (the classic crisis indicator)
- `corr_dispersion_60`
- `return_dispersion` — cross-sectional std of daily returns
- `breadth` — fraction of sectors above their 50d MA
- `first_eigenvalue_share_60` — share of variance in PC1 of the correlation matrix (market-mode dominance)
- Shrunk covariance matrix (Ledoit-Wolf), 60d window — stored separately for the allocator, not flattened into the state vector

### 5.3 Macro features (`features/macro.py`)

- `vix_level`, `vix_change`, `vix_term_structure` (VIX/VIX3M if available)
- `curve_slope_10y_2y`, `curve_change`
- `credit_proxy` — log(HYG/LQD) and its 20d change
  - *Note:* FRED's ICE BofA high-yield OAS series has restricted history; use the ETF-ratio proxy or verify any FRED series' coverage before relying on it.
- `dollar_return_20`, `oil_vol_20`

### 5.4 Forward targets (`features/targets.py`)

**Used only for probes and evaluation — never inside a state vector.**
- `fwd_vol_5`, `fwd_vol_20` — forward realised volatility
- `fwd_ret_5`, `fwd_ret_20`
- `fwd_max_drawdown_20`
- `fwd_corr_20` — forward average pairwise correlation

These are the reason the embargo exists (§6).

### 5.5 Feature hygiene

- Drop constant/degenerate columns (`SPY_spycorr ≡ 1.0` in the reference pipeline).
- Prune by correlation: within a family (`vol_20`/`vol_60`/`volume_ratio`; `mom_20`/`mom_60`), drop members with \|ρ\| > 0.95 and document which survived.
- Winsorise at the 1st/99th percentile **using training-set quantiles only**, applied to all splits.
- Scale with a `StandardScaler` (or robust scaler) **fitted on the training split only**, or refit walk-forward — never on the full sample.
- Persist the final feature list with a schema hash so state-vector assembly is verifiable.

---

## 6. Splits, embargo, and walk-forward

### 6.1 Split semantics

See §3.2 for the two-universe fit schedule. Universe B carries the splits:

```
train        2007-04-04 .. 2017-12-31     (GFC + recovery)
[embargo 25 trading days]
val          2018-01-01 .. 2018-12-31     (model selection, early stopping, K choice)
[embargo 25 trading days]
test         2019-01-01 .. 2023-12-31     (reported; analysed by episode too)
[embargo 25 trading days]
holdout      2024-01-01 .. 2026-09-30     (LOCKED — one use, at the very end)
```

Universe A supplies `fit_early` (1999-01-04 .. 2006-12-31) for the HMM and encoder, entirely before B's training window.

**Embargo rationale (important nuance):** lookback features (60d vol) legitimately read data *before* a boundary — that is not leakage. The embargo exists because **forward-looking targets and multi-day rewards** near a boundary overlap the next split. Size the embargo to `max(forward_target_horizon, reward_horizon, rebalance_horizon)` plus margin.

### 6.2 Walk-forward folds

For HMM and encoder refits inside the training period, and for probe evaluation:
- **Expanding window** (not rolling-5-year). A fixed 5-year window starting in 2013 contains no crisis, which cripples the HMM's crisis state.
- Refit cadence: HMM monthly, encoder annually (both configurable).
- Each fold records `(fit_start, fit_end, apply_start, apply_end)` and asserts `fit_end < apply_start`.

### 6.3 Holdout lock

`loaders.load_holdout()` raises unless called with `final=True` **and** an environment variable `PRISM_ALLOW_HOLDOUT=1`. `scripts/99_final_holdout.py` requires `--i-am-sure` and logs the invocation permanently.

---

## 7. Step 0 deliverable — the test suite

This is the highest-value part of the build. Write these **before** the model code.

### 7.1 `test_causality.py`

```python
def assert_causal(build_fn, raw_panel, cut_dates, rtol=1e-8):
    """Output for t must be invariant to all data after t."""
    full = build_fn(raw_panel)
    for t in cut_dates:
        # (a) truncation: rebuilding on data up to t reproduces rows up to t
        cut = build_fn(raw_panel.loc[:t])
        pd.testing.assert_frame_equal(full.loc[:t], cut, rtol=rtol)

        # (b) perturbation: randomising the future leaves the past unchanged
        noisy = raw_panel.copy()
        m = noisy.index > t
        noisy.loc[m] *= rng.uniform(0.5, 1.5, noisy.loc[m].shape)
        pd.testing.assert_frame_equal(full.loc[:t], build_fn(noisy).loc[:t], rtol=rtol)
```

Apply to **every** stage:

| Stage | Expected outcome on current logic |
|---|---|
| `build_features` | **PASS** (verified on a replica: rolling windows and positive shifts are causal) |
| HMM posteriors (params frozen) | **FAIL** until §8 is implemented — `predict_proba` is smoothed |
| Encoder latents (weights frozen) | FAIL if the off-by-one is ported |
| State vector assembly | FAIL if either upstream fails |

The test must be **validated against known-bad implementations**: inject (i) a full-sample scaler, (ii) a `bfill`, (iii) a `shift(-1)` and confirm the test fails on each. A causality test that never fails is worthless. *(This validation was run on a replica of the reference feature builder: clean → pass; global scaler → fail; look-ahead → fail.)*

### 7.2 `test_fit_scope.py`

- Any fitted object records the `(start, end)` of its fit data; assert `end <= train_end` (or `<= fold.fit_end`).
- Mutate data after `train_end`; assert fitted parameters are byte-identical.
- Assert scalers are never refit on transform.

### 7.3 `test_alignment.py`

- Row dated `D` of the latent frame equals `encoder(window whose last row is D)`.
- Window count equals `len(features) - window + 1` (the reference implementation dropped the last window and shifted labels by one).
- Reward for decision at `t` uses return from `t → t+1` (with execution lag), verified on a synthetic price path with a known answer.
- Feature index, target index, and state index are identical after alignment.

### 7.4 `test_splits.py`

- Splits are ordered, disjoint, and separated by ≥ embargo trading days.
- `load_holdout()` raises without the explicit flag.
- Walk-forward folds satisfy `fit_end < apply_start` for every fold.
- No hard-coded date strings anywhere outside config (grep test).

### 7.5 Other tests

- `test_costs.py` — zero turnover implies zero cost; cost is monotone in turnover; round-trip cost matches an analytic value.
- `test_env.py` — weights sum to 1 and respect bounds; episode length matches the calendar; no NaN observations; a deterministic buy-and-hold policy reproduces the analytic benchmark return.
- `test_hmm.py` — filtered posteriors at `t` equal the last row of a prefix-run; relabeling is deterministic and permutation-invariant; log-likelihood and parameter counts match hand-computed values on a tiny synthetic model.
- `test_metrics.py` — Sharpe, Sortino, max drawdown and turnover verified against hand-computed values on a small fixture.

**Acceptance for Step 0:** the suite runs, passes on features, and fails for documented reasons on the components still to be rebuilt.

---

## 8. HMM component

### 8.1 What the HMM is for

A Hidden Markov Model posits an unobserved discrete state that persists over time; each state emits observations from its own distribution. Baum-Welch (EM) fits transition probabilities, emission parameters, and per-day state posteriors without labels. "Regime classification" means `P(state | observations)`.

**The reference implementation's central flaw:** it fed `spy_vol20` (a 20-day rolling statistic) as an observation. Rolling statistics are strongly autocorrelated, which violates the conditional-independence assumption, double-counts evidence, and saturates posteriors to 0/1. The result was a chain-structured transition matrix with near-identical state means — i.e. a **volatility ladder**, not economically distinct regimes. It also duplicated information already in the feature set, so the HMM could add nothing.

### 8.2 Observation design

Use conditionally-independent, low-autocorrelation observables:

- **Primary:** `spy_return_1d`
- **Plus (recommended):** `vix_change`, `credit_proxy_change`, `curve_change`, `avg_pairwise_corr_change`

Build at least two specifications and compare:
- **H1 (univariate):** returns only, Gaussian emissions, full covariance. The volatility structure should emerge from state-dependent variance.
- **H2 (multivariate):** returns + VIX change + credit change + correlation change.

Optionally cross-check against `statsmodels` `MarkovRegression`/`MarkovAutoregression` as an independent implementation.

### 8.3 Causal (filtered) posteriors — mandatory

`hmmlearn`'s `predict_proba` runs forward-backward and returns **smoothed** posteriors `P(s_t | x_1..x_T)`, which use the future. On a synthetic two-regime series, smoothed and filtered posteriors differed by > 0.25 on ~12% of days — a material contamination, not a rounding issue.

Implement `filtered_posteriors(model, X)` returning `P(s_t | x_1..x_t)`:
- Forward recursion in log space (log-sum-exp) for numerical stability.
- Must exactly reproduce `model.predict_proba(X[:t+1])[-1]` for all `t` (the backward pass is trivial at the final step). Assert this in tests.
- Return both posteriors and per-step log-likelihood.

**Every downstream consumer uses filtered posteriors. No exceptions.**

### 8.4 Model selection

- Sweep `K = 2..8`.
- ≥ 20 random restarts per `K`; report the log-likelihood distribution, not just the best.
- **Correct information criteria.** `model.score(X)` already returns the *total* log-likelihood — do not multiply by `n`. Free parameter count for a Gaussian HMM with full covariance and `d` observation dims:
  ```
  n_params = (K - 1)          # initial distribution
           + K * (K - 1)      # transition matrix rows
           + K * d            # means
           + K * d * (d + 1) / 2   # covariances
  ```
  (The reference code used `K² + 2K + 4K` with an inflated log-likelihood; recomputing correctly gave BIC 8935 / 6520 / 5394 / 4470 for K = 2..5 — still monotone decreasing, i.e. K = 5 was simply the edge of the range searched, not a justified optimum.)
- **Primary selection criterion: validation-set (2018) log-likelihood**, with BIC/AIC reported as secondary.
- Reject degenerate solutions: any state with expected duration < 5 days or unconditional probability < 2%.

### 8.5 Deterministic relabeling

EM state ordering is arbitrary and changes across restarts and refits (label switching). Implement `canonical_labels(model)`:
- Sort states by a stable, economically meaningful statistic — recommended: state-conditional return standard deviation (ascending).
- Apply the permutation to `startprob_`, `transmat_`, `means_`, `covars_`, and posterior columns together.
- Names are derived from position after sorting, never hard-coded.

**Delete every hard-coded label list.** The reference code had three (HMM cell 16 live-regime readout, LSTM PCA legend, LSTM regime bar chart) plus the Streamlit dashboard, all assuming an ordering that did not match the fitted model. The "live regime" readout was reporting the bull-state probability under a crisis label.

### 8.6 Walk-forward regime generation

`models/hmm/walkforward.py`:
- Expanding-window refit, monthly cadence (configurable).
- Carry the filter state across refit boundaries — do not restart from `startprob_` each month.
- The refit date must not be inside its own fit window.
- Relabel canonically after every refit.
- Produce a **single, de-duplicated, calendar-aligned** posterior series. (The reference rolling loop used an inclusive slice and produced 1,557 rows for 1,509 trading days.)
- Scalers are refit per fold on fold data only — never mutated in place on a shared object.

### 8.7 Evaluation

Out-of-sample, against independent references:
- **State characterisation:** mean return, volatility, max drawdown, average duration, transition matrix per state.
- **Economic validity:** overlap with NBER recession dates and with drawdown-based bear flags (e.g. SPY ≥ 20% off its high); report detection lag and false-alarm rate.
- **Posterior quality:** mean entropy (saturation check), distribution of max posterior, stability of the assigned state.
- **Beats-baseline test:** compare against `threshold_regime.py` (VIX quantile or realised-vol quantile rules). **If the HMM cannot beat a two-state volatility threshold on both detection lag and downstream probe performance, that is a finding — report it.**

**Acceptance for Step 2:** filtered posteriors pass causality tests; K is selected on validation likelihood with restart variance reported; states are canonically labelled; the HMM is characterised against NBER/drawdown labels and the threshold baseline.

---

## 9. LSTM encoder component

### 9.1 What it is for

An LSTM is a sequence model: it reads inputs step by step and maintains a hidden state. Its association with text is incidental — it is a general-purpose sequence architecture. Here it compresses a rolling window of multivariate market features into a low-dimensional vector, so the agent receives a summary of recent *dynamics* (vol trending up vs down, momentum building vs fading) rather than a single-day snapshot.

This is a **hypothesis, not a given**: reconstruction fidelity is not the same as decision-relevant information, and daily returns are close to white noise. Tier 1 exists to test it.

### 9.2 Known problems in the reference implementation

1. **Off-by-one alignment** — the window ending at row `k+29` was labelled with the date at `k+30`, and dataset length was `len - window` (dropping the final window).
2. **No validation set** — early stopping and "best weights" were driven by *training* loss, making both meaningless; loss was still falling at epoch 100.
3. **ReLU on the latent** — produced permanently-dead units (`latent_0`, `latent_6` identically zero) and a visible activation-distribution shift after ~2018 (train/test drift).
4. **Capacity mismatch** — ~507k parameters against ~2,400 heavily-overlapping windows.
5. **In-sample evaluation only** — reconstruction quality reported on the first batch (2008 data).
6. **Circular validation** — PCA-by-regime separation is guaranteed when features contain volatility and regimes are a function of volatility.

### 9.3 Required design

- **Windowing:** window `W = 30` (sweep 10/20/30/60). Row dated `D` = encoder output for the window **ending at** `D`. Unit-tested.
- **Architecture:** encoder LSTM → latent (linear or `tanh`, **no ReLU**) → decoder. Latent dim swept over {8, 16, 32}. Hidden size swept over {32, 64}. Dropout and weight decay as regularisation.
- **Variants to compare:**
  - `AE` — plain reconstruction
  - `DAE` — denoising (Gaussian input noise)
  - `VAE` — with a small KL weight (gives a smoother, better-behaved latent for RL)
  - `PRED` — predictive head(s): forward 5d/20d realised vol and forward 20d correlation, optionally multi-task with reconstruction
- **Training:** validation loss (2018) drives early stopping and checkpointing. Train to convergence, not a fixed epoch count. Log train/val curves.
- **Evaluation:**
  - reconstruction MSE **per feature family** and **per period** (train/val/each test episode)
  - number of active latent dimensions (variance above a threshold)
  - latent distribution drift across periods (KS or MMD, train vs test)
- **Walk-forward refits** (annual). The agent must train on latents produced the same way it will see them at test time — otherwise it learns on clean in-sample encodings and is tested on noisier out-of-sample ones.
- **Seeds:** ≥ 5. Latents are not comparable across seeds (rotation/permutation), so **compare downstream probe performance, not raw latent values**.
- **Normalisation:** standardise latents (train-fold statistics) before they enter any state vector.

### 9.4 Mandatory baselines

Every claim about the LSTM is relative to these, evaluated identically:
- `pca_encoder` — PCA to the same dimension on the flattened window
- `random_encoder` — untrained LSTM with random weights (frozen)
- `rolling_stats` — hand-picked summary statistics of the same window
- `raw_window` — the flattened window itself (upper bound on available information)

**Acceptance for Step 3:** the encoder has a validation-driven training loop, no dead units, walk-forward refits, and beats PCA and the random encoder on the Tier-1 probe with non-overlapping confidence intervals. **If it does not, simplify or drop it and report that as a finding.**

---

## 10. State vector assembly

`src/prism/state.py` builds state vectors per variant. All variants share the same index, scaling policy, and portfolio-state block. **This module is built in Phase A** — Tier 1 consumes the same variants Tier 2 will.

| Variant | Contents | Isolates |
|---|---|---|
| **V1** | current-day features (baseline) | — |
| **V1′** | same features over the window, flattened or PCA-compressed | "more history" vs "the LSTM" |
| **V2** | V1 + LSTM latent | — |
| **V3** | V1 + HMM filtered posteriors | — |
| **V4** | V1 + latent + posteriors (full model) | — |
| **C1** | V1 + random-encoder latent (frozen, untrained) | "more dimensions" vs "learned representation" |
| **C2** | V1 + threshold-regime one-hot | "the HMM" vs "any regime signal" |
| **C3** | V1 + *shuffled* HMM posteriors | permutation control |
| **O1** *(optional)* | V1 + **smoothed** posteriors — deliberately leaky | upper bound on perfect regime knowledge |

**Why the controls are mandatory.** With V1–V4 alone, "V2 beats V1" has at least three explanations: the LSTM learned something; thirty days of history helps and any summary would do; or the state vector simply got 32 dimensions wider. The design cannot separate them, and an examiner will ask. With the controls, the claims become falsifiable:

- LSTM adds value ⟺ **V2 > V1′ and V2 > C1**
- HMM adds value ⟺ **V3 > C2 and V3 > C3**
- Regime conditioning adds beyond temporal ⟺ **V4 > V2** ← *this is the research question*

**O1 (oracle)** is diagnostic only and must never be reported as a result. If even the leaky oracle barely beats V1, that is a strong finding about how much regime information is worth in principle, and it reframes any null result from the honest variants.

**Portfolio block.** In Phase B every variant additionally carries current weights, time since last rebalance, and cumulative turnover — without current weights an agent cannot reason about transaction costs (absent from the reference design). In Phase A the allocator supplies these directly, so `state.py` must accept them as an optional block, wired but unused.

---

## 11. RL environment — **PHASE B, DEFERRED**

> Not implemented in Phase A. Create `src/prism/env/` with `__init__.py` and docstring stubs only. Specified here so Phase A's interfaces anticipate it.

`src/prism/env/portfolio_env.py` — Gymnasium-compatible.

- **Observation:** the variant's state vector (float32, finite, standardised).
- **Action:** continuous vector over `n_assets + 1` (cash). Map to weights with a softmax or a simplex projection; enforce bounds (e.g. each ≤ 0.35) and long-only by construction, not by penalty.
- **Timing:** observe at close `t` → action → execute at close `t + execution_lag` → reward from the realised return over the holding period. Assert in tests that the reward window does not overlap the observation window.
- **Costs (`env/costs.py`):** proportional cost on turnover (default 5 bps per side, configurable), plus a spread/slippage term scaled by prevailing volatility. Report results at 0 / 5 / 10 / 20 bps as a sensitivity.
- **Reward variants (`env/rewards.py`):**
  - `log_return_net` — log of net-of-cost portfolio return (default)
  - `dsr` — differential Sharpe ratio
  - `mv_penalty` — return − λ·variance
  - `drawdown_penalty` — adds a penalty term on running drawdown
  Reward choice is a documented experimental factor, not a silent default.
- **Episodes:** sample start points within the training period (random-start episodes) so the agent does not memorise one path; fixed full-period episodes at evaluation.
- **No look-ahead in the env:** the env holds a pre-built state array and a pre-built return array; it may never index beyond the current step.

---

## 12. Agents — **PHASE B, DEFERRED**

> Not implemented in Phase A. Create `src/prism/agents/` with docstring stubs only.

- **Algorithm:** SAC (continuous actions, entropy-regularised, sample-efficient). `stable-baselines3` is acceptable; if implemented from scratch, test against a known toy problem first.
- **Shared hyperparameter budget:** every variant gets the *same* search budget over the *same* grid, tuned on validation only. Unequal tuning invalidates the ablation.
- **Seeds:** ≥ 10 per variant (the config's `seeds.runs`).
- **Logging:** learning curves, entropy coefficient, actor/critic losses, gradient norms, and validation performance per checkpoint.
- **Checkpoint selection:** on validation performance, never test.
- **Sanity gates before Tier 2 results are trusted:**
  - the agent beats a random policy
  - the agent can learn a degenerate task (e.g. one asset with a deterministic edge)
  - performance is stable across seeds within a reported band

---

## 13. Ablation protocol

### 13.1 Tier 1 — representation ablation (no RL) — **PHASE A DELIVERABLE**

This is the headline output of Phase A. Fast, deterministic, and a legitimate research result on its own.

- **Probes (`probes/probe.py`):** walk-forward ridge/logistic regression from each feature set to forward targets (`fwd_vol_5`, `fwd_vol_20`, `fwd_max_drawdown_20`, `fwd_corr_20`, `fwd_ret_20`). Report out-of-sample R² / AUC with block-bootstrap CIs.
  - Probe hyperparameters (ridge α) tuned on validation folds only, with the same grid for every variant.
  - Report `fwd_ret_*` results with explicit caution: near-zero or negative out-of-sample R² on return prediction is the normal, expected outcome and is not evidence of a broken pipeline. Risk targets (`fwd_vol_*`, `fwd_corr_20`, `fwd_max_drawdown_20`) are where regime information should show up.
- **Allocator (`probes/allocator.py`):** drive a classical allocator (mean-variance with shrunk covariance; vol-target; regime-conditional vol-target) from each feature set. Report net-of-cost performance metrics. This is the economic counterpart to the statistical probe — a variant can win on R² and lose here, which is itself informative.
- **Feature sets compared:** V1, V1′, V2, V3, V4, C1, C2, C3 (+ O1 diagnostically).

**Gate rule:** a component that does not beat its own control (V2 > V1′ and V2 > C1; V3 > C2 and V3 > C3) with non-overlapping CIs is redesigned or dropped **before** Phase B. Record the decision either way in `DECISIONS.md`.

**Pre-registration applies here too.** Write the hypotheses and metrics into `reports/tables/preregistration.md`, committed, before the test period is touched.

### 13.2 Tier 2 — policy ablation (with RL) — **PHASE B, DEFERRED**

- Train SAC on V1, V1′, V2, V3, V4 (+ controls if Tier 1 was ambiguous), ≥ 10 seeds each.
- Evaluate on the test period, reported three ways: full period, calendar sub-periods, and **drawdown episodes** (see §14.3).
- **Pre-register hypotheses and metrics before running the test period.** Write them into `reports/tables/preregistration.md` with a git commit timestamp.

### 13.3 Benchmarks (`backtest/benchmarks.py`)

Equal-weight, 60/40, minimum-variance, risk-parity, vol-target, and buy-and-hold SPY. The reference project compared only against equal-weight and 60/40, which is too weak a bar for a regime-aware claim — min-variance and vol-target are the honest competitors.

---

## 14. Evaluation and statistics

### 14.1 Metrics (`backtest/metrics.py`)

Annualised return, annualised volatility, Sharpe, Sortino, Calmar, max drawdown, drawdown duration, VaR/CVaR (95%), turnover (annualised), net-vs-gross return, hit rate, and tail ratio. **All headline metrics are net of costs.**

### 14.2 Uncertainty (`analysis/`)

- **Block bootstrap** (stationary or circular, block ≈ 20 days) for CIs on all metrics.
- **Seed variance** reported explicitly: mean ± std and the full distribution across seeds. A variant "wins" only if its CI does not overlap the comparator's.
- **Permutation test:** shuffle HMM posteriors (C3) and confirm the V3/V4 advantage disappears.
- **Multiple-testing awareness:** report deflated Sharpe or an equivalent adjustment; you are running many variants.

### 14.3 Episode-based reporting (`analysis/episodes.py`)

Calendar blocks mislead: the reference "COVID stress" test period (2020–21) contains a violent crash *and* a strong bull market; the "rate shock" period (2022–23) contains the 2023 rebound. Segment the test period by **drawdown episodes** (peak → trough → recovery, e.g. ≥ 10% drawdowns) and report per-episode performance alongside calendar results. This is where a regime-aware model should earn its keep, and where it can actually be seen.

---

## 15. Reporting

### 15.1 Expected outcomes — what Phase A is likely to find

State these as hypotheses *before* running, so the result is interpretable either way. Written here because a project whose only acceptable outcome is "it worked" is not research.

| # | Hypothesis | Prior | If confirmed | If refuted |
|---|---|---|---|---|
| H1 | The rebuilt HMM's states differ in more than volatility level (distinguishable mean return, correlation structure, credit behaviour) | **Uncertain — depends on the new observation set.** The reference version collapsed to a vol ladder because `vol20` was an input | Regime conditioning has a mechanism to add value | Report that market regimes on this data are approximately one-dimensional in volatility — a real, citable finding |
| H2 | HMM beats a VIX/vol-threshold rule on detection lag and probe performance | **Roughly even.** Threshold rules are strong baselines and frequently competitive | HMM earns its complexity | The honest headline: probabilistic regime detection adds little over a threshold on daily ETF data |
| H3 | LSTM latent beats PCA and the random encoder on risk probes (`fwd_vol_*`, `fwd_corr_20`) | **Likely yes, modestly.** Volatility and correlation are persistent and compressible | Temporal representation carries information | Reconstruction-based encoding is the wrong objective; pivot to the predictive head (§9.3 `PRED`) |
| H4 | LSTM latent beats baselines on **return** probes (`fwd_ret_20`) | **Likely no.** Near-zero out-of-sample R² on returns is the normal outcome | Would be a notable result, treat with suspicion and re-audit for leakage | Expected — report plainly, do not tune toward it |
| H5 | **V4 > V2** — regime conditioning adds beyond temporal (*the research question*) | **Genuinely uncertain.** This is why the project is worth doing | The thesis claim holds | Equally publishable: LSTM latents already encode regime information implicitly, making explicit HMM conditioning redundant |
| H6 | Regime-conditioned allocation improves risk metrics (max drawdown, CVaR) more than return metrics | **Likely yes** | Frame the contribution as risk management, not alpha | — |
| H7 | Any advantage concentrates in drawdown episodes, not full-period averages | **Likely yes** — this is why §14.3 exists | Report per-episode; it is the honest place to look | — |

**The single most probable overall outcome, stated plainly:** modest, statistically fragile improvements that are visible in risk metrics during crisis episodes and invisible in full-period Sharpe. **This is a normal and publishable result for this class of research, and the report must be written to accommodate it.** The apparatus in §14 (bootstrap CIs, permutation tests, deflated Sharpe) exists precisely so that a modest true effect can be distinguished from noise — and so that a null result is a *conclusion* rather than a failure.

**What would indicate something is wrong**, rather than a genuine null:
- Out-of-sample R² above ~0.15 on 20-day forward *returns* → almost certainly leakage; re-audit before celebrating.
- Sharpe above ~2.0 net of costs on the test period → same.
- A variant's advantage that survives the C3 shuffle control → the advantage is not coming from regimes.
- Perfectly separated HMM posteriors (entropy near zero) → the conditional-independence violation has returned.

### 15.2 Report contents

`scripts/07_report.py` regenerates **every** figure and table from stored results.

**Phase A report** (the deliverable now):

- Data quality report, universe and split diagram
- HMM: state characterisation table, transition matrix, regime timeline vs SPY drawdowns, detection-lag table vs NBER/drawdown labels, posterior entropy, K-selection curve with restart spread
- Encoder: train/val curves, per-feature reconstruction MSE, active-dimension count, latent drift, probe comparison vs baselines
- Tier 1 table: all variants × probe targets, with CIs
- Allocator results per variant: equity curves, drawdowns, weight heatmaps, turnover
- Per-episode performance table (§14.3)
- Cost-sensitivity table (0/5/10/20 bps)

**Phase B additions (deferred):** Tier 2 table with seed distributions, agent learning curves, final holdout results.

**Reproducibility statement** in the README: snapshot hash, git commit, config hash, seed list, and the exact command sequence.

---

## 16. Build order and acceptance criteria

Work strictly in this order. Do not begin a step before the previous one's acceptance criteria pass.

| # | Step | Phase | Acceptance |
|---|---|---|---|
| **0** | Repo, config, snapshot, splits, **tests** | A | Suite runs; causality passes on features; validated against injected leaks; holdout lock works |
| **1** | Dataset expansion: two universes (§3.1), defensive assets, macro, cross-sectional + QA | A | Both universes build; QA report clean; no constant/degenerate features; all features pass causality; A-pipeline provably never touches B-only tickers |
| **2** | HMM rebuild (fit on Universe A, §3.2) | A | Filtered posteriors verified against prefix runs; K chosen on validation; canonical labels; characterised vs NBER/drawdown and threshold baseline; 2007+-only robustness refit reported |
| **3** | Encoder rebuild (fit on Universe A) | A | Alignment test passes; val-driven training; no dead units; walk-forward; beats PCA and random encoder on probe |
| **3b** | `state.py` — all 9 variants | A | Every variant reproducible from config; schema hash recorded; portfolio block wired but unused |
| **4a** | Tier 1 ablation + Phase A report | A | Full variant × target table with CIs; allocator results; gate decisions in `DECISIONS.md`; report regenerates from one command |
| — | **PHASE A ENDS — review before continuing** | — | Exit criteria in §0.3 all met |
| **4b** | Environment + costs | B | Env tests pass; buy-and-hold reproduces analytic benchmark; cost model verified |
| **4c** | SAC + Tier 2 ablation | B | Sanity gates pass; ≥ 10 seeds per variant; pre-registration committed before test evaluation |
| **5** | Final report + holdout | B | One command regenerates everything; holdout evaluated exactly once |

**Ordering note:** Step 2 (HMM) is placed before Step 1's completion in earlier discussion because the leakage fix is urgent; however, since the dataset change alters the HMM's observation set, **do Step 1 first** in the final build. The leakage fix is then applied once, to the correct data.

---

## 17. Instructions for the coding assistant

0. **Scope: Phase A only (§0.3).** Build steps 0, 1, 2, 3, 3b, 4a. Do not write RL environment or agent code, do not install `gymnasium` or `stable-baselines3`, and stop at the Phase A exit criteria for review rather than continuing into Phase B.
1. **Read this document fully before writing code.** Ask clarifying questions on the open decisions in §3 before implementing them.
2. **Do not port notebook code.** The notebooks are reference material with documented defects (Appendix A).
3. **Tests first** for Step 0. The causality test must be shown to fail on deliberately-injected leaks before it is trusted.
4. **Small, reviewable commits**, one per acceptance criterion, each with tests.
5. **Surface, don't silently resolve, any ambiguity** — especially anything affecting the timing convention, fit scope, or split boundaries.
6. **When a result is negative, report it.** "The HMM adds nothing beyond a volatility threshold" is a legitimate and publishable outcome for this project. Do not tune until the desired answer appears; that is the failure mode this entire spec exists to prevent.
7. **Never touch the holdout** — not in Phase A, and not until §16 Step 5.
8. Write a short `DECISIONS.md` log: every non-obvious choice, the alternatives, and why.
9. **Report progress against the §16 table**, naming the step and its acceptance criteria, so review points are unambiguous.

---

## Appendix A — Defects in the reference notebooks

Documented so they are not reproduced. Notebook cell numbers refer to the uploaded reference versions.

### `pipeline.ipynb`
| # | Defect | Impact |
|---|---|---|
| A1 | Re-downloads and overwrites "frozen" data each run; no hash, date, or version recorded | Non-reproducible; adjusted prices are restated as dividends accrue |
| A2 | `fillna(method='bfill')` in the QA cell | Writes future prices into the past (did not fire, but is a live trap) |
| A3 | Overlap check compares only train vs others, using hard-coded `'2008':'2017'` | Proves nothing; splits are disjoint by construction |
| A4 | No embargo, no holdout; data ends 2023-12-31 | ~2.75 years of free out-of-sample data unused |
| A5 | Splits contain features only; no aligned reward series | Reward alignment left to downstream code — easy to leak |
| A6 | Feature warm-up starts 2008-03-31 but HMM input starts 2008-01-31 | Inconsistent training windows across components |
| A7 | `SPY_spycorr ≡ 1.0` | Constant feature carried into V1 |
| A8 | Extreme-return threshold of 50%; no stale-price or calendar checks | QA too permissive to catch real problems |
| A9 | `warnings.filterwarnings('ignore')` | Hides deprecations and convergence warnings |
| A10 | Cumulative-return plot compounds log returns as simple returns | Cosmetic (plots only) |
| A11 | All-equity universe; no defensive assets, no macro, no cross-sectional features | Regime knowledge has nothing to act on |

*Verified as correct:* `auto_adjust=True` (total-return prices), and the feature builder is causal (passes the truncation and perturbation tests).

### `prismhmm.ipynb`
| # | Defect | Impact |
|---|---|---|
| B1 | `predict_proba` over the full sample (smoothed posteriors) feeds `state_vector.parquet` | **Look-ahead leakage into the RL state.** Any V3/V4 advantage would be an artefact |
| B2 | `model.score` multiplied by `n`; wrong free-parameter count | BIC invalid; K = 5 unjustified (corrected BIC is monotone to the edge of the search range) |
| B3 | `spy_vol20` used as an HMM observation | Violates conditional independence; saturates posteriors; states collapse to a volatility ladder |
| B4 | Hard-coded state-name lists assuming the wrong ordering (cell 16, plus two in the LSTM notebook and the dashboard) | Live "crisis probability" readout was actually the bull-state probability |
| B5 | Rolling refit: label switching, duplicate dates (1,557 rows for 1,509 days), filter restarted each month, refit date inside its own window, scaler mutated in place | The rolling path is unusable as written — and it is not what fed the state vector anyway |
| B6 | Validation = eyeballing smoothed in-sample crisis probabilities during GFC/COVID; 2022 never checked | Circular and incomplete |
| B7 | Fixed 5-year rolling window | Post-2013 windows contain no crisis |

### `LSTM.ipynb`
| # | Defect | Impact |
|---|---|---|
| C1 | Window ending at `k+29` labelled with date `k+30`; length `len - window` | Off-by-one; last window dropped |
| C2 | Early stopping and checkpointing on *training* loss; no validation set | Model selection meaningless; not converged at epoch 100 |
| C3 | ReLU on the latent | Permanently dead units; post-2018 activation drift |
| C4 | ~507k parameters vs ~2,400 overlapping windows | Overfitting risk |
| C5 | Reconstruction evaluated on the first training batch | In-sample only |
| C6 | PCA-by-regime "validation" | Circular: features contain vol, regimes are a function of vol |
| C7 | No baselines (PCA, random encoder, rolling stats) | No evidence the LSTM adds anything |
| C8 | Single seed | No uncertainty quantification |

### Design-level gaps (all notebooks)
| # | Gap |
|---|---|
| D1 | No portfolio state (current weights, turnover) — agent cannot reason about costs |
| D2 | No transaction-cost or slippage model |
| D3 | V1 baseline sees one day; V2 sees 30 — unfair comparison without V1′ |
| D4 | Encoders and HMM fit in-sample for the training period, applied out-of-sample at test — train/test distribution mismatch for the agent |
| D5 | One crisis in the training period; single-seed RL comparisons would be noise — addressed by the two-universe design (§3.1) |
| D6 | Benchmarks limited to equal-weight and 60/40 |
| D7 | Test periods are calendar blocks that mix crashes with rallies |

---

## Appendix B — Glossary

- **Filtered posterior** — `P(state_t | data up to t)`. Causal. What the agent may see.
- **Smoothed posterior** — `P(state_t | all data)`. Uses the future. Useful for *post hoc* analysis only.
- **Embargo** — a gap between splits sized to the longest forward-looking horizon, preventing target/reward overlap across boundaries.
- **Walk-forward** — refitting periodically using only past data, then applying forward.
- **Label switching** — the arbitrary ordering of latent states across EM runs.
- **Tier 1 / Tier 2** — representation ablation (no RL) / policy ablation (with RL).
- **Deflated Sharpe** — a Sharpe ratio adjusted for the number of strategies tried.
