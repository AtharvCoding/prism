# PRISM Dashboard — build specification

**Audience of this file:** the coding session that builds the dashboard. It is a spec, like `PRISM_BUILD_SPEC.md`, for one deliverable:
an interactive Streamlit app that explains PRISM end to end, shows the frozen system running on fresh data, and reports the findings honestly.
Read §0–§3 before writing any code. §4–§6 say how to get the data; §7 says what each page shows; §8 onward say how to build, test and finish.

---

## 0. Goal and framing

**Goal.** A smooth, interactive app that

1. **explains PRISM end to end**: the research question, the data, the HMM regimes, the LSTM latent, the SAC agent and the results;
2. **shows the machine working live**: this week's 14 portfolio weights computed from fresh market data through the *frozen* pipeline, and a
   "what-if" lab where the viewer changes the regime input and watches the agent respond;
3. **reports the findings honestly**: the null result replicated on the holdout, and the agents do not beat the simple benchmarks after costs.
   Present it as rigour, not failure.

**Framing used throughout (use these words on the Home page and the Verdict page):**
> *We built the full system end to end, and tested it rigorously enough to know what it does and doesn't do.*

**Tone rules (non-negotiable):**
* Never imply the agents work, predict, or should be followed. The live weights are a *demonstration of a frozen system*, not a recommendation.
  Every page that shows live or historical weights carries this permanent caption:
  > *Demonstration of the frozen system. Not a recommendation; the agents did not beat these benchmarks after costs.*
* The negative result is the headline, stated plainly, and framed as a *pre-registered, replicated, properly powered-as-far-as-possible test*. No spin
  in either direction: do not call it a failure, do not hide it, do not bury it on the last page.
* Numbers shown must be the stored, pre-registered ones. The dashboard **computes no new statistic for the verdict**; it displays
  `reports/final_report.md` and the tables behind it. Where a chart needs a derived series (an equity curve, a per-seed Sharpe) it is derived
  from the stored daily returns and checked against the stored tables (§6).
* Plain language first; jargon is defined in a "How to read this" expander. A smart non-specialist should follow every page.

---

## 1. What the project is (the facts the app must get right)

**Research question (spec §0.1).** Does explicit probabilistic regime conditioning (an HMM) improve portfolio allocation beyond what a temporal
representation model (an LSTM autoencoder) and raw features already capture? Answered by a two-tier ablation: Tier 1 (no RL; probes and a classical
allocator) and Tier 2 (SAC agents).

**Tradable universe (14 weights):**
* 9 sector ETFs: XLK, XLE, XLV, XLF, XLU, XLI, XLP, XLY, XLB
* Bonds: SHY, IEF, TLT
* Gold: GLD
* Cash (accrues at the prior day's `^IRX`)

Rules: long-only, each *risky* asset capped at 35% (cash is not capped), weights sum to 100%.

**Display sleeves (for the donut and the stacked area):** Equity = the 9 sectors; Bonds = SHY, IEF, TLT (shown as a duration ladder: short, mid, long);
Gold = GLD; Cash.

**Non-tradable signals:** SPY (benchmark and signal, *not* an agent asset), VIX, VIX3M, ^TNX, ^FVX, ^IRX, the HYG/LQD credit proxy, DX-Y.NYB, CL=F.
(Excluded for short history: XLRE, XLC.)

**Two universes.** Universe A (1999 onward): the sectors, SPY and macro series; used to *fit* the HMM and the LSTM (they see the dot-com crisis, 2008 and
2020). Universe B (2007-04-04 onward, bound by HYG's inception): adds bonds, gold, credit and the dollar; the allocation universe.

**Splits (Universe B):**

| Split | Period | Status |
|---|---|---|
| Train | 2007-04-04 .. 2017-11-22 (effective, after a 25-session embargo) | the agents learn here |
| Validation | 2018-01-01 .. 2018-11-21 | tuning and checkpoint choice |
| Test | 2019-01-01 .. 2023-11-22 (effective) | **exploratory** (viewed in Tier 1 before Tier 2) |
| Holdout | 2024-01-09 .. 2026-09-30 (first sessions of the window) | **confirmatory, spent**: evaluated exactly once, 2026-10-05 |

**Agents.** Four variants, **10 seeds each = 40 frozen SAC agents**, rebalancing **weekly** (decision at the week's last session close, executed at the next close):
* **V1** — the 184 raw features (+ the agent's own current weights and turnover);
* **V2** — V1 + the 32-dim LSTM (denoising autoencoder) latent;
* **V4** — V1 + latent + the 2 HMM posteriors (`state_0`, `state_1`);
* **C4** — V1 + latent + a 2-column **VIX-threshold** one-hot (the control that isolates "HMM vs any regime signal").
Reward: weekly log net return. Costs: 5 bps per side per risky leg + 0.02 × trailing daily vol per unit traded. SAC (stable-baselines3), 250 000 steps/run,
reward scaled ×10 *during training only*, checkpoint chosen on validation. Frozen configs in `data/processed/tier2/chosen_configs.json`
(γ = 0.97 for V1, V2, V4; 0.9 for C4; all networks 64×64).

**HMM.** Specification **H1**: K = 2, SPY daily returns only. States are ordered by volatility in every fold: `state_0` = **Calm**, `state_1` = **Volatile**.
Initial fit window 1999–2006, then **204 monthly walk-forward folds** (expanding window, filtered posteriors only — they use data up to that day).
*Why K = 2:* K = 2..8 were searched on validation log-likelihood; K = 3 was ahead by under 2 points (noise); K = 2 won on BIC and was the only K with real dwell
times; the multivariate spec H2 collapsed into short-lived "shock" states. (DECISIONS.md D-024, D-025; `reports/tables/hmm_k_selection_H1.md`.)

**Verdict (pre-registered; `reports/final_report.md`).** The three comparisons — **V4 vs V2** (HMM beyond the LSTM: the research question), **V4 vs C4**
(HMM vs a VIX threshold), **V2 vs V1** (the LSTM latent) — **all fail on both the test split and the holdout** (0 of 4 metrics favourable, 0 of 4 adverse; every
CI contains zero). On the holdout the sign of the difference flipped for two of the three. Seed-to-seed Sharpe spread (about 0.3) exceeds every between-variant gap;
no variant's deflated Sharpe reaches 0.95. The agents **do not beat equal weight, 60/40 or risk parity after costs**, mainly because they trade 23–38% of the portfolio a
week against 1–4% for the benchmarks. Tier 1 said the same earlier: the HMM added nothing over a two-bin VIX threshold, and the trained LSTM latent was no better than an
untrained random projection.

---

## 2. Hard constraints (read these twice)

1. **The holdout is spent.** Never use it to choose, tune or re-select anything. The dashboard shows it, labelled, and nothing else changes. Do not re-run
   any *evaluation* statistic on it. Descriptive replay of the frozen policies to record *weights* is allowed only as specified in §5.4 (amendment first).
2. **Frozen artifacts are immutable.** Never overwrite or regenerate: `data/processed/states/*`, `data/processed/tier2/*` (runs, `chosen_configs.json`,
   `eval_daily.parquet`, `tables/`), `data/processed/holdout/*`, the three pre-registrations, any `*_features.parquet`, the raw snapshot. They are pinned by SHA-256
   (`reports/tables/preregistration_tier2.md` §2, `preregistration_holdout.md` §1). New outputs go in new directories (`dashboard/artifacts/`, `data/live/`).
3. **No fitting or training at request time.** The app loads precomputed artifacts and runs forward passes only (SAC inference on 40 tiny MLPs, an HMM filter step,
   an LSTM forward pass). A page must render in under ~1 s from cache.
4. **No look-ahead anywhere in the live path.** Everything for date *t* uses data to the close of *t* (the repo's causality tests are the model: `tests/test_causality.py`).
5. **The frozen snapshot is write-once.** Live data goes to a separate cache and never into `data/raw/`.
6. **Do not edit the pre-registrations in place.** Amendments are appended, dated and committed (§5.4).
7. Match the repo's conventions: Python 3.11, `.venv`, pinned dependencies in `pyproject.toml`, tests for new logic, `DECISIONS.md` entries for every non-obvious choice,
   small reviewable commits. Surface ambiguity; record choices with alternatives.

---

## 3. Architecture

```
dashboard/
  app.py                    # st.navigation, theme, global caption, as-of/freshness header
  pages/                    # one file per page, in story order (§7)
  components/               # pipeline_diagram, gauges, donuts, whisker bars, captions, "how to read" helper
  artifacts/                # PRECOMPUTED, committed (small): see §5. Regenerated only by scripts/10_dashboard_data.py
  assets/                   # css, any static images
src/prism/live.py           # live data refresh, splice, features, frozen-model forward pass, 40-agent inference (pure functions, tested)
src/prism/dashboard_data.py # the precompute: replay, fold parameters, derived series, validation against stored tables (pure functions, tested)
scripts/10_dashboard_data.py
tests/test_live.py  tests/test_dashboard_data.py  tests/test_dashboard_app.py   # streamlit.testing.v1.AppTest smoke tests
```

* **Stack:** Streamlit (multipage via `st.navigation`) + Plotly for charts. Add an optional dependency group `dashboard` to `pyproject.toml` (pinned `streamlit`,
  `plotly`); `make install` stays unchanged, add `make dashboard-install`. `make dashboard-data` runs the precompute; `make dashboard` runs `streamlit run dashboard/app.py`.
* **Data flow:** `scripts/10_dashboard_data.py` (offline, minutes) → `dashboard/artifacts/*` (parquet/JSON) → pages read through `st.cache_data`. The live path
  (`prism.live`) is the only code that touches the network, and only on the Live weights page and Home.
* **Performance budget:** cold start < 5 s; page switch < 1 s; the refresh button < 20 s (network + one forward pass), with a spinner and a clear error if offline.
* **Offline behaviour:** if the network is unavailable or yfinance fails, the app shows the last cached live result with a red "stale as of …" badge. It never crashes
  and never fabricates a number.

---

## 4. The live pipeline — what "fresh data through the frozen pipeline" really means

This is the hard part; do it carefully.

### 4.1 Problems you must solve (they are not obvious)

1. **The walk-forward models are not persisted.** The HMM refits monthly and the encoder annually, and steps 2–3 stored only their *outputs* (posteriors, latents).
   The live path needs the *last fold's* models. **The precompute must re-run the walk-forward (`prism.holdout.build_extended_states` already does, in ~5 min, and
   is replay-verified) and persist the final fold's parameters:** HMM means, variances, transition matrix, start probabilities, the fold's `FeatureScaler`, the
   state ordering; encoder weights + its `FeatureScaler` + architecture (window 10, latent 32, hidden 64, DAE). The fold objects expose `.model`, `.scaler`, `.order`
   (`WalkforwardFoldOutput`, `EncoderFoldOutput`). Save to `data/live/models/` (not `data/processed/`).
2. **"Frozen" means the last fold.** The research pipeline would refit the HMM monthly and the encoder annually. The live app deliberately does **not**: it freezes the models as of
   the last holdout fold (HMM fit through ~Aug–Sep 2026, encoder fit through end of 2025) and states this on the page ("models last refit: …; the live view does not refit").
   The age of the models is shown in the freshness badge.
3. **Adjusted prices restate.** `auto_adjust=True` prices change retroactively when a dividend is paid, so appending a fresh download to the frozen snapshot creates a *level seam*.
   Returns are what matter: download an **overlap window** (the last ~60 sessions of the snapshot plus everything new), and **rescale the new series to equal the frozen close on the
   last common session** (per ticker), then append only the genuinely new sessions. Assert the overlap returns agree to ~1e-3 before accepting the splice; otherwise show "data seam check failed" and
   keep the last good result.
4. **The agent's observation includes its own portfolio.** Each agent sees its 14 current (drifted) weights and the *mean one-way turnover so far in the episode* (`PortfolioEnv` observation,
   D-035). A live weight vector therefore depends on a rollout. **Define the live rollout as the holdout's own deterministic episode, continued:** start in cash on 2024-01-09 (the holdout
   window's first decision, the same episode the pre-registered evaluation ran), step weekly, and the *latest* decision's output is "this week's weights". Say so on the page. This makes the live
   number exactly consistent with the stored holdout results up to 2026-09-30, and extends it.
5. **What is "this week"?** Decisions happen at the week's last session close. "As-of" = the latest completed weekly decision date; if today is mid-week, also offer a clearly labelled
   *preview as of the latest close* (the weights the agent would choose if the week ended now). Never present a preview as a decision.
6. **The scaler.** State assembly standardises with a `FeatureScaler` fit on the Universe-B train split (`prism.holdout.build_extended_states` reproduces it); refit it once in the precompute and persist it.

### 4.2 `prism.live` — the pure functions to write (each unit-tested)

| Function | Does |
|---|---|
| `fetch_recent(tickers, start) -> raw_panel` | yfinance download (`auto_adjust=True`), same tickers as `configs/data.yaml`, returns the repo's raw frame format |
| `splice(frozen_snapshot, fresh) -> panel` | the overlap-and-rescale splice of §4.1.3, with the return-agreement assertion |
| `build_live_features(panel) -> features_a, features_b` | `build_features` + the step-1 pruning with the persisted pruner/fit windows (reuse `prism.holdout._prune`, `_load_set`) |
| `filter_hmm(model_params, obs, carry_state) -> posteriors` | the forward (filtering) recursion with the persisted last-fold parameters, continuing the filter from the stored posterior at the holdout end |
| `encode(model, scaler, features_a) -> latents` | the frozen DAE forward pass for the new windows |
| `assemble_state(variant, ...) -> row` | `prism.state.build_state` logic for V1/V2/V4/C4 for the new dates |
| `load_agents() -> dict[variant][seed] -> SAC` | the 40 `best.zip` (CPU), cached with `st.cache_resource`; verify SHA-256 against `preregistration_holdout.md` §1 on load |
| `rollout(variant, states, ...) -> weights_by_week` | the deterministic episode of §4.1.4 through `PortfolioEnv` (eval mode), returning every week's target weights, drifted weights and turnover |
| `ensemble(weights_by_seed) -> mean, min, max, spread` | the V4 ensemble (mean over the 10 seeds) and the whiskers |

**Verification (required, automated):** replaying the pipeline over 2019-01-01..2026-09-30 must reproduce the stored `eval_daily.parquet` / `data/processed/holdout/eval_daily.parquet`
daily returns to 1e-9 per series; and the live HMM filter / encoder forward pass over the holdout must reproduce `states_extended.parquet` to 1e-9. These checks run in `make dashboard-data`
and gate it.

---

## 5. Data contract — what is precomputed, from where

`scripts/10_dashboard_data.py` writes everything below to `dashboard/artifacts/` (small parquet/JSON; commit them; the script is idempotent and refuses to overwrite a differing
frozen source). **Each artifact lists its source and which page uses it.**

### 5.1 Straight from stored results (no recomputation)

| Artifact | Source | Page |
|---|---|---|
| verdict tables (gates, paired differences, claimable) | `data/processed/{tier2,holdout}/tables/paired_block{10,20,40}.csv`, `claims.csv` | 10, 11 |
| seed distributions | `…/tables/seed_metrics_{V1,V2,V4,C4}.csv`, `seed_band.csv` | 10 |
| benchmarks with CIs | `…/tables/benchmark_ci.csv`, `benchmark_metrics.csv` | 10 |
| deflated Sharpe | `…/tables/dsr.csv` | 10 |
| cost sensitivity, turnover | `…/tables/cost_sensitivity.csv`, `turnover.csv` | 6, 10 |
| episodes, calendar years | `…/tables/episodes.csv`, `calendar.csv` | 9, 10 |
| daily net returns, all seeds/variants/benchmarks at 0/5/10/20 bps | `data/processed/tier2/eval_daily.parquet` (test), `data/processed/holdout/eval_daily.parquet` | 10 |
| HMM posteriors (to 2023) and states through the holdout | `data/processed/hmm_posteriors.parquet`; V4's `state_0/state_1` columns in `data/processed/holdout/states_extended.parquet` | 4, 8, 9 |
| encoder latents (to 2023) / V2 latent columns through the holdout | `data/processed/encoder_latents.parquet`, `states_extended.parquet` | 5 |
| K-selection | `reports/tables/hmm_k_selection_H1.md`, `hmm_selection_dwell.csv`, `data/processed/hmm_summary.json` | 4 |
| Tier 1 gates | `data/processed/tier1/gates.json` | 5, 11 |
| NBER / drawdown labels | `configs/reference/nber_recessions.yaml` | 4 |
| learning curves | `data/processed/tier2/runs/final/*/*/seed*/curve.csv` | 6 |

### 5.2 Derived, with a validation check against the stored tables

| Artifact | Derivation | Validate against |
|---|---|---|
| equity curves with seed bands | cumulate `eval_daily.parquet` columns (headline 5 bps) | final NAV matches the stored annualised return |
| per-seed Sharpe, per-seed deflated Sharpe | `prism.backtest.metrics`, `prism.analysis.significance.deflated_sharpe` with the stored SR₀ (`dsr.csv` / report §5) | medians equal `dsr.csv`; Sharpe equals `seed_band.csv` |
| HMM-vs-VIX overlay and agreement rate | `state_1` posterior > 0.5 vs `vix_level` above/below its walk-forward threshold (the C2 control, `prism.models.baselines.threshold_regime`) | the C2 states in `data/processed/states/C2.parquet` |
| 2-D latent map | PCA of the 32-dim latent (fit on the persisted latents, not refit in-app), coloured by regime | — |

### 5.3 Needs a one-off re-run (outputs were never stored)

| Artifact | How | Page |
|---|---|---|
| **per-fold HMM parameters** (means, vols, transition matrix, dwell times, fit end), 204 folds + the holdout folds | re-run `hmm_walkforward` (≈100 s) and persist `fold.model` parameters; assert the posteriors reproduce `hmm_posteriors.parquet` | 4 (transition matrix, dwell times, vol evolution) |
| **final-fold HMM and encoder models** | §4.1.1, to `data/live/models/` | 1, 7, 8 |
| **weekly weights of every agent, 2019 → latest** | replay the 40 frozen policies deterministically on the test and holdout windows (and the live extension), store weekly target weights, drifted weights, turnover | 7, 8, 9 |

### 5.4 The one place the pre-registration is touched — do this first, in this order

Recording *weekly weights* requires running the frozen agents on the **holdout**, which `preregistration_holdout.md` §3 says is evaluated "exactly once". A replay of the same deterministic policies on
the same inputs yields **identical returns** (so no new statistic and no second evaluation) but it does call the holdout env builder, which is behind two keys. Therefore:

1. Append a dated **Amendment 1** to `reports/tables/preregistration_holdout.md` (append only; commit it before anything else): *"Descriptive replay for the dashboard: the 40 frozen agents are replayed
   deterministically on the holdout window solely to record weekly weights; the replay must reproduce `data/processed/holdout/eval_daily.parquet` to 1e-9 per series; no statistic, test or decision uses it;
   the holdout evaluation results are those already stored."* State that holdout results had been seen when this was written.
2. Run the replay through the same two-key path (`final_holdout=True`, `PRISM_ALLOW_HOLDOUT=1`) with a recorded reason, so the access is logged in `reports/logs/holdout_access.jsonl`.
   The user types this command themselves (it is the same human gate as step 5): prepare it, tell them, wait.
3. Abort the whole precompute if the replay does not reproduce the stored daily returns.
4. Record the decision in `DECISIONS.md`.
Everything for the *test* window is not gated (it was already exposed), but use the same replay-and-reproduce check.

---

## 6. Honesty and integrity checks the app itself enforces

* A **"Verified" footer** on the Results and Verdict pages: *"All numbers on this page are read from `reports/final_report.md`'s source tables; the regenerated figures matched them at build time (hash: …)."* The build-time check compares every
  displayed headline number with the stored CSVs and fails the precompute on any mismatch.
* A global **"What this is / isn't"** modal reachable from the header on every page.
* **Freshness badge** on every live element: as-of date, data age, models last refit, status (live / cached / stale / seam-check-failed).
* The **permanent caption** of §0 on every page with weights.
* The V4-vs-V2 and V2-vs-V1 *sign flip* is shown visually and never explained away: the page says the point estimates are noise, because the CIs contain zero and the signs changed.
* No chart may use the word *outperform* about an agent. No "alpha", no "signal strength", no performance projection, no "buy/sell".

---

## 7. Pages (sidebar in story order)

Every page has: a **one-line takeaway** at the top, the interactive content, and a **"How to read this"** expander. Use one colour per sleeve, one per variant, consistently across pages
(Calm = a cool tone, Volatile = a warm tone; colour-blind safe; light and dark friendly).

### 1. Home — *The machine, live*
* An **animated pipeline**: Prices → features → HMM (Calm/Volatile) → LSTM latent → state → SAC ×10 seeds → 14 weights. Each node shows **today's real values**: the regime bar fills to P(Volatile),
  the latent sparkline draws, the weights fan out into a donut. (Animate with Plotly frames or a lightweight SVG/CSS component; keep it under ~2 s.)
* **Metric cards:** as-of date, current regime, equity/defensive split (equity vs bonds+gold+cash), turnover from last week.
* A link to the verdict page; the §0 framing sentence.
* Freshness badge + refresh button.

### 2. The question
* The research question in one paragraph and the three comparisons (V4 vs V2, V4 vs C4, V2 vs V1).
* A visual of **V1, V2, V4 and C4** showing *what each one sees*: stacked blocks (features | latent | HMM posteriors | VIX one-hot | own weights), highlighting what differs between each pair.
* The two tiers in two sentences (probes, then agents), and why Tier 1 exists (so a null at Tier 2 is interpretable).

### 3. Data and universe
* A **timeline** of Universe A (1999 →) and B (2007-04-04 →) and the four splits. The **holdout is drawn sealed, then opened once** (a small animation: lock → "opened 2026-10-05, once").
* A **map of tradable assets vs signal-only series** (sleeves; SPY marked "benchmark & signal, not tradable"; excluded XLRE/XLC with reasons).
* The allocation rules (long-only, 35% cap, sums to 100%, weekly).

### 4. Regimes
* **SPY with Calm/Volatile shading** and a range slider; the fit window (1999–2006) greyed; NBER recessions optionally overlaid.
* A **current P(Volatile) gauge**.
* The **latest 2×2 transition matrix** and the **expected dwell times, 1/(1−pᵢᵢ)**, per state.
* How the **Volatile state's volatility estimate evolved across the 204+ folds** (a line with the crises marked).
* **The key overlay:** HMM Calm/Volatile against **VIX low/high on one timeline, with the agreement rate**. This is the visual "why" behind the null: the two regime signals are largely the same thing.
* A **"Why two states?"** expander: the K-selection chart (validation log-likelihood and BIC by K, with the sub-2-point margin of K = 3 shown) and the H2 note (collapsed into short "shock" states).

### 5. LSTM latent
* A **2-D latent map** (PCA of the 32-dim latent) coloured by regime, with an **animation slider** moving through time and **crises highlighted** (2008, 2020, 2022, the holdout's 2025 drop).
* A plain-language description of the denoising autoencoder (window 10 sessions → 32 numbers → reconstruct).
* **The Tier 1 note:** the trained latent was no better than a random projection of the same size (V2 vs C1: 1 of 4 targets favourable, 1 of 4 adverse). Say what that means: the *recurrent feature map* carried the information, not the training.

### 6. The agent
* **How SAC works as a 4-step explainer:** state → actor distribution → mean action → projection to the constraints (softmax of 3×action, then capped-simplex projection). An interactive: drag a few inputs, see weights.
* **What a weight means:** the target allocation for the coming week.
* **The two horizons:** a one-week decision horizon vs a planning horizon of about **1/(1−γ)** weeks, with γ read from `chosen_configs.json` per variant (0.97 → ~33 weeks; C4's 0.9 → ~10).
* **Turnover:** why it is high (the policy re-weights every week in response to noisy inputs), why it hurts after costs (cost model; the 0/5/10/20 bps sensitivity), and the 23–38% vs 1–4% comparison.
* The learning curves: train (in-sample, rising to ~+0.02/decision) vs validation (flat): the overfitting picture, and why checkpoints were chosen on validation.

### 7. Live weights (the centrepiece)
* A **sleeve donut** (Equity / Bonds / Gold / Cash) and a **14-asset bar chart** with **seed min–max whiskers** and the **policy-spread band** (how much the 10 seeds disagree — a visual of how unsettled the "decision" is).
* A **strategy toggle:** V4 ensemble / V2 / V1 / C4 / equal weight / 60/40 / risk parity, with **animated transitions** between selections. (Benchmarks use their stored definitions: equal weight 1/13 across the risky assets; 60/40 = 60% SPY / 40% IEF — note SPY is benchmark-only; risk parity = inverse-vol capped.)
* **Turnover from last week** (one-way, with the cost it implies at 5 bps).
* A **freshness badge** and a **refresh button**.
* The **permanent caption** (§0).

### 8. What-if lab
* A **P(Volatile) slider** with presets **Calm (0), Volatile (1), Today**. For **V4**: replace the two HMM posterior inputs (`state_0 = 1−p`, `state_1 = p`) in today's observation, keep *everything else fixed* (features, latent, own weights), run the 10 agents.
  For **C4**: the same slider drives the 2-column VIX-threshold one-hot (0/1 per the sleeve's threshold, with a note that this control is binary). V1 and V2 have no regime input: show them as flat reference lines
  ("by construction, the regime slider changes nothing here").
* **Live weights**, the **sleeve change from today**, and a **change-by-asset chart**.
* **The sweep chart of sleeves against P(Volatile)** (p from 0 to 1). **If it is flat or noisy, the caption says that is the finding**: it is the visible form of "regime input adds nothing the agent uses." Report the sweep's total sleeve range and compare it with the seed spread (one number: *the regime effect is X pp; seed disagreement is Y pp*).
* Caveat shown: an off-distribution input (e.g. P = 1 on a calm day) is a counterfactual the agent never trained on in combination with those features; it illustrates sensitivity, not a forecast.

### 9. Allocation through time — *descriptive*
* A **stacked area of the 4 sleeves from 2019 to the present**, with **regime shading behind it** and the **holdout boundary marked** (test vs holdout). Strategy selectable (default V4 ensemble).
* **Select a week** (slider/click) to see its 14 weights.
* A **scatter of defensive share (bonds+gold+cash) against P(Volatile)** per week, with the correlation printed.
* A **SHY/IEF/TLT duration ladder**, with **2022 highlighted** (the year the bond sleeve did not protect).
* Caption: descriptive, not evidence of skill.

### 10. Results — *pre-registered*
* **Forest plot:** Sharpe-difference CIs for the 3 comparisons, **test and holdout side by side**, so the flipping signs are visible. A metric selector (return, Sharpe, max drawdown, CVaR) and a block-length toggle (10/20/40).
* **Seed noise:** a per-seed **Sharpe strip plot per variant** against the between-variant gaps (draw the gaps as bars on the same axis).
* **Deflated Sharpe:** each seed against the **0.95 bar** (none clears it).
* **Benchmarks:** the table and **equity curves with seed bands**, with **drawdown episodes zoomable** (2020, 2022, 2025).
* **Cost slider (0–20 bps):** strategies **re-rank live** (a rank chart / sorted bar; use the stored 0/5/10/20 columns and interpolate linearly between them, saying so).
* The gate rule in one sentence, and the "claimable" column.

### 11. Verdict and what's next
* **The null replicated on unseen data.** The three-line verdict with the CIs.
* **What it does and doesn't show:** it shows no detectable benefit of HMM regimes or the LSTM latent for this agent *in this setup* and that the benchmarks are hard to beat after costs; it does *not* show regime information is worthless, nor that RL cannot work.
* **Limitations:** statistical power (CIs about ±0.35 Sharpe), a narrow universe (US sectors, Treasuries, gold), a single algorithm (SAC) and weekly decisions, ~550 training weeks and one training crisis (heavy overfitting), K = 2 on SPY returns only, one test window (exploratory) and one holdout (spent).
* **Future work:** a turnover-penalised agent, a broader universe and longer history, other algorithms/frequencies — each needing **fresh data and a fresh pre-registration**.
* Links to `reports/final_report.md`, `reports/tier1_report.md`, `reports/tier2_report.md`, the three pre-registrations and `DECISIONS.md`.

---

## 8. Build order (milestones; stop and show the user after each)

Treat each as a review gate, as in `PRISM_BUILD_SPEC.md` §16. Surface open decisions *before* coding a milestone; record choices in `DECISIONS.md`.

| # | Milestone | Acceptance |
|---|---|---|
| **D0** | Plan and decisions: read the repo (§9), list the open questions of §11 with a recommended default, confirm the stack and layout. Install `streamlit`/`plotly` into the dashboard dependency group. | decisions recorded; deps pinned |
| **D1** | **Static story pages from stored results**: pages 2, 3, 6 (static parts), 10, 11 and the shell (`app.py`, navigation, theme, captions, "How to read" helper, freshness-badge component). No network, no models. | all pages render offline in < 1 s; every number matches the stored CSVs (automated check) |
| **D2** | **Precompute**: amendment (§5.4) → user runs the replay → `scripts/10_dashboard_data.py`: fold parameters, final-fold models, weekly weights for the 40 agents, derived series. Replay validation gates the build. | replay reproduces stored daily returns to 1e-9; artifacts + checksums written; tests pass |
| **D3** | **Regime and latent pages** (4, 5) from the artifacts, including the HMM-vs-VIX overlay and agreement rate. | overlay agreement rate reproducible from C2 states; K-selection chart matches `hmm_k_selection_H1.md` |
| **D4** | **Allocation through time** (9) and **Results interactivity** (10: forest toggle, strip plot, DSR bars, equity curves, cost slider). | selected-week weights equal the stored replay; cost slider endpoints equal the stored 0/20 bps numbers |
| **D5** | **`prism.live`**: fetch, splice, features, HMM filter, encoder, state assembly, agent inference, rollout; **Live weights** (7) and **Home** (1) with the animated pipeline. | live pipeline replayed over the holdout reproduces `states_extended.parquet` and the stored weights to 1e-9; splice check passes; offline fallback works |
| **D6** | **What-if lab** (8) and the sweep chart. | at "Today" the lab equals the live weights exactly; V1/V2 flat by construction (tested) |
| **D7** | Polish and hardening: animations, light/dark, mobile width, copy review against §0/§6, `AppTest` smoke tests for every page, README section, `make dashboard`. | all tests pass; a clean-machine run follows the README; reviewer checklist (§10) ticked |

---

## 9. What to read first (the new session's reading list)

1. `README.md` — current status and commands. 2. `reports/final_report.md` — the verdict. 3. `DECISIONS.md` — especially D-001 (open decisions), D-024/D-025 (HMM spec, K), D-032 (Phase B variants), D-034..D-036 (costs, env, reward),
D-038..D-042 (Tier 2 and holdout design), "Gate decisions (Tier 1/Tier 2/holdout)". 4. `PRISM_BUILD_SPEC.md` §3 (universes), §8 (HMM), §9 (encoder), §10 (variants), §11–§12 (env, agents), §14 (statistics). 5. The three pre-registrations
(`reports/tables/preregistration*.md`). 6. `reports/tier1_report.md`, `reports/tier2_report.md`. 7. Code to reuse, in this order: `src/prism/holdout.py`, `src/prism/env/portfolio_env.py`, `src/prism/env/actions.py`, `src/prism/agents/{sac,data,evaluate,tier2}.py`,
`src/prism/state.py`, `src/prism/models/hmm/{walkforward,filtered}.py`, `src/prism/models/encoder/walkforward.py`, `src/prism/backtest/{benchmarks,metrics}.py`, `src/prism/analysis/{tier2,significance,episodes}.py`.
8. `configs/*.yaml` and `data/processed/tier2/chosen_configs.json`.

---

## 10. Reviewer checklist (the definition of done)

* [ ] Every page has a one-line takeaway and a "How to read this" expander.
* [ ] The permanent caption is on every page that shows weights; no page says or implies the agents outperform, predict or should be followed.
* [ ] Every headline number matches `reports/final_report.md` (automated).
* [ ] The null is the headline of Home, Results and Verdict, framed as rigour; the sign flip is visible; the benchmark comparison and the turnover explanation are on the Results page.
* [ ] The holdout is labelled spent; nothing selects or tunes on it; the §5.4 amendment exists and is committed.
* [ ] No frozen artifact was modified (`git status` clean on pinned paths; SHA-256 re-verified).
* [ ] Live weights: freshness badge, splice check, offline fallback, "models last refit" shown; rollout definition stated.
* [ ] What-if at "Today" equals live weights; sweep and seed-spread numbers shown together.
* [ ] Cold start < 5 s, page switch < 1 s, refresh < 20 s; works offline from cache.
* [ ] Tests for `prism.live` and `prism.dashboard_data`; `AppTest` smoke test for each page; the existing 385 tests still pass.
* [ ] Colour-blind-safe palette; works in light and dark and at phone width.
* [ ] `DECISIONS.md` entries for every non-obvious choice; README section; `make dashboard-install`, `make dashboard-data`, `make dashboard`.

---

## 11. Open decisions (recommend a default, surface before coding the milestone)

1. **Live "as-of" semantics:** weekly decision only (default) vs also a mid-week preview (default: both, preview clearly labelled).
2. **Hosting:** local Streamlit only (default) vs Streamlit Community Cloud. If hosted, the app must run entirely from committed artifacts and `data/live/models/` (small), with the refresh button using the public yfinance path and a rate limit; do **not** commit the raw snapshot or the 148 MB state files.
3. **Where the frozen live models live and how they are versioned:** `data/live/models/` with a manifest (fit end dates, SHA-256) committed; large binaries only if small (the MLPs are ~100 KB each).
4. **Animation technology** for the Home pipeline: Plotly frames (default) vs a small custom component.
5. **The what-if on C4:** binary threshold slider (default) vs not offered.
6. **Interpolation between the four stored cost levels** on the cost slider (default: linear, labelled) vs recomputing from daily returns (not possible: the cost enters the path).
7. **Scope of "fresh":** how many weeks of history the live panel needs beyond the snapshot (default: splice with a 60-session overlap; features need ≥ 252 sessions of warm-up, already in the snapshot).

---

## 12. Out of scope

Trading, brokerage links, alerts, any "signal" subscription, new model training, re-tuning, any new pre-registered statistic, evaluating anything new on the holdout, and any wording that turns the demonstration into advice.
