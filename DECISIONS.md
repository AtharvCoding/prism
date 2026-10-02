# DECISIONS

Every non-obvious choice, the alternatives, and why. Spec §17.8.

Entries are append-only. A decision that turns out wrong gets a new entry
superseding the old one, not an edit.

---

## Step 0 — repository, config, splits, tests

### D-001 · The three open decisions in §3.3
**Date:** 2026-10-01 · **Decided by:** principal investigator

All three resolved to the spec's stated defaults:

1. **Weekly rebalancing, Friday close.** Daily becomes a robustness check.
   ~570 decisions over Universe B's training window. Recorded in
   `configs/data.yaml` under `decision.frequency`.
2. **Long-only with cash, weights summing to 1, each in [0, 0.35].** The cash
   line is what lets a regime signal express "de-risk" without shorting; the
   0.35 cap prevents single-sector concentration. Enforced by simplex
   projection, not by penalty. Recorded under `allocation`.
3. **Encoder fitted on Universe A**, with the §3.2 robustness check (refit on
   B's 2007–2017 training window) reported alongside. Recorded under
   `encoder.fit`.

**Why not the alternatives.** Daily rebalancing makes cost assumptions
dominate the result; limited shorting adds borrow costs and a harder
projection for an unclear gain; fitting the encoder on B alone reintroduces
the train/test distribution mismatch (design gap D4) that the two-universe
design exists to remove.

---

### D-002 · Embargo is applied by purging the END of the earlier split
**Date:** 2026-10-01 · **Status:** implemented in `src/prism/splits.py`

**The problem.** The spec declares splits that are calendar-**contiguous**
(train ends 2017-12-31, val starts 2018-01-01) while also declaring
`embargo_days: 25`. Taken literally these contradict each other: zero trading
sessions separate the two boundaries. `n_sessions_between('2017-12-31',
'2018-01-01')` returns 0.

**Alternatives.**
- **(a) Purge the end of the earlier split.** Train's usable range ends 25
  sessions before val begins.
- **(b) Shift the later split's start forward by 25 sessions.** Val would
  become 2018-02-07 .. 2018-12-31.

**Decision: (a).** §6.1 says the embargo exists because "forward-looking
targets and multi-day rewards near a boundary overlap the next split" — it is
a property of the *earlier* split's tail, so that is where the cost belongs.
(a) also preserves the reported evaluation windows exactly as specified: val
is the 2018 calendar year, test is 2019–2023, holdout is 2024 onward. Under
(b) every table in the report would describe a different period than it
claims to. This is standard purging (López de Prado).

**Consequence.** Each of train, val and test loses exactly 25 sessions from
its end:

| Split | Declared | Effective | Sessions | Purged |
|---|---|---|---|---|
| train | 2007-04-04 .. 2017-12-31 | 2007-04-04 .. 2017-11-22 | 2681 | 25 |
| val | 2018-01-01 .. 2018-12-31 | 2018-01-01 .. 2018-11-21 | 226 | 25 |
| test | 2019-01-01 .. 2023-12-31 | 2019-01-01 .. 2023-11-22 | 1233 | 25 |
| holdout | 2024-01-01 .. 2026-09-30 | unchanged | 689 | 0 |

Validation loses 10% of its length, which is the largest relative cost. It is
accepted: a validation set whose last 25 days' targets reach into the test
period cannot be used for honest model selection, which is the only thing it
is for. `Split.effective_*` is what every consumer must read;
`Split.declared_*` exists only for reporting.

**Pinned by** `tests/test_splits.py::test_embargo_is_applied_by_purging_the_earlier_split`.

---

### D-003 · `warmup_days` is 252, not 60
**Date:** 2026-10-01 · **Status:** implemented in `configs/data.yaml`

The §3.3 config sketch says `warmup_days: 60  # longest lookback`. That
predates the §5.1 feature list, which includes `dist_from_52w_high` — a
252-session lookback. 60 would leave that feature NaN for the first year and
the frame would fail the QA `feature_nan_or_inf_after_warmup` hard check.

**Two consequences, both properties of the data rather than choices:**

- **Universe A features begin 1999-12-15, not 1999-01-04.** The nine sector
  ETFs launched 1998-12-16, so a trailing 52-week high does not exist before
  late 1999. The effective `fit_early` window is therefore ~7.0 years
  (1999-12-15 .. 2006-12-31, ~1770 sessions) and still contains the entire
  dot-com episode, which is what §3.1 wanted it for.
- **Universe B features begin 2007-05-03, not 2007-04-04.** HYG's first trade
  *is* 2007-04-04, so `credit_proxy_change_20` cannot exist until 20 sessions
  later. B's equity and macro features are warm long before this; the credit
  proxy is the binding constraint. Cost: 20 sessions out of 2681.

**Alternative considered and rejected:** drop `dist_from_52w_high` to keep the
60-day warm-up. Distance from the 52-week high is one of the few genuinely
informative non-volatility features in the set, and 20–250 sessions at the
start of a 25-year panel is a much smaller loss than the feature.

Both dates are reported by `FeatureSet.warm_start` and written into the run
manifest rather than absorbed by a `dropna()`.

---

### D-004 · The 10y–2y curve slope is not obtainable; 10y–3m is used instead
**Date:** 2026-10-01 · **Status:** implemented in `src/prism/features/macro.py`

§5.3 asks for `curve_slope_10y_2y`. Yahoo's Treasury yield indices are `^IRX`
(13-week), `^FVX` (5-year), `^TNX` (10-year) and `^TYX` (30-year). **There is
no 2-year series.**

**Decision.** The primary slope is `curve_slope_10y_3m` (`^TNX − ^IRX`), with
`curve_slope_10y_5y` reported alongside. 10y–3m is the other classical
recession signal and is in fact the better-documented one for that purpose.
`curve_change` is the daily change in the primary slope, and is the HMM's
`curve_change` observation.

**Alternative:** source a 2-year yield from FRED (`DGS2`). Rejected for now
because it introduces a second data vendor with its own calendar, revision
policy and hashing story, for a feature whose substitute is at least as good.
Worth revisiting if the curve turns out to carry the HMM.

**To verify at snapshot time:** that `^TNX`/`^IRX` are quoted in percentage
points (e.g. 4.25 = 4.25%) and not as 10× the yield, which older Yahoo
history sometimes was. `macro.py` uses arithmetic differences throughout, so a
constant scale factor would not break the *change* features, but it would
misstate the slope's units.

---

### D-005 · `^VIX3M` added to Universe B only
**Date:** 2026-10-01 · **Status:** implemented in `configs/data.yaml`

§5.3 asks for `vix_term_structure (VIX/VIX3M if available)`. `^VIX3M`'s
history begins 2002-12-04, which postdates Universe A's 1999-01-04 start.

**Decision.** `^VIX3M` goes in Universe B's `macro_extra`, making
`vix_term_structure` a B-only feature. This is consistent with §3.1's design
rather than an exception to it: B is precisely the universe defined by
"series with shorter history that the allocator needs". `build_macro_features`
degrades gracefully with a logged message if the series is absent, so a vendor
outage at Step 1 does not abort the run.

**To verify at snapshot time:** that `^VIX3M` is available from Yahoo for the
full 2007-04-04 onward range. If it is not, the feature drops out
automatically and B loses one of 185 columns.

---

### D-006 · Yields are never log-transformed
**Date:** 2026-10-01 · **Status:** implemented in `src/prism/features/macro.py`

`^IRX` was effectively at zero from 2009–2015 and again in 2020–2021, and
short-rate series can print negative. Every yield-derived feature therefore
uses **arithmetic** differences. `log(yield)` is both dimensionally wrong and
undefined at the zero bound.

The synthetic test panel generates `^IRX` with a floor of 0.01 percentage
points specifically so that any code taking a log of a yield fails in the test
suite rather than in production.

---

### D-007 · Fitted operations are separated from feature construction
**Date:** 2026-10-01 · **Status:** implemented

`build_features` is a **pure causal transform**. Winsorising, scaling and
correlation pruning — the only parts of feature construction that *learn*
anything — live in `prism.features.scaling` behind `FittedArtifact`.

This is a design consequence, not a style preference. If winsorisation
quantiles were computed inside the feature builder, the builder would no
longer be truncation-invariant: rebuilding on data up to `t` would change
values for rows before `t`, because the quantiles would differ. The causality
test would catch it and the fix would be exactly this separation.

`FittedArtifact` gives every fitted object a recorded fit range, a parameter
hash, no `fit_transform`, and a refusal to re-fit implicitly. The last of
those is what makes defect B5 — a shared scaler mutated in place inside a
rolling refit loop — inexpressible by accident.

---

### D-008 · Step 0 tests run on synthetic data; no network call yet
**Date:** 2026-10-01 · **Decided by:** principal investigator

§16 lists "snapshot" under Step 0, but Step 0's acceptance criteria are all
about the test suite. `scripts/00_snapshot.py` is written and dry-run; the real
download happens at the top of Step 1, where the two-universe feature build
needs it.

The whole suite runs against `prism.data.synthetic`, a deterministic
regime-switching generator with the config's real inception dates, the NYSE
calendar, and yields that approach zero. Rationale: the suite asserts
properties of the *code*, and those must not depend on what the vendor served
today. `create_snapshot` accepts pre-fetched panels precisely so the
write-once logic can be tested offline.

---

### D-009 · `pandas` 3.0.6 and the `SettingWithCopyWarning` filter
**Date:** 2026-10-01 · **Status:** implemented in `pyproject.toml`

The resolver selected pandas 3.0.6. All dependencies import and the NYSE
calendar, parquet round-trip, rolling correlation and `assert_frame_equal`
paths were smoke-tested before committing to it. Copy-on-write is now the
default, which removes an entire class of silent-mutation bug — a net gain for
this project.

One consequence: `pandas.errors.SettingWithCopyWarning` no longer exists, so
it was removed from the pytest `filterwarnings` list. `error::FutureWarning`
is retained, so a deprecation cannot ride along unnoticed into a result. There
is **no** blanket `ignore` anywhere in the project (defect A9).

A related trap found during the build: storing a pandas `Series` in
`DataFrame.attrs` breaks pandas 3, which compares `attrs` with `==` inside
`__finalize__`, making the comparison elementwise. The synthetic generator now
exposes `synthetic_true_states()` instead of attaching the latent state path
to the frame.

---

### D-010 · `turnover` accepts `initial_weights`
**Date:** 2026-10-01 · **Status:** implemented in `src/prism/backtest/metrics.py`

Turnover is one-way (`0.5 · Σ|Δw|`), so a complete switch between two assets
is 100%, not 200%, and "5 bps per side" means what a practitioner expects.
The first row is charged as a move from all-cash, so initiating a portfolio is
not free.

That default is wrong at a walk-forward fold boundary: evaluating fold 2
onward would bill a fresh initiation for a portfolio already in place,
inflating the reported cost of every fold after the first. `initial_weights`
lets the caller state what is already held. Pinned by
`test_initial_weights_prevent_a_spurious_reinitiation_at_a_fold_boundary`.

---

### D-011 · Deferred components get `xfail(strict=True)` tests now
**Date:** 2026-10-01 · **Status:** implemented

`tests/test_hmm.py`, `test_encoder.py` and `test_state.py` are written against
the interfaces steps 2, 3 and 3b must provide, and marked
`xfail(strict=True)`. The suite runs green today and documents exactly what
each step owes.

`strict=True` is the point: when a step lands, its tests turn XPASS, which
pytest reports as a **failure**, so the marker must be removed and the test
becomes a real guard. The suite tells you the stub is gone rather than relying
on anyone remembering. Phase B files (`test_costs.py`, `test_env.py`) are
module-level `skip` with the §0.3 reason, not `xfail` — they are not merely
unimplemented, they are out of scope.

---

## Step 1 — dataset expansion, universes, QA

### D-012 · `^VIX3M` inception corrected from 2002-12-04 to 2006-07-17
**Date:** 2026-10-01 · **Status:** implemented in `configs/data.yaml`

The real snapshot (2026-10-01, hash `9de525958b07...`) showed Yahoo's
`^VIX3M` series has no data before 2006-07-17 — four years later than the
2002-12-04 placeholder configured in Step 0 from CBOE's own publication date.
This is a vendor-coverage gap wider than the "a fund's first few trading
days are sometimes missing" case `clean.py`'s `max(declared, observed)`
availability logic exists for (see the docstring on
`prism.data.clean.availability_mask`); 2002-12-04 was simply wrong for this
vendor, so the config is corrected to match reality rather than left to be
silently overridden every time.

No feature coverage is affected: Universe B does not start until
2007-04-04, which both dates precede.

### D-013 · The snapshot's `MANIFEST.json` is committed; the data is not
**Date:** 2026-10-01 · **Status:** implemented in `.gitignore`

`data/raw/` holds the actual parquet files (prices, volume, macro levels).
These are never committed — not because of size, but because of what §4.1
says about them: adjusted prices are *restated* whenever a new dividend is
paid, so **re-running the download script does not reproduce this snapshot**.
A later run is a new, differently-hashed snapshot, not a reproduction of this
one; committing a "reproduce by re-running" instruction here would be a
promise the data layer cannot keep.

What *is* committed is `data/raw/snapshot_20261001/MANIFEST.json` — the
per-file SHA-256 hashes, ticker list, row counts, and the full `pip freeze`.
This is the audit trail: anyone who obtains a copy of the actual parquet
files (out-of-band — this machine, a shared drive, cloud storage, whatever
the team sets up) can verify byte-for-byte that it is the same snapshot every
result in this repository traces to, by matching those hashes. Without the
real files, `make install && make test` still runs the full causality and
fit-scope suite against the synthetic generator (spec intent: tests assert
properties of the code, not of one researcher's local disk) and
`scripts/01_build_features.py` simply cannot run until a snapshot — this one
or a newly-taken one — exists locally.

**Bug found while implementing this:** the original Step 0 `.gitignore` had
`data/raw/` (the whole directory) ignored, with a nested `!data/**/MANIFEST.json`
intended to re-admit the manifest. That negation silently did nothing — git
does not descend into a directory it has already decided to ignore, so a
nested un-ignore pattern inside one is dead code. Fixed with the standard
three-step idiom: ignore the directory's *children* (`data/raw/*`), re-admit
directories so git will look inside them (`!data/raw/*/`), ignore everything
inside those directories (`data/raw/*/*`), then re-admit the one file that
matters (`!data/raw/*/MANIFEST.json`). Verified with `git add -n data/`
before trusting it.

### D-014 · `oil_vol_20` uses a simple return, not a log return
**Date:** 2026-10-01 · **Status:** implemented in `src/prism/features/macro.py`

Running the real pipeline against the real 2026-10-01 snapshot produced a
**hard** QA failure: `feature_nan_or_inf_after_warmup` on `oil_vol_20`, 21
non-finite values. WTI front-month futures (`CL=F`) printed **-$37.63 on
2020-04-20** — a real, storage-constraint-driven event during the COVID
demand collapse, not a data error. `oil_vol_20` was computed as the rolling
std of the oil log return, masked to `price > 0` the same way the yield
features are (D-006). That mask turns the one negative day into NaN, and a
20-day rolling std needs 20 consecutive finite inputs, so the NaN propagates
into a ~20-session hole spanning the entire following month — discarding
exactly the day, and the month after it, that a crisis-sensitive volatility
feature exists to capture.

**Decision.** `oil_vol_20` uses a **simple** (arithmetic, `pct_change()`)
return for oil instead. Unlike a log return, a simple return is well-defined
across a sign change — the -$37.63 → recovery transition — so the feature
correctly reports an extreme realised volatility for the following month
rather than a gap. No other series in the project needs this: equities never
go negative (log returns stay), and yields use arithmetic differences already
(D-006) for the same underlying reason this decision generalises from.

**Verification:** `test_oil_vol_tolerates_a_negative_print` in
`tests/test_features.py` constructs a synthetic path through zero and asserts
`oil_vol_20` stays finite across it; `test_data.py`'s
`test_each_hard_check_can_actually_fire` still provokes
`feature_nan_or_inf_after_warmup` independently via an injected `np.inf`, so
the hard check itself is still proven capable of firing.

### D-015 · `DX-Y.NYB`'s off-NYSE-calendar sessions are dropped, not a bug
**Date:** 2026-10-01 · **Status:** no-op, documented

`clean.py` logged "115 dated rows fall outside the NYSE calendar and are
dropped" while building the real pipeline. Investigated rather than ignored:
110 of the 115 are `DX-Y.NYB` (the US Dollar Index), on dates that are NYSE
holidays — Presidents' Day, July 4th, Thanksgiving, Christmas, New Year's.
The dollar index trades on an FX-market calendar, which does not observe
NYSE equity holidays, so Yahoo legitimately has a value for it on those days.
The remaining 5 are `CL=F` (also on a different futures-market calendar) and
2 are `^VIX`.

This is `align_to_calendar`'s designed behaviour (spec §4.2: the project's one
calendar is NYSE, full stop) working as intended, not a defect: admitting
these rows would introduce dates into the panel that no equity in the
universe ever traded on, which every downstream rolling-window and
rebalance-schedule computation assumes cannot happen. No code change; logged
here so the warning is not mistaken for noise on a future read of the logs.

---

## Step 2 — HMM rebuild

### D-016 · `credit_proxy_change` dropped from H2 (not substituted)
**Date:** 2026-10-01 · **Status:** implemented in `configs/hmm.yaml`

Spec §8.2 lists `credit_proxy_change` among H2's five observations. Building
the real observation frame surfaced a conflict Step 0's config didn't catch:
`credit_proxy` derives from HYG/LQD (`log(HYG/LQD)`), and both are
Universe-B-only tickers (HYG's inception, 2007-04-04, is Universe B's start
date). The HMM fits on Universe A throughout — `fit_early` (§3.2) and every
walk-forward refit after it, since a single `GaussianHMM`'s observation
dimensionality is fixed for its whole life and cannot change mid-series when
the chain crosses into 2007.

**Alternatives considered.**
- **Restrict the whole HMM to post-2007 dates**, where credit data exists.
  Rejected: defeats §3.2's entire reason for fitting on Universe A — the
  dot-com crisis (2000-02) is unrepresented in B, and losing it from the
  HMM's training history is exactly the "one crisis in training" problem
  (design gap D5) the two-universe design exists to fix.
- **Violate universe isolation** and let the A-side HMM read HYG/LQD anyway.
  Rejected outright: isolation is structural and tested (`build_features`
  filters the raw panel to a universe's own tickers before any feature code
  runs; `test_fit_scope.py` corrupts every B-only series and asserts
  Universe A's output is unaffected). Making the HMM an exception would mean
  "never touches a B-only ticker" is no longer actually true of Phase A.
- **Substitute a Universe-A-available proxy for credit spread.** Unlike the
  curve slope (D-004, 10y-3m for 10y-2y) or the VIX term structure (D-005,
  made B-only), there is no equity-universe proxy for credit spread — it is
  intrinsically a bond-market quantity.

**Decision: drop it.** H2 is `[SPY_return_1d, vix_change, curve_change,
avg_pairwise_corr_change]` — four observations, not five. All four are
genuinely available across Universe A's full 1999-2006 `fit_early` window and
every date thereafter, so the walk-forward chain never has to change
dimensionality. `credit_proxy_change_20` remains a Universe-B feature and is
available to the Tier 1 probes and the allocator from step 4a onward, where
Universe B is the right universe to be in anyway.

### D-017 · Detection search margin corrected from 20 to 60 sessions
**Date:** 2026-10-01 · **Status:** implemented in `configs/hmm.yaml`, `evaluate.py`

`detect_episodes`'s original default searched up to 20 sessions (~1 month)
before an episode's dated start for the crisis signal crossing 0.5. Running
the real evaluation showed this was a **ceiling, not a design choice**: both
the HMM and the VIX-threshold baseline crossed 0.5 well before 20 sessions
ahead of the 2007-12 GFC NBER date (re-checked at margins up to 250 sessions,
where both detectors were still showing negative lag). A 20-session margin
was silently reporting the search window's edge, not the true first
crossing.

**Decision:** raised to 60 sessions (~1 quarter) — long enough to capture
genuine early signal without reaching so far back that "detection" becomes
indistinguishable from ambient volatility. Even at 60, the baseline still
hits the margin ceiling for the GFC and 2022 rate-shock episodes specifically
(both preceded by months of gradually elevated VIX), which is itself worth
reporting rather than hiding: for a slow build-up, "detection lag" measured
this way partly reflects how long volatility was already elevated, not a
crisp early-warning event. The evaluation report states the margin used.

### Finding · H2's richer observation set finds short-duration "shock" states at every K
**Date:** 2026-10-01 · **Phase A finding, not a defect**

The real K-sweep for H2 (`SPY_return_1d, vix_change, curve_change,
avg_pairwise_corr_change`) flagged **every** restart at **every** K in
`2..8` as degenerate by the §8.4 duration criterion — including K=2, the
simplest possible case. Investigated rather than dismissed: at K=2, EM
consistently finds one persistent state (~9-day duration, ~85% of mass) and
one short-lived state (~1.5-day duration, ~15% of mass); at the selected
K=4, two persistent states (~10.5 and ~13-day durations, 91% combined) and
two short-lived ones (~1.1 and ~1.6-day durations, 9% combined).

This is a real, interpretable property of the data and the spec's own
observation design, not a bug: §8.1/§8.2 deliberately require low-
autocorrelation, conditionally-independent *change* observations (to avoid
defect B3's volatility-ladder collapse). Changes are inherently closer to
white noise than the levels they're computed from, so a single-day outlier
— one sharp VIX jump, one wide correlation swing — genuinely looks like its
own brief "shock" state to EM rather than blending into a persistent
regime's tail. H1 (univariate, raw `SPY_return_1d`) shows far fewer
degenerate restarts (1/20 at K=2, climbing with K) by contrast, which is
itself informative: a single volatility-clustered return series retains more
autocorrelation than a hand-picked basket of independent changes does.

**Not resolved by relaxing the threshold.** Spec §8.4's criterion is stated
as a firm rule, and loosening it until a result looks clean is exactly the
"tune until the desired answer appears" failure mode §17.6 exists to
prevent. The fallback behaviour already in `select_k` (use the overall-best
restart, flagged `degenerate=True`) is what ran, and every K-sweep report
states the fraction of degenerate restarts per K plainly rather than hiding
it behind a single pass/fail bit. Worth revisiting at step 4a: does
state 3 behave like a genuine "shock" the probes benefit from distinguishing
from state 1's more persistent elevated-vol regime, or is it noise the
model should have fewer states to avoid? The Tier 1 probe comparison is the
honest way to answer that, not another look at the duration statistic alone.

### Finding · The HMM does not beat the two-state threshold baseline on detection lag
**Date:** 2026-10-01 · **Phase A finding — the §8.7 beats-baseline test, detection half**

On the real 2026-10-01 snapshot, walk-forward H2 (K=4) does **not** win spec
§8.7's beats-baseline comparison against a two-state VIX-quantile threshold,
on either reference:

- **NBER recessions** (2 in the walk-forward window — dot-com predates it,
  the walk-forward only applies from 2007 onward): GFC lag HMM -51 vs
  baseline -60 (baseline faster, both near the 60-session search ceiling);
  COVID lag HMM +20 vs baseline -5 (baseline notably faster — the COVID
  crash was violent and fast, and VIX itself reacted immediately, while the
  HMM's filter needed three weeks of observations to become confident).
- **Drawdown episodes** (≥20% off high; 19 found, though several are
  overlapping sub-episodes of the same 2007-2011 structural downturn, not
  19 independent crises — see the evaluation report): the baseline is
  faster or tied on every episode.
- False-alarm rate clearly favours the HMM (3.3% vs 38.6% on NBER windows;
  2.6% vs 29.4% on drawdown windows) — the HMM is far more **precise** about
  when it calls crisis, even though it is not **faster**.

**This is the honest result, stated plainly per spec §0.4.4 and §15.1: a
negative result is a publishable result.** It does not by itself mean the
HMM adds nothing — §8.7 explicitly asks for detection lag **and** downstream
probe performance, and the second half of that test is step 4a's Tier 1
ablation (`configs/experiments/tier1_probes.yaml`'s `gates.hmm_adds_value`:
`V3 > C2` and `V3 > C3`). A model that is slower to raise the alarm but far
less prone to false alarms could still carry information a regression probe
exploits better than a hard threshold does — or it could not. That is
exactly what Tier 1 is for, and this finding is the reason to take its
result seriously rather than assume the HMM wins by construction.

### Finding · Walk-forward robustness refit on B-train-only selects a different K
**Date:** 2026-10-01 · **§3.2 robustness check, §16 step 2 acceptance**

Refitting H1 and H2 on Universe B's training window alone (2007-04 to
2017-11, no dot-com crisis) and selecting K the same way:

| Spec | A-fitted (`fit_early`, 1999-2006) | B-train-only (2007-2017) |
|---|---|---|
| H1 | K=3, val_ll=816.59 | K=3, val_ll=828.36 |
| H2 | K=4, val_ll=2350.09 | K=8, val_ll=2498.54 |

H1's selection is **stable** across the two fit windows (same K, comparable
validation likelihood) — encouraging for the univariate specification.
H2's is **not**: fit on 2007-2017 alone, the validation criterion keeps
climbing all the way to the edge of the K-range searched (K=8), the exact
"monotone to the edge of the sweep" symptom that made the reference
project's K=5 unjustified (defect B2). Fit on 1999-2006 instead (which
includes dot-com), H2 settles cleanly at K=4 with K=5-8 all degenerate.

Plausible reading: the dot-com crisis in the A fit window gives H2 a second,
structurally different crisis to generalise across, which regularises the
state count; without it, EM has more freedom to carve the single GFC-plus-
rate-shock history into additional states that don't generalise. This is
exactly the crisis-diversity argument spec §3.2 makes for fitting the HMM on
Universe A in the first place, now visible in a concrete number rather than
asserted. The A-fitted H2 (K=4) is what the walk-forward and downstream
steps use; the B-train-only sweep is reported here as the required
robustness check, not adopted.

---

## Step 3 — encoder rebuild

### D-018 · Staged, reduced hyperparameter sweep (not the full 480-run grid)
**Date:** 2026-10-01 · **Status:** implemented in `scripts/03_train_encoder.py`

Spec §9.3's grid — window ∈ {10,20,30,60} × latent ∈ {8,16,32} × hidden ∈
{32,64} × variant ∈ {AE,DAE,VAE,PRED}, ≥5 seeds each — is 480 individual
training runs. Benchmarked at ~13s per run on the real Universe A data
(window=30, hidden=64, latent=16, 1767-row fit window), the full grid would
take roughly two hours of serial training.

**Decision**, the same shape as D-002's HMM walk-forward restart budget:
three sequential stages, each fixing what the previous stage selected rather
than re-sweeping it — window (4 candidates × 2 seeds), then variant at the
selected window (4 × 5 seeds, meeting spec's seed floor exactly for the
comparison that matters most), then latent_dim × hidden_dim at the selected
window and variant (6 × 2 seeds). 40 runs total, ~6 minutes. The reduction's
assumption — that window, variant and architecture size do not interact
sharply enough to require a joint search — is checked in the order most
likely to matter (window first: it changes what information later stages
even have access to), not assumed silently.

### Finding · Hyperparameter selection by validation RECONSTRUCTION loss chose window=10, DAE — and that choice does not transfer to downstream usefulness
**Date:** 2026-10-01 · **Phase A finding**

The three-stage sweep selected window=10, variant=DAE, latent_dim=32,
hidden_dim=64 — by validation reconstruction loss, the only criterion
available before a Tier 1 probe harness exists (this step built a lean one;
see D-019). This is a real limitation worth stating plainly: reconstruction
quality and downstream predictive usefulness are not the same objective, and
spec §15.1 H3 names exactly this risk ("reconstruction-based encoding is the
wrong objective"). The finding below is evidence the risk materialised here.

### Finding · The encoder does not beat PCA or the random encoder on the probe — confirmed across two training objectives, after fixing a real bug in the comparison
**Date:** 2026-10-01 · **Phase A finding — the §9.4 beats-baseline test**

Probing each representation (latent dimensions alone — see the scope note
below) against `fwd_vol_20` with the walk-forward ridge probe built this
step (D-019):

| Representation | R² (test) | Beats PCA? |
|---|---|---|
| DAE (window=10, selected by reconstruction loss) | -0.049 | No |
| PRED, purely predictive (`pred_reconstruction_weight=0`, directly trained toward `fwd_vol_20` among its targets) | -0.203 | No |
| PCA (walk-forward refit, same dimensionality) | 0.179 | — |
| Random encoder (frozen, untrained, same dimensionality) | 0.214 | — |

Neither a reconstruction-trained encoder nor one trained **directly** on the
evaluation target beats an untrained random linear projection. This is spec
§9.4's exact "if it does not, simplify or drop it and report that as a
finding" scenario, and it is reported as such — not tuned away.

**A real bug was caught and fixed before trusting this.** The first version
of this comparison fit PCA **once**, on the 1999-2006 `fit_early` window, and
applied it unrefit through 2023 — while the LSTM walk-forward refits
annually. That is not a fair comparison in either direction. Fixed with
`pca_encoder_walkforward` (refits PCA on the identical expanding, embargoed
fold schedule the LSTM uses). The correction mattered: PCA's R² dropped from
0.296 (static) to 0.179 (walk-forward-refit) — part of PCA's apparent edge
really was the asymmetry. The corrected, fairer comparison still shows the
LSTM behind both baselines, which is why the finding above is reported with
confidence rather than as a methodology artifact.

**Scope, precisely.** This probes the **latent alone** against PCA-alone and
random-alone at matching dimensionality — not spec §10's actual gate
(`V2 = V1 + latent` vs `C1 = V1 + random-encoder latent`, both carrying the
full raw feature set too). That comparison needs state assembly (step 3b)
and is step 4a's `gates.lstm_adds_value` (`V2 > V1'` and `V2 > C1`) in
`configs/experiments/tier1_probes.yaml`. This step's finding is a strong
prior for how that gate will likely resolve, not a substitute for running it
— the raw V1 features carried alongside the latent in V2 and C1 could change
the outcome, since V1 and C1 together might already contain what the probe
needs regardless of what the latent itself adds.

**Not resolved by further tuning here.** A longer window might let the
LSTM's recurrent structure earn more of its complexity (window=10 was
selected by reconstruction loss, which may itself be the wrong criterion —
see the finding above), and this is worth revisiting if step 4a's formal
gate also fails. Re-running the hyperparameter sweep with probe performance
as the selection criterion, rather than reconstruction loss, is the natural
next step if so — but that is a materially larger undertaking (an
inner-loop probe evaluation inside every candidate's scoring) than this
step's budget covers, and doing it now, before knowing whether step 4a's
full-feature-set gate even needs it, would be exactly the kind of
unprincipled extra tuning spec §17.6 warns against.

### D-019 · `prism.probes.probe` and `prism.analysis.bootstrap` built now, not deferred to step 4a
**Date:** 2026-10-01 · **Status:** implemented

Spec §9.4's own acceptance text ("beats PCA and random encoder on the
Tier-1 probe with non-overlapping confidence intervals") and §16's step 3
acceptance row both name the probe comparison as THIS step's bar, not step
4a's — unlike the HMM's "downstream probe performance," which §8.7 and
§13.1 together make unambiguously a step 4a concern. Built a lean, reusable
core now: `ridge_probe` (train/val/test split, train-only standardisation,
alpha selected on validation from one grid) and
`stationary_block_bootstrap_ci` (Politis-Romano stationary bootstrap, the
exact method `configs/experiments/tier1_probes.yaml` names). Step 4a extends
this to every variant and target with a walk-forward refit schedule; it does
not replace or re-architect it.

### D-020 · V1' carries the full flattened window, not a PCA compression
**Date:** 2026-10-01 · **Status:** implemented — surfaced to the user before coding, per spec §17.5

Spec §10 describes V1' as "the same features over the window, flattened
**or** PCA-compressed." Compressing to `latent_dim` would make V1' dimension-
matched to V2 — but C1 (the random encoder) already isolates exactly that
case ("more dimensions" vs "a learned representation") at matched
`latent_dim`. If V1' were also compressed, the two mandatory gates
(`V2 > V1'` and `V2 > C1`) would stop being independent falsifications: a
failure of either would partly retest the other. V1' is therefore the full
window flattened (`window * n_features` columns, via
`prism.models.baselines.pca_encoder.flatten_windows`) — a pure "more history,
unlearned" control, confirmed with the user as the first open decision of
this step.

### D-021 · `state.py` assembles from precomputed artifacts; it does not fit anything
**Date:** 2026-10-01 · **Status:** implemented

`build_state` takes a `StateArtifacts` bundle (encoder latents, HMM
posteriors, random-encoder latents, threshold-regime one-hot, O1's smoothed
posteriors) as already-computed, already-causal, date-indexed frames, and
raises `MissingArtifactError` naming exactly what is absent for a given
variant rather than refitting anything itself. This matches the module
boundary everywhere else in the project (walk-forward modules fit; `state.py`
assembles) and is what keeps `tests/test_state.py` fast — hand-built
synthetic artifacts stand in for a real HMM/encoder fit, the same style
`test_threshold_regime.py` and `test_encoder_baselines.py` already use for
the modules that produce those artifacts in production.
`scripts/03b_build_states.py` is the one place that wires real,
step-2/step-3-persisted artifacts into it; C1 and C2 are cheap enough
(no backprop, a handful of walk-forward folds) that the script recomputes
them directly rather than requiring a fourth/fifth persisted parquet.

### D-022 · O1 (the leaky oracle) is per-fold smoothed, computed by extending `hmm_walkforward`
**Date:** 2026-10-01 · **Status:** implemented — surfaced to the user before coding, per spec §17.5

No smoothed-posteriors function existed; only the causal `filtered_posteriors`
(spec §8.3's fix for defect B1). O1 needs `hmmlearn`'s own forward-backward
(`predict_proba`) run over each walk-forward fold's fit+apply window, apply
rows kept — leaky only *within* a fold, not across the whole sample. Rather
than a second, parallel fitting path, `hmm_walkforward` grew an
`also_smoothed: bool = False` flag that reuses the SAME already-fitted,
canonically-relabelled fold model to also emit `smoothed_posteriors` — one
extra `predict_proba` call per fold, no duplicate EM fit, and the filtered
output is unperturbed (asserted in
`tests/test_hmm.py::test_also_smoothed_reuses_the_fold_model_and_differs_from_filtered`).
`scripts/03b_build_states.py --with-oracle` re-runs the walk-forward for the
winning spec/K (read back from `hmm_summary.json`) with this flag set, since
step 2 does not persist per-fold model objects; it asserts the re-derived
filtered posteriors match the persisted ones exactly before trusting the
smoothed output. O1 stays out of `cfg.tier1.variants` (enforced at the config
layer already) and is written to its own file, never merged with the eight
reportable variants.

### D-023 · C2's one-hot cardinality tracks the HMM's actual selected K; its signal is `vix_level`
**Date:** 2026-10-01 · **Status:** implemented

C2 must be column-count-comparable to V3 for the gate (`V3 > C2`) to compare
like with like, so `scripts/03b_build_states.py` takes
`k = hmm_posteriors.shape[1]` (the real, already-selected K) rather than a
hardcoded or re-swept value. The threshold signal is `vix_level` — the same
series step 2's own HMM-vs-baseline detection comparison already uses (spec
§8.7) — rather than inventing a second baseline signal.

### D-024 · HMM specification pinned to H1; H2 dropped — cross-spec log-likelihood comparison is invalid
**Date:** 2026-10-02 · **Status:** implemented

The original step-2 design fit H1 (returns only, 1 observation dim) and H2
(returns + vix_change + curve_change + avg_pairwise_corr_change, 4 dims) and
picked whichever had the higher validation log-likelihood
(`scripts/02_fit_hmm.py`'s `max(sweep_results, key=val_loglik)`). This
comparison was invalid and is removed, not patched: a multivariate Gaussian
density sits on a structurally different scale than a univariate one on the
same data, so the higher-dimensional spec (H2) wins such a comparison
regardless of whether its regimes are any good — which they were not (see
below). `configs/hmm.yaml` now pins the active spec explicitly
(`hmm.fit.specification`, validated at config-load time to name a declared
specification), and `scripts/02_fit_hmm.py` fits and walk-forwards only that
one spec; there is no code path left that compares `val_loglik` across
specs of different dimensionality. H2 stays declared in `configs/hmm.yaml`
for the historical K-sweep record (`reports/tables/hmm_k_selection_H2.md`),
not as a selectable alternative.

**Why H2 specifically, independent of the dimensionality problem.** A
diagnostic refit of every K in H2's own sweep (2-8) found a degenerate best
restart — expected state duration ≈ 1-2 days against the 5-day floor, on
essentially every state — at **every single K**, including K=4, the one the
old (invalid) cross-spec comparison had selected. H2 would have been dropped
on this basis even if its dimensionality were comparable to H1's.

### D-025 · K=2 selected for H1 via a corrected within-spec algorithm (discard degenerate, near-tie BIC)
**Date:** 2026-10-02 · **Status:** implemented

`select_best_k` (`src/prism/models/hmm/fit.py`) previously preferred
non-degenerate `K` by validation log-likelihood but fell back to the full
pool (flagged) if every `K` was degenerate — i.e. a degenerate `K` could
still win if nothing else survived. It now **discards any `K` whose best
restart is degenerate outright**, before any comparison, and raises
`RuntimeError` if that leaves nothing (no silent least-bad fallback — a
sweep with zero non-degenerate `K` is itself a finding that must stop the
pipeline, not get quietly worked around). Among the non-degenerate
survivors, the highest validation log-likelihood wins **unless** another
survivor is within `hmm.selection.near_tie_margin` (2.0 points) of it, in
which case the tie is broken by BIC (lower is better) rather than by a raw
log-likelihood difference too small to be meaningful at this sample size.

Applied to H1's real sweep: K=4 through K=8 are degenerate and discarded
outright. Of the two survivors, K=3's val_loglik (816.59) edges out K=2's
(815.07) by 1.52 points — inside the 2.0-point near-tie margin — and K=2
has the better (more negative) BIC (-11177.82 vs -11148.28), so **K=2 wins**.
K=2 is also, independently, the only fit in the entire H1/H2 sweep (both
specs, every K) with a long realised dwell time (pooled median decoded dwell
9 days, vs 1-4 days everywhere else) and the lowest per-state duration
fail rate (5%) of any candidate — the statistical tiebreak and the
qualitative "does this look like a regime" check agree.

### D-026 · `hmm_walkforward` raises above a 10% all-restarts-degenerate fold rate
**Date:** 2026-10-02 · **Status:** implemented

Before this change, a walk-forward fold where every restart was degenerate
silently fell back to the least-bad restart and kept going — correct
per-fold behaviour (there is no better restart to pick), but with no
aggregate check on how OFTEN that was happening. A diagnostic refit of H2's
2019-07..2023-12 folds (54 of 204) found **0/54 with any clean restart at
all**, and a backward search from fold 180 found no clean fold anywhere back
to fold 0 — the "least-bad fallback" was not a rare safety valve for H2, it
was running on effectively every fold.

`hmm_walkforward` now counts folds where every restart was degenerate and
raises `RuntimeError` if that fraction exceeds
`hmm.walkforward.max_degenerate_fold_fraction` (0.10) after the full
walk-forward completes. This is a guardrail against exactly the H2 failure
mode recurring silently for H1 or any future spec/K — it does not change
what any individual fold does, only whether the aggregate result is trusted
enough to return at all.

**Consequence, and resolution.** At the time D-024/D-025/D-026 were written,
`data/processed/hmm_posteriors.parquet` and everything under
`data/processed/states/` were still the OLD H2/K=4 artifacts and were left
unregenerated pending review. Step 2 (D-024/D-025/D-026) was reviewed and
accepted on 2026-10-02; `scripts/02_fit_hmm.py` was re-run and
`hmm_posteriors.parquet` now reflects H1/K=2 (confirmed: 0/204 walk-forward
folds fully degenerate, well under the 10% guard — see D-027 for how its
dwell time is characterised). `scripts/03b_build_states.py --with-oracle`
was then re-run so all nine state variants (`data/processed/states/`) are
rebuilt consistently against the new posteriors; `encoder_latents.parquet`
is untouched and did not need rebuilding (the encoder does not consume HMM
output).

### D-027 · Full walk-forward dwell time: report the time-weighted mean, not the run-count median — the median is distorted by single-day boundary flips
**Date:** 2026-10-02 · **Status:** documentation only — no model, prior, or threshold change

The accepted H1/K=2 walk-forward's decoded state path (4278 days, 257 runs)
was first summarized by a run-count median: 4 days pooled (Q1=1, Q3=12),
treating each of the 257 runs as one equally-weighted observation. That
number understates the model's actual persistence. A chain that is mostly
stable but flickers back and forth for a day or two right at a genuine
regime boundary — a filtered posterior crossing 0.5 and re-crossing a day or
two later, not a new regime starting — contributes several extra *short*
runs around every *one* boundary. Counting runs, rather than days, lets a
handful of boundary flips outvote the long stable stretches between them,
which is exactly backwards for a persistence claim.

**The time-weighted figure is the persistence evidence to use instead:**
mean run length ≈ 17 days (16.65 pooled; total days / total runs — equal to
the expected length of the run a randomly-chosen *day* falls in), max 350
days. Per state: state 0 (the lower-return-std state) ≈ 23.3 days, state 1
≈ 9.9 days — consistent with state 0 being the more persistent, calmer
regime. This is the number `reports/tables/hmm_evaluation_H1.md` and any
future report should quote for "how long do regimes last", not the run-count
median.

**No stickiness prior.** This is a reporting choice about which summary
statistic characterises an already-fitted model's output — nothing in
fitting or decoding changed. No Dirichlet/sticky transition-matrix prior or
other persistence-favouring regularisation was added anywhere; the
posteriors and the decoded path are exactly what the accepted D-024/D-025/
D-026 walk-forward produced.

### D-028 · Defect fixed: C1 and V1' were built from config defaults, not the encoder step 3 selected
**Date:** 2026-10-02 · **Status:** fixed before the Tier 1 pre-registration was committed

Found while writing the pre-registration. `scripts/03b_build_states.py` built
C1 (random encoder) with `cfg.encoder.window.size` (30) and
`cfg.encoder.architecture.latent_dim` (16), and `build_state` built V1' with
`cfg.encoder.window.size` (30). Step 3 had selected window **10** and
latent_dim **32** (`encoder_summary.json`); the config values are only the
defaults the sweep started from. Consequences, had it gone uncorrected: C1
added 16 columns against V2's 32, so the V2-vs-C1 gate would not have isolated
"learned" from "more dimensions" (the whole purpose of C1, spec §10); and V1'
carried 30 sessions of history against the 10 V2's encoder saw, breaking the
same-information control (design gap D3). The script's own docstring claimed
"same window/dims as the real encoder", which the code did not do.

Fix: the script reads the selection from `encoder_summary.json`; C1 uses the
selected window / hidden / latent; `build_state` takes an explicit
`v1p_window` (passed from the same selection) instead of defaulting silently;
the script aborts if C1's width differs from V2's. V1' is now window 10
(2024 columns, 4182 sessions), C1 is window 10 / 32 dims. Both state files
were rebuilt; their schema hashes changed, and the Tier 1 pre-registration
records the rebuilt files' hashes. The earlier accepted rebuild (D-023's
neighbours, this date) is superseded for V1' and C1 only; V1, V2, V3, V4, C2,
C3 and O1 were unaffected.

### D-029 · Step 3 did not meet its acceptance criterion and was carried forward anyway; the Tier 1 gate is the deciding test
**Date:** 2026-10-02 · **Status:** recorded before the Tier 1 run

Spec §16 gives step 3's acceptance as "beats PCA and random encoder on probe"
(§9.4: with non-overlapping confidence intervals). **It did not.** The
latent-only probe of `fwd_vol_20` scored R² (test) of -0.049 for the DAE
latent and -0.203 for the purely predictive PRED latent, against 0.179 for
walk-forward PCA and 0.214 for the frozen random encoder (see "The encoder does
not beat PCA or the random encoder on the probe", above). §9.4's own text says
that in this case the encoder is simplified or dropped, and the result reported.

Step 3 was nevertheless carried forward, knowingly, for two reasons stated here
so it is a decision rather than an omission. First, that probe scored the
latent *alone*; the gate that matters, V2 (V1 plus latent) against V1' and C1
(each also carrying the full feature set), is a different and harder-to-read
comparison that only exists once state assembly does (step 3b). Second, the
Tier 1 gate (`tier1.gates.lstm_adds_value`: V2 > V1' and V2 > C1) is the
pre-registered test of whether the LSTM earns its place, and it is run once,
under `reports/tables/preregistration.md`, rather than being pre-judged by a
single-target preview.

**This does not rescue the encoder.** The step-3 result is a strong prior that
the LSTM gate fails, and the pre-registration says so (Hyp-3). If the gate fails
the LSTM is redesigned or dropped before Phase B per §13.1; carrying step 3
forward changes the order of work, not the standard it is held to.

### D-030 · Code version behind the pre-registered state files
**Date:** 2026-10-02 · **Status:** recorded

The state files whose SHA-256 values are listed in
`reports/tables/preregistration.md` §2 (committed as `242053811b9e04a667e541394b031f0c96a20035`)
were built by the code at commit **`444509807d75bec1698381be0a699d634dedc692`**
("Steps 2 and 3b: HMM pinned to H1/K=2, state assembly, diagnostics"), run as
`scripts/03b_build_states.py --with-oracle` against the accepted H1/K=2
walk-forward posteriors and the step-3 encoder latents. The pre-registration
was committed first, but the working tree it described was that code; this
entry fixes the correspondence. Anything that changes `src/prism/state.py`,
the 03b script, the HMM code or `configs/hmm.yaml` after this commit changes the
code version and requires an amendment to the pre-registration.

### D-031 · Code version behind the Tier 1 results
**Date:** 2026-10-02 · **Status:** recorded

The Tier 1 results (`reports/tier1_report.md`, `reports/tables/tier1_*.csv`, the
gate decisions below) were produced by the code at commit
**`77cc6bcd2754ec636bde4e92b98238f2a790e0b5`** ("Step 4a part 2: run the Tier 1
ablation as pre-registered; both gates fail"; parent `b99f4aa`), run as
`PYTHONHASHSEED=0 python scripts/04_tier1_ablation.py` (`make tier1`) against the
state files whose SHA-256 values are fixed in `reports/tables/preregistration.md`
(commit `242053811b9e04a667e541394b031f0c96a20035`), with state-building code at
`444509807d75bec1698381be0a699d634dedc692` (D-030).

Two things to know about that correspondence. The run itself happened with those
files uncommitted, so the run manifest in `reports/logs/` records the git tree as
dirty; the commit was made immediately afterwards with no edits in between, so
`77cc6bcd2754ec636bde4e92b98238f2a790e0b5` contains exactly the code that ran. And the stored results were
re-generated once after the last code change (moving the first validation year
into config) and reproduced the gates, probe metrics and allocator metrics
byte-for-byte, so the results do not depend on which of the two final runs is
read.

Any change to `src/prism/probes/`, `src/prism/analysis/`, `src/prism/backtest/`,
`src/prism/reporting/`, `scripts/04_tier1_ablation.py`, `configs/experiments/tier1_probes.yaml`
or the state-building code after this commit changes the code version behind these
results and needs an amendment to the pre-registration before any re-run is reported.

---

## Gate decisions (Tier 1)

### Gate decisions · LSTM: FAIL · HMM: FAIL · (research question V4 > V2: passes the paired rule, fails the spec's)
**Date:** 2026-10-02 · **Phase A result** · run under `reports/tables/preregistration.md` (commit `242053811b9e04a667e541394b031f0c96a20035`); full tables in `reports/tier1_report.md`, regenerated by `make tier1` (or `make report` from stored results). Test split 2019-01-01 .. 2023-11-22, 1233 sessions on the common index. The holdout was not read.

Rule, as pre-registered: a comparison passes iff the paired-difference 95% CI (stationary block bootstrap, block 20, seed 20260101, same resampled days for both variants) is favourable on at least 3 of the 4 primary risk targets and adverse on none; a gate passes iff both of its comparisons pass. dR2 below is R2(X) - R2(Y).

**LSTM adds value: FAIL.**
* V2 vs V1' passes (3/4, 0 adverse): favourable on `fwd_vol_20` (dR2 +0.069), `fwd_max_drawdown_20` (+0.041), `fwd_corr_20` (+0.128); indeterminate on `fwd_vol_5`.
* V2 vs C1 fails (1/4 favourable, 1/4 adverse). The random-encoder control, dimension-matched at 32, forecasts as well or better than the trained latent on every target but `fwd_corr_20` (dR2 +0.127, favourable) and is *better* on `fwd_vol_5` (dR2 -0.018 [-0.038, -0.001], adverse).
* The spec's non-overlapping-CI rule passes neither comparison (0/4 each).
* Reading: V2's edge over V1' is an edge over a 2024-column raw-window probe that ridge struggles with; an untrained 32-dimensional recurrent projection (C1) reproduces it. The value is in the recurrent feature map, not in the training. This is what D-029 anticipated (step 3's latent-only probe lost to PCA and the random encoder) and Hyp-3 predicted.
* Consequence (spec §13.1): the LSTM is redesigned or dropped before Phase B. It is not carried forward on the strength of V2 beating V1'.

**HMM adds value: FAIL.**
* V3 vs C2 fails (0/4 favourable, 0 adverse): every dR2 is within +/-0.01 and every CI includes zero. The K=2 HMM posteriors carry no more probe-relevant information than a two-bin VIX threshold.
* V3 vs C3 passes (3/4, 0 adverse): favourable on `fwd_vol_5` (+0.0032), `fwd_vol_20` (+0.0049), `fwd_max_drawdown_20` (+0.0029); `fwd_corr_20` indeterminate. So V3's regime columns are not noise (a shuffle of them is worse), but that shows only that they carry *some* information, which C2 carries too.
* The spec's non-overlapping rule passes neither comparison (0/4 each).
* Reading: consistent with Hyp-2, whose detection-lag arm was already negative before this run (HMM did not beat the VIX threshold against NBER or drawdown episodes). Neither arm of "HMM beats a volatility threshold" holds on this data.
* Consequence (spec §13.1): the HMM is redesigned or dropped before Phase B.

**Research question, V4 > V2 (reported; not a Phase B entry gate).** Paired rule: pass, 4/4 favourable, dR2 +0.0028 to +0.0055, CIs exclude zero. Spec rule: fail, 0/4. Two cautions travel with it. (1) The size is a half-point of R2 or less; under strong shrinkage the forecasts are nearly identical, so a low-variance loss differential makes tiny gaps "significant". (2) Because the HMM gate fails (V3 is indistinguishable from C2), the V4 gain cannot be attributed to the HMM rather than to *any* two-column regime signal. No "V2 plus threshold regime" control was pre-registered, so that attribution is untested here. The pre-registered design therefore does not support the thesis claim that HMM conditioning adds beyond the LSTM; it supports only that two extra regime columns move the forecasts a little.

**Diagnostics that frame the size of the effect (not used in any decision).** O1, the deliberately leaky oracle, beats V3 on all four risk targets (dR2 +0.003 to +0.074, largest on `fwd_corr_20`) and V1 on vol by about +0.02: even perfect regime knowledge buys only about two points of R2 on volatility here. Probe levels are sensible (OOS R2 about 0.44 on `fwd_vol_5`, about 0.22 on `fwd_vol_20`, about 0.01 on drawdown, negative on `fwd_corr_20` for V1/V3/C2/C3); `fwd_ret_20` is a mean forecast for every variant (R2 = 0, alpha at the grid ceiling), as expected. None of the section 15.1 audit triggers fired (no return R2 above 0.15; no net Sharpe above 2.0; no regime advantage surviving the C3 shuffle in a way that flatters V3; entropy not near zero).

**Allocator results (secondary, descriptive; no allocator result is a gate).**
* 0 of 45 pre-specified paired contrasts are claimable (CI excludes zero favourably AND DSR >= 0.95). Highest DSR among reportable strategies: 0.85 (V3/V4 regime-conditional). All Sharpe CIs span roughly -0.3 to 1.6 on 1230 sessions.
* Regime-conditional vol-target (RVT) with the HMM columns: V3 Sharpe 0.70, max drawdown -14.3%, against V1's 0.62 and -16.9%; in the 2022 episode V3's decline was -8.4% against -11.7%. The matching controls did not do better on both counts: C2 RVT 0.55 and -14.8%, C3 RVT 0.48 and -19.0%. But V3 - C2 Sharpe is +0.157 [-0.093, +0.381] and V3 - C3 is +0.221 [-0.014, +0.441]; both CIs include zero. Hyp-6 (risk metrics improve more than return) and Hyp-7 (advantage concentrated in drawdown episodes) are directionally consistent for V3/V4 RVT and statistically unsupported.
* The forecast-driven vol-target legs (VT) are near-identical across all eight variants (Sharpe 0.59-0.62): the probe forecasts differ little. They do beat the forecast-free trailing-vol benchmark on drawdown (-16.9% against -22.6%), but equally for V1, so that is the shared forecaster, not any component.
* No variant beats the simple benchmarks on Sharpe: buy-and-hold SPY 0.75, equal-weight 0.74, 60/40 0.72, risk-parity 0.72 (max drawdown -13.2%, as good as V3 RVT's -14.3% with higher Sharpe), against 0.70 for the best variant. What the variants buy is drawdown relative to SPY (-14 to -17% against -34%), at much lower return, and risk-parity and minimum-variance achieve comparable or better drawdowns without any forecast.
* Costs matter for the mean-variance leg (net Sharpe about 0.49 at 0 bps falling to about 0.21 at 20 bps; turnover about 10 per year) and little for the others.

**Limitations recorded with the decision.** (a) The mean-variance gamma sits at the top of its pre-registered grid (32; V1's validation exposure 0.82 against the 0.70 target); the grid was not widened after test output existed. (b) Ridge alpha is at the grid ceiling for `fwd_ret_20` for every variant, which is the pre-registered edge limitation and is benign (it means predict the mean). (c) One test split, already viewed once at step 3 (D-029): a redesigned LSTM or HMM scored on it again is exploratory, and confirmatory evidence needs the holdout under a new pre-registration. (d) Sharpe is on total net returns, not excess of cash. (e) Run history: two development runs preceded the final run; the only change between them was a bug in a weight-constraint check on the 60/40 benchmark; no pre-registered choice changed.

**Phase A ends here, for review (spec §16).** Per §13.1 both components are redesigned or dropped before Phase B; this entry records that they did not clear their gates, not what to do next.
