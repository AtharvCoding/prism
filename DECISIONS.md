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

---

## Phase B

### D-032 · Phase B scope: variants V1, V2, V4 and the new control C4; LSTM and HMM kept only as required comparisons
**Date:** 2026-10-02 · **Decided by:** principal investigator · **Status:** recorded; Phase A-only restriction (spec §0.3, §17.0) lifted

Phase A is accepted. Both Tier 1 component gates **failed** (see "Gate decisions (Tier 1)" above): the LSTM
(V2 vs C1) and the HMM (V3 vs C2). Spec §13.1 says a component that fails is "redesigned or dropped before Phase B".
The decision is that neither is presumed to help. **The LSTM and the HMM stay in Phase B only as the comparisons the
research question requires** (V4 vs V2 and V2 vs V1 need them), not as presumed improvements; a Phase B result in
which either adds nothing is a finding, not a defect of the build. Nothing was redesigned.

**Phase B variants: V1, V2, V4 and C4.** V1' , V3, C1, C2 and C3 are Tier 1 variants and are not trained in Phase B.

**C4 = V2 + the C2 threshold-regime columns** (V1 features, the 32-dim DAE latent, the two-column VIX-threshold
one-hot). Why it exists: the Tier 1 research-question comparison V4 > V2 passed the paired rule, but because V3 was
indistinguishable from C2, the gain "cannot be attributed to the HMM rather than to *any* two-column regime signal,
and no 'V2 plus threshold regime' control was pre-registered" (gate-decision entry above). C4 is that control. At the
policy level, **V4 vs C4** isolates whether the HMM's probabilistic posteriors matter beyond a threshold once a
temporal model is present; V4 vs V2 stays the research question as written.

**Implementation.** `prism.state.PHASE_B_VARIANTS = ("C4",)`; C4 needs `StateArtifacts.latents` and
`threshold_states` and is deliberately **not** in `cfg.tier1.variants` or `diagnostic_variants`: the Tier 1
pre-registration fixed those lists and `configs/experiments/tier1_probes.yaml`, and D-031 ties any change there to an
amendment. C4's schema is written to `states/schema_phase_b.json`, not `states/schema.json` (whose SHA-256 is
pinned). Tests: `tests/test_state.py` (C4 = V2 columns + C2 columns, each block byte-identical to the sibling
variant's; carries no HMM information; never a Tier 1 variant; names the missing artifact).

**Caveat carried forward.** The Tier 1 test split (2019-01-01 .. 2023-11-22) has been viewed (D-029, D-031 limitation c).
Phase B evaluation on it is exploratory unless a new pre-registration is committed before any agent is scored (spec
§13.2, build step 4c); confirmatory evidence needs the holdout under its own pre-registration.

### D-033 · O1 is not byte-reproducible: Amendment 1, a numeric check, and a guard against overwriting pre-registered files
**Date:** 2026-10-02 · **Decided by:** principal investigator · **Status:** implemented

**What happened.** Rebuilding the states to add C4 (`03b_build_states.py --with-oracle`) reproduced 13 of the 14
pre-registered SHA-256 values exactly (all of V1, V1p, V2, V3, V4, C1, C2, C3, `schema.json` and the inputs) and
changed one: `states/O1.parquet`, `79cf1166…` -> a different hash. O1 is built from per-fold *smoothed* posteriors,
which the script re-fits; a second consecutive run gave different bytes again, at most 2.3e-11 apart in value. The
refit is not byte-reproducible, so the pre-registered O1 bytes could not have been restored even in principle.
The rebuild also **overwrote the original O1 file, and I did not back it up first**; that was my error and is what
prompted the guard below. The original is unrecoverable.

**Amendment 1** (appended to `reports/tables/preregistration.md`, nothing above it edited): O1's exact hash is replaced
by a numeric check — same index and columns as, and within 1e-9 of, the frozen reference copy
`states/O1.amendment1.parquet` (`a631c8ab…`, itself verified by exact SHA-256). All 13 other hashes stay exact.
`scripts/04_tier1_ablation.py` implements it (`_verify_o1`; shared parser `prism.utils.prereg`). Tests in `tests/test_tier1.py`.
The check is against the post-loss rebuild, so it cannot show equivalence to the *original* O1; the evidence for
that is the next paragraph.

**Re-run.** `make tier1` after the amendment: input verification passed (15 hashes, O1 numeric), and the stored
results reproduce: `gates.json` byte-identical; every probe, allocator, DSR, cost-sensitivity, episode and paired
table within 9.3e-12 of the stored values (the O1 diagnostic rows included), non-numeric columns identical. They are
not byte-identical in the last digits (the earlier regeneration in D-031 was); the cause was not isolated and the
differences are far below any reported precision. **The committed `reports/tier1_report.md` and `reports/tables/tier1_*.csv`
and the original `data/processed/tier1/` are kept as the results of record**; the re-run output was discarded.

**Guard.** `03b_build_states.py` now refuses (before doing any work) to write any file whose SHA-256 the
pre-registration lists, unless `--allow-overwrite-preregistered` is passed. `--phase-b-only` builds C4 alone and
touches no Tier 1 file; it is the normal way to rebuild C4. Because O1 is listed, `--with-oracle` now needs the flag.
Anything that flag overwrites invalidates those hashes and needs an amendment first.

### D-034 · Transaction cost: 5 bps per side on each risky leg, plus volatility-scaled slippage; not the Tier 1 allocator's convention
**Date:** 2026-10-02 · **Decided by:** principal investigator (convention); implementation choices mine · **Status:** implemented, step 4b

**Convention (decided).** `cost = sum over risky i of |dw_i| * (per_side_bps/1e4 + slippage_vol_coef * sigma_i)`, as a
fraction of portfolio value, where `dw` is the trade from the weights drifted to the execution close to the target.
The per-side term is charged on **each leg's traded notional**; the cash line is free. This is the literal spec §11 /
§7.5 reading: a round trip into and out of a position costs 10 bps of that position at 5 bps per side.

**It differs from the Tier 1 allocator.** `prism.probes.allocator.net_returns` charged `bps * 0.5 * sum|dw|` with cash
included. The two agree on a trade between cash and risky assets and differ by exactly **2x on a risky-to-risky swap**
(env: 10 bps for a full swap; allocator: 5 bps). Consequences: (a) Tier 1 allocator and benchmark net numbers are *not*
comparable to env numbers and must not be mixed in one table; (b) any Phase B benchmark is to be re-run through the env
cost model (the env's `run_episode` with a fixed policy does it). The Tier 1 results were not changed; they are correct
under the convention they were pre-registered with. `tests/test_costs.py::test_a_swap_costs_twice_the_tier1_allocator_convention`
pins the relationship.

**Slippage (my choice, not in the spec beyond "scaled by prevailing volatility").** `slippage = slippage_vol_coef * sigma_i`
with `sigma_i` the trailing 20-session standard deviation of asset i's daily simple returns known at the execution
close (causal), `slippage_vol_coef = 0.02`: 2 bps per unit traded at a 1% daily vol, 8 bps at 4%. Chosen as
plausible for liquid ETFs and not tuned on any split; it is one knob in `configs/env.yaml`, and the sensitivity grid
below is how its effect is reported. Cost on the first step from cash is a real cost (initiation is a trade).

**Sensitivity grid (spec §11).** 0 / 5 / 10 / 20 bps per side, scaling **both** terms by `bps / per_side_bps` (so 0 is
frictionless and 10 doubles slippage too), rather than adding a second grid. A reader who wants the proportional term
alone should know slippage moves with it.

**Scale seen on the train split** (`reports/tables/env_check.md`, random action weights, all-train episode): a
near-constant mix pays ~78 bps of NAV in total costs over 550 weeks at the 5 bps level; a policy that re-draws weights
every week (mean one-way turnover 0.63 per step) pays ~45% of NAV and ends below 1.3 against 1.98 gross. Costs will dominate any
high-turnover policy; this is the intended pressure on the agents.

### D-035 · Environment design (step 4b)
**Date:** 2026-10-02 · **Status:** implemented · open items flagged below

Most of the §11 text is followed literally; these are the choices it left open, each of them recorded rather than
silently picked.

1. **One step is one weekly decision** (D-001), with the daily path simulated *inside* the step. The decision schedule,
   execution lag (1 session) and holding periods come from `prism.backtest.engine.holding_periods`, the single statement of
   the observe -> decide -> execute -> earn contract. Decisions fall on the last session of each calendar week on or before
   Friday (a Thursday when Friday is a holiday; `tests/test_env.py::test_a_holiday_friday_moves_the_decision_to_thursday`).
   A step's reward window is `(e_k, e_{k+1}]`, strictly after the decision close `d_k`; the drift between `d_k` and `e_k`
   belongs to the previous step, and the trade at `e_k` starts from weights drifted to `e_k`, not to `d_k`. A spike test
   asserts each one-session return lands in exactly the step whose window contains it, and mutation checks confirmed that
   starting the window one day early, dropping the cost, trading from the wrong weights, or dropping drift each fail a test.
2. **Assets: the 13 allocatable risky assets plus cash** (`data.allocatable.B`), as the Tier 1 allocator. **SPY is not an
   asset**: it is the benchmark and a signal, overlapping the nine sectors.
3. **Action to weights.** `a in [-1, 1]^(n+1)` -> `softmax(3 * a)` -> Euclidean projection onto {0 <= w_i <= 0.35 for risky,
   0 <= w_cash <= 1, sum = 1}. Long-only, the cap and full investment hold by construction (D-001), no penalty. **The cap
   applies to risky assets and not to cash**, the allocator's interpretation: a policy must be able to hold mostly cash. The
   projection is the identity whenever the softmax is feasible, so the map is smooth except where the cap binds; where it binds
   the excess is shared in equal additive amounts across every line with slack (cash is not singled out). The zero action is equal
   weight over all 14 lines (about 7% each, cash included). `logit_scale = 3` is a design constant, not tuned; it sets how
   concentrated the policy can get (at 3, the logit range is +/-3, so the extreme weight ratio before the cap is e^6, about 400x).
4. **Portfolio block** (appended to the variant's state vector, built by the env, not by `state.py`): the `n+1` drifted current
   weights and the **mean** one-way turnover per step so far. Two deviations from spec §10's "weights, time since last
   rebalance, cumulative turnover": (a) *time since last rebalance is omitted*: with a decision every week it is constant, and a
   constant input is a degenerate feature (§4.3 treats a constant feature as a hard failure); (b) *cumulative turnover is divided by the
   step count*, so it stays in [0, 1] and does not leak the position in the episode. `state.py`'s date-indexed `portfolio_block`
   hook cannot carry any of this, because it depends on the agent's own actions; it stays as the Phase A hook, unused.
5. **Episodes.** Training: random start inside the split and 104 decisions (about two years), ended by *truncation* so the
   next observation exists for bootstrapping; evaluation: one fixed episode over the whole split ending in *termination* at its
   last decision. The portfolio starts in cash. Splits are cut to the **effective (embargo-purged) range** so no reward window
   crosses an embargo or the next split; the holdout is refused by the builder (`PermissionError`), and `assert_not_holdout` runs on
   every built dataset.
6. **Cash return** is the prior session's `^IRX` close / 100 / 252, as in the allocator (asserted equal in the tests), so no rate is used
   before it was published.
7. **Observations are the variant's already-scaled state** (the train-split scaler of step 3b) in float32; `Box(-inf, inf)` bounds
   (gymnasium's checker warns about that; the warning is left visible).

**Known accounting difference from the Tier 1 simulator.** `allocator.simulate` earns the first execution close's day return
while still in cash; the env starts earning after its first trade. The tests and the check script compare from the following
session; the difference is one day of cash return at episode start.

**No look-ahead is tested, and the test is tested.** Every state, return and volatility row after what a step may read is
replaced with garbage; observations and rewards must be bit-identical. The identical check fails on a deliberately leaky
environment that reads three sessions ahead (spec §7.1: a causality test that never fails is worthless).

### D-036 · Reward variants and their parameters
**Date:** 2026-10-02 · **Status:** implemented; the default is the only reward intended for headline results

Default `log_return_net`: `log(1 + net return)` of the holding period, net of the cost paid to reach the weights (so costs are in the
reward by construction). Spec §11 lists four variants and says reward choice is a documented experimental factor, so all four exist, selected by
name in `configs/env.yaml`, and none is tuned:

* `dsr`: Moody & Saffell differential Sharpe ratio on the net period return, moments `A, B` with rate `eta = 0.05`, evaluated before the
  update, zero while the variance estimate is degenerate. Hand-computed in `tests/test_env.py`.
* `mv_penalty`: log net return minus `lambda * sum (r_t - mean)^2` over the period's daily portfolio returns, `lambda = 1.0`. (Realised
  variance, not annualised; at a typical weekly variance of about 5e-4 the penalty is a fifth of a typical weekly return.)
* `drawdown_penalty`: log net return minus `lambda * DD`, `DD = 1 - NAV/peak NAV` within the episode, `lambda = 0.02`.

Which variants run in Tier 2, and with what values, is a step 4c question: a pre-registration is needed before any agent is scored on the
test split, and a reward variant added after seeing test output would be a forking path.

### D-037 · Code version behind the step 4b results
**Date:** 2026-10-02 · **Status:** recorded

`reports/tables/env_check.md` (21 checks, all PASS, train split only) was produced by `scripts/04b_env_check.py` run as
`PYTHONHASHSEED=0 python scripts/04b_env_check.py` (`make env-check`) on the code at commit
**`23c9423c209584e8952bc3b60b44408c933f7f81`** ("Step 4b: environment, cost model, ..."), with the state files `V1`, `V2`, `V4` and
`C4` whose SHA-256 values are in the run manifest in `reports/logs/`. The run happened with those files uncommitted (the manifest
records a dirty tree) and the commit followed with no change to `src/prism/env/`, `src/prism/config.py`, `configs/env.yaml` or the
script in between; only the README, Makefile, pyproject pin, `DECISIONS.md` and a test were touched after the final run. C4 itself
was built by commit `415947d`; `03b_build_states.py --phase-b-only` rebuilds it byte-identically.

### D-038 · Step 4c design: SAC, the Tier 2 contract and how a "variant" is compared
**Date:** 2026-10-03 · **Decided by:** principal investigator (variants, grid, seeds, reward, runner contract); implementation choices mine · **Status:** implemented; fixed in `reports/tables/preregistration_tier2.md`

What was decided for me: variants V1, V2, V4, C4; reward `log_return_net` only; grid γ ∈ {0.9, 0.97} × network ∈ {[64,64], [128,128]} ×
lr ∈ {3e-4, 1e-3}, 3 seeds per config, identical for every variant, tuned on validation only; best config per variant, 10 seeds,
checkpoint selection on validation; comparisons V4-V2, V4-C4, V2-V1; Tier 1's paired-bootstrap rule, deflated Sharpe and
drawdown episodes; benchmarks re-run through the env; an overfitting report; `make tier2` resumable and aborting before the test split
if the sanity gates fail. What I chose, each one fixed in the pre-registration before any run:

1. **A variant is a set of seeds; the Tier 1 gate rule is carried over by changing the comparison unit.** Tier 1's unit was squared forecast
   error per day on four risk targets. Here: four primary metrics of each seed's daily net return series (annualised return, Sharpe, max
   drawdown, CVaR 95%, all oriented larger-is-better); a variant's statistic is the mean over its 10 seeds; the CI comes from the stationary
   block bootstrap with **days and seeds both resampled** (one shared day path per replicate for every series, seeds resampled
   independently per variant); a comparison passes iff favourable on >= 3 of 4 metrics and adverse on none; the spec's non-overlapping-CI
   version is reported beside it; the paired rule decides. A resample of days alone would treat 10 seeds as one estimate and understate the
   uncertainty; resampling seeds as well is what makes "seed variance" part of the verdict.
2. **Tuning statistic.** Per run: the best-checkpoint validation mean per-decision `log_return_net` of the deterministic policy; per config: the
   mean over the 3 tuning seeds; ties go to the earlier grid position. It is the training objective, evaluated on 46 validation decisions, so it is
   noisy; the full tuning table is reported. Tuning seeds (1000-1002) are disjoint from final seeds (0-9), enforced by config validation, so
   the final seeds never chose a configuration.
3. **Deflated Sharpe trial count.** N = 40 (every final run is a trial; per-run Sharpe variance across the 40) as the headline, N = 136 (all runs
   trained, tuning included) as the sensitivity. "Claimable" needs the gate AND a median DSR >= 0.95 across the candidate's seeds, as Tier 1 needed
   the CI AND DSR >= 0.95.
4. **Env additions (behaviour-preserving).** `PortfolioEnv.step_weights` (the same code path as `step` after the action map; `step` now calls it),
   `drifted_weights` (what a pure hold keeps), `info["daily_returns"]`, and a `risky=` override on `build_env_data`. They exist so the six benchmarks,
   whose weights (an SPY line, a hold) are not reachable through the action map, run through the env's cost and drift code, and so a daily net return
   series can be built that compounds exactly to the env's NAV (asserted at run time; the cost paid at an execution close is charged on the first session of
   the holding period it opens). `make env-check` was re-run after the change: `reports/tables/env_check.md` is byte-identical.
5. **Benchmarks** reuse the Tier 1 weight functions (`prism.backtest.benchmarks`) on the env's own decision dates, with SPY as a benchmark-only line
   (D-035: SPY is not an agent asset). Buy-and-hold SPY trades once and then holds (zero cost after initiation, tested).
6. **Dependencies.** `stable-baselines3==2.9.0` pinned. 2.8.0 would have downgraded `gymnasium` to 1.2.3 (it requires `<1.3`); 2.9.0 allows `<2.0`, so
   `gymnasium==1.3.0` is unchanged.
7. **Runner.** Process pool (spawn), one torch thread per process; every run is a directory whose `result.json` is written last and atomically, so a
   run killed midway is retrained from scratch on resume (a replay buffer is not worth checkpointing for a 30-minute run). The frozen configs are
   written once and refused if edited or overwritten; the single test evaluation leaves a marker and does not repeat. `--smoke` runs the same code at toy
   size, with the validation split standing in for test, into a separate directory, and never writes the report or `DECISIONS.md`. The contract check
   (pre-registration committed and unmodified; input SHA-256 values match) applies to every stage but `--stage sanity`, which reads train/validation
   only and may run before the pre-registration exists. `make tier2` is wrapped in `caffeinate -i` where available.
8. **Reward scale 10 during training only.** SAC's automatic entropy coefficient starts at 1 against rewards of order 1e-3, which left the agent barely off
   equal weight at 20 000 steps (attempt 0 below). The scale is a positive constant, so it does not change the optimal policy; every checkpoint score,
   evaluation and report reads the unscaled reward. It is not a reward-sensitivity experiment.

### D-039 · SAC sanity gates: what was tried, two gate-design corrections, and what the passing record does not show
**Date:** 2026-10-03 · **Status:** gates pass at the production budget; full attempt table in `reports/tables/tier2_sanity.md` and pre-registration §5

Gate 1 (degenerate task) and gate 2 (beats random on validation), train/validation only. Attempt 0 (reward scale 1) and attempt 1 (scale 100) both
**failed** both gates. Exploratory runs on the synthetic task then showed the gate getting **worse with more training**, which is not what an optimiser
fault looks like: random state features let the network memorise the single noise path. **That was a defect in my gate, not in SAC**, so gate 1 was redesigned
(constant state, 3 000 sessions), not "fixed" with one of the three allowed knobs. Gate 2's bar was also corrected: at attempt 0 a constant equal-weight mix
scored -0.00059 per decision on validation, below the 95th percentile of random policies (+0.00001), because 2018 was a falling year; a 95th-percentile bar
measures the market. It moved to the **median** random policy, evaluated on the validation-selected checkpoint (the pipeline's product); the 95th percentile and the
final checkpoint are reported, not gating. Both corrections were made before the production-length run and are disclosed as corrections. Attempt 2 (scale 10,
300 000 steps) passed; the budget was then cut to 250 000 and the cadence refined (D-040), and **attempt 3, scale 10 at the production budget, passed**:
degenerate task 1.00 / 1.00 / 1.00 of the weight cap and 0.98 / 0.83 / 0.92 of the oracle's return; beats random with selected-checkpoint ranks 0.93 / 0.82 / 0.92
among 200 random policies. Learning rate and observation normalisation were not needed.

**What the pass does not show.** No selected checkpoint beat the 95th percentile of random policies. The final checkpoints fit the train split at about +0.020
per decision and predict validation at -0.0008 to -0.0017; two of three selected checkpoints in each of attempts 2 and 3 were at or near the first
checkpoint (10 000-15 000 steps). SAC on about 550 weekly decisions overfits heavily, and checkpoint selection on a 46-decision validation window is noisy. This is why the
overfitting report is part of the pre-registered output and why the Tier 2 result is read with it.

### D-040 · Training budget: 250 000 steps per run, six workers, about 12.3 hours
**Date:** 2026-10-03 · **Status:** fixed in pre-registration §6

Timed on this MacBook Air M4: 412 steps/s for one process; with six processes about 156 (network [128,128]) and 173 ([64,64]) steps/s each, ≈ 990 aggregate;
four processes ≈ 700 aggregate and eight ≈ 660-800, so **six workers**. A full 250 000-step run (the sanity gate, six in parallel) took 1 818-1 833 s. 136 runs
(96 tuning + 40 final) in 23 rounds of six ≈ 12.3 h, ≈ 15.4 h with a 25% derating for sustained throttling of a fanless machine. The first choice, 300 000 steps,
would have crossed 16 h under that derating; checkpoints every 5 000 steps (50 per run) cost 0.3 s each, so the finer early grid is free. Run:
`make tier2` (it is resumable: re-running skips finished runs).


### D-041 · Code version behind the step 4c pre-registration and sanity record
**Date:** 2026-10-03 · **Status:** recorded

`reports/tables/preregistration_tier2.md` was committed with the step 4c code as **`e0dfc11062569dd49c2068754534b0fb7c63e622`** ("Step 4c: SAC pipeline,
Tier 2 pre-registration, sanity gates, make tier2"), before any tuning or final run and before the test split was read by this step. The sanity record
(`data/processed/tier2/sanity.json`, rendered in `reports/tables/tier2_sanity.md`) was produced by `scripts/05_train_agents.py --stage sanity` with the
working tree uncommitted; between that run's start and the commit only the sanity *report text* (`prism.reporting.tier2_report`), the Makefile and the README
changed, not `prism.agents.sac`, `.jobs`, `.sanity`, the env or `configs/experiments/tier2.yaml`. The earlier 300 000-step sanity run (attempt 2) is kept
under `data/processed/tier2/sanity_300k_cadence10k.json` (untracked) and is in the attempt table. `make tier2-smoke` exercised the full pipeline (tune, freeze,
final, evaluate, report, resume, abort-on-failed-sanity) on the validation split; the 379-test suite passes. The full run is the user's to start: `make tier2`.

Anything that changes `src/prism/agents/`, `src/prism/analysis/tier2.py`, `src/prism/env/`, `src/prism/reporting/tier2_report.py`, `scripts/05_train_agents.py`,
`configs/experiments/tier2.yaml` or the four state files after this commit changes the code version behind the Tier 2 results and needs an amendment to the
pre-registration (appended, committed) before the run is reported; a change to the pre-registration itself is refused by the runner until it is committed.

---

## Gate decisions (Tier 2)

### Gate decisions · V4>V2: FAIL · V4>C4: FAIL · V2>V1: FAIL
**Date:** 2026-10-05 · **Phase B result, exploratory (D-032, preregistration_tier2 §0)** · run under `reports/tables/preregistration_tier2.md`; code at commit `de2eaaff2a7e96c7eff73d54330e60d3c1948d07`; full tables in `reports/tier2_report.md`, regenerated by `make tier2`. Evaluation window 2019-01-08 .. 2023-11-22 (1229 sessions). The holdout was not read.

Rule, as pre-registered: a comparison X vs Y passes iff the paired 95% CI (stationary block bootstrap, block 20, 2000 replicates, shared day paths, seeds resampled within each variant) is favourable on at least 3 of the 4 primary metrics (annualised net return, net Sharpe, max drawdown, CVaR 95%) and adverse on none. Candidate statistic = mean over the variant's seeds of the metric.

* **V4 vs V2: FAIL** (0/4 favourable, 0/4 adverse); spec's non-overlap version: fail (0/4). Median DSR of V4 across seeds 0.53 -> not claimable (needs the gate AND DSR >= 0.95). Differences: `annualised_return` -1.0% [-5.6%, 3.6%] indeterminate; `sharpe` -0.08 [-0.42, 0.27] indeterminate; `max_drawdown` -2.1% [-7.3%, 5.5%] indeterminate; `cvar_95` 0.0% [-0.2%, 0.3%] indeterminate.
* **V4 vs C4: FAIL** (0/4 favourable, 0/4 adverse); spec's non-overlap version: fail (0/4). Median DSR of V4 across seeds 0.53 -> not claimable (needs the gate AND DSR >= 0.95). Differences: `annualised_return` 0.6% [-4.7%, 5.9%] indeterminate; `sharpe` 0.02 [-0.38, 0.41] indeterminate; `max_drawdown` -1.1% [-7.5%, 7.8%] indeterminate; `cvar_95` 0.0% [-0.2%, 0.3%] indeterminate.
* **V2 vs V1: FAIL** (0/4 favourable, 0/4 adverse); spec's non-overlap version: fail (0/4). Median DSR of V2 across seeds 0.55 -> not claimable (needs the gate AND DSR >= 0.95). Differences: `annualised_return` 1.3% [-4.6%, 6.7%] indeterminate; `sharpe` 0.11 [-0.25, 0.51] indeterminate; `max_drawdown` 1.9% [-5.8%, 8.7%] indeterminate; `cvar_95` 0.0% [-0.2%, 0.3%] indeterminate.

Block-length sensitivity (verdict of the paired rule): block 10: V4>V2 fail, V4>C4 fail, V2>V1 fail; block 40: V4>V2 fail, V4>C4 fail, V2>V1 fail.

**Limitations recorded with the decision.** (a) The evaluation split is the Tier 1 test split, already viewed in Tier 1 and used to choose the variant set (D-029, D-032); this result is exploratory, and confirmatory evidence needs the holdout under its own pre-registration (build step 5). (b) Hyperparameters were tuned on one validation year (about 46 weekly decisions), so selection is noisy. (c) One test window; per-episode numbers are descriptive. (d) Seeds share the data, so the seed band is optimiser variance, not sampling variance of the market.

### D-042 · Step 5 design: the holdout is spent once, on the frozen Tier 2 agents
**Date:** 2026-10-05 · **Decided by:** principal investigator (protocol: frozen agents, confirmatory); implementation mine · **Status:** built and rehearsed; the holdout has NOT been read

The principal investigator chose, from three options, to evaluate the 40 frozen Tier 2 agents on the holdout under a new pre-registration (`reports/tables/preregistration_holdout.md`), not to
hold the holdout for a redesigned candidate. Choices that followed:

1. **Inputs for 2024-2026.** The step 1-3b states stop at the test split's end by design, so the holdout rows of V1, V2, V4 and C4 are produced by the same causal walk-forward pipeline extended through
   the holdout (`prism.holdout.build_extended_states`: features and pruning with the step-1 fit windows, the pinned H1/K=2 HMM, the selected DAE encoder, the VIX threshold, the train-split scaler),
   with everything read from the persisted step 2-3 summaries; nothing is re-selected. It writes to `data/processed/holdout/`, never to a pre-registered file.
2. **A replay check as a hard gate.** Run with its end at the test split's end (no holdout row read), the extended pipeline reproduced all four stored state files with a maximum absolute difference of
   3.2e-11 (V4 only, from the HMM posteriors; V1, V2 and C4 exactly), in 5 minutes. The real run requires the same, to 1e-9, on every row up to 2023-12-31, and aborts without evaluating if not; no fallback
   and no widened tolerance. This is what makes "the agents see the same inputs" a checked fact, and the extension a pure addition of rows.
3. **Everything frozen is pinned by SHA-256 in the pre-registration**: the four state files and schema, `chosen_configs.json` and all 40 `best.zip` checkpoints (46 hashes). The runner verifies them before it
   reads the holdout, and refuses if the pre-registration is uncommitted or modified.
4. **Locks.** `99_final_holdout.py` needs `--i-am-sure` and `PRISM_ALLOW_HOLDOUT=1`, logs the opening in `reports/logs/holdout_access.jsonl`, and refuses once `data/processed/holdout/eval_done.json` exists. The env
   builder's old unconditional refusal became a two-key refusal (`final_holdout=True` and the environment variable); `load_close` still refuses outright. A failure before the marker evaluated nothing and may be repeated.
5. **Same evaluation, same rule.** `tier2.evaluate_test` and `analyse` are reused unchanged in logic (injected inputs, `runs_dir` separating the Tier 2 agents from the holdout outputs): same three comparisons, four metrics, bootstrap
   (days and seeds), deflated Sharpe, episodes, benchmarks and costs. Reading rules for each outcome (null replicated / unconfirmed signal / pass on both) are fixed in advance in the pre-registration.
6. **Rehearsal.** `scripts/99_final_holdout.py --rehearse` ran the entire path (replay check, evaluation of the 40 agents and 6 benchmarks, bootstrap, report) on the VALIDATION split into
   `data/processed/holdout_rehearsal/`, reading no holdout row. `make final-report` regenerates `reports/final_report.md` from stored results (Tier 1, Tier 2, holdout if present) and never opens the holdout.
7. **Not done:** no retraining, tuning or re-selection; no change to any choice after the holdout is read; no second evaluation.

The command to spend the holdout is typed by hand, deliberately (there is no `make` target): `PRISM_ALLOW_HOLDOUT=1 .venv/bin/python scripts/99_final_holdout.py --i-am-sure`.

---

## Gate decisions (holdout)

### Holdout · V4>V2: FAIL · V4>C4: FAIL · V2>V1: FAIL
**Date:** 2026-10-05 · **Phase B, step 5; confirmatory, one use** · run under `reports/tables/preregistration_holdout.md`; code at commit `442e9db10297ee8692ee98599d9806091723c169`; window 2024-01-09 .. 2026-09-30 (684 sessions); the 40 frozen Tier 2 agents, nothing retrained. Replay check on the extended inputs: max |difference| to the stored states 2.15e-11.

* **V4 vs V2: FAIL** (0/4 favourable, 0/4 adverse); spec non-overlap: fail; median DSR 0.74 -> not claimable. `annualised_return` 1.7% [-3.3%, 6.3%] indeterminate; `sharpe` 0.18 [-0.31, 0.71] indeterminate; `max_drawdown` -0.2% [-4.5%, 4.1%] indeterminate; `cvar_95` -0.0% [-0.3%, 0.2%] indeterminate.
* **V4 vs C4: FAIL** (0/4 favourable, 0/4 adverse); spec non-overlap: fail; median DSR 0.74 -> not claimable. `annualised_return` 0.6% [-4.5%, 5.3%] indeterminate; `sharpe` 0.15 [-0.33, 0.64] indeterminate; `max_drawdown` 0.4% [-3.1%, 4.1%] indeterminate; `cvar_95` 0.1% [-0.1%, 0.3%] indeterminate.
* **V2 vs V1: FAIL** (0/4 favourable, 0/4 adverse); spec non-overlap: fail; median DSR 0.59 -> not claimable. `annualised_return` -1.0% [-5.7%, 3.2%] indeterminate; `sharpe` -0.14 [-0.57, 0.33] indeterminate; `max_drawdown` -1.8% [-6.0%, 2.4%] indeterminate; `cvar_95` -0.1% [-0.3%, 0.2%] indeterminate.

The holdout is spent. No result of this evaluation changes a choice; the final report is `reports/final_report.md`.

---

## Dashboard (DASHBOARD.md)

Not a research step. The dashboard displays the stored, pre-registered results and runs the frozen system forward as a demonstration; it selects,
tunes and evaluates nothing, and the holdout stays spent.

### D-043 · Dashboard D0: the open decisions of DASHBOARD.md §11
**Date:** 2026-10-06 · **Decided by:** principal investigator (2: hosting); the rest are the spec's stated defaults, taken by me · **Status:** recorded, nothing built

1. **Live "as-of": both.** The headline is the latest completed weekly decision; a mid-week *preview as of the latest close* is offered, labelled as a preview and never
   as a decision. Alternative: decision only (simpler; rejected because a stale Friday number mid-week invites the reader to guess).
2. **Hosting: local only** (`make dashboard`). The agents, the snapshot and the live models are read where they already are. Alternative: Streamlit Community Cloud. Rejected
   for now: the 40 `best.zip` files are about 950 KB each (37 MB, not the ~100 KB the spec assumed), and the live features need the raw snapshot as warm-up history, which must not be committed.
3. **Live models: `data/live/models/`, binaries not committed, a manifest committed** (fold index, fit and apply dates, SHA-256 of each file). They are regenerated by the gated
   precompute and verified against the manifest on load. Follows from 2. Encoder weights are not stored as `*.pt`/`*.pkl` (gitignored and pickle-based); format chosen at D2.
4. **Home-page animation: Plotly frames.** Alternative: a custom component (more control, a JavaScript build to maintain; not justified).
5. **What-if on C4: offered, as a binary switch** (its two regime columns are a 0/1 one-hot), with the note that this control has no intermediate values. Alternative: not offered.
6. **Cost slider: linear interpolation between the stored 0 / 5 / 10 / 20 bps levels, labelled as interpolation.** The endpoints and the four stored levels are exact. Recomputing
   from daily returns is not possible (the cost enters the path).
7. **"Fresh" data: splice with a 60-session overlap** onto the frozen snapshot; the feature warm-up (252 sessions) comes from the snapshot.

**Stack:** Streamlit multipage (`st.navigation`) + Plotly, layout as DASHBOARD.md §3.

### D-044 · Dashboard: where DASHBOARD.md does not match the code, and what is done instead
**Date:** 2026-10-06 · **Decided by:** principal investigator (items 1, 2); the rest found on reading the code, recorded so they are not silently resolved · **Status:** 1-2 decided; the others are raised again at the milestone named

1. **Verification of the live HMM filter and encoder: final fold only (decided).** DASHBOARD.md §4.2 and D5 ask a single last-fold model to reproduce `states_extended.parquet` over the
   holdout to 1e-9. It cannot: the holdout posteriors come from 33 monthly refits (folds 204-236) and the latents from 3 annual refits, each a different model. The last HMM fold
   (index 236) is fit 1999-01-04 .. 2026-07-27 and applies to 2026-09-01 .. 2026-09-30; the last encoder fold (index 19) is fit .. 2025-11-24 and applies to 2026-01-02 .. 2026-09-30 (the spec's
   "fit through ~Aug-Sep 2026 / end of 2025" is approximate; the page shows these dates). **The live functions must reproduce the stored states to 1e-9 on those apply windows**, and the precompute
   separately checks the whole 237-fold HMM re-run against the stored posteriors. Alternative: persist every holdout-era fold and select the model by date, so the check passes as worded; rejected as
   code and artifacts for a path the live view never takes after 2026-09-30.
2. **Live rollout: re-run and disclose (decided).** `build_env_data` cuts every split at its effective end, so sessions after 2026-09-30 can only be run through `EnvData.from_arrays`, which is outside
   the two-key holdout gate. Each refresh therefore re-runs the stored holdout episode from cash and continues it (DASHBOARD.md §4.1.4), reading holdout-period inputs descriptively. **Amendment 1 to
   `preregistration_holdout.md` will say so**, in addition to the one-off gated replay of §5.4; no statistic is computed from either. Alternative: resume from a saved end-of-holdout portfolio state;
   rejected because it needs a state-injection hook in `src/prism/env/`, code behind the results. `src/prism/env/`, `src/prism/agents/` and `src/prism/analysis/tier2.py` are not edited by the dashboard.
3. **Dates.** The holdout episode's first decision is 2024-01-05, executed 2024-01-08; 2024-01-09 is the first return day, not "the first decision" (§4.1.4). 143 decisions, the last on 2026-09-25. The test
   episode: 255 decisions, 2019-01-04 .. 2023-11-17.
4. **Turnover figures.** §1 quotes 23-38% a week against 1-4% (the holdout: 23.3-37.8%, benchmarks 0.7-3.7%); `final_report.md` §6 quotes 25-40% against 1-3% (the test split: 24.5-41.4%, 0.4-2.9%). Both are
   right for their window. The dashboard shows each window's numbers from its own `turnover.csv`, labelled.
5. **Tier 1's V4-vs-V2 comparison passed the paired rule** (4/4 favourable, at 0.3-0.6 points of R²; not attributable to the HMM, which is why C4 exists). DASHBOARD.md §1 summarises Tier 1 as agreeing with the
   null and omits this. The dashboard reports it wherever Tier 1 is summarised: "no spin in either direction" applies to a pass as well.
6. **Not listed in §4.1.1 but needed by the live path:** the VIX-threshold breakpoints of the last fold (C4's regime columns) and the Universe-B train-split scaler; both are persisted at D2 rather than refit at request time.
7. **Open, for D2:** `build_extended_states` does not return the fold objects (`wf.folds`, `ewf.folds`) the precompute must persist. Either two optional fields are added to `ExtendedStates` (additive; `holdout.py`
   is the code behind the holdout inputs) or the precompute repeats its calls. Also for D2: the HMM is fit on per-fold standardised returns, so a fold's state volatility must be multiplied by that fold's scaler to be shown in return units.
8. **Open, for D3:** latent coordinates are not comparable across the annual encoder refits. The stored latents move a median 0.22 (99th percentile 0.68) from one session to the next and a median 3.15 (minimum 2.52)
   across each of the 16 year boundaries. One PCA map animated through time (§7 page 5) would show a jump every January that is the refit, not the market. Options: a map per encoder fold, or one map with the refits marked.
9. **Open, for D5:** (a) the splice of §4.1.3 rescales to the frozen close; that is right for adjusted ETF prices and wrong for level series (`^VIX`, `^VIX3M`, the yields; `^IRX` can be near zero), which should be
   checked for equality and not rescaled; (b) a refresh during market hours returns a partial bar for the current session, which must be dropped until the close (exchange calendar), or the "close" is not a close.
10. **Tests.** The date-literal grep test covers `src/` and `scripts/`, so `prism.live` and `prism.dashboard_data` take every date from the config and the stored artifacts. `make install` stays without Streamlit, so the
    app tests must skip when it is absent (the README's "no skips" then holds for a dashboard install). A probe of `AppTest` with a Plotly chart raised nothing under the suite's `error::FutureWarning` / `Pandas4Warning` policy.
11. **Column names.** C4's two VIX-threshold columns are also named `state_0`, `state_1` in the state files; the dashboard renames them on display so they are not read as HMM posteriors.

### D-045 · Dashboard dependency group, and a SHA-256 baseline of the frozen files
**Date:** 2026-10-06 · **Decided by:** principal investigator (baseline now, in D0); pins mine · **Status:** implemented

**Dependencies.** `pyproject.toml` gains the optional group `dashboard` = `streamlit==1.65.0`, `plotly==7.1.0`; `make dashboard-install` installs `.[dev,dashboard]` into the existing venv; `make install` is unchanged.
Checked with a dry run first, then by diffing `pip freeze` before and after: 18 packages added (streamlit, plotly, altair, pydeck, starlette, uvicorn and their dependencies), **no existing pin changed or removed**,
`pip check` clean. As elsewhere in this repo only direct dependencies are pinned in `pyproject.toml`. The suite was run before the install and after it: 385 passed both times.

**Baseline.** Only the inputs were pinned (the snapshot; the state files, `chosen_configs.json` and the 40 checkpoints in the pre-registrations). The results (`eval_daily.parquet`, `tables/`, `data/processed/holdout/*`),
the posteriors, latents and feature files had no recorded hash and are gitignored, so "no frozen artifact was modified" (DASHBOARD.md §2.2, §10) could not be checked. `dashboard/frozen_sources.sha256` now lists
the SHA-256 of all 532 files under `data/processed/` (except `tier2_smoke/` and `holdout_rehearsal/`, which are rehearsal output and not a source of anything shown) and `data/raw/snapshot_20261001/`, in `shasum` format,
written before any dashboard code existed. `make dashboard-verify-frozen` checks it. It agrees with the 46 hashes of `preregistration_holdout.md`, the 5 of `preregistration_tier2.md`, the snapshot manifest and the Tier 1
pre-registration (whose original `states/O1.parquet` hash differs, as D-033 and its Amendment 1 record). `reports/logs/holdout_access.jsonl` is append-only and the pre-registrations are tracked by git; neither is in the baseline.
The baseline is evidence from 2026-10-06 onward only: it cannot show that nothing changed between the runs and today.

### D-046 · Dashboard D1: static pages from stored results, the artifact stage, and three more spec mismatches
**Date:** 2026-10-06 · **Status:** implemented; milestone D1 of DASHBOARD.md §8 · choices mine unless stated

**What the pages read.** `scripts/10_dashboard_data.py` (`make dashboard-data`) has one stage so far, "stored": it copies the stored tables of the test split and the holdout,
the Tier 1 gates and the frozen configurations **byte for byte** into `dashboard/artifacts/` (40 files, 388 KB, committed), after checking each source against the D-045 baseline
and refusing on any difference. A copy's SHA-256 in `manifest.json` therefore equals the baseline hash of its source, which a test asserts. Three files are derived and are not
statistics: the 40 final runs' learning curves in one CSV; `facts.json` (configuration values and table summaries the page copy quotes, so no page hard-codes a research number);
`report_check.json`. The manifest carries no timestamp or commit, so rebuilding unchanged inputs is byte-identical. The stage reads no raw data, state file or model and never opens
the holdout; the replay and the live models (D2) will be separate, gated stages. Alternative: pages read `data/processed` directly; rejected because those files are gitignored and
the app would then show numbers nothing committed can vouch for.

**"Every number matches" is checked against the report, not against a copy of itself.** The headline tables the pages show (gate verdicts, all 24 paired differences, variants and
benchmarks, cost sensitivity, episodes, Tier 1 gates) are built from the artifacts with `final_report.py`'s own row builders and formatters, and each of the resulting 130 markdown
rows must be a line of the committed `reports/final_report.md`. The precompute fails otherwise, `make dashboard-check` repeats it without writing, and tests show that one changed
number or one changed verdict fails it. `final_report.py` is imported, not edited. The app verifies the artifacts against the manifest once per session and shows nothing if that fails.

**Page scripts are in `dashboard/views/`, not `pages/` (deviation from DASHBOARD.md §3).** A directory named `pages/` beside the main script switches on Streamlit's legacy page
lookup. Under `AppTest`, and plausibly on a cold server opened at a page URL, that ran the page script alone, without `app.py`: no artifact verification, no header, no holdout
status. Found because the header test failed on every page but the first. With `views/` the frame runs on every page (tested).

**Scope taken in D1.** Pages 2, 3, 6, 10, 11 are registered; pages 1, 4, 5, 7, 8, 9 appear when their milestones are built (no placeholder pages). Page 6 includes the action-map
interactive: it calls the environment's own `action_to_weights`, loads no model, and no later milestone names it. Page 10 shows the stored tables and the headline forest plot (Sharpe,
block 20, test beside holdout); its selectors, per-seed plots, equity curves and cost slider stay in D4. The Verdict page offers the reports as downloads, since a local app cannot link to repository files.

**Presentation.** Stored tables are static tables (every row visible, text wrapped), not scrolling grids: the twelve-row difference table was hiding two rows. Colours: variants
V1/V2/V4/C4 = blue/orange/aqua/violet in both themes, validated for colour-vision deficiency on adjacent pairs; no four-hue set passes for every pair, so C4 is also dashed with a
diamond marker. Benchmarks are grey. The two windows are not hues: holdout filled in the primary ink, test split open in grey. No theme is forced. `.streamlit/config.toml` binds
the server to localhost (D-043: local only; the default serves the LAN) and turns Streamlit's usage statistics off.

**Three more places where DASHBOARD.md does not match the stored results** (the pages show the stored values):
1. §7 page 11 gives the power as "CIs about ±0.35 Sharpe". That was the pre-registered expectation. The stored half-widths are ±0.35 to ±0.39 on the test split and ±0.45 to ±0.51 on the
   shorter holdout; the Verdict page states both.
2. §1 says the agents lose to the benchmarks "mainly because" of turnover. The stored cost table supports that on the test split (at 0 bps the best variant's mean Sharpe, 0.89, is above
   equal weight, 60/40 and risk parity; at 5 bps it is not) but not on the holdout, where the best variant at 0 bps (1.19) is below all three. The Agent page says costs account for part
   of the gap, not necessarily all of it, and prints which benchmarks the best variant's mean exceeds at 0 and 5 bps for the selected window, labelled as untested point estimates.
3. The minimum-variance benchmark's deflated Sharpe on the holdout is 0.9487. At two decimals it prints as 0.95, the bar. The deflated-Sharpe table uses three decimals.

**Tests.** `tests/test_dashboard_data.py` (17; needs no Streamlit) and `tests/test_dashboard_app.py` (40; `AppTest` on the real app, skipped as a module when the `dashboard` group
is absent, as `test_holdout.py` already does for gymnasium). They cover: copies are the frozen bytes; a modified source, artifact or manifest is refused; report rows; every page
renders with title, takeaway and "How to read this", in under 1 s from cache (measured 0.01 to 0.06 s; cold start 1.1 s); the stored-results and holdout-spent badges on every page;
the forbidden-wording list of §6 over page text and chart text; the permanent weights caption exactly where a page shows weights (none yet); the Results tables equal the report's rows;
the sign-flip sentence; the Tier 1 V4-vs-V2 pass is shown (D-044 item 5). Suite: 442 passed with the dashboard group installed (385 before).

### D-047 · Dashboard D2: Amendment 1, the replay, fold parameters and the frozen models; what is and is not gated
**Date:** 2026-10-06 · **Decided by:** principal investigator (work on through the milestones and decide; the holdout replay stays theirs to run); the choices below are mine · **Status:** ungated stages run and verified; **the gated holdout stage has NOT been run**

**Amendment 1** was appended to `reports/tables/preregistration_holdout.md` and committed alone (`c14c9f4`) before any replay code existed. It covers the one-off descriptive replay and the
live demo's re-read of holdout-period inputs (D-044 items 1-2), states that all holdout output had been seen, and fixes the 1e-9 gates. The 46 pinned hashes still parse.

**Stages of `scripts/10_dashboard_data.py`.** `stored`, `derived`, `replay-test`, `folds-test` are not gated and are what `make dashboard-data` runs; none reads a holdout row from the raw
data. `holdout-replay` is gated: it checks `--i-am-sure` and `PRISM_ALLOW_HOLDOUT=1` before reading any file, then that the amendment is committed and unmodified and all 46 pinned files match,
logs the access through `load_holdout`, and writes only under `dashboard/artifacts/` and `data/live/`. The command, to be typed by hand:

    PRISM_ALLOW_HOLDOUT=1 .venv/bin/python scripts/10_dashboard_data.py --stage holdout-replay --i-am-sure

It takes about seven minutes, adds one line to `reports/logs/holdout_access.jsonl`, and writes `weights/holdout_*.parquet`, `regimes/hmm_folds_holdout.csv`, `live/models_manifest.json`,
`live/models_check.json` and `data/live/models/`. **The same code path was run on the test split, which is its rehearsal** (below); only the window differs.

**Results of the ungated stages (2026-10-06).**
* `replay-test`: all 184 daily series (40 agents and 6 benchmarks at 0/5/10/20 bps) equal `data/processed/tier2/eval_daily.parquet` with maximum difference 0.0; the mean weights and
  turnover equal those `eval_done.json` recorded. 255 decisions, 2019-01-04 to 2023-11-17. Weekly target weights, pre-trade (drifted to the execution close) weights, turnover and cost are in
  `weights/test_agents.parquet`; the benchmarks' targets in `weights/test_benchmarks.parquet`.
* `folds-test`: the walk-forward re-run to 2023-12-31 reproduces the four stored state files (V1, V2, C4 exactly; V4 to 1.8e-11); 204 HMM folds' parameters in return units in
  `regimes/hmm_folds_test.csv`. Rehearsal of the frozen models (HMM fold 203, encoder fold 16, written to `data/live/models_rehearsal/`): from the persisted parameters alone they reproduce the
  stored states on their own apply windows to 1.1e-12 (HMM), exactly (encoder, threshold, state scaler).
* `derived`: per-seed deflated Sharpe (medians, minima and counts equal `dsr.csv`); 46 equity curves per window end at the stored annualised return; the regime series equal
  `hmm_posteriors.parquet` (2.1e-11) and the C2 states (exactly); the latent series equal `encoder_latents.parquet` exactly.

**Choices.**
1. **`holdout.py` gains four output fields** on `ExtendedStates` (the two walk-forward results, the train-split scaler, the Universe-A features), resolving D-044 item 7. Additive; no value
   computed changes, and the re-run reproducing the stored states is the evidence. Alternative: repeat its sixty lines in the precompute; rejected as two copies of the pipeline that could drift.
2. **Replay inputs are the stored ones**: the Tier 2 state files for the test split, `states_extended.parquet` for the holdout, each the exact frame the stored evaluation ran on. The fresh
   walk-forward is used for its fold objects and must match those frames to 1e-9.
3. **Pre-trade weights are captured through the environment's public `drifted_weights`**, by a recording wrapper around the policy; no environment or agent code is edited.
4. **The frozen models are plain data**: `models.json` (HMM parameters in canonical order, its filter state at the fold's last day, the three scalers, the threshold edges, the encoder's
   architecture) and `encoder.npz` (weights), both hashed in a manifest whose copy is committed. No pickle. `prism.live` rebuilds a `GaussianHMM` and the encoder from them and reuses
   `filtered_posteriors` and `compute_latents`.
5. **The live encoder runs the fold's whole fit window through as well as the new rows.** Encoding only the last windows is the same arithmetic in different float32 batches and differed from
   the stored latents by 3e-8, above the 1e-9 gate; batching as the walk-forward did reproduces them exactly. Found by the synthetic test.
6. **The latent map is one PCA per encoder fold** (D-044 item 8). The page shows a year at a time and says that years are not comparable.
7. **The HMM-versus-VIX agreement is a count, reported by window, and it is lower than DASHBOARD.md §7 implies.** The HMM's call (P(Volatile) > 0.5) and the VIX-threshold state agree on 80.8% of
   days overall: 86.7% on train, 88.1% on validation, 70.3% on test, 72.9% on the holdout. The disagreement is one-sided: when the HMM says Volatile the VIX is high 93% of the time (1266 of 1359
   days), but the threshold says "high" far more often (43% of days against 28%). The page says this and does not call the two "largely the same thing" without the numbers.
8. **The SPY price line** is the snapshot's adjusted close to the test split's end joined to the stored holdout SPY series, indexed to 100 (`regimes/spy.parquet`, about 7 000 values). It is a
   derived extract of the raw snapshot committed for display; nothing else from the snapshot is.
9. **File layout.** The model stages are in `src/prism/dashboard_replay.py` (heavy imports), the model forward passes in `src/prism/live.py`, the stored and derived stages in
   `src/prism/dashboard_data.py`, which the pages import and which therefore stays light. `data/live/` is gitignored.

**Tests** (`tests/test_dashboard_replay.py`, 16, plus 8 more in `test_dashboard_data.py`): the gated stage refuses without either key and logs nothing; the amendment is appended and the hashes
parse; one series over tolerance, a NaN or another index fails the replay; the recorded weights are valid (sum to one, cap, start in cash, turnover consistent) and equal what the stored
evaluation summarised; fold parameters map to return units; on a synthetic walk-forward the persisted last fold reproduces the stored states, continuing the filter equals filtering the whole
history, the forward passes do not read the future, and a changed file or parameter is refused.
