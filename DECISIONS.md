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

## Gate decisions (Tier 1)

*Empty until Step 4a. Per §13.1, a component that does not beat its own
control with non-overlapping confidence intervals is redesigned or dropped
before Phase B, and the decision is recorded here **either way**.*
