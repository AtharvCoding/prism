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

## Gate decisions (Tier 1)

*Empty until Step 4a. Per §13.1, a component that does not beat its own
control with non-overlapping confidence intervals is redesigned or dropped
before Phase B, and the decision is recorded here **either way**.*
