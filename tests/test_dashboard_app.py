"""The dashboard app: every page renders offline and fast, carries its fixed furniture, and shows the stored numbers.

DASHBOARD.md §6, §7, §8 (milestone D1), §10. Runs the real app through ``streamlit.testing.v1.AppTest`` on the
committed artifacts. Skipped when the optional ``dashboard`` dependency group is not installed.
"""

from __future__ import annotations

import re
import sys
import time
from datetime import date
from pathlib import Path

import pytest

pytest.importorskip("streamlit")
pytest.importorskip("plotly")

from streamlit.testing.v1 import AppTest  # noqa: E402

from prism import dashboard_data as dd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
APP = ROOT / "dashboard" / "app.py"
sys.path.insert(0, str(APP.parent))

from components import alloc, charts, theme, ui  # noqa: E402
from components.registry import PAGES  # noqa: E402

ARTIFACTS = ROOT / dd.ARTIFACTS
#: DASHBOARD.md §6: wording no page or chart may use about an agent. ("Buy and hold SPY" is a benchmark's name.)
FORBIDDEN = re.compile(r"outperform|\balpha\b|signal strength|\b(buy|sell)\b(?! and hold)|price target|will (rise|fall|beat)", re.I)


def _open(page) -> AppTest:  # noqa: ANN001
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    if page is not PAGES[0]:
        at.switch_page(page.script)
        at.run()
    return at


@pytest.fixture(scope="module")
def rendered() -> dict[str, AppTest]:
    return {p.key: _open(p) for p in PAGES}


def _texts(at: AppTest) -> list[str]:
    out = [e.value for kind in ("title", "header", "subheader", "markdown", "caption") for e in getattr(at, kind)]
    out += [f"{m.label} {m.value}" for m in at.metric] + [e.label for e in at.expander] + [b.label for b in at.button]
    out += [t.value.to_string() for t in at.table]
    return [str(t) for t in out]


# --------------------------------------------------------------------------- every page
@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_the_page_renders_with_its_title_takeaway_and_how_to_read(rendered, page):
    at = rendered[page.key]
    assert not at.exception and not at.error
    assert [t.value for t in at.title] == [page.title]
    takeaway = ui.takeaway_text(page.key)
    assert "{" not in takeaway and f"**{takeaway}**" in [m.value for m in at.markdown]
    assert ui.HOW_TO_READ in [e.label for e in at.expander]


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_the_page_renders_from_cache_in_under_a_second(rendered, page):
    at = rendered[page.key]
    start = time.perf_counter()
    at.run()
    assert time.perf_counter() - start < 1.0 and not at.exception


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_the_page_says_the_results_are_stored_and_the_holdout_is_spent(rendered, page):
    header = " ".join(m.value for m in rendered[page.key].markdown)
    assert "Stored results" in header and "Holdout spent: evaluated once, " + dd_facts()["windows"]["holdout"]["evaluated_utc"][:10] in header


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_no_page_uses_wording_that_reads_as_a_claim_or_as_advice(rendered, page):
    hits = [(m.group(0), t[:80]) for t in _texts(rendered[page.key]) for m in [FORBIDDEN.search(t)] if m]
    assert not hits


@pytest.mark.parametrize("page", PAGES, ids=lambda p: p.key)
def test_a_page_that_shows_weights_carries_the_permanent_caption(rendered, page):
    captions = [c.value for c in rendered[page.key].caption]
    assert (f"*{ui.WEIGHTS_CAPTION}*" in captions) == page.shows_weights


def test_the_fixed_wording_is_the_specifications():
    assert ui.WEIGHTS_CAPTION == ("Demonstration of the frozen system. Not a recommendation; the agents did not beat these "
                                  "benchmarks after costs.")
    assert ui.FRAMING == "We built the full system end to end, and tested it rigorously enough to know what it does and doesn't do."


def test_the_about_dialog_opens_from_the_header_of_any_page():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    [button] = [b for b in at.button if b.label == "What this is / isn't"]
    button.click().run()
    text = " ".join(m.value for m in at.markdown)
    assert not at.exception and "**What it isn't.** Not a recommendation, not a forecast and not a trading system." in text
    assert "evaluated exactly once" in text and not FORBIDDEN.search(text)


def test_the_app_stops_without_showing_anything_if_an_artifact_was_changed(tmp_path, monkeypatch):
    import shutil

    from components import data

    broken = tmp_path / "artifacts"
    shutil.copytree(ARTIFACTS, broken)
    table = broken / "holdout" / "tables" / "claims.csv"
    table.write_text(table.read_text().replace("False", "True"))
    monkeypatch.setattr(data, "ARTIFACTS", broken)
    data.manifest.clear()
    try:
        at = AppTest.from_file(str(APP), default_timeout=60).run()
        assert len(at.error) == 1 and "could not be verified" in at.error[0].value
        assert not at.title and not at.table and not at.metric
    finally:
        data.manifest.clear()


# --------------------------------------------------------------------------- Results and Verdict: the stored numbers
def dd_facts() -> dict:
    import json

    return json.loads((ARTIFACTS / "facts.json").read_text())


def test_the_results_page_shows_the_final_reports_rows_for_both_windows(rendered):
    at = rendered["results"]
    tables = [t.value for t in at.table]
    assert [t.label for t in at.tabs] == ["Holdout (confirmatory, one use)", "Test split (exploratory)"]
    tables = [t for t in tables if "comparison" in t.columns or "candidate" in t.columns or "variant" in t.columns
              or "strategy" in t.columns or t.index.name == "strategy"]
    for i, window in enumerate(("holdout", "test")):
        gates, differences, variants, costs, dsr = tables[5 * i: 5 * i + 5]
        res = dd.load_results(ARTIFACTS, window)
        stored = dd.gate_table(res)
        assert gates.drop(columns="claimable").equals(stored.drop(columns="claimable"))
        assert gates.claimable.tolist() == ["no" if not c else "yes" for c in stored.claimable]
        assert differences.equals(dd.difference_table(res))                     # 12 rows: every comparison on every metric
        assert variants.drop(columns="variant").equals(dd.variant_table(res).drop(columns="variant"))
        assert (costs.to_numpy() == dd.cost_table(res).to_numpy()).all()
        assert dsr["median (40 trials)"].tolist()[:4] == res["dsr"]["dsr_median_n40"].tolist()[:4]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Comparisons passed, holdout"] == "0 of 3" and metrics["Comparisons passed, test split"] == "0 of 3"
    assert metrics["Results that can be claimed"] == "0"


def test_the_results_page_names_the_comparisons_whose_sign_flipped(rendered):
    flips = dd_facts()["sign_flips"]["sharpe"]
    flipped = ", ".join(f"{f['candidate']} vs {f['control']}" for f in flips if f["flipped"])
    text = " ".join(m.value for m in rendered["results"].markdown)
    assert f"For {sum(f['flipped'] for f in flips)} of the 3 comparisons ({flipped})" in text and "**changed sign**" in text


def test_results_and_verdict_carry_the_verified_footer_and_the_other_pages_do_not(rendered):
    for page in PAGES:
        footers = [c.value for c in rendered[page.key].caption if "All numbers on this page are read from the source tables" in c.value]
        assert len(footers) == (1 if page.key in ("results", "verdict") else 0)


def test_the_verdict_page_states_the_framing_and_each_stored_sharpe_difference(rendered):
    at = rendered["verdict"]
    text = " ".join(m.value for m in at.markdown)
    assert f"> *{ui.FRAMING}*" in [m.value for m in at.markdown]
    for window, label in (("test", "Test split"), ("holdout", "Holdout")):
        d = dd.difference_table(dd.load_results(ARTIFACTS, window))
        for cell in d[d.metric == "Sharpe"]["difference [95% CI]"]:
            assert f"{label}: **FAIL** · Sharpe difference {cell}" in text
    assert "does not show" in text.lower() and "fresh data and a fresh plan" in text


def test_the_question_page_reports_that_tier_1s_research_comparison_passed(rendered):
    """D-044 item 5: the Tier 1 V4-vs-V2 pass is shown, not folded into 'Tier 1 agreed with the null'."""
    at = rendered["question"]
    tier1 = at.table[0].value
    row = tier1[tier1.comparison == "V4 vs V2"].iloc[0]
    assert (row.favourable, row.adverse, row.result) == ("4/4", "0/4", "pass")
    assert any("did pass at this stage" in c.value and "could not be credited to the HMM" in c.value for c in at.caption)


def test_the_agent_page_action_map_and_window_switch_work():
    at = _open(next(p for p in PAGES if p.key == "agent"))
    [cash] = [s for s in at.slider if s.key == "action_CASH"]
    cash.set_value(1.0).run()
    assert not at.exception
    at.button_group[0].set_value("test").run()
    text = " ".join(m.value for m in at.markdown)
    f = dd_facts()["windows"]["test"]
    assert not at.exception and f"**{ui.pct_range(f['turnover_agents'])}**" in text


# --------------------------------------------------------------------------- components
def test_freshness_label_states_the_state_the_date_the_age_and_the_model_age():
    label = ui.freshness_label("stale", date(2026, 10, 2), today=date(2026, 10, 6), models_refit=date(2026, 7, 27))
    assert label == "Stale · as of 2026-10-02 · 4 days old · models last refit 2026-07-27"
    assert ui.freshness_label("live", date(2026, 10, 6), today=date(2026, 10, 6)) == "Live · as of 2026-10-06 · today"
    assert set(ui.FRESHNESS) == {"stored", "live", "cached", "stale", "seam-check-failed"}


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_each_variant_has_its_own_colour_and_the_control_a_second_encoding(mode):
    c = theme.palette(mode)
    assert set(c["variants"]) == set(theme.VARIANTS) and len(set(c["variants"].values())) == 4
    assert c["context"] not in c["variants"].values() and len(c["ordinal"]) == 4
    assert theme.VARIANT_DASH["C4"] != theme.VARIANT_DASH["V4"] and len(set(theme.VARIANT_SYMBOL.values())) == 4


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_forest_plot_draws_the_stored_differences_and_intervals(mode):
    c = theme.palette(mode)
    test, hold = (dd.load_results(ARTIFACTS, w)["paired_block20"] for w in ("test", "holdout"))
    fig = charts.forest(test, hold, "sharpe", c)
    assert [t.name for t in fig.data] == ["Test split (exploratory)", "Holdout (confirmatory, one use)"]
    for trace, table in zip(fig.data, (test, hold)):
        rows = [table[(table.candidate == x) & (table.control == y) & (table.metric == "sharpe")].iloc[0] for x, y in charts.COMPARISON_LABEL]
        assert list(trace.x) == [r["diff"] for r in rows]
        assert list(trace.x + trace.error_x.array) == pytest.approx([r["ci_high"] for r in rows])
        assert list(trace.x - trace.error_x.arrayminus) == pytest.approx([r["ci_low"] for r in rows])
    assert fig.data[0].marker.symbol != fig.data[1].marker.symbol          # the window is never colour alone
    assert not FORBIDDEN.search(fig.to_json())


def test_no_chart_text_uses_forbidden_wording():
    c, f = theme.palette("light"), dd_facts()
    res = dd.load_results(ARTIFACTS, "holdout")
    import numpy as np
    import pandas as pd

    curves = pd.read_csv(ARTIFACTS / "tier2" / "learning_curves.csv").groupby(["variant", "step"], as_index=False).mean(numeric_only=True)
    figs = [charts.timeline(f, c), charts.turnover_bars(res["turnover"], c), charts.cost_lines(res["cost_sensitivity"], c),
            charts.learning_curves(curves, c), charts.action_weights(["A", "CASH"], np.array([0.35, 0.65]), 0.35, c)]
    for fig in figs:
        assert not FORBIDDEN.search(fig.to_json())
    assert len(charts.cost_lines(res["cost_sensitivity"], c).data) == 10       # six benchmarks in grey, four variants


# --------------------------------------------------------------------------- D3: regimes and latent
def test_the_regimes_page_shows_the_stored_agreement_and_the_latest_stored_fold(rendered):
    import json

    import pandas as pd

    at = rendered["regimes"]
    summary = json.loads((ARTIFACTS / "regimes" / "summary.json").read_text())["agreement"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Days the two agree"] == ui.pct(summary["all"]["agreement"], 1)
    table = next(t.value for t in at.table if "Agree" in t.value.columns)
    assert table.Period.tolist() == ["Train", "Validation", "Test", "Holdout", "All days"]
    assert table.Agree.tolist() == [ui.pct(summary[k]["agreement"], 1) for k in ("train", "val", "test", "holdout", "all")]
    folds = pd.read_csv(ARTIFACTS / "regimes" / "hmm_folds_test.csv")
    last = folds.iloc[-1]
    matrix = next(t.value for t in at.table if "to Volatile" in t.value.columns)
    assert matrix.to_numpy().tolist() == [[last.p_00, last.p_01], [last.p_10, last.p_11]]
    assert metrics["Expected stay in Calm"] == f"{last.dwell_0:.0f} trading days"
    assert "Why two states?" in [e.label for e in at.expander]
    text = " ".join(m.value for m in at.markdown)
    assert "K = 3 scored 1.52 points higher" in text and "K = 2 has the better BIC" in text


def test_the_latent_page_states_that_years_are_not_comparable_and_reports_the_tier_1_control(rendered):
    at = rendered["latent"]
    assert any("Positions are not comparable between years" in i.value for i in at.info)
    text = " ".join(m.value for m in at.markdown)
    assert "reliably better\non **1/4** and reliably worse on **0/4**" not in text      # the stored row is 1/4 favourable, 1/4 adverse
    assert "**1/4** and reliably worse on **1/4**" in text and "**failed**" in text
    at.select_slider[0].set_value(2022).run()
    assert not at.exception and any("trading day of 2022" in c.value for c in at.caption)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_regime_charts_shade_each_day_once_and_never_use_forbidden_wording(mode):
    import pandas as pd

    c = theme.palette(mode)
    regimes = pd.read_parquet(ARTIFACTS / "regimes" / "daily.parquet")
    spy = pd.read_parquet(ARTIFACTS / "regimes" / "spy.parquet")["spy_index"]
    f = dd_facts()
    fig = charts.regime_price(spy, regimes, f["universe"]["fit_early"], None, c)
    runs = charts._runs(regimes.p_volatile > 0.5)
    assert sum((b - a).days + 1 for a, b, _ in runs) >= len(regimes) and len(fig.layout.shapes) == len(runs) + 1
    assert all(x[2] != y[2] and x[1] < y[0] for x, y in zip(runs, runs[1:]))          # runs alternate and do not overlap
    strips = charts.regime_strips(regimes, c)
    assert strips.data[0].z.shape == (2, regimes.vix_high.notna().sum())
    coords = pd.read_parquet(ARTIFACTS / "latents" / "pca.parquet")
    one = coords[coords.fold == coords.fold.max()]
    lm = charts.latent_map(one, c)
    assert len(lm.frames) == one.index.to_period("M").nunique() and len(lm.data[0].x) == len(one)
    for figure in (fig, strips, lm, charts.fold_volatility(pd.read_csv(ARTIFACTS / "regimes" / "hmm_folds_test.csv", parse_dates=["apply_start"]), c),
                   charts.k_selection(pd.read_csv(ARTIFACTS / "regimes" / "k_selection.csv"), 2.0, c)):
        assert not FORBIDDEN.search(figure.to_json())


# --------------------------------------------------------------------------- D4: allocation through time, results interactivity
@pytest.fixture(scope="module")
def recorded_weights():
    import pandas as pd

    return (pd.read_parquet(ARTIFACTS / "weights" / "test_agents.parquet"), pd.read_parquet(ARTIFACTS / "weights" / "test_benchmarks.parquet"))


def test_an_ensemble_is_the_seed_average_of_the_recorded_weights_and_sleeves_add_up(recorded_weights):
    import numpy as np

    agents, bench = recorded_weights
    u = dd_facts()["universe"]
    w = alloc.strategy_weights(agents, bench, "V4 ensemble")
    week = w["mean"].index[100]
    rows = agents[(agents.variant == "V4") & (agents.decision_date == week)]
    for line in alloc.lines(u):
        assert w["mean"].loc[week, line] == pytest.approx(rows[f"w_{line}"].mean(), abs=1e-15)
        assert w["low"].loc[week, line] == rows[f"w_{line}"].min() and w["high"].loc[week, line] == rows[f"w_{line}"].max()
    sl = alloc.sleeves(w["mean"], u)
    assert list(sl.columns) == ["Equity", "Bonds", "Gold", "Cash"] and np.allclose(sl.sum(axis=1), 1.0)
    assert sl.loc[week, "Equity"] == pytest.approx(w["mean"].loc[week, u["sectors"]].sum())
    assert np.allclose(alloc.defensive_share(w["mean"], u), 1.0 - sl["Equity"])
    sixty = alloc.sleeves(alloc.strategy_weights(agents, bench, "60/40")["mean"], u)
    assert np.allclose(sixty["Equity"], 0.6) and np.allclose(sixty["Bonds"], 0.4)       # the S&P 500 fund counts as equity
    assert alloc.strategies(["V1", "V2", "V4", "C4"])[0] == "V4 ensemble"


def test_the_allocation_page_shows_the_recorded_weights_for_the_selected_week(recorded_weights):
    """D4 acceptance: selected-week weights equal the stored replay."""
    agents, _ = recorded_weights
    at = _open(next(p for p in PAGES if p.key == "allocation"))
    dates = sorted(agents.decision_date.unique())
    week = dates[57]
    at.select_slider[0].set_value(week).run()
    assert not at.exception
    table = next(t.value for t in at.table if "target weight" in t.value.columns)
    rows = agents[(agents.variant == "V4") & (agents.decision_date == week)]
    for line in table.index:
        assert table.loc[line, "target weight"] == pytest.approx(rows[f"w_{line}"].mean(), abs=1e-15)
        assert table.loc[line, "lowest seed"] == rows[f"w_{line}"].min()
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Turnover that week"] == ui.pct(rows.turnover.mean(), 1)
    assert f"*{ui.WEIGHTS_CAPTION}*" in [c.value for c in at.caption]
    at.selectbox[0].set_value("Risk parity").run()
    assert not at.exception and {m.label: m.value for m in at.metric}["Turnover that week"] == "n/a"


def test_the_cost_slider_is_exact_at_the_stored_levels_and_linear_between_them():
    """D4 acceptance: the slider's endpoints (and every stored level) equal the stored numbers."""
    import pandas as pd

    for window in ("test", "holdout"):
        cost = pd.read_csv(ARTIFACTS / window / "tables" / "cost_sensitivity.csv")
        for bps in sorted(cost.bps.unique()):
            got = alloc.interpolate_costs(cost, bps)
            stored = cost[cost.bps == bps].set_index("strategy").sharpe
            assert (got[stored.index] == stored).all() and got.is_monotonic_decreasing
        mid = alloc.interpolate_costs(cost, 7.5)
        lo, hi = (cost[cost.bps == b].set_index("strategy").sharpe for b in (5.0, 10.0))
        assert mid["V4"] == pytest.approx((lo["V4"] + hi["V4"]) / 2)
        with pytest.raises(ValueError):
            alloc.interpolate_costs(cost, 25.0)


def test_equity_curves_are_the_cumulated_stored_returns():
    import pandas as pd

    daily = pd.read_parquet(ARTIFACTS / "holdout" / "eval_daily.parquet")
    curves = alloc.equity_curves(daily, ["V4|s3", "BM|EqualWeight"], 5.0)
    assert curves["V4|s3"].iloc[-1] == pytest.approx((1 + daily["V4|s3|5"]).prod()) and list(curves.columns) == ["V4|s3", "BM|EqualWeight"]


def test_the_results_selectors_draw_the_stored_tables_and_the_slider_labels_interpolation():
    import pandas as pd

    at = _open(next(p for p in PAGES if p.key == "results"))
    groups = {g.key: g for g in at.button_group}
    groups["forest_metric"].set_value("max_drawdown").run()
    groups = {g.key: g for g in at.button_group}
    groups["forest_block"].set_value(40).run()
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    test, hold = (pd.read_csv(ARTIFACTS / w / "tables" / "paired_block40.csv") for w in ("test", "holdout"))
    flips = dd.sign_flips(test, hold, "max_drawdown")
    crossing = sum(int(r.ci_low < 0 < r.ci_high) for t in (test, hold) for r in t[t.metric == "max_drawdown"].itertuples())
    assert f"**{crossing} of the 6 intervals shown cross zero.**" in text and "difference in maximum drawdown" in text
    assert f"For {int(flips.flipped.sum())} of the 3 comparisons" in text
    at.slider[0].set_value(7.5).run()
    captions = " ".join(c.value for c in at.caption)
    assert not at.exception and "**Interpolated**: 7.5 basis points is between two stored cost levels" in captions
    at.slider[0].set_value(20.0).run()
    assert "Stored values at 20 basis points." in " ".join(c.value for c in at.caption)


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_d4_charts_draw_what_they_are_given(mode, recorded_weights):
    import numpy as np
    import pandas as pd

    c, u = theme.palette(mode), dd_facts()["universe"]
    agents, bench = recorded_weights
    w = alloc.strategy_weights(agents, bench, "V4 ensemble")
    sl = alloc.sleeves(w["mean"], u)
    p = pd.read_parquet(ARTIFACTS / "regimes" / "daily.parquet").p_volatile.reindex(sl.index)
    area = charts.allocation_area(sl, p, None, c)
    assert [t.name for t in area.data[1:]] == list(theme.SLEEVES) and list(area.data[1].y) == sl["Equity"].tolist()
    assert area.data[0].type == "heatmap"                                    # the regime is a strip above, never behind the sleeves
    week = sl.index[-1]
    bar = charts.weights_bar(w["mean"].loc[week, alloc.lines(u)], w["low"].loc[week, alloc.lines(u)], w["high"].loc[week, alloc.lines(u)],
                             alloc.sleeve_of(u), 0.35, c)
    assert sum(len(t.x) for t in bar.data) == 14
    daily = pd.read_parquet(ARTIFACTS / "holdout" / "eval_daily.parquet")
    names = [f"V4|s{s}" for s in range(10)] + ["BM|EqualWeight"]
    eq = charts.equity_chart(alloc.equity_curves(daily, names, 5.0), ["V4"], list(range(10)), ["EqualWeight"], c)
    median = alloc.equity_curves(daily, names, 5.0)[[f"V4|s{s}" for s in range(10)]].median(axis=1)
    assert np.allclose(eq.data[-1].y, median)
    strip = charts.seed_strip({v: np.linspace(0.1, 0.9, 10) for v in theme.VARIANTS}, {"BM|EqualWeight": 0.8}, "x", c, bar=0.95)
    assert len(strip.data) == 5 and all(len(t.y) == 10 for t in strip.data[:4])
    rank = charts.cost_rank(alloc.interpolate_costs(pd.read_csv(ARTIFACTS / "holdout" / "tables" / "cost_sensitivity.csv"), 5.0), c)
    assert list(rank.data[0].x) == sorted(rank.data[0].x)
    for fig in (area, bar, eq, strip, rank, charts.size_bars({"V1": 0.3}, {"V4 vs V2": 0.1}, c), charts.defensive_scatter(1 - sl["Equity"], p, c),
                charts.duration_ladder(w["mean"], u["bonds"], None, c)):
        assert not FORBIDDEN.search(fig.to_json())


# --------------------------------------------------------------------------- D5/D6: live weights, home, what-if
def _page(key):  # noqa: ANN001, ANN202
    return next(p for p in PAGES if p.key == key)


def test_home_leads_with_the_framing_and_the_null_and_is_the_first_page(rendered):
    at = rendered["home"]
    assert PAGES[0].key == "home" and f"> *{ui.FRAMING}*" in [m.value for m in at.markdown]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Comparisons passed, holdout"] == "0 of 3" and metrics["Comparisons passed, test split"] == "0 of 3"
    text = " ".join(m.value for m in at.markdown)
    assert "no detectable benefit" in text and "did not beat" in text


def test_without_the_frozen_models_the_live_page_shows_recorded_weights_and_says_so(rendered, recorded_weights):
    """No gated replay has been run in the repository: nothing is fetched, the button is off, the source is named."""
    agents, _ = recorded_weights
    at = rendered["live"]
    last = agents.decision_date.max()
    assert not (ARTIFACTS / "live" / "models_manifest.json").exists()
    captions = " ".join(c.value for c in at.caption)
    assert "Not live." in captions and f"{last:%d %b %Y}" in captions and "gated holdout replay" in captions
    [refresh] = [b for b in at.button if b.label == "Refresh"]
    assert refresh.disabled
    table = next(t.value for t in at.table if "weight" in t.value.columns)
    rows = agents[(agents.variant == "V4") & (agents.decision_date == last)]
    for line in table.index:
        assert table.loc[line, "weight"] == pytest.approx(rows[f"w_{line}"].mean(), abs=1e-15)
        assert table.loc[line, "highest seed"] == rows[f"w_{line}"].max()
    assert {m.label: m.value for m in at.metric}["Turnover from last week"] == ui.pct(rows.turnover.mean(), 1)
    assert not any("not a decision" in w.value for w in at.warning)


def test_the_live_strategy_toggle_shows_the_benchmarks_by_their_definitions():
    at = _open(_page("live"))
    {g.key: g for g in at.button_group}["live_strategy"].set_value("60/40").run()
    table = next(t.value for t in at.table if "weight" in t.value.columns)
    assert not at.exception and table.loc["SPY", "weight"] == pytest.approx(0.6) and table.loc["IEF", "weight"] == pytest.approx(0.4)
    assert {m.label: m.value for m in at.metric}["Equity"] == "60.0%"


def _fake_cache(monkeypatch, *, preview: bool):
    """A live cache built from recorded weights, to exercise the live branches without models or network."""
    import json

    import pandas as pd

    from components import current, data

    agents = pd.read_parquet(ARTIFACTS / "weights" / "test_agents.parquet")
    bench = pd.read_parquet(ARTIFACTS / "weights" / "test_benchmarks.parquet")
    seen = json.loads((ARTIFACTS / "weights" / "test_last_observations.json").read_text())
    dates = sorted(agents.decision_date.unique())
    drop = ["decision_date", "execution_date", "cost"]
    cache = {
        "as_of": "2023-11-22" if preview else str(dates[-1].date()), "decision_date": str(dates[-1].date()), "is_preview": preview, "window": "test",
        "models": {"end": "2023-10-31", "hmm_fit_end": "2023-08-24", "encoder_fit_end": "2022-11-23"},
        "decision": agents[agents.decision_date == dates[-1]].drop(columns=drop),
        "preview": agents[agents.decision_date == dates[-2]].drop(columns=drop) if preview else None,
        "benchmarks": bench[(bench.decision_date == dates[-1]) & bench.benchmark.isin(["EqualWeight", "SixtyForty", "RiskParity"])].drop(columns=["decision_date", "turnover", "cost"]),
        "columns": seen["columns"], "observations": seen["observations"], "latent": [0.1] * 32,
        "p_volatile": {"2023-11-16": 0.2, "2023-11-17": 0.4, "2023-11-22": 0.9},
    }
    monkeypatch.setattr(data, "live_models_manifest", lambda: {"end": "2023-10-31"})
    monkeypatch.setattr(current.live, "read_cache", lambda _dir: cache)
    return cache


def test_a_cached_live_result_is_badged_and_a_mid_week_preview_is_never_called_a_decision(monkeypatch):
    _fake_cache(monkeypatch, preview=True)
    at = _open(_page("live"))
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Cached · as of 2023-11-22" in text and "models last refit 2023-08-24" in text
    assert [s.value for s in at.subheader][:1] == ["The weekly decision of 17 Nov 2023"]
    assert "Preview as of the close of 22 Nov 2023" in [s.value for s in at.subheader]
    assert any("This is **not a decision**" in w.value for w in at.warning)
    assert "**the live view does not refit anything**" in text
    [refresh] = [b for b in at.button if b.label == "Refresh"]
    assert not refresh.disabled


def test_a_failed_refresh_keeps_the_cached_result_and_turns_the_badge_stale(monkeypatch):
    from components import current

    _fake_cache(monkeypatch, preview=False)

    def offline(*args, **kwargs):  # noqa: ANN002, ANN003
        raise ConnectionError("no route to host")

    monkeypatch.setattr(current.live, "refresh", offline)
    at = _open(_page("live"))
    before = next(t.value for t in at.table if "weight" in t.value.columns).copy()
    [refresh] = [b for b in at.button if b.label == "Refresh"]
    refresh.click().run()
    assert not at.exception
    assert "Stale · as of 2023-11-17" in " ".join(m.value for m in at.markdown)
    assert any("ConnectionError: no route to host" in c.value and "last good result" in c.value for c in at.caption)
    assert next(t.value for t in at.table if "weight" in t.value.columns).equals(before)

    def seam(*args, **kwargs):  # noqa: ANN002, ANN003
        raise current.live.SeamError("data seam check failed for XLK")

    monkeypatch.setattr(current.live, "refresh", seam)
    at2 = _open(_page("live"))
    [refresh] = [b for b in at2.button if b.label == "Refresh"]
    refresh.click().run()
    assert "Data seam check failed · as of 2023-11-17" in " ".join(m.value for m in at2.markdown)


needs_agents = pytest.mark.skipif(not (ROOT / "data" / "processed" / "tier2" / "runs" / "final").exists(),
                                  reason="the 40 trained checkpoints are not in the repository (data/processed is not committed)")


@needs_agents
def test_the_what_if_lab_at_today_equals_the_live_weights_exactly(recorded_weights):
    """D6 acceptance: with the input untouched the lab shows the same weights as the Live page."""
    agents, _ = recorded_weights
    at = _open(_page("whatif"))
    assert not at.exception and any("**Showing the actual input, untouched.**" in c.value for c in at.caption)
    metrics = {m.label: (m.value, m.delta) for m in at.metric}
    last = agents.decision_date.max()
    rows = agents[(agents.variant == "V4") & (agents.decision_date == last)]
    u = dd_facts()["universe"]
    equity = sum(rows[f"w_{t}"].mean() for t in u["sectors"])
    assert metrics["Equity"] == (ui.pct(equity, 1), "+0.0 pp") and all(m[1] in ("+0.0 pp", "-0.0 pp") for k, m in metrics.items() if k in theme.SLEEVES)
    live_at = _open(_page("live"))
    assert {m.label: m.value for m in live_at.metric}["Equity"] == metrics["Equity"][0]


@needs_agents
def test_the_what_if_slider_moves_v4_and_the_page_compares_the_effect_with_seed_disagreement():
    at = _open(_page("whatif"))
    before = {m.label: m.value for m in at.metric}
    at.slider[0].set_value(1.0).run()
    after = {m.label: (m.value, m.delta) for m in at.metric}
    assert not at.exception and any("Showing P(Volatile) = 1.00" in c.value for c in at.caption)
    assert any(after[k][0] != before[k] for k in theme.SLEEVES)
    assert "Regime effect" in after and "Seed disagreement" in after and after["Regime effect"][0].endswith(" pp")
    text = " ".join(m.value for m in at.markdown) + " ".join(c.value for c in at.caption)
    assert "by construction the slider changes nothing for them" in text and "is a counterfactual" in text
    [today] = [b for b in at.button if b.label == "Today"]
    today.click().run()
    assert any("**Showing the actual input, untouched.**" in c.value for c in at.caption)
    {g.key: g for g in at.button_group}["whatif_variant"].set_value("C4").run()
    assert not at.exception and any("This control is binary" in c.value for c in at.caption)
    at.radio[0].set_value("VIX high").run()
    assert not at.exception


@pytest.mark.parametrize("mode", ["light", "dark"])
def test_the_live_and_lab_charts_draw_what_they_are_given(mode):
    import numpy as np
    import pandas as pd

    c = theme.palette(mode)
    donut = charts.sleeve_donut(pd.Series({"Equity": 0.5, "Bonds": 0.3, "Gold": 0.2, "Cash": 0.0}), c)
    assert list(donut.data[0].labels) == ["Equity", "Bonds", "Gold"] and sum(donut.data[0].values) == pytest.approx(1.0)
    sweep = pd.DataFrame({"Equity": [0.6, 0.5], "Bonds": [0.2, 0.3], "Gold": [0.1, 0.1], "Cash": [0.1, 0.1]}, index=[0.0, 1.0])
    lines_ = charts.sweep_lines(sweep, {"V1 equity (no regime input)": 0.55}, 0.2, c)
    assert [t.name for t in lines_.data] == [*theme.SLEEVES, "V1 equity (no regime input)"] and list(lines_.data[-1].y) == [0.55, 0.55]
    change = charts.change_bars(pd.Series({"XLK": 0.02, "CASH": -0.02}), {"XLK": "Equity", "CASH": "Cash"}, c)
    assert np.allclose(change.data[0].y, [2.0, -2.0])
    pipe = charts.pipeline([("Prices", "23 series"), ("Features", "184 numbers"), ("Weights", "14")], c)
    assert len(pipe.frames) == 3
    for fig in (donut, lines_, change, pipe, charts.latent_spark([0.1, -0.2], c)):
        assert not FORBIDDEN.search(fig.to_json())
