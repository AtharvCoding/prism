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

from components import charts, theme, ui  # noqa: E402
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
