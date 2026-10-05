"""The pages, in story order (DASHBOARD.md §7). One place for each page's title and one-line takeaway.

A page appears here when its milestone is built; the numbering follows the spec so later pages slot in.
The page scripts live in ``dashboard/views/``, not ``pages/``: a directory with that name switches on Streamlit's
legacy page lookup, which can run a page on its own, without the frame in ``app.py`` (DECISIONS.md D-046).
Takeaways that quote a research number are templates filled from ``facts.json`` (see ``ui.page_header``).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PageInfo:
    key: str
    script: str
    title: str
    icon: str
    takeaway: str
    shows_weights: bool = False


PAGES: tuple[PageInfo, ...] = (
    PageInfo(
        "question", "views/02_question.py", "The question", ":material/help:",
        "Does telling an allocation agent which market regime it is in help, once it already has a learned summary of "
        "recent history? That was tested in advance, twice, and no benefit was detected.",
    ),
    PageInfo(
        "data", "views/03_data.py", "Data and universe", ":material/database:",
        "Fourteen weights chosen once a week, judged on data the models had not seen, with a final holdout that was "
        "opened exactly once.",
    ),
    PageInfo(
        "regimes", "views/04_regimes.py", "Regimes", ":material/thermostat:",
        "The HMM sorts each day into a Calm or a Volatile regime. A plain VIX threshold makes the same call on "
        "{regime_agreement} of days, and the VIX is high on {volatile_overlap} of the days the HMM calls Volatile.",
    ),
    PageInfo(
        "latent", "views/05_latent.py", "LSTM latent", ":material/hub:",
        "The LSTM compresses the last {encoder_window} days of market features into {latent_dim} numbers. In the Tier 1 test "
        "a trained LSTM was no more useful than an untrained one of the same shape.",
    ),
    PageInfo(
        "agent", "views/06_agent.py", "The agent", ":material/smart_toy:",
        "Each agent re-weights the portfolio heavily every week ({agents_turnover} of it, against {bench_turnover} for the "
        "benchmarks), and trading costs take a large part of its return.",
    ),
    PageInfo(
        "allocation", "views/09_allocation.py", "Allocation through time", ":material/stacked_line_chart:",
        "What the frozen agents held, week by week. It describes their behaviour; it is not evidence of skill, and these "
        "allocations did not beat the simple benchmarks after costs.",
        shows_weights=True,
    ),
    PageInfo(
        "results", "views/10_results.py", "Results", ":material/fact_check:",
        "{passed_test} of 3 pre-registered comparisons passed on the test split and {passed_holdout} of 3 on the holdout. "
        "The gaps between variants are smaller than the gaps between random seeds.",
    ),
    PageInfo(
        "verdict", "views/11_verdict.py", "Verdict and what's next", ":material/gavel:",
        "The null result replicated on unseen data: neither the HMM regimes nor the LSTM latent gave this agent a "
        "detectable benefit.",
    ),
)

BY_KEY = {p.key: p for p in PAGES}
