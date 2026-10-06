"""The frozen models, applied forward: persisted last-fold parameters and the functions that run them. DASHBOARD.md §4.

"Frozen" means the last walk-forward fold (DASHBOARD.md §4.1.2): the research pipeline would refit the HMM every
month and the encoder every year; the live view does not. The dashboard's precompute persists that fold's
parameters once (:func:`save_models`), and everything here is a forward pass on them: an HMM filter step, an
encoder forward pass, a threshold lookup, a standardisation. Nothing in this module fits or trains anything.

Each function is checked against the stored states on the fold's own apply window (:func:`verify_models`,
1e-9; DECISIONS.md D-044 item 1), which is the only window one frozen fold can reproduce: earlier dates were
produced by earlier folds' models.

The second half of the file is the path from fresh prices to weights: which sessions are complete
(:func:`completed_sessions`), the splice onto the frozen snapshot (:func:`splice`), the new state rows
(:func:`live_states`), and the agents' deterministic episode continued to the present (:func:`rollout`,
:func:`latest_weights`). No function here looks past the session it is computing for; the tests perturb the future
and check nothing earlier moves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from prism.utils.hashing import sha256_file

__all__ = [
    "MODELS_DIR", "LiveModels", "save_models", "load_models", "hmm_model", "filter_hmm", "encoder_model", "encode",
    "threshold_states", "scale_features", "verify_models", "LiveModelError", "SeamError", "is_level_series",
    "completed_sessions", "fetch_recent", "splice", "live_states", "decision_calendar", "rollout", "latest_weights", "ensemble",
    "regime_positions", "what_if_weights", "load_policies", "benchmark_weights", "refresh", "read_cache", "CACHE_DIR",
]

MODELS_DIR = "data/live/models"
CACHE_DIR = "data/live/cache"
REGIME_COLUMNS = ("state_0", "state_1")


class LiveModelError(RuntimeError):
    """The persisted models are missing, differ from their manifest, or do not reproduce the stored states."""


@dataclass(frozen=True)
class LiveModels:
    """The last fold's parameters, as plain arrays. ``meta`` is ``models.json``; ``encoder_state`` the weights."""

    meta: dict[str, Any]
    encoder_state: dict[str, np.ndarray]

    @property
    def hmm(self) -> dict[str, Any]:
        return self.meta["hmm"]

    @property
    def encoder(self) -> dict[str, Any]:
        return self.meta["encoder"]

    @property
    def threshold(self) -> dict[str, Any]:
        return self.meta["threshold"]

    @property
    def state_scaler(self) -> dict[str, Any]:
        return self.meta["state_scaler"]


# --------------------------------------------------------------------------- #
# persist / load
# --------------------------------------------------------------------------- #
def _fold_dates(fold) -> dict[str, Any]:  # noqa: ANN001
    return {"fold": int(fold.index), "fit_start": str(fold.fit_start.date()), "fit_end": str(fold.fit_end.date()),
            "apply_start": str(fold.apply_start.date()), "apply_end": str(fold.apply_end.date())}


def _scaler(scaler) -> dict[str, Any]:  # noqa: ANN001
    p = scaler._params()
    return {"columns": [str(c) for c in p["centre"].index], "centre": p["centre"].to_numpy(dtype="float64").tolist(),
            "scale": p["scale"].to_numpy(dtype="float64").tolist()}


def save_models(cfg, ext, out_dir: Path, *, spec: str, k: int, encoder_selection: dict[str, Any],  # noqa: ANN001
                threshold_column: str) -> dict[str, Any]:
    """Persist the LAST fold of ``ext``'s walk-forwards (HMM, encoder, VIX threshold) and the state scaler.

    ``ext`` is a :class:`prism.holdout.ExtendedStates`. Called by the dashboard's precompute only. Returns the
    manifest (dates and SHA-256 of both files), which is also written beside them.
    """
    from prism.models.baselines.threshold_regime import fit_threshold_breakpoints
    from prism.models.hmm.filtered import filtered_posteriors

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hf, ef = ext.hmm.folds[-1], ext.encoder.folds[-1]
    observations = list(cfg.hmm.specifications[spec].observations)
    obs = ext.features_a[observations].dropna()
    combined = pd.concat([hf.scaler.transform(obs.loc[hf.fold.fit_start: hf.fold.fit_end]),
                          hf.scaler.transform(obs.loc[hf.fold.apply_start: hf.fold.apply_end])])
    carried = filtered_posteriors(hf.model, combined.to_numpy(dtype="float64"))
    signal = ext.features_a[threshold_column]
    edges = fit_threshold_breakpoints(signal.loc[hf.fold.fit_start: hf.fold.fit_end], k, scope="live models").edges
    meta = {
        "end": str(ext.end.date()),
        "hmm": {
            **_fold_dates(hf.fold), "specification": spec, "k": int(k), "observations": observations,
            "covariance_type": hf.model.covariance_type, "scaler": _scaler(hf.scaler),
            "startprob": hf.model.startprob_.tolist(), "transmat": hf.model.transmat_.tolist(),
            "means": hf.model.means_.tolist(), "covars_raw": np.asarray(hf.model._covars_).tolist(),
            "carry_date": str(combined.index[-1].date()), "carry_log_alpha": carried.final_log_alpha.tolist(),
        },
        "encoder": {
            **_fold_dates(ef.fold), "variant": encoder_selection["variant"], "window": int(encoder_selection["window"]),
            "hidden_dim": int(encoder_selection["hidden_dim"]), "latent_dim": int(encoder_selection["latent_dim"]),
            "dropout": cfg.encoder.architecture.dropout, "latent_activation": cfg.encoder.architecture.latent_activation,
            "dae_noise_std": cfg.encoder.dae_noise_std, "vae_kl_weight": cfg.encoder.vae_kl_weight, "scaler": _scaler(ef.scaler),
        },
        "threshold": {**_fold_dates(hf.fold), "column": threshold_column, "k": int(k), "edges": np.asarray(edges, dtype="float64").tolist()},
        "state_scaler": _scaler(ext.scaler),
    }
    (out_dir / "models.json").write_text(json.dumps(meta, indent=1) + "\n", encoding="utf-8")
    state = {name: tensor.detach().cpu().numpy() for name, tensor in ef.model.state_dict().items()}
    np.savez(out_dir / "encoder.npz", **state)
    manifest = {
        "end": meta["end"], "hmm": _fold_dates(hf.fold), "encoder": _fold_dates(ef.fold),
        "files": {name: sha256_file(out_dir / name) for name in ("models.json", "encoder.npz")},
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return manifest


def load_models(models_dir: Path, *, expected: dict[str, Any] | None = None) -> LiveModels:
    """Load the persisted parameters, verifying both files against the manifest (and ``expected``, the committed copy)."""
    models_dir = Path(models_dir)
    path = models_dir / "manifest.json"
    if not path.exists():
        raise LiveModelError(f"no frozen live models in {models_dir}; they are written by the dashboard precompute")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if expected is not None and manifest != expected:
        raise LiveModelError("the live models' manifest differs from the committed one")
    for name, digest in manifest["files"].items():
        if sha256_file(models_dir / name) != digest:
            raise LiveModelError(f"{name} differs from the live models' manifest")
    with np.load(models_dir / "encoder.npz") as z:
        state = {name: z[name] for name in z.files}
    return LiveModels(meta=json.loads((models_dir / "models.json").read_text(encoding="utf-8")), encoder_state=state)


# --------------------------------------------------------------------------- #
# forward passes
# --------------------------------------------------------------------------- #
def _standardise(frame: pd.DataFrame, scaler: dict[str, Any]) -> pd.DataFrame:
    cols = scaler["columns"]
    missing = [c for c in cols if c not in frame.columns]
    if missing:
        raise KeyError(f"columns the frozen scaler was fitted on are absent: {missing[:8]}")
    out = frame.copy()
    out[cols] = (out[cols] - np.asarray(scaler["centre"])) / np.asarray(scaler["scale"])
    return out


def hmm_model(params: dict[str, Any]):  # noqa: ANN201
    """The fold's Gaussian HMM rebuilt from its persisted parameters (already in canonical state order)."""
    from hmmlearn.hmm import GaussianHMM

    means = np.asarray(params["means"], dtype="float64")
    model = GaussianHMM(n_components=int(params["k"]), covariance_type=params["covariance_type"])
    model.n_features = means.shape[1]
    model.startprob_ = np.asarray(params["startprob"], dtype="float64")
    model.transmat_ = np.asarray(params["transmat"], dtype="float64")
    model.means_ = means
    model._covars_ = np.asarray(params["covars_raw"], dtype="float64")
    return model


def filter_hmm(params: dict[str, Any], obs: pd.DataFrame, *, initial_log_alpha: np.ndarray | None = None
               ) -> tuple[pd.DataFrame, np.ndarray]:
    """Filtered posteriors for ``obs`` (raw units; standardised here with the fold's scaler), and the carried filter state.

    With ``initial_log_alpha`` (the persisted ``carry_log_alpha``, or a previous call's second return value) the
    recursion continues from where the fold stopped, so each row uses data up to its own date and nothing later.
    """
    from prism.models.hmm.filtered import filtered_posteriors

    x = _standardise(obs[params["observations"]], params["scaler"])
    if x.isna().any().any():
        raise ValueError("the HMM observations contain NaN")
    res = filtered_posteriors(hmm_model(params), x.to_numpy(dtype="float64"),
                              initial_log_alpha=None if initial_log_alpha is None else np.asarray(initial_log_alpha, dtype="float64"))
    columns = [f"state_{i}" for i in range(int(params["k"]))]
    return pd.DataFrame(res.posteriors, index=obs.index, columns=columns), res.final_log_alpha


def encoder_model(models: LiveModels):  # noqa: ANN201
    """The fold's encoder rebuilt from its persisted weights, in eval mode."""
    import torch

    from prism.models.encoder.models import build_model

    e = models.encoder
    model = build_model(e["variant"], input_dim=len(e["scaler"]["columns"]), hidden_dim=e["hidden_dim"], latent_dim=e["latent_dim"],
                        window=e["window"], dropout=e["dropout"], latent_activation=e["latent_activation"],
                        dae_noise_std=e["dae_noise_std"], vae_kl_weight=e["vae_kl_weight"])
    model.load_state_dict({k: torch.from_numpy(np.asarray(v)) for k, v in models.encoder_state.items()})
    return model.eval()


def encode(models: LiveModels, features_a: pd.DataFrame, *, model=None) -> pd.DataFrame:  # noqa: ANN001
    """Latents for every date from the fold's ``apply_start`` on.

    The windows are built, and batched, exactly as the walk-forward built them: the fold's whole fit window, then
    its apply window (the embargoed sessions between them are absent), so the first apply dates' windows reach back
    into the fit tail. Encoding only the tail would be the same arithmetic but not the same float32 batches, and
    differs from the stored latents by about 3e-8; running the fit window through as well reproduces them exactly.
    """
    from prism.models.encoder.dataset import WindowDataset
    from prism.models.encoder.evaluate import compute_latents

    e = models.encoder
    fit_start, fit_end, apply_start = pd.Timestamp(e["fit_start"]), pd.Timestamp(e["fit_end"]), pd.Timestamp(e["apply_start"])
    cols = e["scaler"]["columns"]
    frame = features_a[cols]
    combined = pd.concat([frame.loc[fit_start:fit_end], frame.loc[apply_start:]])
    scaled = _standardise(combined, e["scaler"])
    latents = compute_latents(model if model is not None else encoder_model(models), WindowDataset(scaled, e["window"]))
    return latents.loc[latents.index >= apply_start]


def threshold_states(params: dict[str, Any], signal: pd.Series) -> pd.DataFrame:
    """The VIX-threshold control's one-hot columns from the fold's fixed breakpoints."""
    bins = np.searchsorted(np.asarray(params["edges"], dtype="float64"), signal.to_numpy(dtype="float64"), side="right")
    out = np.zeros((len(signal), int(params["k"])), dtype="float64")
    out[np.arange(len(signal)), bins] = 1.0
    return pd.DataFrame(out, index=signal.index, columns=[f"state_{i}" for i in range(int(params["k"]))])


def scale_features(models: LiveModels, features_b: pd.DataFrame) -> pd.DataFrame:
    """The base (V1) block: Universe-B features standardised with the train-split scaler."""
    return _standardise(features_b[models.state_scaler["columns"]], models.state_scaler)


# --------------------------------------------------------------------------- #
# verification against the stored states
# --------------------------------------------------------------------------- #
def verify_models(models: LiveModels, features_a: pd.DataFrame, features_b: pd.DataFrame, stored: dict[str, pd.DataFrame],
                  *, atol: float = 1e-9) -> dict[str, Any]:
    """Each frozen model must reproduce the stored states on its own fold's apply window, from persisted parameters alone.

    ``stored`` maps V1, V2, V4, C4 to their stored state frames. Returns the maximum absolute differences; raises
    :class:`LiveModelError` if any exceeds ``atol``.
    """
    end = pd.Timestamp(models.meta["end"])
    h, e = models.hmm, models.encoder
    out: dict[str, Any] = {"atol": atol}

    obs = features_a[h["observations"]].dropna()
    fresh = pd.concat([obs.loc[pd.Timestamp(h["fit_start"]): pd.Timestamp(h["fit_end"])],
                       obs.loc[pd.Timestamp(h["apply_start"]): pd.Timestamp(h["apply_end"])]])
    posteriors, log_alpha = filter_hmm(h, fresh)
    apply = posteriors.loc[pd.Timestamp(h["apply_start"]):]
    want = stored["V4"].loc[apply.index, list(REGIME_COLUMNS)]
    out["hmm"] = {"window": [h["apply_start"], h["apply_end"]], "rows": int(len(apply)),
                  "max_abs_diff": float(np.abs(apply.to_numpy() - want.to_numpy()).max()),
                  "carry_matches": bool(np.allclose(log_alpha, h["carry_log_alpha"], rtol=0, atol=1e-9))}

    latents = encode(models, features_a.loc[:end])
    latents = latents.loc[latents.index.intersection(stored["V2"].index)]
    want = stored["V2"].loc[latents.index, list(latents.columns)]
    out["encoder"] = {"window": [e["apply_start"], e["apply_end"]], "rows": int(len(latents)),
                      "max_abs_diff": float(np.abs(latents.to_numpy() - want.to_numpy()).max())}

    t = models.threshold
    sig = features_a[t["column"]].loc[pd.Timestamp(t["apply_start"]): pd.Timestamp(t["apply_end"])]
    states = threshold_states(t, sig)
    states = states.loc[states.index.intersection(stored["C4"].index)]
    out["threshold"] = {"rows": int(len(states)),
                        "max_abs_diff": float(np.abs(states.to_numpy() - stored["C4"].loc[states.index, list(REGIME_COLUMNS)].to_numpy()).max())}

    base = scale_features(models, features_b)
    base = base.loc[base.index.intersection(stored["V1"].index)]
    out["state_scaler"] = {"rows": int(len(base)),
                           "max_abs_diff": float(np.abs(base.to_numpy() - stored["V1"].loc[base.index, list(base.columns)].to_numpy()).max())}

    worst = max(out[k]["max_abs_diff"] for k in ("hmm", "encoder", "threshold", "state_scaler"))
    out["ok"] = bool(worst <= atol and out["hmm"]["carry_matches"] and all(out[k]["rows"] > 0 for k in ("hmm", "encoder", "threshold", "state_scaler")))
    if not out["ok"]:
        raise LiveModelError(f"the persisted models do not reproduce the stored states: {json.dumps(out)}")
    return out


# --------------------------------------------------------------------------- #
# fresh data: complete sessions, and the splice onto the frozen snapshot
# --------------------------------------------------------------------------- #
class SeamError(RuntimeError):
    """Fresh data does not join the frozen snapshot cleanly; nothing may be computed from it."""


def is_level_series(ticker: str) -> bool:
    """Index levels, yields and futures prices are not dividend-adjusted, so they are never rescaled at the splice."""
    return ticker.startswith("^") or "=" in ticker or "." in ticker


def completed_sessions(index: pd.DatetimeIndex, now: pd.Timestamp, *, exchange: str = "NYSE", buffer_minutes: int = 30) -> pd.DatetimeIndex:
    """The dates in ``index`` that are exchange sessions whose close is at least ``buffer_minutes`` before ``now``.

    A download taken during market hours carries a partial bar for the current session. It is not a close, and a
    state computed from it would use a price nobody could have traded at; it is dropped here.
    """
    import pandas_market_calendars as mcal

    idx = pd.DatetimeIndex(index).normalize()
    if len(idx) == 0:
        return idx
    now = pd.Timestamp(now)
    now = now.tz_localize("UTC") if now.tzinfo is None else now.tz_convert("UTC")
    schedule = mcal.get_calendar(exchange).schedule(start_date=idx.min(), end_date=idx.max())
    done = schedule.index[schedule["market_close"] + pd.Timedelta(minutes=buffer_minutes) <= now]
    return idx[idx.isin(pd.DatetimeIndex(done).normalize())]


def fetch_recent(tickers: list[str], start: pd.Timestamp, end: pd.Timestamp) -> dict[str, pd.DataFrame]:
    """Daily bars from the vendor (``auto_adjust=True``), through the project's one network function."""
    from prism.data.download import download_panel

    return download_panel(list(tickers), start, end)


def splice(frozen_close: pd.DataFrame, frozen_volume: pd.DataFrame, fresh_close: pd.DataFrame, fresh_volume: pd.DataFrame, *,
           overlap: int = 60, tol: float = 1e-3) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Append the genuinely new sessions of ``fresh`` to the frozen panel, after checking the two agree where they overlap.

    Adjusted prices are restated when a dividend is paid, so a fresh download sits at a slightly different level
    from the snapshot. Returns are what matter: over the last ``overlap`` common sessions the daily returns must
    agree to ``tol``, and the new rows are rescaled so the series is continuous at the last common close. Level
    series (:func:`is_level_series`) must agree in level and are appended unscaled. Volume is rescaled by the
    median overlap ratio, which is 1 unless a split occurred. Raises :class:`SeamError` otherwise.
    """
    missing = [t for t in frozen_close.columns if t not in fresh_close.columns]
    if missing:
        raise SeamError(f"the fresh download lacks {missing}")
    common = frozen_close.index.intersection(fresh_close.index)
    new_index = fresh_close.index[fresh_close.index > frozen_close.index[-1]]
    if len(common) < min(overlap, 20):
        raise SeamError(f"only {len(common)} sessions overlap the frozen snapshot; need at least {min(overlap, 20)}")
    window = common[-overlap:]
    report: dict[str, Any] = {"anchor": str(window[-1].date()), "overlap_sessions": int(len(window)), "new_sessions": int(len(new_index)),
                              "tol": tol, "tickers": {}}
    new_close = pd.DataFrame(index=new_index, columns=frozen_close.columns, dtype="float64")
    for t in frozen_close.columns:
        pair = pd.concat([frozen_close[t].reindex(window), fresh_close[t].reindex(window)], axis=1, keys=["frozen", "fresh"]).dropna()
        if len(pair) < 5:
            raise SeamError(f"{t}: fewer than 5 overlapping observations")
        if is_level_series(t):
            diff = float((np.abs(pair.fresh - pair.frozen) / np.maximum(np.abs(pair.frozen), 1.0)).max())
            scale = 1.0
        else:
            diff = float(np.abs(pair.fresh.pct_change() - pair.frozen.pct_change()).max())
            scale = float(pair.frozen.iloc[-1] / pair.fresh.iloc[-1])
        report["tickers"][t] = {"scale": scale, "max_diff": diff, "level": is_level_series(t)}
        if not diff <= tol:
            report["ok"] = False
            raise SeamError(f"data seam check failed for {t}: the fresh download differs from the frozen snapshot by {diff:.2e} "
                            f"over the last {len(pair)} common sessions (tolerance {tol:g})")
        new_close[t] = fresh_close[t].reindex(new_index) * scale
    new_volume = pd.DataFrame(index=new_index, columns=frozen_volume.columns, dtype="float64")
    for t in frozen_volume.columns:
        pair = pd.concat([frozen_volume[t].reindex(window), fresh_volume[t].reindex(window)], axis=1, keys=["frozen", "fresh"]).dropna()
        pair = pair[(pair.frozen > 0) & (pair.fresh > 0)]
        ratio = float((pair.frozen / pair.fresh).median()) if len(pair) else 1.0
        new_volume[t] = fresh_volume[t].reindex(new_index) * ratio
    report["ok"] = True
    return pd.concat([frozen_close, new_close]), pd.concat([frozen_volume, new_volume]), report


# --------------------------------------------------------------------------- #
# new state rows from the frozen models
# --------------------------------------------------------------------------- #
def live_states(cfg, raw: pd.DataFrame, models: LiveModels, reference_columns: dict[str, list[str]], *,  # noqa: ANN001
                start: pd.Timestamp | None = None) -> dict[str, pd.DataFrame]:
    """State rows of V1, V2, V4 and C4 from ``start`` to the end of ``raw``, produced by the frozen last-fold models.

    ``start=None`` means "after the models' last day": the HMM filter continues from its carried state. An earlier
    ``start`` (not before the HMM fold's apply window) re-derives rows that are already stored, which is how the
    function is verified. ``reference_columns`` are the stored state files' column lists; the result must match
    them exactly.
    """
    from prism.features.build import build_features
    from prism.holdout import _load_set, _prune

    end = raw.index[-1]
    features = {}
    for universe in ("A", "B"):
        fs = build_features(raw, cfg, universe)
        features[universe] = _load_set(cfg, universe, _prune(cfg, fs, universe, end), fs).frame
    h = models.hmm
    carry_date = pd.Timestamp(h["carry_date"])
    obs = features["A"][h["observations"]].dropna()
    if start is None or pd.Timestamp(start) > carry_date:
        start = carry_date + pd.Timedelta(days=1) if start is None else pd.Timestamp(start)
        new = obs.loc[obs.index > carry_date]
        posteriors = filter_hmm(h, new, initial_log_alpha=np.asarray(h["carry_log_alpha"]))[0] if len(new) else pd.DataFrame(columns=list(REGIME_COLUMNS))
    else:
        start = pd.Timestamp(start)
        if start < pd.Timestamp(h["apply_start"]):
            raise ValueError("the frozen HMM fold cannot reproduce dates before its own apply window")
        history = pd.concat([obs.loc[pd.Timestamp(h["fit_start"]): pd.Timestamp(h["fit_end"])], obs.loc[pd.Timestamp(h["apply_start"]):]])
        posteriors = filter_hmm(h, history)[0]
    latents = encode(models, features["A"])
    threshold = threshold_states(models.threshold, features["A"][models.threshold["column"]])
    base = scale_features(models, features["B"])
    index = base.index[base.index >= start].intersection(latents.index).intersection(posteriors.index).intersection(threshold.index)
    blocks = {"V1": [base], "V2": [base, latents], "V4": [base, latents, posteriors], "C4": [base, latents, threshold]}
    out = {}
    for variant, parts in blocks.items():
        frame = pd.concat([part.loc[index] for part in parts], axis=1)
        if [str(c) for c in frame.columns] != list(reference_columns[variant]):
            raise LiveModelError(f"{variant}: the live state columns differ from the stored state file's")
        if not np.isfinite(frame.to_numpy(dtype="float64")).all():
            raise LiveModelError(f"{variant}: the live state contains NaN or inf")
        out[variant] = frame
    return out


# --------------------------------------------------------------------------- #
# the agents' episode, continued
# --------------------------------------------------------------------------- #
def decision_calendar(sessions: pd.DatetimeIndex, *, exchange: str = "NYSE") -> tuple[pd.Timestamp, bool]:
    """The latest completed weekly decision date in ``sessions``, and whether it is the last session.

    A decision is taken at the last session of a week. The last session of ``sessions`` is one only if the next
    exchange session falls in a later week; otherwise the latest decision is the previous week's last session
    and anything computed at the last session is a mid-week *preview*.
    """
    from prism.utils.calendar import rebalance_dates, trading_days

    sessions = pd.DatetimeIndex(sessions)
    last = sessions[-1]
    upcoming = trading_days(last + pd.Timedelta(days=1), last + pd.Timedelta(days=10), exchange)
    week = lambda d: tuple(d.isocalendar())[:2]  # noqa: E731
    if len(upcoming) and week(upcoming[0]) != week(last):
        return last, True
    weekly = rebalance_dates(sessions, "weekly", "FRI")
    earlier = weekly[weekly < last]
    eligible = earlier[[week(d) != week(last) for d in earlier]]
    if len(eligible) == 0:
        raise ValueError("no completed week in the sessions")
    return eligible[-1], False


def rollout(cfg, states: pd.DataFrame, close: pd.DataFrame, policy, *, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, Any]:  # noqa: ANN001
    """One agent's deterministic episode from cash at ``start`` through ``end``, and what it would choose at ``end``.

    The episode is the environment's own (same arrays, costs, action map) built over the sessions directly, so it
    continues past the last split. Returns every executed decision (target and pre-trade weights, turnover, cost)
    and ``latest``: the weights the policy chooses on the terminal observation, i.e. the state at ``end`` with the
    portfolio drifted to ``end``. When ``end`` is a week's last session that is the weekly decision.
    """
    from prism.backtest.engine import TimingConvention
    from prism.env.actions import action_to_weights, upper_bounds
    from prism.env.costs import one_way_turnover
    from prism.env.data import CASH, EnvData, line_returns
    from prism.env.portfolio_env import make_env

    risky = list(cfg.data.allocatable[cfg.env.universe])
    lr_full = line_returns(close, risky)
    vol_full = lr_full[risky].rolling(cfg.env.costs.vol_window).std()
    sessions = pd.DatetimeIndex(states.index).intersection(close.index).sort_values()
    sessions = sessions[(sessions >= pd.Timestamp(start)) & (sessions <= pd.Timestamp(end))]
    if (np.diff(close.index.get_indexer(sessions)) != 1).any():
        raise ValueError("the state sessions are not consecutive in the price panel")
    data = EnvData.from_arrays(sessions, states.loc[sessions].to_numpy(), lr_full.reindex(sessions).to_numpy(), vol_full.reindex(sessions).to_numpy(),
                               lines=(*risky, CASH), state_columns=tuple(str(c) for c in states.columns), convention=TimingConvention.from_config(cfg))
    env = make_env(cfg, data, mode="eval")
    obs, _ = env.reset(seed=0)
    rows = []
    while True:
        pre = env.drifted_weights
        obs, _, terminated, truncated, info = env.step(policy(obs))
        rows.append({"decision_date": info["decision_date"], "execution_date": info["execution_date"], "turnover": info["turnover"],
                     "cost": info["cost"], **{f"w_{c}": w for c, w in zip(data.lines, info["weights"])},
                     **{f"p_{c}": w for c, w in zip(data.lines, pre)}})
        if terminated or truncated:
            break
    held = env.drifted_weights
    chosen = action_to_weights(policy(obs), upper_bounds(len(risky), cfg.data.allocation.weight_max), cfg.env.action.logit_scale)
    latest = {"date": sessions[-1], "weights": pd.Series(chosen, index=list(data.lines)), "held": pd.Series(held, index=list(data.lines)),
              "turnover": float(one_way_turnover(held, chosen)), "observation": np.asarray(obs, dtype="float32")}
    return {"decisions": pd.DataFrame(rows), "latest": latest, "lines": list(data.lines), "state_columns": list(data.state_columns)}


def latest_weights(cfg, states: dict[str, pd.DataFrame], close: pd.DataFrame, policies: dict[tuple[str, int], Any], *,  # noqa: ANN001
                   start: pd.Timestamp, recorded: pd.DataFrame | None = None, atol: float = 1e-9) -> dict[str, Any]:
    """Every agent's latest weekly decision and, mid-week, its preview as of the latest close.

    Returns ``decision_date``, ``as_of`` (the last session), ``is_preview`` (a preview exists) and two frames,
    ``decision`` and ``preview`` (``None`` when the last session is itself the decision), one row per agent.

    ``recorded`` is the replay's weekly weights for the same episode. When given, every agent's continued episode
    must contain the recorded one (same decision dates, target weights within ``atol``), or :class:`LiveModelError`
    is raised: the live number is then, checkably, the continuation of the stored evaluation and nothing else.
    """
    sessions = pd.DatetimeIndex(next(iter(states.values())).index)
    sessions = sessions[sessions >= pd.Timestamp(start)].intersection(close.index)
    decision_date, complete = decision_calendar(sessions)

    worst = 0.0

    def at(end: pd.Timestamp, check: bool = False) -> tuple[pd.DataFrame, dict[str, list[float]]]:
        nonlocal worst
        rows, observations = [], {}
        for (variant, seed), policy in policies.items():
            full = rollout(cfg, states[variant], close, policy, start=start, end=end)
            if check and recorded is not None:
                want = recorded[(recorded.variant == variant) & (recorded.seed == seed)].set_index("decision_date")
                got = full["decisions"].set_index("decision_date")
                cols = [c for c in want.columns if c.startswith("w_")]
                if len(want) == 0 or not want.index.isin(got.index).all():
                    raise LiveModelError(f"{variant} seed {seed}: the continued episode does not contain the recorded decisions")
                worst = max(worst, float(np.abs(got.loc[want.index, cols].to_numpy() - want[cols].to_numpy()).max()))
            res = full["latest"]
            rows.append({"variant": variant, "seed": seed, "turnover": res["turnover"], **{f"w_{k}": v for k, v in res["weights"].items()},
                         **{f"p_{k}": v for k, v in res["held"].items()}})
            observations[f"{variant}|s{seed}"] = [float(x) for x in res["observation"]]
        return pd.DataFrame(rows), observations

    decision, observations = at(decision_date, check=True)
    if recorded is not None and not worst <= atol:
        raise LiveModelError(f"the continued episode differs from the recorded one by {worst:.2e}; the live weights are not shown")
    preview = None if complete else at(sessions[-1])[0]
    return {"decision_date": decision_date, "as_of": sessions[-1], "is_preview": not complete, "decision": decision, "preview": preview,
            "observations": observations, "recorded_max_abs_diff": worst if recorded is not None else None}


def ensemble(frame: pd.DataFrame, prefix: str = "w_") -> pd.DataFrame:
    """Across the seeds of one variant: the mean weight of each holding, the lowest and highest seed, and their gap."""
    w = frame[[c for c in frame.columns if c.startswith(prefix)]].rename(columns=lambda c: c[len(prefix):])
    out = pd.DataFrame({"mean": w.mean(), "min": w.min(), "max": w.max()})
    out["spread"] = out["max"] - out["min"]
    return out


# --------------------------------------------------------------------------- #
# what-if: the regime inputs of an observation, replaced
# --------------------------------------------------------------------------- #
def regime_positions(columns: list[str]) -> tuple[int, int]:
    """Positions of the two regime inputs (``state_0``, ``state_1``) in an observation's column list."""
    if "state_0" not in columns or "state_1" not in columns:
        raise ValueError("this observation has no regime inputs")
    return columns.index("state_0"), columns.index("state_1")


def what_if_weights(cfg, policy, observation: np.ndarray, columns: list[str], p_volatile: float | None) -> np.ndarray:  # noqa: ANN001
    """The weights a frozen policy chooses when its two regime inputs are set to ``(1 - p, p)`` and nothing else changes.

    ``p_volatile=None`` leaves the observation untouched. A variant with no regime inputs returns the same weights
    for every ``p`` by construction.
    """
    from prism.env.actions import action_to_weights, upper_bounds

    obs = np.array(observation, dtype="float32", copy=True)
    if p_volatile is not None and "state_1" in columns:
        if not 0.0 <= p_volatile <= 1.0:
            raise ValueError("p_volatile must be a probability")
        i0, i1 = regime_positions(columns)
        obs[i0], obs[i1] = 1.0 - p_volatile, p_volatile
    n_lines = sum(c.startswith("weight_") for c in columns)
    return action_to_weights(policy(obs), upper_bounds(n_lines - 1, cfg.data.allocation.weight_max), cfg.env.action.logit_scale)


# --------------------------------------------------------------------------- #
# one refresh: fresh prices -> frozen models -> the 40 agents' latest weights, cached
# --------------------------------------------------------------------------- #
def load_policies(cfg, root: Path) -> dict[tuple[str, int], Any]:  # noqa: ANN001
    """The 40 frozen agents as deterministic policies, after checking every pinned file against its SHA-256."""
    from prism.agents import tier2 as t2
    from prism.agents.jobs import job_dir
    from prism.agents.sac import greedy_policy, load_agent
    from prism.dashboard_replay import verify_agents

    verify_agents(root)
    plan = t2.Tier2Plan.from_config(cfg, root)
    frozen = t2.load_frozen(plan)
    return {(v, s): greedy_policy(load_agent(job_dir(plan.runs, "final", v, frozen[v].cfg_id, s) / "best.zip"))
            for v in plan.variants for s in plan.final_seeds}


def benchmark_weights(cfg, close: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:  # noqa: ANN001
    """Equal weight, 60/40 and risk parity at one decision date, by their stored definitions (one row each)."""
    from prism.backtest import benchmarks as bm
    from prism.probes import allocator as al

    params = al.AllocatorParams(risky=tuple(cfg.data.allocatable[cfg.env.universe]), equity_sectors=tuple(cfg.data.universes["A"].equity_sectors),
                                cap=cfg.data.allocation.weight_max)
    inputs, _ = al.prepare_inputs(close, params, pd.DatetimeIndex([date]))
    frames = {"EqualWeight": bm.equal_weight(inputs, params), "SixtyForty": bm.sixty_forty(inputs, params), "RiskParity": bm.risk_parity(inputs, params)}
    return pd.concat({name: f.iloc[0] for name, f in frames.items()}, axis=1).T.add_prefix("w_").rename_axis("benchmark").reset_index()


def refresh(cfg, root: Path, *, now: pd.Timestamp, window: str = "holdout", models_dir: Path | None = None,  # noqa: ANN001
            cache_dir: Path | None = None, fetch=fetch_recent, policies: dict[tuple[str, int], Any] | None = None,
            expected_manifest: dict[str, Any] | None = None, overlap: int = 60, recorded: pd.DataFrame | None = None) -> dict[str, Any]:
    """Fetch, splice, compute the new states with the frozen models, continue every agent's episode, and cache the result.

    ``window`` names the stored evaluation whose episode is continued (``holdout`` for the real thing; ``test`` only for
    the rehearsal). The snapshot is used up to the models' last day and never written. Raises on any failure (network,
    :class:`SeamError`, :class:`LiveModelError`) without touching the previous cache; the caller decides what to show.
    """
    from prism.dashboard_replay import stored_states
    from prism.data.loaders import assert_not_holdout, load_snapshot
    from prism.features.build import build_features, make_raw_frame
    from prism.splits import build_split_plan

    root = Path(root)
    models = load_models(models_dir if models_dir is not None else root / MODELS_DIR, expected=expected_manifest)
    end = pd.Timestamp(models.meta["end"])
    snap = load_snapshot(cfg)
    assets, macro = list(snap.close.columns), list(snap.macro.columns)
    frozen_close = pd.concat([snap.close, snap.macro], axis=1).loc[:end]
    frozen_volume = snap.volume.loc[:end]
    if window != "holdout":
        assert_not_holdout(cfg, frozen_close.index, context="live rehearsal")
    start = frozen_close.index[-(overlap + 20)]
    today = pd.Timestamp(now)
    today = (today.tz_convert("UTC").tz_localize(None) if today.tzinfo is not None else today).normalize()
    panels = fetch([*assets, *macro], start, today)
    done = completed_sessions(panels["Close"].index, now)
    fresh_close, fresh_volume = panels["Close"].loc[done], panels["Volume"].reindex(columns=assets).loc[done]
    close, volume, report = splice(frozen_close, frozen_volume, fresh_close, fresh_volume, overlap=overlap)
    raw = make_raw_frame(close[assets], volume, close[macro])
    stored = stored_states(root, window)
    stored = {v: f.loc[:end] for v, f in stored.items()}
    if report["new_sessions"]:
        new = live_states(cfg, raw, models, {v: [str(c) for c in f.columns] for v, f in stored.items()})
        states = {v: pd.concat([stored[v], new[v]]) for v in stored}
    else:
        states = stored
    close_b = build_features(raw, cfg, "B").close
    episode_start = build_split_plan(cfg)[window].effective_start
    policies = policies if policies is not None else load_policies(cfg, root)
    if recorded is not None:
        recorded = recorded[recorded.decision_date < end]              # only what the frozen models' own period covers
    res = latest_weights(cfg, states, close_b, policies, start=episode_start, recorded=recorded)
    regime = states["V4"][list(REGIME_COLUMNS)].iloc[-260:]
    latent = states["V2"][[c for c in states["V2"].columns if c.startswith("latent_")]].iloc[-1]
    risky = len(cfg.data.allocatable[cfg.env.universe])
    columns = {v: [*map(str, f.columns), *(c.replace("w_", "weight_") for c in res["decision"].columns if c.startswith("w_")), "mean_turnover"]
               for v, f in states.items()}
    status = {
        "as_of": str(res["as_of"].date()), "decision_date": str(res["decision_date"].date()), "is_preview": bool(res["is_preview"]),
        "fetched_utc": str(pd.Timestamp(now).tz_convert("UTC") if pd.Timestamp(now).tzinfo is not None else pd.Timestamp(now)),
        "window": window, "models": {"end": models.meta["end"], "hmm_fit_end": models.hmm["fit_end"], "encoder_fit_end": models.encoder["fit_end"]},
        "recorded_episode_max_abs_diff": res["recorded_max_abs_diff"],
        "new_sessions": report["new_sessions"], "splice": {"anchor": report["anchor"], "overlap_sessions": report["overlap_sessions"],
                                                           "max_diff": max(t["max_diff"] for t in report["tickers"].values())},
        "n_lines": risky + 1, "columns": columns, "observations": res["observations"],
        "p_volatile": {str(d.date()): float(p) for d, p in regime["state_1"].items()}, "latent": [float(x) for x in latent],
    }
    cache = Path(cache_dir) if cache_dir is not None else root / CACHE_DIR
    scratch = cache.with_name(cache.name + ".tmp")
    if scratch.exists():
        import shutil

        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    res["decision"].to_parquet(scratch / "decision.parquet")
    if res["preview"] is not None:
        res["preview"].to_parquet(scratch / "preview.parquet")
    benchmark_weights(cfg, close_b, res["decision_date"]).to_parquet(scratch / "benchmarks.parquet")
    (scratch / "status.json").write_text(json.dumps(status, indent=1) + "\n", encoding="utf-8")
    if cache.exists():
        import shutil

        shutil.rmtree(cache)
    scratch.rename(cache)                                           # the previous good result is replaced only by a complete new one
    return {**status, "decision": res["decision"], "preview": res["preview"]}


def read_cache(cache_dir: Path) -> dict[str, Any] | None:
    """The last successful refresh, or ``None`` if there has not been one."""
    cache = Path(cache_dir)
    if not (cache / "status.json").exists():
        return None
    out = json.loads((cache / "status.json").read_text(encoding="utf-8"))
    out["decision"] = pd.read_parquet(cache / "decision.parquet")
    out["preview"] = pd.read_parquet(cache / "preview.parquet") if (cache / "preview.parquet").exists() else None
    out["benchmarks"] = pd.read_parquet(cache / "benchmarks.parquet")
    return out
