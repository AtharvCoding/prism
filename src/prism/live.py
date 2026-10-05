"""The frozen models, applied forward: persisted last-fold parameters and the functions that run them. DASHBOARD.md §4.

"Frozen" means the last walk-forward fold (DASHBOARD.md §4.1.2): the research pipeline would refit the HMM every
month and the encoder every year; the live view does not. The dashboard's precompute persists that fold's
parameters once (:func:`save_models`), and everything here is a forward pass on them: an HMM filter step, an
encoder forward pass, a threshold lookup, a standardisation. Nothing in this module fits or trains anything.

Each function is checked against the stored states on the fold's own apply window (:func:`verify_models`,
1e-9; DECISIONS.md D-044 item 1), which is the only window one frozen fold can reproduce: earlier dates were
produced by earlier folds' models.

This file holds the model half of the live path. Fetching fresh prices, the splice and the agents' rollout are
milestone D5.
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
    "threshold_states", "scale_features", "verify_models", "LiveModelError",
]

MODELS_DIR = "data/live/models"
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
