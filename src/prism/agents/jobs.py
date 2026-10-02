"""One training run as a self-contained, resumable, picklable job. Build step 4c.

``run_job(spec)`` runs in a worker process (spawn), builds its own environments
from the train and validation splits only, trains, and writes
``<job dir>/result.json`` **last and atomically**: a directory without it is an
unfinished run and is retrained from scratch on resume. Nothing here can reach
the test split: the close panel is cut at the validation split's last session.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from prism.agents.sac import SacConfig, SacSettings, train_run
from prism.utils.seeding import derive_seed

__all__ = ["make_spec", "job_dir", "is_done", "run_job", "read_result"]

_CACHE: dict[str, Any] = {}


def make_spec(root: Path, stage: str, variant: str, config: SacConfig, seed: int, steps: int,
            eval_every: int, settings: SacSettings, log_dir: Path, out_dir: Path) -> dict[str, Any]:
    return {
        "root": str(root), "stage": stage, "variant": variant,
        "config": asdict(config), "seed": int(seed), "steps": int(steps), "eval_every": int(eval_every),
        "settings": asdict(settings), "log_dir": str(log_dir), "out_dir": str(out_dir),
    }


def job_dir(out_dir: Path, stage: str, variant: str, cfg_id: str, seed: int) -> Path:
    return out_dir / "runs" / stage / variant / cfg_id / f"seed{seed}"


def is_done(directory: Path) -> bool:
    return (directory / "result.json").exists()


def read_result(directory: Path) -> dict[str, Any]:
    return json.loads((directory / "result.json").read_text())


def _worker_data(root: str, variant: str):  # noqa: ANN202
    from prism.agents.data import load_close, variant_env_data
    from prism.config import load_config
    from prism.splits import build_split_plan

    if "cfg" not in _CACHE:
        cfg = load_config(Path(root) / "configs" / "base.yaml")
        _CACHE["cfg"] = cfg
        _CACHE["plan"] = build_split_plan(cfg)
        _CACHE["close"] = load_close(cfg, "val")  # train + validation only: the test split is never read here
    key = f"data:{variant}"
    if key not in _CACHE:
        cfg, plan, close = _CACHE["cfg"], _CACHE["plan"], _CACHE["close"]
        _CACHE[key] = {s: variant_env_data(cfg, variant, close, s, plan=plan) for s in ("train", "val")}
    return _CACHE["cfg"], _CACHE[key]


def run_job(spec: dict[str, Any]) -> dict[str, Any]:
    """Train one (stage, variant, config, seed). Skips if already finished."""
    from prism.env.portfolio_env import make_env

    config = SacConfig(gamma=spec["config"]["gamma"], hidden=tuple(spec["config"]["hidden"]), lr=spec["config"]["lr"])
    settings = SacSettings(**spec["settings"])
    directory = job_dir(Path(spec["out_dir"]), spec["stage"], spec["variant"], config.cfg_id, spec["seed"])
    if is_done(directory):
        return read_result(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for stale in directory.glob("*"):  # an unfinished earlier attempt is discarded, not resumed
        stale.unlink()
    log_path = Path(spec["log_dir"]) / f"{spec['stage']}_{spec['variant']}_{config.cfg_id}_s{spec['seed']}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", buffering=1)

    def emit(msg: str) -> None:
        log.write(f"{time.strftime('%H:%M:%S')} pid={os.getpid()} {msg}\n")

    cfg, data = _worker_data(spec["root"], spec["variant"])
    train_env = make_env(cfg, data["train"], mode="train")
    train_eval = make_env(cfg, data["train"], mode="eval")
    val_env = make_env(cfg, data["val"], mode="eval")
    seed = derive_seed(cfg.data.seeds.master, "sac", spec["variant"], config.cfg_id, spec["seed"])
    emit(f"start {spec['stage']} {spec['variant']} {config.cfg_id} seed={spec['seed']} sac_seed={seed} steps={spec['steps']}")
    result = train_run(
        train_env, val_env, config, settings, seed=seed, total_steps=spec["steps"], eval_every=spec["eval_every"],
        out_dir=directory, train_eval_env=train_eval,
        on_checkpoint=lambda r: emit(f"step {r['step']} train {r['train_mean_log_return']:+.5f} val {r['val_mean_log_return']:+.5f}"),
    )
    result.update({"stage": spec["stage"], "variant": spec["variant"], "model_seed": seed, "data_seed": spec["seed"]})
    emit(f"done {result['wall_seconds']:.0f}s best step {result['best']['step']} val {result['best']['val_mean_log_return']:+.5f}")
    log.close()
    tmp = directory / "result.json.tmp"
    tmp.write_text(json.dumps(result, indent=2, default=float))
    tmp.replace(directory / "result.json")
    return result
