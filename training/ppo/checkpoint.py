"""Training checkpoints, format "ckpt-1" (docs/MODULE_CONTRACTS.md section 8.1).

Path: runs/rl-training-engineer/<run-id>/checkpoints/ckpt_<update:06d>.pt (torch.save).

ckpt-1 holds Python/numpy RNG state objects (tuples with numpy arrays), so it is loaded with
`torch.load(weights_only=False)`. Only load checkpoints you produced. The EXPORT (policy.pt) is plain
tensors and loads with weights_only=True (training/ppo/export.py).

Exact resume: model, optimiser, scheduler, update / global_step counters, RNG states (python, numpy,
torch global, trainer generator), return-norm statistics and per-env return accumulators, and the env
carry (last obs, episode_start, hidden state). The carry is only usable when the vector env can be
snapshotted (`get_state`/`set_state`, the test fake). A real SubprocVecEnv cannot, so a real resume
restarts episodes (envs reset, hidden state zero), which changes the data stream but not the learner.
"""

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any

import torch

FORMAT_VERSION = "ckpt-1"
REPO_ROOT = Path(__file__).resolve().parents[2]


def git_info(repo: Path = REPO_ROOT) -> tuple[str, bool]:
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
                                check=True, timeout=10).stdout.strip()
        dirty = bool(subprocess.run(["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True,
                                    check=True, timeout=30).stdout.strip())
        return commit, dirty
    except Exception:
        return "unknown", True


def obs_metadata() -> dict:
    from doom_agent.obs import OBS_SCHEMA_VERSION, obs_pipeline_hash

    return {"obs_schema_version": OBS_SCHEMA_VERSION, "obs_pipeline_hash": obs_pipeline_hash()}


def build_checkpoint(trainer, *, policy_config: dict, obs_config: dict, action_space_version: str,
                     train_manifest_ids: list[str], aux_state: dict | None = None,
                     privileged_critic_state: dict | None = None, wall_time_s: float = 0.0,
                     extra: dict | None = None) -> dict[str, Any]:
    cfg = trainer.cfg
    commit, dirty = git_info()
    st = trainer.state_dict()
    ckpt = {
        "format_version": FORMAT_VERSION,
        "policy_state": st["policy_state"],
        "policy_config": dict(policy_config),
        "aux_state": aux_state,
        "privileged_critic_state": privileged_critic_state,
        "optimizer_state": st["optimizer_state"],
        "scheduler_state": st["scheduler_state"],
        "rng_states": st["rng_states"],
        "return_norm_state": st["return_norm_state"],
        "update": st["update"],
        "global_step": st["global_step"],
        "wall_time_s": float(wall_time_s),
        "counters": st["counters"],
        "env_carry": st["env_carry"],
        **obs_metadata(),
        "obs_config": dict(obs_config),
        "action_space_version": action_space_version,
        "train_manifest_ids": list(train_manifest_ids),
        "train_seeds": {"seed_base": cfg.env.seed_base, "run_seed": cfg.run.seed},
        "ppo_config": cfg.to_dict()["ppo"],
        "reward_config": cfg.to_dict()["reward"],
        "env_config": cfg.to_dict()["env"],
        "train_config": cfg.to_dict(),
        "git_commit": commit,
        "git_dirty": dirty,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        ckpt.update(extra)
    for k, v in ckpt["train_seeds"].items():
        if not 0 <= int(v) < 10000:
            raise ValueError(f"training seed {k}={v} >= 10000 is reserved for val/test")
    return ckpt


def save_checkpoint(path: str | Path, ckpt: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(ckpt, tmp)
    tmp.replace(path)  # atomic: a crash mid-save never leaves a truncated checkpoint
    return path


def load_checkpoint(path: str | Path, map_location: str = "cpu") -> dict:
    ckpt = torch.load(path, map_location=map_location, weights_only=False)
    if ckpt.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {ckpt.get('format_version')!r} != {FORMAT_VERSION!r}")
    return ckpt


def ckpt_path(run_dir: str | Path, update: int) -> Path:
    return Path(run_dir) / "checkpoints" / f"ckpt_{update:06d}.pt"


def latest_checkpoint(run_dir: str | Path) -> Path | None:
    cks = sorted((Path(run_dir) / "checkpoints").glob("ckpt_*.pt"))
    return cks[-1] if cks else None


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


__all__ = ["FORMAT_VERSION", "build_checkpoint", "ckpt_path", "file_sha256", "git_info", "latest_checkpoint",
           "load_checkpoint", "obs_metadata", "save_checkpoint"]
