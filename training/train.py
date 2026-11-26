"""PPO training entrypoint (owner: rl-training-engineer). See docs/TRAINING.md.

    uv run --no-sync python -m training.train --config training/configs/h1.json --run-id h1-s0
    uv run --no-sync python -m training.train --fake --updates 5            # engine smoke, no ViZDoom
    uv run --no-sync python -m training.train --resume runs/rl-training-engineer/h1-s0 --updates 100

Real runs import the real modules (doom_agent.models from B2, doom_agent.obs.ObsConfig from A2,
training.envs.vector.SubprocVecEnv from A3) and fail with a clear message if one is missing.
`--fake` swaps in training/ppo/testing.py doubles and is marked `"fake": true` in run.json.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT / "submission", REPO_ROOT):  # same import identity as the scorer (MODULE_CONTRACTS 0)
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch  # noqa: E402

from training.configs import TrainConfig  # noqa: E402
from training.ppo.adapters import get_action_api  # noqa: E402
from training.ppo.checkpoint import (  # noqa: E402
    build_checkpoint, ckpt_path, git_info, latest_checkpoint, load_checkpoint, obs_metadata, save_checkpoint)
from training.ppo.logger import MetricsLogger  # noqa: E402
from training.ppo.trainer import PPOTrainer, seed_everything  # noqa: E402

DEFAULT_CONFIG = REPO_ROOT / "training" / "configs" / "h1.json"


class MissingIntegration(ImportError):
    pass


def _need(what: str, owner: str, exc: Exception) -> MissingIntegration:
    return MissingIntegration(f"{what} is not importable ({exc}). It is owned by the {owner} and has not landed "
                              f"yet. Use --fake to smoke-test the PPO engine without it.")


def manifest_ids(paths: list[str]) -> list[str]:
    ids = []
    for p in paths:
        path = Path(p) if Path(p).is_absolute() else REPO_ROOT / p
        data = json.loads(path.read_text())
        if data.get("split") in ("val", "test"):
            raise ValueError(f"{p}: refusing to train on a {data['split']} manifest (EVALUATION_PROTOCOL section 7)")
        ids.extend(m["id"] for m in data.get("maps", []))
    return ids


def build_real(cfg: TrainConfig):
    """Real B2 + A2 + A3 wiring. Returns (policy, vec_env, obs_specs, policy_config, obs_config, action_api, asv)."""
    try:
        from doom_agent.models.action_codec import ACTION_SPACE_VERSION
        from doom_agent.models.config import PolicyConfig
        from doom_agent.models.policy import DoomPolicy
    except ImportError as e:
        raise _need("doom_agent.models (DoomPolicy, PolicyConfig, action_codec)", "model-architect (B1/B2)", e)
    try:
        from doom_agent.obs import ObsConfig
    except ImportError as e:
        raise _need("doom_agent.obs.ObsConfig", "environment-engineer (A2)", e)
    try:
        from training.envs.vector import SubprocVecEnv
    except ImportError as e:
        raise _need("training.envs.vector.SubprocVecEnv", "environment-engineer (A3)", e)
    from doom_agent.obs import FEATURE_SPECS

    try:
        from training.rewards.base import RewardComposer
    except ImportError as e:
        raise _need("training.rewards.base.RewardComposer", "environment-engineer (C1)", e)
    try:  # validate the reward config in the parent, before any worker starts
        RewardComposer(list(cfg.reward.terms))
    except KeyError as e:
        raise MissingIntegration(f"reward config names an unregistered term: {e}. cell_visit is owned by the "
                                 "exploration-engineer, kill/damage_taken by the combat-and-survival-engineer; "
                                 "override reward.terms to registered terms until they land.") from e

    obs_cfg = ObsConfig(**cfg.obs)
    pdict = PolicyConfig().to_dict()
    pdict.update(cfg.model)
    # Flag authority (MODULE_CONTRACTS 3.2): PolicyConfig.use_* derives from ObsConfig.enable_*.
    pdict["use_depth_band"] = bool(getattr(obs_cfg, "enable_depth_band", False))
    pdict["use_egomap"] = bool(getattr(obs_cfg, "enable_egomap", False))
    pcfg = PolicyConfig.from_dict(pdict)
    policy = DoomPolicy(pcfg)
    e = cfg.env
    if e.num_workers * e.envs_per_worker != cfg.ppo.num_envs:
        raise ValueError(f"env.num_workers*envs_per_worker={e.num_workers * e.envs_per_worker} != ppo.num_envs")
    sampler_cfg = {"manifests": e.manifests, "randomize_ticks": e.randomize_ticks, "render_buffers": e.render_buffers,
                   "mods": e.mods, "return_privileged": e.return_privileged,
                   "episodes_per_map": e.episodes_per_map, "step_threads": e.step_threads}
    vec = SubprocVecEnv(sampler_cfg, num_workers=e.num_workers, envs_per_worker=e.envs_per_worker,
                        seed_base=e.seed_base, reward_cfg=cfg.reward.terms, obs_cfg=obs_cfg)
    specs = {k: (tuple(s.shape), s.dtype) for k, s in FEATURE_SPECS.items()}
    api = get_action_api(pcfg.action_head, allow_fallback=False)
    obs_cfg_dict = obs_cfg.to_dict() if hasattr(obs_cfg, "to_dict") else dataclasses.asdict(obs_cfg)
    return policy, vec, specs, pcfg.to_dict(), obs_cfg_dict, api, ACTION_SPACE_VERSION


def build_fake(cfg: TrainConfig):
    from training.ppo.testing import FakePolicy, FakeVecEnv, obs_specs

    hidden = int(cfg.model.get("hidden_size", 16))
    policy = FakePolicy(hidden_size=hidden)
    vec = FakeVecEnv(cfg.ppo.num_envs, seed=cfg.env.seed_base, episode_len=(23, 41, 17))
    api = get_action_api("factored", allow_fallback=True)
    return policy, vec, obs_specs(), {"hidden_size": hidden, "num_logits": 15}, dict(cfg.obs), api, "1.0.0"


def run(cfg: TrainConfig, *, fake: bool = False, resume: str | None = None, max_updates: int | None = None,
        run_dir: str | Path | None = None, max_minutes: float | None = None) -> Path:
    if resume:
        rpath = Path(resume)
        ck = latest_checkpoint(rpath) if rpath.is_dir() else rpath
        if ck is None:
            raise FileNotFoundError(f"no checkpoint under {resume}")
        ckpt = load_checkpoint(ck)
        cfg = TrainConfig.from_dict(ckpt["train_config"])
        run_dir = Path(run_dir) if run_dir else (ck.parent.parent if rpath.is_file() else rpath)
    else:
        ckpt = None
    torch.set_num_threads(cfg.run.torch_threads)
    seed_everything(cfg.run.seed)

    policy, vec, specs, pcfg, ocfg, api, asv = (build_fake if fake else build_real)(cfg)
    if run_dir is None:
        rid = cfg.run.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + ("-fake" if fake else "")
        run_dir = REPO_ROOT / cfg.run.out_root / rid
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = MetricsLogger(run_dir, tensorboard=cfg.run.tensorboard)
    trainer = PPOTrainer(policy, vec, cfg, api, obs_specs=specs, device=cfg.run.device, logger=logger)
    if ckpt is not None:
        if not fake and ckpt["obs_pipeline_hash"] != obs_metadata()["obs_pipeline_hash"]:
            raise RuntimeError("resume refused: checkpoint obs_pipeline_hash differs from the current tree")
        trainer.load_state_dict(ckpt, restore_env=True)

    manifests = [] if fake else manifest_ids(cfg.env.manifests)
    commit, dirty = git_info()
    run_meta = {"config": cfg.to_dict(), "fake": fake, "git_commit": commit, "git_dirty": dirty,
                "action_api_source": api.source, "tensorboard": logger.tb_status, "device": cfg.run.device,
                "torch_threads": torch.get_num_threads(), "torch_version": torch.__version__,
                "train_manifest_ids": manifests, "policy_config": pcfg, "obs_config": ocfg,
                "resumed_from": str(resume) if resume else None, **obs_metadata()}
    (run_dir / "run.json").write_text(json.dumps(run_meta, indent=2, sort_keys=True, default=str))

    def save(wall):
        ck = build_checkpoint(trainer, policy_config=pcfg, obs_config=ocfg, action_space_version=asv,
                              train_manifest_ids=manifests, wall_time_s=wall, extra={"fake": fake})
        return save_checkpoint(ckpt_path(run_dir, trainer.update_idx), ck)

    total = cfg.ppo.total_updates
    target = total if max_updates is None else min(total, trainer.update_idx + max_updates)
    t0 = time.time()
    try:
        while trainer.update_idx < target:
            if max_minutes is not None and time.time() - t0 >= 60.0 * max_minutes:
                print(f"[stop] wall-clock limit {max_minutes} min reached at update {trainer.update_idx}", flush=True)
                break
            s = trainer.train_iteration()
            print(f"[update {trainer.update_idx}/{total}] step={trainer.global_step} sps={s['sps/total']:.0f} "
                  f"ep_ret={s.get('episode/return_shaped_mean', float('nan')):.3f} "
                  f"compl={s.get('episode/completion_rate', float('nan')):.3f} kl={s.get('ppo/approx_kl', 0):.4f} "
                  f"ratio_dev={s['ppo/ratio_check_max_dev']:.1e} crashed={s['env/crashed_total']}", flush=True)
            if trainer.update_idx % max(1, cfg.run.ckpt_every) == 0:
                save(time.time() - t0)
        save(time.time() - t0)
    finally:
        logger.close()
        vec.close()
    return run_dir


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Custom recurrent PPO training (DOOMSDAY)")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG))
    ap.add_argument("--set", action="append", default=[], metavar="group.key=value", help="config override (JSON value)")
    ap.add_argument("--run-id", default=None)
    ap.add_argument("--resume", default=None, help="run dir (latest ckpt) or checkpoint file")
    ap.add_argument("--updates", type=int, default=None, help="stop after this many updates (this invocation)")
    ap.add_argument("--max-minutes", type=float, default=None, help="stop (and checkpoint) after this wall time")
    ap.add_argument("--fake", action="store_true", help="fake policy + fake vec env (engine smoke only)")
    a = ap.parse_args(argv)
    cfg = TrainConfig.load(a.config).with_overrides(a.set)
    if a.run_id:
        cfg.run.run_id = a.run_id
    try:
        run_dir = run(cfg, fake=a.fake, resume=a.resume, max_updates=a.updates, max_minutes=a.max_minutes)
    except MissingIntegration as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    print(f"run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
