"""PPO learner / rollout-inference throughput benchmark (owner: rl-training-engineer).

    uv run --no-sync python -m training.benchmarks.train_throughput --envs 16 64 --threads 1 2 4 --devices cpu mps

For each (device, threads, N): builds the REAL DoomPolicy if importable (else ProxyPolicy, labelled),
runs a PPOTrainer on FakeVecEnv for `--iters` iterations after one warm-up iteration, and reports:
  inference_sps  decisions/s of policy.step + sampling + trunc_value during collection (env time excluded)
  learner_sps    decisions/s of the PPO update (epochs x minibatches of forward_sequence + backward)
  env_sps_fake   the fake env only (NOT ViZDoom; use training/benchmarks/env_*.py for real env SPS)
The real-env pipeline rate is bounded by 1 / (1/env_sps + 1/inference_sps + 1/learner_sps).
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import warnings
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO / "submission", REPO):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch  # noqa: E402

from training.configs import TrainConfig  # noqa: E402
from training.ppo.adapters import get_action_api  # noqa: E402
from training.ppo.testing import FakeVecEnv, ProxyPolicy, obs_specs  # noqa: E402
from training.ppo.trainer import PPOTrainer, seed_everything  # noqa: E402


def make_policy():
    try:
        from doom_agent.models.config import PolicyConfig
        from doom_agent.models.policy import DoomPolicy

        return DoomPolicy(PolicyConfig()), "DoomPolicy(v1 default)"
    except ImportError:
        return ProxyPolicy(), "ProxyPolicy (B2 missing; PROXY numbers)"


def bench(device: str, threads: int, n: int, T: int, iters: int, epochs: int, minibatches: int) -> dict:
    torch.set_num_threads(threads)
    seed_everything(0)
    cfg = TrainConfig()
    cfg.ppo.num_envs, cfg.ppo.rollout_len, cfg.ppo.epochs = n, T, epochs
    cfg.ppo.num_minibatches = minibatches
    cfg.ppo.target_kl = 0.0  # measure all minibatches
    cfg.run.device = device
    cfg.validate()
    pol, label = make_policy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api = get_action_api("factored", allow_fallback=True)
    env = FakeVecEnv(n, episode_len=(300, 500))
    tr = PPOTrainer(pol, env, cfg, api, obs_specs=obs_specs(), device=device)
    tr.train_iteration()  # warm-up
    inf = env_t = learn = 0.0
    dev = None
    for _ in range(iters):
        s = tr.train_iteration()
        inf += s["time/policy_inference_s"]
        env_t += s["time/env_s"]
        learn += s["time/learner_s"]
        dev = s["ppo/ratio_check_max_dev"]
    steps = iters * n * T
    return {"device": device, "threads": threads, "num_envs": n, "rollout_len": T, "epochs": epochs,
            "minibatches": minibatches, "policy": label, "params": sum(p.numel() for p in pol.parameters()),
            "inference_sps": steps / inf, "learner_sps": steps / learn, "env_sps_fake": steps / env_t,
            "learner_s_per_iter": learn / iters, "inference_s_per_iter": inf / iters, "ratio_dev_last": dev}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--devices", nargs="+", default=["cpu", "mps"])
    ap.add_argument("--rollout-len", type=int, default=128)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--minibatches", type=int, default=4)
    ap.add_argument("--iters", type=int, default=2)
    ap.add_argument("--out", default=None, help="JSONL output path")
    a = ap.parse_args(argv)
    print(json.dumps({"platform": platform.platform(), "torch": torch.__version__,
                      "mps_available": torch.backends.mps.is_available()}))
    rows = []
    for device in a.devices:
        if device == "mps" and not torch.backends.mps.is_available():
            print(json.dumps({"device": "mps", "skipped": "not available"}))
            continue
        for th in (a.threads if device == "cpu" else [a.threads[0]]):
            for n in a.envs:
                try:
                    r = bench(device, th, n, a.rollout_len, a.iters, a.epochs, a.minibatches)
                except Exception as e:  # report, never hide
                    r = {"device": device, "threads": th, "num_envs": n, "error": f"{type(e).__name__}: {e}"}
                rows.append(r)
                print(json.dumps(r), flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
