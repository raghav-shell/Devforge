"""Where does an env decision's time go? Per-stage cost of the training env path (single process), plus IPC cost.

Stages (microseconds per decision, mean over the timed steps):
  raw_step      doom_env DoomEnv.step (engine 4 tics + official sensors: rays, entities)
  privileged    env.privileged_state() (needed twice per step by StepContext: before is cached, after is fresh)
  reward        RewardComposer(ctx) with the given terms
  preprocess    ObsPreprocessor.commit_action + process (the eval-identical feature path)
  trainenv      TrainEnv.step end-to-end (all of the above + checks + bookkeeping)
Then IPC: SubprocVecEnv(1 worker x k envs) SPS vs the same k TrainEnvs stepped in-process.

    uv run --no-sync python -m training.benchmarks.env_step_profile --maps two_rooms,gen_keys,fd2_map01 --steps 1500
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "submission"), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

MAPS = {
    "two_rooms": ("maps/two_rooms.wad", "MAP01"),
    "gen_keys": ("maps/generated/keys/keys-a-000000.wad", "MAP01"),
    "gen_full": ("maps/generated/full/full-a-000000.wad", "MAP01"),
    "fd2_map01": ("@vizdoom/freedoom2.wad", "MAP01"),
}


def _terms(name: str) -> list[dict]:
    if name == "none":
        return []
    if name == "env_default":
        return [{"name": "env_default", "weight": 1.0}]
    from training.rewards.base import REWARD_TERMS, load_term_modules

    load_term_modules()
    return [{"name": n, "weight": 1.0} for n in sorted(REWARD_TERMS) if not n.startswith("test_")]


def profile_stages(map_key: str, steps: int, terms: str) -> dict:
    from doom_agent.obs import ObsConfig, ObsPreprocessor
    from doom_env import make_env

    from training.envs.factory import EnvSpec, build_composer, make_train_env
    from training.envs.maps import resolve_wad
    from training.rewards.base import StepContext

    wad, m = MAPS[map_key]
    rng = np.random.default_rng(0)
    acts = rng.integers(0, 2, (steps, 9), dtype=np.int8)
    acts[:, 6] = 0  # no attack, keeps episodes alive longer on monster maps

    env = make_env(str(resolve_wad(wad)), m, episode_ticks=100_000, render_buffers=False)
    obs, info = env.reset(seed=1)
    pre = ObsPreprocessor(ObsConfig())
    comp = build_composer(_terms(terms))
    from training.envs.inventory import map_inventory

    priv = env.privileged_state()
    comp.reset(priv, obs, map_inventory(env), EnvSpec(map_key, wad, m, 100_000))
    pre.process(obs)
    t = {"raw_step": 0.0, "privileged": 0.0, "reward": 0.0, "preprocess": 0.0}
    n = 0
    for k, a in enumerate(acts):
        t0 = time.perf_counter()
        obs, r, te, tr, info = env.step(a)
        t1 = time.perf_counter()
        p2 = env.privileged_state()
        t2 = time.perf_counter()
        comp(StepContext(priv, p2, obs, a, te, tr, r, info, k))
        t3 = time.perf_counter()
        pre.commit_action(a)
        pre.process(obs)
        t4 = time.perf_counter()
        priv = p2
        t["raw_step"] += t1 - t0
        t["privileged"] += t2 - t1
        t["reward"] += t3 - t2
        t["preprocess"] += t4 - t3
        n += 1
        if te or tr:
            break
    env.close()

    tenv = make_train_env(EnvSpec(map_key, wad, m, 100_000), seed=1, reward_cfg=_terms(terms))
    tenv.reset()
    t0 = time.perf_counter()
    k2 = 0
    for a in acts[:n]:
        out = tenv.step(a)
        k2 += 1
        if out[2] or out[3]:
            break
    t["trainenv"] = time.perf_counter() - t0
    tenv.close()
    us = {k: round(v / (k2 if k == "trainenv" else n) * 1e6, 1) for k, v in t.items()}
    return {"map": map_key, "terms": terms, "steps": n, **{f"{k}_us": v for k, v in us.items()},
            "trainenv_sps": round(1e6 / us["trainenv"], 0), "raw_sps": round(1e6 / us["raw_step"], 0)}


def profile_ipc(map_key: str, k: int, steps: int, terms: str) -> dict:
    from training.envs.factory import EnvSpec, make_train_env
    from training.envs.vector import SubprocVecEnv

    wad, m = MAPS[map_key]
    rng = np.random.default_rng(0)
    entry = {"id": map_key, "wad": wad, "map": m, "wad_sha256": "x", "source_group": f"bench:{map_key}",
             "episode_ticks": 4200}
    envs = [make_train_env(EnvSpec(map_key, wad, m, 4200), seed=i, reward_cfg=_terms(terms)) for i in range(k)]
    for e in envs:
        e.reset()
    t0 = time.perf_counter()
    for _ in range(steps):
        a = rng.integers(0, 2, (k, 9), dtype=np.int8)
        for i, e in enumerate(envs):
            out = e.step(a[i])
            if out[2] or out[3]:
                e.reset()
    inproc = steps * k / (time.perf_counter() - t0)
    for e in envs:
        e.close()
    with SubprocVecEnv({"entries": [entry], "randomize_ticks": False}, num_workers=1, envs_per_worker=k,
                       reward_cfg=_terms(terms)) as v:
        v.reset()
        t0 = time.perf_counter()
        for _ in range(steps):
            v.step(rng.integers(0, 2, (k, 9), dtype=np.int8))
        vec = steps * k / (time.perf_counter() - t0)
    return {"map": map_key, "envs": k, "inproc_sps": round(inproc), "vec_1worker_sps": round(vec),
            "ipc_overhead_pct": round(100 * (1 - vec / inproc), 1)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.benchmarks.env_step_profile")
    p.add_argument("--maps", default="two_rooms,gen_keys,fd2_map01")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--terms", default="all", choices=("none", "env_default", "all"))
    p.add_argument("--ipc-envs", type=int, default=4)
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    res = {"loadavg_start": os.getloadavg(), "stages": [], "ipc": []}
    print("| map | terms | steps | raw step µs | privileged µs | reward µs | preprocess µs | TrainEnv.step µs | raw SPS | TrainEnv SPS |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for mk in a.maps.split(","):
        r = profile_stages(mk, a.steps, a.terms)
        res["stages"].append(r)
        print(f"| {mk} | {a.terms} | {r['steps']} | {r['raw_step_us']} | {r['privileged_us']} | {r['reward_us']} | "
              f"{r['preprocess_us']} | {r['trainenv_us']} | {r['raw_sps']:.0f} | {r['trainenv_sps']:.0f} |", flush=True)
    print("\n| map | envs | in-process SPS | SubprocVecEnv(1 worker) SPS | IPC overhead |")
    print("|---|---|---|---|---|")
    for mk in a.maps.split(","):
        r = profile_ipc(mk, a.ipc_envs, max(200, a.steps // 4), a.terms)
        res["ipc"].append(r)
        print(f"| {mk} | {r['envs']} | {r['inproc_sps']} | {r['vec_1worker_sps']} | {r['ipc_overhead_pct']} % |", flush=True)
    res["loadavg_end"] = os.getloadavg()
    print(f"\nloadavg start {res['loadavg_start']} end {res['loadavg_end']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
