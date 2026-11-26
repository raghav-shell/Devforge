"""SubprocVecEnv throughput: env steps/s (decisions/s, summed over envs) vs workers x envs_per_worker.

The measured loop is the real training worker path: random MultiBinary actions -> TrainEnv.step (official env, reward
composer, ObsPreprocessor, auto-reset) -> stacked features back to the parent. No policy, so this is the env ceiling.

    uv run --no-sync python -m training.benchmarks.env_vector_throughput                       # full grid
    uv run --no-sync python -m training.benchmarks.env_vector_throughput --workers 8 --epw 8 --maps fd2_map01 --seconds 10
    ... --out runs/environment-engineer/bench/vector.json

Map sets: `generated` = every map in maps/generated (all stages, both families, map sampled per episode);
`fd2_map01` = freedoom2 MAP01 only. Numbers depend on machine load: the script records os.getloadavg().
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_ROOT / "submission"), str(_ROOT)):  # top-level `doom_agent`, as the scorer imports it
    if _p not in sys.path:
        sys.path.insert(0, _p)


def sampler_cfg(maps: str, *, render: bool, episodes_per_map: int, step_threads: bool = True) -> dict:
    from training.envs import maps as M

    if maps == "generated":
        pool = M.generated_entries()
    elif maps == "fd2_map01":
        pool = [e for e in M.freedoom2_entries() if e.map == "MAP01"]
    elif maps in ("smoke", "two_rooms"):  # the trainer's real-env smoke workload (docs/TRAINING.md §6.1)
        man = json.loads((M.ROOT / "maps" / "splits" / "smoke.json").read_text())["maps"]
        return {"entries": [m for m in man if maps == "smoke" or m["id"] == "two_rooms"], "randomize_ticks": True,
                "render_buffers": render, "episodes_per_map": episodes_per_map, "step_threads": step_threads}
    else:
        raise ValueError(maps)
    ids, groups, pairs = M.heldout_keys()  # never benchmark (or train) on val/test maps
    entries = [e.manifest("train") for e in pool
               if e.id not in ids and e.source_group not in groups and f"{e.wad_sha256}:{e.map}" not in pairs]
    if not entries:
        raise ValueError(f"{maps}: every map is held out in val/test")
    return {"entries": entries, "randomize_ticks": True, "render_buffers": render, "episodes_per_map": episodes_per_map,
            "step_threads": step_threads}


def measure(maps: str, workers: int, epw: int, seconds: float, *, render: bool = False, episodes_per_map: int = 1,
            warmup: int = 30, seed_base: int = 0, step_threads: bool = True) -> dict:
    from training.envs.vector import SubprocVecEnv

    t0 = time.perf_counter()
    v = SubprocVecEnv(sampler_cfg(maps, render=render, episodes_per_map=episodes_per_map, step_threads=step_threads),
                      num_workers=workers,
                      envs_per_worker=epw, seed_base=seed_base, reward_cfg=[{"name": "env_default", "weight": 1.0}])
    try:
        v.reset()
        startup = time.perf_counter() - t0
        rng = np.random.default_rng(0)
        n = v.num_envs
        for _ in range(warmup):
            v.step(rng.integers(0, 2, (n, 9), dtype=np.int8))
        steps, dones = 0, 0
        t1 = time.perf_counter()
        while time.perf_counter() - t1 < seconds:
            r = v.step(rng.integers(0, 2, (n, 9), dtype=np.int8))
            steps += 1
            dones += int(r.episode_start.sum())
        dt = time.perf_counter() - t1
        st = v.stats()
    finally:
        v.close()
    return {"maps": maps, "workers": workers, "envs_per_worker": epw, "num_envs": n, "render_buffers": render,
            "step_threads": step_threads,
            "episodes_per_map": episodes_per_map, "sps": round(steps * n / dt, 1), "vec_steps_per_s": round(steps / dt, 2),
            "seconds": round(dt, 2), "episodes_done": dones, "startup_s": round(startup, 2), "crashes": st["crashes"],
            "engine_opens": st["engine_opens"]}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.benchmarks.env_vector_throughput")
    p.add_argument("--workers", default="1,2,4,8,10")
    p.add_argument("--epw", default="1,4,8")
    p.add_argument("--maps", default="generated,fd2_map01")
    p.add_argument("--seconds", type=float, default=6.0)
    p.add_argument("--render", action="store_true", help="render_buffers=True (rgb/depth filled)")
    p.add_argument("--episodes-per-map", type=int, default=1)
    p.add_argument("--no-threads", action="store_true", help="step slots sequentially inside a worker (old behaviour)")
    p.add_argument("--out", default="")
    a = p.parse_args(argv)
    rows = []
    meta = {"platform": platform.platform(), "cpu_count": os.cpu_count(), "python": sys.version.split()[0],
            "loadavg_start": os.getloadavg()}
    print("| maps | workers | envs/worker | N | threads | render | eps/map | SPS | vec steps/s | engine opens | startup s |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for maps in a.maps.split(","):
        for w in (int(x) for x in a.workers.split(",")):
            for e in (int(x) for x in a.epw.split(",")):
                r = measure(maps, w, e, a.seconds, render=a.render, episodes_per_map=a.episodes_per_map,
                            step_threads=not a.no_threads)
                rows.append(r)
                print(f"| {maps} | {w} | {e} | {r['num_envs']} | {r['step_threads']} | {r['render_buffers']} | {r['episodes_per_map']} | "
                      f"{r['sps']:.0f} | {r['vec_steps_per_s']:.1f} | {r['engine_opens']} | {r['startup_s']} |", flush=True)
    meta["loadavg_end"] = os.getloadavg()
    print(f"\nloadavg start {meta['loadavg_start']} end {meta['loadavg_end']}")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps({"meta": meta, "rows": rows}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
