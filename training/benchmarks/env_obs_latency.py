"""Microbenchmark: ObsPreprocessor.process + commit_action latency on 1 thread (OBSERVATION_SCHEMA section 5.9).

    OMP_NUM_THREADS=1 uv run --no-sync python -m training.benchmarks.env_obs_latency [--steps 2000]

Raw observations are recorded first from a random-policy rollout on maps/two_rooms.wad through the
official `doom_env.make_env`, so only the preprocessor is timed.
"""

from __future__ import annotations

import argparse
import copy
import os
import platform
import sys
import time
from pathlib import Path

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_var, "1")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "submission") not in sys.path:
    sys.path.insert(0, str(REPO / "submission"))

from doom_agent.obs import ObsPreprocessor, obs_pipeline_hash  # noqa: E402


def record(steps: int, seed: int) -> list[tuple[dict, np.ndarray | None]]:
    from doom_env import make_env

    rng = np.random.default_rng(seed)
    env = make_env(str(REPO / "maps" / "two_rooms.wad"), "MAP01", render_buffers=False)
    out: list[tuple[dict, np.ndarray | None]] = []
    try:
        obs, _ = env.reset(seed=seed)
        while len(out) < steps:
            a = (rng.random(9) < 0.3).astype(np.int8)
            out.append((copy.deepcopy(obs), a))
            obs, _, term, trunc, _ = env.step(a)
            if term or trunc:
                out.append((copy.deepcopy(obs), None))
                obs, _ = env.reset(seed=int(rng.integers(0, 10000)))
    finally:
        env.close()
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seq = record(args.steps, args.seed)
    pre = ObsPreprocessor()
    times = []
    for _ in range(args.repeats):
        pre.reset()
        for raw, a in seq:
            t0 = time.perf_counter()
            pre.process(raw)
            if a is None:
                pre.reset()
            else:
                pre.commit_action(a)
            times.append(time.perf_counter() - t0)
    ms = np.asarray(times[100:]) * 1e3
    p50, p90, p99 = np.percentile(ms, [50, 90, 99])
    print(f"machine {platform.machine()} {platform.platform()} python {platform.python_version()} numpy {np.__version__}")
    print(f"obs_pipeline_hash {obs_pipeline_hash()}")
    print(f"calls {ms.size}  p50 {p50:.4f} ms  p90 {p90:.4f} ms  p99 {p99:.4f} ms  max {ms.max():.4f} ms")


if __name__ == "__main__":
    main()
