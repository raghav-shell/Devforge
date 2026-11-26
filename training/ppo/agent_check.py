"""Load an export through submission/agent.py's own code path, WITHOUT touching submission/weights/.

    uv run --no-sync python -m training.ppo.agent_check --export runs/.../export --wad maps/two_rooms.wad --steps 300

Copies `submission/` (without its weights) into `<export>/../agent_check/submission/`, places the export's
policy.pt + policy.json in that copy's `weights/`, then, in a fresh subprocess with DOOM_AGENT_STRICT=1,
imports that `agent.py`, builds `DoomAgent` and plays `--steps` decisions on the vanilla `doom_env.make_env`.
Reports per-act latency and checks every action is a (9,) int8 in {0, 1}. Any warning about hash or
version mismatch printed by DoomAgent shows up in the stderr echoed here.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

_CHILD = r"""
import importlib.util, json, sys, time
import numpy as np
sub, wad, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
sys.path[:0] = [sub]
spec = importlib.util.spec_from_file_location("agent", f"{sub}/agent.py")
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
assert mod.WEIGHTS_DIR.resolve().as_posix().startswith(sub), mod.WEIGHTS_DIR
import doom_env
env = doom_env.make_env(wad)
agent = mod.DoomAgent(env.observation_space, env.action_space, "cpu")
obs, _ = env.reset(seed=0)
agent.reset()
lat, n, outcome = [], 0, "running"
for t in range(steps):
    t0 = time.perf_counter(); a = agent.act(obs); lat.append(time.perf_counter() - t0)
    assert isinstance(a, np.ndarray) and a.shape == (9,) and a.dtype == np.int8 and set(np.unique(a)) <= {0, 1}, a
    obs, r, term, trunc, info = env.step(a); n += 1
    if term or trunc:
        outcome = "terminated" if term else "truncated"; break
env.close()
lat = np.array(lat) * 1000
print(json.dumps({"agent_py": f"{sub}/agent.py", "weights_dir": str(mod.WEIGHTS_DIR), "steps": n, "outcome": outcome,
                  "act_ms_p50": float(np.percentile(lat, 50)), "act_ms_p99": float(np.percentile(lat, 99)),
                  "act_ms_max": float(lat.max())}))
"""


def check(export_dir: str | Path, wad: str = "maps/two_rooms.wad", steps: int = 300) -> dict:
    export_dir = Path(export_dir).resolve()
    dst = export_dir.parent / "agent_check" / "submission"
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(REPO / "submission", dst, ignore=shutil.ignore_patterns("weights", "__pycache__"))
    (dst / "weights").mkdir()
    for f in ("policy.pt", "policy.json"):
        shutil.copy2(export_dir / f, dst / "weights" / f)
    env = dict(os.environ, DOOM_AGENT_STRICT="1")
    env.pop("PYTHONPATH", None)
    wad_path = str((REPO / wad) if not Path(wad).is_absolute() else Path(wad))
    r = subprocess.run([sys.executable, "-c", _CHILD, str(dst), wad_path, str(steps)], cwd=dst.parent,
                       capture_output=True, text=True, timeout=600, env=env)
    if r.stderr.strip():
        print(r.stderr, file=sys.stderr)
    if r.returncode != 0:
        raise RuntimeError(f"agent check failed (rc={r.returncode}):\n{r.stderr[-4000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--export", required=True)
    ap.add_argument("--wad", default="maps/two_rooms.wad")
    ap.add_argument("--steps", type=int, default=300)
    a = ap.parse_args(argv)
    print(json.dumps(check(a.export, a.wad, a.steps)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
