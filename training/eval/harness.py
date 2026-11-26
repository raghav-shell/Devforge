"""Evaluation harness skeleton (docs/EVALUATION_PROTOCOL.md §5). Same agent loop as validate.py:
the agent runs in a separate process behind doom_env.remote with THREAD_ENV (1 thread), timeouts and
the per-map compute budget. Unlike validate.py, an agent failure does not abort the run: the player
stands still (NOOP) until tick cap / death / exit and the agent is restarted for the next map, as
docs/EVALUATION.md "When your agent stops" describes.

    uv run --no-sync python -m training.eval.harness --split smoke --seeds 0 1 2
    uv run --no-sync python -m training.eval.harness --split smoke --only two_rooms --seeds 0

Test split: refused unless --test-purpose is given; every test run is appended to
reports/eval/test_access_log.jsonl before any episode is played.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

from . import splits
from .inventory import map_inventory
from .metrics.core import EpisodeTrace, classify_failure, core_metrics
from .proxy_score import PROXY_VERSION, TIER_WEIGHT, proxy_components
from .stats import cluster_bootstrap

ROOT = splits.ROOT
OWNER = "evaluation-and-generalization-researcher"
TEST_LOG = ROOT / "reports" / "eval" / "test_access_log.jsonl"
NOOP = np.zeros(9, np.int8)
DEFAULT_SEEDS = {"smoke": [0, 1, 2], "train": list(range(100, 103)), "val": list(range(10_000, 10_005)),
                 "test": list(range(20_000, 20_010))}


def tree_hash(paths: list[Path], base: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(str(p.relative_to(base)).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


def provenance(repo: Path) -> dict:
    import doom_env
    import vizdoom
    sub = repo / "submission"
    code = [p for p in sub.rglob("*.py") if "__pycache__" not in p.parts]
    weights = [p for p in (sub / "weights").rglob("*") if p.is_file()] if (sub / "weights").is_dir() else []
    try:
        commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip() or "none"
        dirty = bool(subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--", "submission"],
                                    capture_output=True, text=True).stdout.strip())
    except OSError:
        commit, dirty = "none", True
    envfiles = [p for p in (ROOT / "env" / "doom_env").glob("*.py")]
    return {"git_commit": commit, "submission_dirty": dirty, "agent_code_hash": tree_hash(code, repo),
            "weights_hash": tree_hash(weights, repo) if weights else "none", "doom_env_hash": tree_hash(envfiles, ROOT),
            "contract_version": doom_env.CONTRACT_VERSION, "vizdoom_version": vizdoom.__version__,
            "python": platform.python_version(), "platform": platform.platform(), "proxy_version": PROXY_VERSION}


class AgentProc:
    """One agent server process, started exactly like validate.dry_run()."""

    def __init__(self, repo: Path, device: str, extra_env: dict):
        from doom_env import limits as L
        from doom_env import remote
        self.L, self.remote = L, remote
        self.tmp = Path(tempfile.mkdtemp(prefix="doom-eval-"))
        self.server = remote.listen(self.tmp / "agent.sock")
        self.log = open(self.tmp / "agent.log", "wb")
        env_vars = {**os.environ, "PYTHONUNBUFFERED": "1", **L.THREAD_ENV, **extra_env}
        cpu = None
        if hasattr(os, "sched_getaffinity"):
            cpus = sorted(os.sched_getaffinity(0)); cpu = {cpus[-1]} if len(cpus) > 1 else None
        self.pinned = cpu is not None
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "doom_env.remote", "serve", "--socket", remote.address(self.server), "--repo", str(repo), "--device", device],
            stdout=self.log, stderr=subprocess.STDOUT, env=env_vars, cwd=self.tmp,
            preexec_fn=(lambda: os.sched_setaffinity(0, cpu)) if cpu else None)
        t0 = time.perf_counter()
        self.conn = remote.AgentConnection.accept(self.server, L.INIT_TIMEOUT_S, alive=lambda: self.proc.poll() is None)
        self.conn.init(device, timeout=max(1.0, L.INIT_TIMEOUT_S - (time.perf_counter() - t0)))
        self.init_s = time.perf_counter() - t0

    def close(self):
        try:
            self.conn.close()
        finally:
            self.server.close()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait()
            self.log.close()
            shutil.rmtree(self.tmp, ignore_errors=True)


def play(agent: AgentProc | None, entry: dict, seed: int, record: Path | None) -> tuple[dict, EpisodeTrace, dict]:
    """One episode. Returns (metrics, trace, timing). On AgentFailure the player stands still."""
    from doom_env import make_env
    L, remote = (agent.L, agent.remote) if agent else (None, None)
    ticks = int(entry["episode_ticks"])
    wad = entry["wad"]
    if wad.startswith("@vizdoom/"):
        import vizdoom
        wad = str(Path(vizdoom.__file__).parent / wad.removeprefix("@vizdoom/"))
    else:
        wad = str(ROOT / wad)
    env = make_env(wad, entry["map"], episode_ticks=ticks)
    tr = EpisodeTrace(episode_ticks=ticks)
    lat: list[float] = []
    try:
        obs, _ = env.reset(seed=seed, options={"record_path": str(record)} if record else None)
        inv = map_inventory(env)
        door_ids = sorted(inv.door_sectors)
        tr.record(env.privileged_state(), obs, door_ids, None, 0.0)
        failed = agent is None
        if agent is not None:
            try:
                agent.conn.reset(L.RESET_TIMEOUT_S)
            except remote.AgentFailure as e:
                failed, tr.agent_failure, tr.failure_step = True, f"{e.kind}@reset", 0
        budget = L.agent_budget_s(ticks) if L else 0.0
        while True:
            action = NOOP
            if not failed:
                try:
                    remaining = budget - sum(lat)
                    if remaining <= 0:
                        raise remote.AgentFailure("timeout", "agent compute budget used up", "episode")
                    limit = L.FIRST_ACT_TIMEOUT_S if tr.steps == 0 else L.ACT_TIMEOUT_S
                    t = time.perf_counter()
                    action = agent.conn.act(obs, timeout=min(limit, remaining))
                    lat.append(time.perf_counter() - t)
                except remote.AgentFailure as e:
                    failed, tr.agent_failure, tr.failure_step = True, f"{e.kind}@step{tr.steps}", tr.steps
                    action = NOOP
            obs, r, term, trunc, info = env.step(action)
            tr.steps += 1
            st = env.privileged_state()
            tr.record(st, obs, door_ids, action, r)
            if term or trunc:
                tr.outcome = "exit" if info["exit_reached"] else ("death" if term else "timeout")
                want = {"exit": (False, False, True), "death": (True, False, False), "timeout": (False, True, False)}[tr.outcome]
                tr.outcome_consistent = st["finished"] and (st["dead"], st["timeout"], st["exit_reached"]) == want
                break
    finally:
        env.close()
    m = core_metrics(tr, inv)
    m.update(proxy_components(m))
    m["core.failure_class"] = classify_failure(m)
    timing = {"agent_s": sum(lat), "budget_s": budget, "p50_ms": 1000 * float(np.median(lat)) if lat else 0.0,
              "p99_ms": 1000 * float(np.percentile(lat, 99)) if lat else 0.0, "max_ms": 1000 * max(lat) if lat else 0.0}
    return m, tr, timing


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", required=True, choices=splits.SPLITS)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--only", nargs="*", help="map ids to run (default: all, manifest order)")
    ap.add_argument("--mode", default="default", choices=["default", "sample", "greedy"],
                    help="inference mode passed via DOOM_AGENT_INFERENCE (PROPOSED CONTRACT, §9.2)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--repo", default=str(ROOT))
    ap.add_argument("--record", action="store_true", help="save ViZDoom .lmp replays")
    ap.add_argument("--purpose", default="", help="free text, stored with the run")
    ap.add_argument("--test-purpose", default="", help="REQUIRED for --split test; logged")
    ap.add_argument("--run-id", default="")
    args = ap.parse_args()

    if args.split == "test" and not args.test_purpose.strip():
        print("refusing to evaluate on the test split without --test-purpose (see protocol §7)", file=sys.stderr)
        return 2
    probs = splits.validate_all()
    if probs:
        print("manifest problems:\n  " + "\n  ".join(probs), file=sys.stderr)
        return 2
    man = splits.load(args.split)
    entries = [e for e in man["maps"] if not args.only or e["id"] in args.only]
    if not entries:
        print("no maps selected", file=sys.stderr)
        return 2
    seeds = args.seeds if args.seeds is not None and len(args.seeds) else DEFAULT_SEEDS[args.split]
    repo = Path(args.repo).resolve()
    prov = provenance(repo)
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = args.run_id or f"{stamp}-{args.split}-{prov['agent_code_hash'][:8]}-{args.mode}"
    out = ROOT / "runs" / OWNER / run_id
    out.mkdir(parents=True, exist_ok=True)
    header = {"run_id": run_id, "utc": stamp, "split": args.split, "seeds": seeds, "mode": args.mode,
              "maps": [e["id"] for e in entries], "purpose": args.purpose, "argv": sys.argv, **prov}
    if args.split == "test":
        TEST_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_LOG, "a") as f:
            f.write(json.dumps({**header, "test_purpose": args.test_purpose}) + "\n")
    (out / "run.json").write_text(json.dumps(header, indent=2))
    extra_env = {} if args.mode == "default" else {"DOOM_AGENT_INFERENCE": args.mode}
    rows = []
    for seed in seeds:  # one agent init per pass over the suite, maps in manifest order (like the scorer)
        extra = {**extra_env, "DOOM_AGENT_SEED": str(seed)} if extra_env else extra_env
        agent = AgentProc(repo, args.device, extra)
        try:
            for e in entries:
                rec = out / f"{e['id']}_s{seed}.lmp" if args.record else None
                m, tr, timing = play(agent, e, seed, rec)
                np.savez_compressed(out / f"{e['id']}_s{seed}.npz", x=tr.x, y=tr.y, angle=tr.angle, health=tr.health,
                                    kills=tr.kills, damage=tr.damage_taken, keys=np.array(tr.keys),
                                    actions=np.array([a or [0] * 9 for a in tr.actions], np.int8))
                row = {"map": e["id"], "tier": e["tier"], "seed": seed, "mode": args.mode, **m,
                       **{f"time.{k}": v for k, v in timing.items()}}
                rows.append(row)
                print(f"{e['id']:<28} seed={seed:<6} {m['core.outcome']:<8} steps={m['core.steps']:<4} "
                      f"proxy={m['proxy.score']:6.2f} cells={m['core.cells_visited']}/{m['core.cells_total']} "
                      f"area={m['core.area_coverage']:.2f} kills={m['core.kills']}/{m['core.monsters_present']} "
                      f"keys={m['core.keys_picked']}/{m['core.keys_present']} doors={m['core.doors_opened']}/{m['core.doors_total']} "
                      f"locked={m['core.locked_opened']}/{m['core.locked_total']} stuck={m['core.stuck_steps_frac']:.2f} "
                      f"class={m['core.failure_class']} p99={timing['p99_ms']:.1f}ms", flush=True)
                if m["core.agent_failure"]:  # restart the agent for the next map, like the scorer
                    agent.close(); agent = AgentProc(repo, args.device, extra)
        finally:
            agent.close()
    with open(out / "episodes.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r, default=float) + "\n")
    by_map: dict[str, list[float]] = {}
    tiers = {}
    for r in rows:
        by_map.setdefault(r["map"], []).append(r["proxy.score"]); tiers[r["map"]] = r["tier"]
    w = {k: TIER_WEIGHT.get(t, 1.0) for k, t in tiers.items()}
    pt, lo, hi = cluster_bootstrap(by_map, weights=w)
    summary = {
        **header, "n_episodes": len(rows),
        "proxy_suite": {"mean": pt, "ci95": [lo, hi], "weighting": "tier-weighted mean of per-map means"},
        "completion_rate": float(np.mean([r["core.completed"] for r in rows])),
        "survival_rate": float(np.mean([r["core.survived"] for r in rows])),
        "env_return_mean": float(np.mean([r["core.env_return"] for r in rows])),
        "agent_failures": sum(bool(r["core.agent_failure"]) for r in rows),
        "failure_classes": {c: sum(r["core.failure_class"] == c for r in rows) for c in sorted({r["core.failure_class"] for r in rows})},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=float))
    print(json.dumps({k: summary[k] for k in ("run_id", "n_episodes", "proxy_suite", "completion_rate", "survival_rate",
                                              "env_return_mean", "agent_failures", "failure_classes")}, indent=2, default=float))
    print(f"outputs: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
