"""Smoke-load every registry entry through the official `doom_env.make_env` (training/diagnostics only).

Per map: make_env + reset + N seeded random steps (reset on episode end). Records whether it loads, sector counts
(WAD vs engine), geometry source, door / locked-door sectors from mapinfo, how many static lines are classified
door / locked_door at reset, and which ray hit types were actually observed. Generated maps additionally check
their sidecar: mapinfo door map == generator door map, spawn position, and thing classes in the engine.

Reads wrapper internals (`env._geometry`, `env._game`) for diagnostics; never used by agents.

    uv run --no-sync python -m training.envs.smoke_load --sources generated,freedoom2 --jobs 4 \
        --out runs/environment-engineer/smoke/smoke.json
"""

from __future__ import annotations

import argparse
import collections
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

RAY_NAMES = ("none", "wall", "door", "locked", "enemy", "item")


def smoke_one(entry: dict, steps: int = 200, seed: int = 0) -> dict:
    import numpy as np
    from doom_env import make_env

    from training.envs.maps import ROOT, resolve_wad

    res = {"id": entry["id"], "map": entry["map"], "kind": entry.get("kind", ""), "loads": False, "error": ""}
    t0 = time.perf_counter()
    try:
        env = make_env(str(resolve_wad(entry["wad"])), entry["map"], episode_ticks=int(entry["episode_ticks"]),
                       render_buffers=False)
    except Exception as e:  # noqa: BLE001 - recorded, not swallowed: the table shows the failure
        res["error"] = f"init: {type(e).__name__}: {e}"
        return res
    try:
        obs, _ = env.reset(seed=seed)
        res["init_s"] = round(time.perf_counter() - t0, 3)
        state = env._game.get_state()
        geo = env._geometry
        mi = env.mapinfo
        res.update(
            loads=True, wad_sectors=mi.num_sectors, engine_sectors=len(state.sectors), geometry=geo.source,
            doors=sum(1 for v in mi.door_sectors.values() if v == 0), locked=sum(1 for v in mi.door_sectors.values() if v > 0),
            door_lines=int((geo.hit_type == 2).sum()), locked_lines=int((geo.hit_type == 3).sum()),
            monsters=sum(1 for o in state.objects if o.category == "Monster"),
        )
        seen = collections.Counter(int(h) for h in obs["rays"]["hit_type"])
        rng = np.random.default_rng(seed)
        episodes, terminated, exits, done_steps = 1, 0, 0, 0
        t1 = time.perf_counter()
        for _ in range(steps):
            a = rng.integers(0, 2, size=9, dtype=np.int8)
            obs, _r, term, trunc, info = env.step(a)
            done_steps += 1
            seen.update(int(h) for h in obs["rays"]["hit_type"])
            if term or trunc:
                terminated += int(term)
                exits += int(bool(info.get("exit_reached")))
                env.reset(seed=seed + episodes)
                episodes += 1
        res.update(steps=done_steps, steps_per_s=round(done_steps / max(time.perf_counter() - t1, 1e-9), 1),
                   episodes=episodes, terminated=terminated, exits=exits,
                   rays_seen={RAY_NAMES[k]: int(v) for k, v in sorted(seen.items())},
                   door_ray_seen=seen.get(2, 0) > 0, locked_ray_seen=seen.get(3, 0) > 0)
        if entry.get("meta_path"):
            res.update(_check_generated(env, json.loads((ROOT / entry["meta_path"]).read_text()), seed))
    except Exception as e:  # noqa: BLE001
        res["error"] = f"run: {type(e).__name__}: {e}"
    finally:
        env.close()
    return res


def _check_generated(env, meta: dict, seed: int) -> dict:
    """Sidecar vs engine: doors, spawn, thing classes (fresh reset)."""
    env.reset(seed=seed)
    st = env.privileged_state()
    state = env._game.get_state()
    want_doors = {int(d["sector"]): int(d["lock"]) for d in meta["doors"]}
    got = collections.Counter(o.name for o in state.objects if o.category != "Self")
    want = collections.Counter([m["class"] for m in meta["monsters"]] + [k["class"] for k in meta["keys"]]
                               + [i["class"] for i in meta["items"]])
    return {
        "meta_doors_match": env.mapinfo.door_sectors == want_doors,
        "meta_spawn_match": abs(st["x"] - meta["start"]["x"]) < 1 and abs(st["y"] - meta["start"]["y"]) < 1,
        "meta_things_match": got == want,
        "meta_sectors_match": env.mapinfo.num_sectors == meta["num_sectors"],
    }


def run(entries: list[dict], steps: int, jobs: int, timeout: float) -> list[dict]:
    ctx = mp.get_context("spawn")
    out = []
    with ctx.Pool(processes=jobs, maxtasksperchild=8) as pool:
        futs = [(e, pool.apply_async(smoke_one, (e, steps))) for e in entries]
        for e, f in futs:
            try:
                out.append(f.get(timeout=timeout))
            except mp.TimeoutError:
                out.append({"id": e["id"], "map": e["map"], "kind": e.get("kind", ""), "loads": False,
                            "error": f"timeout after {timeout}s"})
            except Exception as ex:  # noqa: BLE001 - worker crash (e.g. engine abort) is recorded
                out.append({"id": e["id"], "map": e["map"], "kind": e.get("kind", ""), "loads": False,
                            "error": f"worker: {type(ex).__name__}: {ex}"})
    return out


def table(results: list[dict]) -> str:
    head = "| id | loads | sectors wad/engine | geo | doors | locked | door lines | locked lines | rays door/locked seen | monsters | eps/exits | meta ok | error |"
    rows = [head, "|" + "---|" * 13]
    for r in results:
        meta_ok = ""
        if "meta_doors_match" in r:
            meta_ok = "yes" if all(r[k] for k in ("meta_doors_match", "meta_spawn_match", "meta_things_match", "meta_sectors_match")) else "NO"
        rows.append(
            f"| {r['id']} | {'yes' if r['loads'] else 'NO'} | {r.get('wad_sectors', '')}/{r.get('engine_sectors', '')} | "
            f"{r.get('geometry', '')} | {r.get('doors', '')} | {r.get('locked', '')} | {r.get('door_lines', '')} | "
            f"{r.get('locked_lines', '')} | {'Y' if r.get('door_ray_seen') else '-'}/{'Y' if r.get('locked_ray_seen') else '-'} | "
            f"{r.get('monsters', '')} | {r.get('episodes', '')}/{r.get('exits', '')} | {meta_ok} | {r.get('error', '').splitlines()[0][:60] if r.get('error') else ''} |")
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> int:
    from dataclasses import asdict

    from training.envs.maps import SOURCES, all_entries

    p = argparse.ArgumentParser(prog="python -m training.envs.smoke_load")
    p.add_argument("--sources", default=",".join(SOURCES))
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--timeout", type=float, default=120.0, help="seconds per map")
    p.add_argument("--only", default="", help="comma-separated id substrings")
    p.add_argument("--out", default="", help="write JSON results here")
    a = p.parse_args(argv)
    entries = [asdict(e) for e in all_entries([s for s in a.sources.split(",") if s])]
    if a.only:
        keys = a.only.split(",")
        entries = [e for e in entries if any(k in e["id"] for k in keys)]
    t0 = time.perf_counter()
    results = run(entries, a.steps, a.jobs, a.timeout)
    print(table(results))
    bad = [r["id"] for r in results if not r["loads"] or r.get("error")]
    meta_bad = [r["id"] for r in results if "meta_doors_match" in r and not all(
        r[k] for k in ("meta_doors_match", "meta_spawn_match", "meta_things_match", "meta_sectors_match"))]
    print(f"\n{len(results)} maps, {len(results) - len(bad)} loaded cleanly, failures: {bad or 'none'}, "
          f"generated sidecar mismatches: {meta_bad or 'none'}, {time.perf_counter() - t0:.1f}s")
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(results, indent=1, sort_keys=True) + "\n")
    return 1 if bad or meta_bad else 0


if __name__ == "__main__":
    sys.exit(main())
