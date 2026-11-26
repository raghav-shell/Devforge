"""Probe 2: dead-reckoning drift over full-length episodes (up to 1,050 steps) for three legal pose
estimators, and what it implies for an ego-centric visit/occupancy grid.

Estimators (definitions in probe_dr_variants.py; all use legal inputs only):
  V0 training/experiments/dead_reckoning.DeadReckoner + the step-2 turn fix (see HeadingTracker)
  V1 per-term heading frames (v_now rotated by post-turn heading, v_prev by pre-turn heading)
  V2 V1 + point-to-line ICP of consecutive wall-hit ray scans (rotation fixed to the action-integrated
     heading change)
Truth: env.privileged_state() (diagnostic only, never an agent input).

Per episode:
  - position error at t = 100, 250, 525, 1050 (n reported at each mark: deaths/exits shorten episodes)
  - heading error (action-integrated)
  - per-step residual on "blocked" steps (true step < 50 % of the V1-predicted step, predicted > 2 u)
    vs unblocked steps -> where the drift comes from
  - teleport/jump count (true step > 64 u)
  - grid-cell agreement with the grid origin offset by half a cell (start pose at a cell centre):
    exact cell and within-1-cell (Chebyshev) for cell = 32, 64, 128 u; visited-set Jaccard
  - relative-displacement error over lags k = 25, 50, 100, 200 (what a fading local map depends on)

--nomonsters runs maps with `-nomonsters` (diagnostic only, to get full-length episodes without
survivorship bias from deaths of the blind probe policy).

Run: .venv/bin/python research/spatial_probes/probe_dead_reckoning.py [--steps 1050] [--seeds 4] [--nomonsters]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from _common import MAPS, POLICIES, dang, out_dir, start_frame
from dead_reckoning import W_FWD, W_SIDE, DeadReckoner
from probe_dr_variants import HeadingTracker, icp_translation, rot, scan_points

MARKS = (100, 250, 525, 1050)
CELLS = (32, 64, 128)
LAGS = (25, 50, 100, 200)
EST = ("V0", "V1", "V2")


def patch_nomonsters():
    import vizdoom as vd
    import doom_env.wrapper as W

    class NoMon(vd.DoomGame):
        def init(self):
            self.add_game_args("-nomonsters")
            return super().init()

    ns = type("vd_patched", (), {k: getattr(vd, k) for k in dir(vd) if not k.startswith("__")})
    ns.DoomGame = NoMon
    W.vd = ns


def episode(wad, mp, pol_name, seed, steps):
    from doom_env import make_env

    env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    to_start = start_frame(p0)
    pol = POLICIES[pol_name](seed)
    v0, ht = DeadReckoner(), HeadingTracker()
    e1, e2, pv = np.zeros(2), np.zeros(2), np.zeros(2)
    prev_scan = scan_points(obs["rays"])
    prev_true = np.zeros(2)
    path = 0.0
    rows = []
    end = "cap"
    for t in range(steps):
        a = pol(obs)
        h_prev = math.radians(ht.h)
        obs, _, te, tr, info = env.step(a)
        if te or tr:
            end = "exit" if info["exit_reached"] else ("trunc" if tr else "death")
            break
        dth = ht.update(a)
        h_now = math.radians(ht.h)
        ps = env.privileged_state()
        true = np.array(to_start(ps["x"], ps["y"]))
        step_true = float(np.linalg.norm(true - prev_true))
        path += step_true
        v0.update(a, obs["vitals"])
        if t == 1 and v0.turn_run == 1:  # same start-quirk fix as HeadingTracker (see probe_dr_variants.py)
            v0.h -= (1 if a[4] else -1) * (7.031 - 1.758)
        vnow = np.array([float(obs["vitals"][14]), -float(obs["vitals"][15])])
        s1 = rot(np.array([W_FWD[0] * vnow[0], W_SIDE[0] * vnow[1]]), h_now) + rot(np.array([W_FWD[1] * pv[0], W_SIDE[1] * pv[1]]), h_prev)
        cur_scan = scan_points(obs["rays"])
        d_local, inl = icp_translation(prev_scan, cur_scan, dth, rot(s1, -h_prev))
        s2 = rot(d_local, h_prev)
        r0 = np.array([v0.x, v0.y]) - np.array(rows[-1][4:6] if rows else (0.0, 0.0))
        e1, e2 = e1 + s1, e2 + s2
        pred = float(np.linalg.norm(s1))
        blocked = pred > 2.0 and step_true < 0.5 * pred
        rows.append((t + 1, path, abs(dang(ht.h, ps["angle"] - p0["angle"])), float(blocked),
                     v0.x, v0.y, e1[0], e1[1], e2[0], e2[1], true[0], true[1],
                     float(np.linalg.norm(r0 - (true - prev_true))), float(np.linalg.norm(s1 - (true - prev_true))),
                     float(np.linalg.norm(s2 - (true - prev_true))), float(step_true > 64), float(inl >= 6)))
        pv, prev_scan, prev_true = vnow, cur_scan, true
    env.close()
    return np.array(rows, np.float64), end


COL = {"V0": 4, "V1": 6, "V2": 8}
RES = {"V0": 12, "V1": 13, "V2": 14}


def summarize(L):
    out = {"len": int(len(L)), "max_hdg_err": float(L[:, 2].max()), "jumps": int(L[:, 15].sum()),
           "blocked_frac": float(L[:, 3].mean()), "icp_ok_frac": float(L[:, 16].mean())}
    T = L[:, 10:12]
    b = L[:, 3].astype(bool)
    for nm in EST:
        E = L[:, COL[nm]:COL[nm] + 2]
        err = np.linalg.norm(E - T, axis=1)
        for k in MARKS:
            if k <= len(L):
                out[f"{nm}_err@{k}"] = float(err[k - 1])
                out[f"{nm}_pct@{k}"] = float(100 * err[k - 1] / max(L[k - 1, 1], 1))
        res = L[:, RES[nm]]
        out[f"{nm}_resid_blocked"] = float(res[b].mean()) if b.any() else 0.0
        out[f"{nm}_resid_free"] = float(res[~b].mean()) if (~b).any() else 0.0
        out[f"{nm}_resid_share_blocked"] = float(res[b].sum() / max(res.sum(), 1e-9))
        for cs in CELLS:
            ec = np.floor(E / cs + 0.5).astype(int)
            tc = np.floor(T / cs + 0.5).astype(int)
            out[f"{nm}_cell{cs}_exact"] = float((ec == tc).all(1).mean())
            out[f"{nm}_cell{cs}_within1"] = float((np.abs(ec - tc).max(1) <= 1).mean())
            es, ts = set(map(tuple, ec)), set(map(tuple, tc))
            out[f"{nm}_cell{cs}_jaccard"] = len(es & ts) / max(len(es | ts), 1)
        for k in LAGS:
            if len(L) > k:
                rel = np.linalg.norm((E[k:] - E[:-k]) - (T[k:] - T[:-k]), axis=1)
                out[f"{nm}_rel{k}_mean"] = float(rel.mean())
                out[f"{nm}_rel{k}_p90"] = float(np.quantile(rel, 0.9))
    out["true_cell64_visited"] = len(set(map(tuple, np.floor(T / 64 + 0.5).astype(int))))
    return out


def agg(items):
    keys = sorted({k for d in items for k in d if k != "map"})
    return {k: {"median": float(np.median([d[k] for d in items if k in d])),
                "p10": float(np.quantile([d[k] for d in items if k in d], 0.1)),
                "p90": float(np.quantile([d[k] for d in items if k in d], 0.9)),
                "n": sum(k in d for d in items)} for k in keys}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--nomonsters", action="store_true")
    ap.add_argument("--groups", default="two_rooms,fixtures,freedoom2")
    args = ap.parse_args()
    if args.nomonsters:
        patch_nomonsters()
    report = {}
    for group in args.groups.split(","):
        for pol in POLICIES:
            per, ends = [], {}
            for wad, mp in MAPS[group]:
                for seed in range(args.seeds):
                    L, end = episode(wad, mp, pol, seed, args.steps)
                    ends[end] = ends.get(end, 0) + 1
                    if len(L) >= 50:
                        s = summarize(L)
                        s["map"] = f"{wad.split('/')[-1]}:{mp}:s{seed}"
                        per.append(s)
            key = f"{group}/{pol}" + ("/nomonsters" if args.nomonsters else "")
            a = agg(per)
            report[key] = {"ends": ends, "agg": a, "per_episode": per}
            print(f"== {key} ends={ends}")
            for k in a:
                if any(s in k for s in ("_err@", "_cell64", "_cell128_within1", "_cell32_within1", "_rel", "_resid", "blocked_frac", "max_hdg", "jumps", "len", "icp_ok")):
                    print(f"   {k:28s} median {a[k]['median']:9.2f}  p10 {a[k]['p10']:9.2f}  p90 {a[k]['p90']:9.2f}  n={a[k]['n']}")
            print(flush=True)
    tag = "_nomonsters" if args.nomonsters else ""
    (out_dir() / f"dead_reckoning{tag}.json").write_text(json.dumps(report, indent=1))
