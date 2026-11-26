"""Probe 8: which LEGAL "blocked / bumping into a wall" cue actually agrees with the privileged label?

Privileged label (diagnostic only): a move button was pressed and the true step < 50 % of the V1
velocity-model step, with the V1 step > 2 u (same definition as probe_dead_reckoning.py).

Legal candidates (computable in submission/ from obs + own actions):
  speed_lt1   move pressed and hypot(vel_fwd, vel_right) < 1.0 u/tic
  icp_ratio   move pressed and |V2 ICP step| < 0.5 * |V1 velocity step| (V1 step > 2 u)
  ray_ddist   forward pressed and the 3 forward rays (indices 63, 0, 1) shrank by < 1 u on average while
              d < 64 u (no approach despite pushing forward), or similarly for the backward rays
Reports precision / recall / F1 and base rate per map group.

Run: .venv/bin/python research/spatial_probes/probe_blocked_flag.py [--steps 1050] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from _common import MAPS, POLICIES, out_dir, start_frame
from dead_reckoning import W_FWD, W_SIDE
from probe_dead_reckoning import patch_nomonsters
from probe_dr_variants import HeadingTracker, icp_translation, rot, scan_points


def episode(wad, mp, pol_name, seed, steps):
    from doom_env import make_env

    env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    to_start = start_frame(p0)
    pol = POLICIES[pol_name](seed)
    ht = HeadingTracker()
    pv = np.zeros(2)
    prev_scan = scan_points(obs["rays"])
    prev_true = np.zeros(2)
    prev_d = obs["rays"]["distance"] * 1024
    rows = []
    for t in range(steps):
        a = pol(obs)
        h_prev = math.radians(ht.h)
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
        dth = ht.update(a)
        h_now = math.radians(ht.h)
        ps = env.privileged_state()
        true = np.array(to_start(ps["x"], ps["y"]))
        vnow = np.array([float(obs["vitals"][14]), -float(obs["vitals"][15])])
        s1 = rot(np.array([W_FWD[0] * vnow[0], W_SIDE[0] * vnow[1]]), h_now) + rot(np.array([W_FWD[1] * pv[0], W_SIDE[1] * pv[1]]), h_prev)
        cur = scan_points(obs["rays"])
        d_local, _ = icp_translation(prev_scan, cur, dth, rot(s1, -h_prev))
        moving = bool(a[0] or a[1] or a[2] or a[3])
        pred = float(np.linalg.norm(s1))
        label = moving and pred > 2.0 and float(np.linalg.norm(true - prev_true)) < 0.5 * pred
        d = obs["rays"]["distance"] * 1024
        f_idx, b_idx = [63, 0, 1], [31, 32, 33]
        f_stall = a[0] and not a[1] and d[f_idx].mean() < 64 and (prev_d[f_idx] - d[f_idx]).mean() < 1.0
        b_stall = a[1] and not a[0] and d[b_idx].mean() < 64 and (prev_d[b_idx] - d[b_idx]).mean() < 1.0
        rows.append((label,
                     moving and math.hypot(vnow[0], vnow[1]) < 1.0,
                     moving and pred > 2.0 and float(np.linalg.norm(d_local)) < 0.5 * pred,
                     bool(f_stall or b_stall)))
        pv, prev_scan, prev_true, prev_d = vnow, cur, true, d
    env.close()
    return rows


def prf(y, p):
    tp = (y & p).sum()
    prec = tp / max(p.sum(), 1)
    rec = tp / max(y.sum(), 1)
    return {"precision": float(prec), "recall": float(rec), "f1": float(2 * prec * rec / max(prec + rec, 1e-9)), "pred_rate": float(p.mean())}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    patch_nomonsters()  # full-length episodes; the label is about walls, not monsters
    out = {}
    for g in ("two_rooms", "fixtures", "freedoom2"):
        for pol in POLICIES:
            R = []
            for wad, mp in MAPS[g]:
                for s in range(args.seeds):
                    R += episode(wad, mp, pol, s, args.steps)
            A = np.array(R, bool)
            key = f"{g}/{pol}/nomonsters"
            out[key] = {"n": len(A), "label_rate": float(A[:, 0].mean()),
                        **{nm: prf(A[:, 0], A[:, i]) for i, nm in ((1, "speed_lt1"), (2, "icp_ratio"), (3, "ray_ddist"))}}
            print(key, json.dumps(out[key]), flush=True)
    (out_dir() / "blocked_flag.json").write_text(json.dumps(out, indent=1))
