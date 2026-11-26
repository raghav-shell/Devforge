"""Probe 9: "local free space" auxiliary target (training-only label) and how much of it is hidden from
the current rays.

Label: an ego-centric 7 x 7 grid of 64 u cells centred on the player, heading-up. A cell is *walkable* if
its centre lies inside some sector polygon (WAD geometry via env.sector_polygons(), parity test) whose
current ceiling - floor >= 56 u (privileged sector heights). Training-only.
Legal counterpart: a cell is *ray-observed free* if its centre is closer than the ray in its direction.
Reports: walkable fraction, observed-free fraction, and hidden-walkable fraction (walkable, not observed),
i.e. what an occupancy-completion head would have to infer (cf. Active Neural SLAM map prediction).

Run: .venv/bin/python research/spatial_probes/probe_free_space.py [--steps 1050] [--seeds 1]
"""
from __future__ import annotations

import argparse
import json
import math

import numpy as np

from _common import MAPS, POLICIES, out_dir

K, CELL = 7, 64.0
r = (np.arange(K) - K // 2) * CELL
FWD, LEFT = np.meshgrid(r[::-1], r[::-1], indexing="ij")
EGO = np.stack([FWD.ravel(), LEFT.ravel()], 1)  # (49, 2) own frame
EGO_D = np.hypot(EGO[:, 0], EGO[:, 1])
EGO_K = np.round(np.arctan2(EGO[:, 1], EGO[:, 0]) / (2 * np.pi / 64)).astype(int) % 64
CENTER = (K * K) // 2


def build_segments(env):
    segs, sid = [], []
    for s, L in env.sector_polygons():
        segs.append(L)
        sid.append(np.full(len(L), s))
    return np.concatenate(segs), np.concatenate(sid)


def inside_sectors(P, segs, sid, n_sec):
    """(M,2) points -> (M, n_sec) bool, even-odd rule per sector."""
    x1, y1, x2, y2 = segs[:, 0], segs[:, 1], segs[:, 2], segs[:, 3]
    px, py = P[:, None, 0], P[:, None, 1]
    cond = (y1[None] > py) != (y2[None] > py)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x1[None] + (py - y1[None]) * (x2 - x1)[None] / (y2 - y1)[None]
    cross = cond & (px < xi)
    out = np.zeros((len(P), n_sec), np.int64)
    for m in range(len(P)):
        out[m] = np.bincount(sid[cross[m]], minlength=n_sec)
    return (out % 2) == 1


def run(group, pol, steps, seeds, every=5):
    from doom_env import make_env

    walk, obsf, hidden, hidden_given_walk = [], [], [], []
    for wad, mp in MAPS[group]:
        for seed in range(seeds):
            env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
            obs, _ = env.reset(seed=seed)
            segs, sid = build_segments(env)
            n_sec = int(sid.max()) + 1
            p = POLICIES[pol](seed)
            for t in range(steps):
                if t % every == 0:
                    ps = env.privileged_state()
                    h = math.radians(ps["angle"])
                    c, s = math.cos(h), math.sin(h)
                    W = np.stack([ps["x"] + EGO[:, 0] * c - EGO[:, 1] * s, ps["y"] + EGO[:, 0] * s + EGO[:, 1] * c], 1)
                    ins = inside_sectors(W, segs, sid, n_sec)
                    tall = (ps["sector_ceiling"] - ps["sector_floor"]) >= 56
                    wk = (ins & tall[None, :n_sec]).any(1)
                    of = EGO_D < obs["rays"]["distance"][EGO_K] * 1024
                    wk[CENTER] = of[CENTER] = False  # the player's own cell is trivial
                    walk.append(wk.mean())
                    obsf.append(of.mean())
                    hidden.append((wk & ~of).mean())
                    if wk.any():
                        hidden_given_walk.append((wk & ~of).sum() / wk.sum())
                obs, _, te, tr, _ = env.step(p(obs))
                if te or tr:
                    break
            env.close()
    return {"group": group, "policy": pol, "n_samples": len(walk), "walkable_frac": float(np.mean(walk)),
            "ray_observed_free_frac": float(np.mean(obsf)), "hidden_walkable_frac": float(np.mean(hidden)),
            "hidden_share_of_walkable": float(np.mean(hidden_given_walk))}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=1)
    args = ap.parse_args()
    res = []
    for g in ("two_rooms", "fixtures", "freedoom2"):
        for pol in POLICIES:
            r_ = run(g, pol, args.steps, args.seeds)
            res.append(r_)
            print(json.dumps(r_), flush=True)
    (out_dir() / "free_space.json").write_text(json.dumps(res, indent=1))
