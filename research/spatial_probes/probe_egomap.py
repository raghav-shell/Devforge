"""Probe 7: the v2 ego-map update rule, its per-step cost, and how much drift corrupts it.

Two identical allocentric grids are written every step from the same rays:
  - grid_est  : pose from the legal V2 odometry (V1 velocity model + ray ICP; see probe_dr_variants.py)
  - grid_v1   : pose from velocity-only odometry (V1), for comparison
  - grid_true : pose from privileged truth (diagnostic oracle; never an agent input)
Both use the episode start pose as origin (truth is expressed in the start frame), so a perfect estimator
yields identical grids. At t = 250, 525, 1050 we compare the heading-up 32x32 ego crops around the agent:
IoU of the wall / free / visited channels, and fraction of cells whose argmax class agrees.

Update rule (also in docs/SPATIAL_OBSERVATION_DESIGN.md section 3.2): visited += 1 at the agent cell; for each
ray, cells strictly before the hit get free=1 (and door/locked cleared), the hit cell gets its hit type.

Run: .venv/bin/python research/spatial_probes/probe_egomap.py [--steps 1050] [--seeds 2] [--nomonsters]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from _common import MAPS, POLICIES, out_dir, start_frame
from dead_reckoning import W_FWD, W_SIDE
from probe_dead_reckoning import patch_nomonsters
from probe_dr_variants import HeadingTracker, icp_translation, rot, scan_points

CELL, N, WIN = 64.0, 512, 32
CH = {"visited": 0, "free": 1, "wall": 2, "door": 3, "locked": 4}
RAY_ANG = np.arange(64) * (2 * np.pi / 64)
SAMPLES = (np.arange(1, 33) / 32.0)  # fractions along each ray (<=16 cells at 64 u for 1024 u -> 2 samples/cell)


class EgoGrid:
    def __init__(self):
        self.g = np.zeros((5, N, N), np.uint8)
        self.off = N // 2 + 0.5  # start pose at a cell centre

    def idx(self, xy):
        return np.floor(xy / CELL + self.off).astype(np.int64).clip(0, N - 1)

    def update(self, pos, heading, rays):
        d = rays["distance"] * 1024.0
        ht = rays["hit_type"]
        ang = heading + RAY_ANG
        u = np.stack([np.cos(ang), np.sin(ang)], 1)  # (64,2)
        c = self.idx(pos)
        self.g[0, c[1], c[0]] = min(int(self.g[0, c[1], c[0]]) + 1, 255)
        # free cells: sample points strictly before the hit (stop 8 u short)
        lens = np.maximum(d - 8.0, 0.0)[:, None] * SAMPLES[None, :]  # (64, 32)
        pts = pos[None, None, :] + lens[:, :, None] * u[:, None, :]
        ij = self.idx(pts.reshape(-1, 2))
        self.g[1, ij[:, 1], ij[:, 0]] = 1
        self.g[3, ij[:, 1], ij[:, 0]] = 0
        self.g[4, ij[:, 1], ij[:, 0]] = 0
        hit = (ht >= 1) & (ht <= 3) & (rays["distance"] < 1.0)
        if hit.any():
            hp = pos[None, :] + (d[hit] + 1.0)[:, None] * u[hit]
            hj = self.idx(hp)
            for k, ch in ((1, 2), (2, 3), (3, 4)):
                m = ht[hit] == k
                self.g[ch, hj[m, 1], hj[m, 0]] = 1

    def crop(self, pos, heading):
        """Heading-up WIN x WIN crop (nearest neighbour). Row 0 = far ahead."""
        r = (np.arange(WIN) - WIN / 2 + 0.5) * CELL
        fwd, left = np.meshgrid(r[::-1], r[::-1], indexing="ij")  # rows: ahead..behind, cols: left..right
        c, s = math.cos(heading), math.sin(heading)
        wx = pos[0] + fwd * c - left * s
        wy = pos[1] + fwd * s + left * c
        ij = self.idx(np.stack([wx, wy], -1).reshape(-1, 2))
        return self.g[:, ij[:, 1], ij[:, 0]].reshape(5, WIN, WIN)


def iou(a, b):
    a, b = a > 0, b > 0
    u = (a | b).sum()
    return float((a & b).sum() / u) if u else 1.0


def dilate(a):
    a = a > 0
    p = np.pad(a, 1)
    return np.max([p[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)], axis=0)


def f1_tol1(a, b):
    """F1 where a cell counts as matched if the other grid has the class within 1 cell (Chebyshev)."""
    a, b = a > 0, b > 0
    if not a.any() and not b.any():
        return 1.0
    prec = (a & dilate(b)).sum() / max(a.sum(), 1)
    rec = (b & dilate(a)).sum() / max(b.sum(), 1)
    return float(2 * prec * rec / max(prec + rec, 1e-9))


def compare(ca, cb):
    out = {f"iou_{k}": iou(ca[v], cb[v]) for k, v in CH.items() if k in ("visited", "free", "wall")}
    out.update({f"f1tol1_{k}": f1_tol1(ca[v], cb[v]) for k, v in CH.items() if k in ("visited", "free", "wall")})
    cls_a = np.where(ca[2] > 0, 2, np.where(ca[1] > 0, 1, 0))
    cls_b = np.where(cb[2] > 0, 2, np.where(cb[1] > 0, 1, 0))
    known = (cls_a > 0) | (cls_b > 0)
    out["class_agree_known"] = float((cls_a == cls_b)[known].mean()) if known.any() else 1.0
    return out


def episode(wad, mp, pol_name, seed, steps):
    from doom_env import make_env

    env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    to_start = start_frame(p0)
    pol = POLICIES[pol_name](seed)
    ht = HeadingTracker()
    est, pv = np.zeros(2), np.zeros(2)
    prev_scan = scan_points(obs["rays"])
    ge, gt, g1 = EgoGrid(), EgoGrid(), EgoGrid()
    est1 = np.zeros(2)
    ge.update(est, 0.0, obs["rays"])
    g1.update(est1, 0.0, obs["rays"])
    gt.update(np.zeros(2), 0.0, obs["rays"])
    res, t_pre, t_crop = {}, [], []
    for t in range(steps):
        a = pol(obs)
        h_prev = math.radians(ht.h)
        obs, _, te, tr, _ = env.step(a)
        if te or tr:
            break
        t0 = time.perf_counter()
        dth = ht.update(a)
        h_now = math.radians(ht.h)
        vnow = np.array([float(obs["vitals"][14]), -float(obs["vitals"][15])])
        s1 = rot(np.array([W_FWD[0] * vnow[0], W_SIDE[0] * vnow[1]]), h_now) + rot(np.array([W_FWD[1] * pv[0], W_SIDE[1] * pv[1]]), h_prev)
        cur = scan_points(obs["rays"])
        d_local, _ = icp_translation(prev_scan, cur, dth, rot(s1, -h_prev))
        est = est + rot(d_local, h_prev)
        ge.update(est, h_now, obs["rays"])
        t1 = time.perf_counter()
        ce = ge.crop(est, h_now)
        t2 = time.perf_counter()
        t_pre.append(t1 - t0)
        t_crop.append(t2 - t1)
        pv, prev_scan = vnow, cur
        ps = env.privileged_state()
        tp = np.array(to_start(ps["x"], ps["y"]))
        th = math.radians(ps["angle"] - p0["angle"])
        gt.update(tp, th, obs["rays"])
        est1 = est1 + s1  # velocity-only (V1) pose, for comparison
        g1.update(est1, h_now, obs["rays"])
        if t + 1 in (250, 525, 1050):
            ct = gt.crop(tp, th)
            res[t + 1] = compare(ce, ct)
            res[t + 1].update({f"V1pose_{k}": v for k, v in compare(g1.crop(est1, h_now), ct).items()})
    env.close()
    return res, 1000 * float(np.median(t_pre)), 1000 * float(np.median(t_crop))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--nomonsters", action="store_true")
    args = ap.parse_args()
    if args.nomonsters:
        patch_nomonsters()
    report = {}
    for group in ("two_rooms", "fixtures", "freedoom2"):
        for pol in POLICIES:
            rows, pre, crop = {250: [], 525: [], 1050: []}, [], []
            for wad, mp in MAPS[group]:
                for seed in range(args.seeds):
                    r, tp, tc = episode(wad, mp, pol, seed, args.steps)
                    pre.append(tp)
                    crop.append(tc)
                    for k, v in r.items():
                        rows[k].append(v)
            key = f"{group}/{pol}" + ("/nomonsters" if args.nomonsters else "")
            summ = {"preproc_ms_median": float(np.median(pre)), "crop_ms_median": float(np.median(crop))}
            for k, lst in rows.items():
                if lst:
                    summ[k] = {m: float(np.median([x[m] for x in lst])) for m in lst[0]}
                    summ[k]["n"] = len(lst)
            report[key] = summ
            print(key, json.dumps(summ), flush=True)
    tag = "_nomonsters" if args.nomonsters else ""
    (out_dir() / f"egomap{tag}.json").write_text(json.dumps(report, indent=1))
