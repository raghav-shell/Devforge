"""Probe 3: can legal dead reckoning be made better? Compares pose estimators that use only legal
inputs (own actions, vitals velocity, rays) against privileged truth (diagnostic only).

  V0  training/experiments/dead_reckoning.DeadReckoner as-is: both velocity terms rotated by the
      heading *before* this step's turn.
  V1  frame fix: v_now is measured in the frame *after* this step's turn and v_prev in the frame
      before it, so rotate each term by its own heading. Same coefficients.
  V2  V1 + ray scan matching: point-to-line ICP of this step's wall-hit ray points against the previous
      step's, rotation fixed to the (exact) action-integrated heading change, translation initialised at
      and regularised towards the V1 prediction. Pure preprocessing (no action is chosen from it).

Also refits the V1 coefficients by least squares on all logged steps.

Run: .venv/bin/python research/spatial_probes/probe_dr_variants.py [--steps 1050] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import math
import time

import numpy as np

from _common import MAPS, POLICIES, dang, out_dir, start_frame
from dead_reckoning import TURN_RAMP, W_FWD, W_SIDE, DeadReckoner
from doom_env import make_env

RAY_ANG = np.arange(64) * (2 * np.pi / 64)
U = np.stack([np.cos(RAY_ANG), np.sin(RAY_ANG)], 1)  # own frame: x forward, y left


def rot(v, h):
    c, s = math.cos(h), math.sin(h)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class HeadingTracker:
    """Exact heading from own turn actions (same ramp as DeadReckoner)."""

    def __init__(self):
        self.h, self.run, self.t = 0.0, 0, 0

    def update(self, a):
        left, right = bool(a[4]), bool(a[5])
        tdir = (1 if left else -1) if left != right else 0
        self.run = self.run + 1 if tdir else 0
        self.t += 1
        deg = 0.0
        if tdir:
            deg = TURN_RAMP[min(self.run, 3) - 1]
            if self.t == 1:
                deg = 0.0
            elif self.t == 2:
                # measured: step 2 applies only the tics after the start freeze -> 1.758 deg if the key was
                # first pressed at step 2, 3.516 deg if already held at step 1 (DeadReckoner gives 7.031
                # for the first case: a constant 5.273 deg offset for the rest of the episode)
                deg = 3.516 if self.run >= 2 else 1.758
        self.h += tdir * deg
        return math.radians(tdir * deg)


def scan_points(rays):
    d = rays["distance"]
    ok = (rays["hit_type"] == 1) & (d < 1.0)
    return d[:, None] * 1024 * U, ok


def normals(P, ok):
    """Per-point normal from the neighbouring ray points (consecutive indices), if both are close."""
    n = np.zeros_like(P)
    good = np.zeros(len(P), bool)
    for i in range(64):
        a, b = (i - 1) % 64, (i + 1) % 64
        if ok[i] and ok[a] and ok[b]:
            t = P[b] - P[a]
            L = np.linalg.norm(t)
            if 1e-6 < L < 200 and np.linalg.norm(P[b] - P[i]) < 120 and np.linalg.norm(P[a] - P[i]) < 120:
                n[i] = (-t[1] / L, t[0] / L)
                good[i] = True
    return n, good


def icp_translation(prev, cur, dtheta, d0, lam=0.05, iters=4, gate=24.0):
    """Translation d (frame t-1) of the player at t, rotation fixed. Returns d, n_inliers."""
    P, okp = prev
    Q, okq = cur
    N, gn = normals(P, okp)
    if gn.sum() < 6 or okq.sum() < 6:
        return d0, 0
    Pg, Ng = P[gn], N[gn]
    c, s = math.cos(dtheta), math.sin(dtheta)
    Qr = Q[okq] @ np.array([[c, s], [-s, c]])  # rotate current points into frame t-1 orientation
    d = d0.copy()
    inl = 0
    for _ in range(iters):
        X = Qr + d
        D = ((X[:, None, :] - Pg[None, :, :]) ** 2).sum(-1)
        j = D.argmin(1)
        r = ((X - Pg[j]) * Ng[j]).sum(1)
        m = np.abs(r) < gate
        inl = int(m.sum())
        if inl < 6:
            return d0, 0
        n = Ng[j][m]
        # minimise sum (n.(q + d - p))^2 + lam*|m|*|d - d0|^2 over d
        e = ((Qr[m] - Pg[j][m]) * n).sum(1)
        A = n.T @ n + lam * inl * np.eye(2)
        b = -(n * e[:, None]).sum(0) + lam * inl * d0
        d = np.linalg.solve(A, b)
    return d, inl


def episode(wad, mp, pol_name, seed, steps, coef=None):
    env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
    obs, _ = env.reset(seed=seed)
    p0 = env.privileged_state()
    to_start = start_frame(p0)
    pol = POLICIES[pol_name](seed)
    v0 = DeadReckoner()
    ht = HeadingTracker()
    est1 = np.zeros(2)
    est2 = np.zeros(2)
    wf, ws = (W_FWD, W_SIDE) if coef is None else coef
    pv = np.zeros(2)
    prev_scan = scan_points(obs["rays"])
    path = 0.0
    log, reg = [], []
    icp_t = 0.0
    prev_true = np.zeros(2)
    for t in range(steps):
        a = pol(obs)
        h_prev = math.radians(ht.h)
        obs, _, te, tr, info = env.step(a)
        if te or tr:
            break
        dth = ht.update(a)
        h_now = math.radians(ht.h)
        ps = env.privileged_state()
        tx, ty = to_start(ps["x"], ps["y"])
        true = np.array([tx, ty])
        path += float(np.linalg.norm(true - prev_true))
        vf, vr = float(obs["vitals"][14]), float(obs["vitals"][15])
        vnow = np.array([vf, -vr])  # own frame, y = left
        # V0
        v0.update(a, obs["vitals"])
        # V1: per-term frames
        d_now = np.array([wf[0] * vnow[0], ws[0] * vnow[1]])
        d_prev = np.array([wf[1] * pv[0], ws[1] * pv[1]])
        step1 = rot(d_now, h_now) + rot(d_prev, h_prev)
        est1 = est1 + step1
        # V2: ICP in frame t-1 (rotation dth), init at V1 step
        cur_scan = scan_points(obs["rays"])
        t0 = time.perf_counter()
        d_local, inl = icp_translation(prev_scan, cur_scan, dth, rot(step1, -h_prev))
        icp_t += time.perf_counter() - t0
        est2 = est2 + rot(d_local, h_prev)
        prev_scan = cur_scan
        # regression row: true displacement in frame t-1 vs per-term features
        true_local = rot(true - prev_true, -math.radians(ps["angle"] - p0["angle"]) + dth)  # frame of h_prev (true)
        vn_prevframe = rot(vnow, dth)  # v_now expressed in frame t-1
        reg.append((true_local[0], true_local[1], vn_prevframe[0], vn_prevframe[1], pv[0], pv[1], dth))
        pv = vnow
        prev_true = true
        log.append((t + 1, path, np.linalg.norm([v0.x - tx, v0.y - ty]), np.linalg.norm(est1 - true),
                    np.linalg.norm(est2 - true), abs(dang(ht.h, ps["angle"] - p0["angle"])), inl))
    env.close()
    return np.array(log), np.array(reg), icp_t / max(len(log), 1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    report = {}
    all_reg = []
    for group, maps in MAPS.items():
        for pol in POLICIES:
            rows = []
            for wad, mp in maps:
                for seed in range(args.seeds):
                    L, R, icp_ms = episode(wad, mp, pol, seed, args.steps)
                    if len(L) < 50:
                        continue
                    all_reg.append(R)
                    r = {"map": f"{wad.split('/')[-1]}:{mp}", "len": len(L), "icp_ms": 1000 * icp_ms,
                         "icp_inlier_frac_steps": float((L[:, 6] >= 6).mean())}
                    for k in (100, 250, 525, 1050):
                        if k <= len(L):
                            r[f"path@{k}"] = float(L[k - 1, 1])
                            for j, nm in ((2, "V0"), (3, "V1"), (4, "V2")):
                                r[f"{nm}_err@{k}"] = float(L[k - 1, j])
                    r["max_hdg_err"] = float(L[:, 5].max())
                    rows.append(r)
            key = f"{group}/{pol}"
            summ = {}
            for k in sorted({k for r in rows for k in r if k != "map"}):
                v = [r[k] for r in rows if k in r]
                summ[k] = {"median": float(np.median(v)), "min": float(np.min(v)), "max": float(np.max(v)), "n": len(v)}
            report[key] = {"summary": summ, "episodes": rows}
            print(key, " | ".join(f"{k}={summ[k]['median']:.1f}[{summ[k]['min']:.0f},{summ[k]['max']:.0f}]n{summ[k]['n']}" for k in summ), flush=True)
    R = np.concatenate(all_reg)
    # least-squares refit: true_local_x ~ a*vn_x + b*vp_x ; true_local_y ~ c*vn_y + d*vp_y
    fx, *_ = np.linalg.lstsq(R[:, [2, 4]], R[:, 0], rcond=None)
    fy, *_ = np.linalg.lstsq(R[:, [3, 5]], R[:, 1], rcond=None)
    res_x = R[:, 0] - R[:, [2, 4]] @ fx
    res_y = R[:, 1] - R[:, [3, 5]] @ fy
    old_x = R[:, 0] - R[:, [2, 4]] @ np.array(W_FWD)
    old_y = R[:, 1] - R[:, [3, 5]] @ np.array(W_SIDE)
    report["refit"] = {"fwd": fx.tolist(), "side": fy.tolist(), "n": len(R),
                       "rmse_refit": float(np.sqrt((res_x ** 2 + res_y ** 2).mean())),
                       "rmse_given_coef": float(np.sqrt((old_x ** 2 + old_y ** 2).mean())),
                       "p99_abs_resid_refit": float(np.quantile(np.hypot(res_x, res_y), 0.99))}
    print("refit", report["refit"])
    (out_dir() / "dr_variants.json").write_text(json.dumps(report, indent=1))
