"""Probe 1: what the legal observation actually contains on real maps.

Measures, per map group and probe policy:
  - ray hit-type distribution (all 64 rays, and the 17 rays inside the 90 deg FOV)
  - ray distance saturation (fraction == 1.0) and quantiles
  - visible-entity count distribution, fraction of steps with >=1 entity / >=1 enemy, entity-type mix
  - rays that hit an enemy vs enemies listed in visible_entities (rays see 360 deg, entities only FOV+LOS)
  - depth buffer: scale vs ray distance, saturation, disagreement with rays, and ridge-regression R^2 of
    pooled depth from ray features (how much of depth is redundant given rays) and vice versa.
  - env step time with and without render_buffers.

Run: .venv/bin/python research/spatial_probes/probe_obs_stats.py [--steps 1050] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter

import numpy as np

from _common import MAPS, POLICIES, out_dir
from doom_env import make_env
from doom_env import spaces as S

FOV_RAYS = np.array(list(range(0, 9)) + list(range(56, 64)))  # -45..+45 deg
FOV_ANG = np.where(FOV_RAYS < 32, FOV_RAYS, FOV_RAYS - 64) * (2 * math.pi / 64)
# screen column that looks along each FOV ray (90 deg hfov, 160 px, positive angle = left)
FOV_COL = np.clip(np.round(80 - 80 * np.tan(FOV_ANG) - 0.5).astype(int), 0, 159)
POOL = (8, 8)  # depth pooled to 15 x 20


def pooled_depth(d):
    h, w = d.shape
    return d.reshape(h // POOL[0], POOL[0], w // POOL[1], POOL[1]).mean(axis=(1, 3))


def ray_feats(obs):
    dist = obs["rays"]["distance"]
    oh = np.eye(6, dtype=np.float32)[obs["rays"]["hit_type"]]
    return np.concatenate([dist, oh.reshape(-1)])  # 64 + 384


def ridge_r2(X, Y, groups, lam=1.0):
    """Held out by episode: the last ~25 % of episodes (by id) are the test set, so near-duplicate
    frames from one episode never straddle the split."""
    ug = np.unique(groups)
    test_g = ug[int(np.ceil(0.75 * len(ug))):] if len(ug) > 1 else ug
    te = np.isin(groups, test_g)
    tr = ~te
    if tr.sum() < 10 or te.sum() < 10:
        return float("nan")
    mx, my = X[tr].mean(0), Y[tr].mean(0)
    Xt, Yt = X[tr] - mx, Y[tr] - my
    W = np.linalg.solve(Xt.T @ Xt + lam * tr.sum() * 1e-2 * np.eye(X.shape[1]), Xt.T @ Yt)
    pred = (X[te] - mx) @ W + my
    ss_res = ((Y[te] - pred) ** 2).sum()
    ss_tot = ((Y[te] - Y[te].mean(0)) ** 2).sum()
    return float(1 - ss_res / max(ss_tot, 1e-9))


def run_group(group, policy_name, steps, seeds):
    hit_all, hit_fov = np.zeros(6), np.zeros(6)
    n_steps, sat = 0, 0
    dists = []
    ent_counts = Counter()
    type_mix = Counter()
    steps_any_ent = steps_any_enemy = steps_ray_enemy = steps_ray_enemy_no_ent = 0
    depth_pairs = []  # (depth value, perpendicular ray distance, hit type)
    Xr, Yd, G = [], [], []
    ep = 0
    horizon_rows = []
    terminations = Counter()
    t_env = 0.0
    for wad, mp in MAPS[group]:
        for seed in range(seeds):
            env = make_env(wad, mp, episode_ticks=4 * steps, render_buffers=True)
            obs, _ = env.reset(seed=seed)
            pol = POLICIES[policy_name](seed)
            for t in range(steps):
                ht = obs["rays"]["hit_type"]
                rd = obs["rays"]["distance"]
                hit_all += np.bincount(ht, minlength=6)
                hit_fov += np.bincount(ht[FOV_RAYS], minlength=6)
                sat += int((rd >= 1.0).sum())
                if t % 5 == 0:
                    dists.append(rd.copy())
                m = obs["visible_entities"]["mask"].astype(bool)
                et = obs["visible_entities"]["entity_type"][m]
                ent_counts[int(m.sum())] += 1
                type_mix.update(S.ENTITY_TYPES[i] for i in et)
                any_enemy = bool(np.isin(et, [1, 2, 3]).any())
                steps_any_ent += int(m.any())
                steps_any_enemy += int(any_enemy)
                ray_enemy = bool((ht == 4).any())
                steps_ray_enemy += int(ray_enemy)
                steps_ray_enemy_no_ent += int(ray_enemy and not any_enemy)
                dep = obs["depth"].astype(np.float32)
                if t % 3 == 0:
                    col = dep[56:64, FOV_COL].mean(axis=0)  # eye-level band around the horizon (row 60)
                    perp = rd[FOV_RAYS] * S.RAY_RANGE * np.cos(FOV_ANG)
                    for k in range(len(FOV_RAYS)):
                        depth_pairs.append((col[k], perp[k], ht[FOV_RAYS[k]], rd[FOV_RAYS[k]]))
                    Xr.append(ray_feats(obs))
                    Yd.append(pooled_depth(dep).reshape(-1))
                    G.append(ep)
                    horizon_rows.append(dep[56:64].mean(axis=0))
                n_steps += 1
                a = pol(obs)
                t0 = time.perf_counter()
                obs, _, te, tr, info = env.step(a)
                t_env += time.perf_counter() - t0
                if te or tr:
                    terminations["exit" if info["exit_reached"] else ("trunc" if tr else "death")] += 1
                    break
            else:
                terminations["cap"] += 1
            env.close()
            ep += 1
    dists = np.concatenate(dists)
    P = np.array(depth_pairs, np.float64)
    res = {
        "group": group, "policy": policy_name, "steps": n_steps, "episodes_end": dict(terminations),
        "env_step_ms_render": 1000 * t_env / n_steps,
        "hit_frac_all": dict(zip(S.RAY_HIT_TYPES, np.round(hit_all / hit_all.sum(), 4).tolist())),
        "hit_frac_fov": dict(zip(S.RAY_HIT_TYPES, np.round(hit_fov / hit_fov.sum(), 4).tolist())),
        "ray_saturated_frac": sat / (64 * n_steps),
        "ray_frac_below_64u": float((dists < 64 / 1024).mean()),
        "ray_frac_below_128u": float((dists < 128 / 1024).mean()),
        "ray_dist_quantiles_units": dict(zip(["p10", "p25", "p50", "p75", "p90"],
                                             np.round(np.quantile(dists, [.1, .25, .5, .75, .9]) * 1024, 1).tolist())),
        "frac_steps_any_entity": steps_any_ent / n_steps,
        "frac_steps_any_enemy_entity": steps_any_enemy / n_steps,
        "frac_steps_enemy_ray": steps_ray_enemy / n_steps,
        "frac_steps_enemy_ray_but_no_enemy_entity": steps_ray_enemy_no_ent / n_steps,
        "entity_count_hist": {k: v / n_steps for k, v in sorted(ent_counts.items())},
        "entity_count_mean": sum(k * v for k, v in ent_counts.items()) / n_steps,
        "entity_count_max": max(ent_counts),
        "entity_count_p99": int(np.quantile(np.repeat(list(ent_counts.keys()), list(ent_counts.values())), 0.99)),
        "entity_type_mix": {k: round(v / max(sum(type_mix.values()), 1), 4) for k, v in type_mix.most_common()},
    }
    # depth scale: fit depth = a * perp on unsaturated wall hits
    w = (P[:, 2] == 1) & (P[:, 3] < 1.0) & (P[:, 0] < 250)
    if w.sum() > 50:
        a = float((P[w, 0] * P[w, 1]).sum() / (P[w, 1] ** 2).sum())
        resid = P[w, 0] - a * P[w, 1]
        res["depth_scale_per_unit"] = a
        res["depth_units_per_level"] = 1 / a
        res["depth_saturation_units"] = 255 / a
        res["depth_wall_fit_mae_levels"] = float(np.abs(resid).mean())
        res["depth_wall_fit_corr"] = float(np.corrcoef(P[w, 0], P[w, 1])[0, 1])
        est = P[:, 0] / a
        valid = (P[:, 0] < 250) & (P[:, 3] < 1.0)
        res["fov_rays_depth_farther_than_ray(>1.25x+16u)"] = float(((est > 1.25 * P[:, 1] + 16) & valid).sum() / max(valid.sum(), 1))
        res["fov_rays_depth_nearer_than_ray(<0.8x-16u)"] = float(((est < 0.8 * P[:, 1] - 16) & valid).sum() / max(valid.sum(), 1))
        res["fov_rays_depth_saturated"] = float((P[:, 0] >= 250).mean())
    Xr, Yd, G = np.array(Xr), np.array(Yd), np.array(G)
    res["n_depth_episodes"] = int(len(np.unique(G)))
    res["n_depth_samples"] = len(Xr)
    res["R2_pooled_depth15x20_from_rays_linear"] = ridge_r2(Xr, Yd, G)
    H = np.array(horizon_rows)
    res["R2_fov_ray_dist_from_horizon_row_linear"] = ridge_r2(H, Xr[:, FOV_RAYS], G)
    # per row band: top (ceiling), middle (walls), bottom (floor)
    Yb = Yd.reshape(-1, 15, 20)
    res["R2_depth_band_from_rays"] = {b: ridge_r2(Xr, Yb[:, sl].reshape(len(Yb), -1), G)
                                      for b, sl in (("rows0-39", slice(0, 5)), ("rows40-79", slice(5, 10)), ("rows80-119", slice(10, 15)))}
    return res


def step_time(wad, mp, render, n=600):
    env = make_env(wad, mp, episode_ticks=4 * (n + 10), render_buffers=render)
    env.reset(seed=0)
    pol = POLICIES["random"](0)
    obs = None
    t0 = time.perf_counter()
    k = 0
    for _ in range(n):
        obs, _, te, tr, _ = env.step(pol(obs))
        k += 1
        if te or tr:
            break
    dt = (time.perf_counter() - t0) / k
    env.close()
    return 1000 * dt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--timing-only", action="store_true", help="run alone: env step time with/without buffers")
    args = ap.parse_args()
    if args.timing_only:
        for g in ("two_rooms", "freedoom2"):
            wad, mp = MAPS[g][0]
            for render in (True, False, True, False):
                print(g, "render" if render else "norender", f"{step_time(wad, mp, render, 1000):.3f} ms/step")
        raise SystemExit
    results = []
    for group in MAPS:
        for pol in POLICIES:
            r = run_group(group, pol, args.steps, args.seeds)
            results.append(r)
            print(json.dumps(r, indent=1))
    (out_dir() / "obs_stats.json").write_text(json.dumps({"results": results}, indent=1))
