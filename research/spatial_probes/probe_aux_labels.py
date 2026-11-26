"""Probe 5: base rates and signal of candidate auxiliary targets built from TRAINING-ONLY ground truth.

Only targets that are NOT a function of the current legal observation are worth an aux head (a head that
re-derives "enemy in view" from mask+type adds nothing). Candidates measured:
  hidden_enemy_512     alive monster within 512 u (any direction, ignoring LOS) and not in visible_entities
  hidden_enemy_any     ... within 1024 u, not visible and not hit by any ray
  door_near_hidden     door/locked-door line within 256 u but no ray currently reports door/locked_door
  locked_seen_before   a locked_door ray hit happened earlier in the episode (memory target; legal-derivable
                       from history but not from the current frame) -- base rate only
  key_hidden_1024      key item within 1024 u, not in visible_entities
  exit_dist            Euclidean distance to the nearest exit linedef midpoint (training-only map info)
  novel_cell64         the true 64 u cell is visited for the first time this episode
  rays_t+4             future ray distances: persistence-baseline MSE vs variance (learnable signal)
  (local free space / occupancy completion is measured separately in probe_free_space.py)
Run: .venv/bin/python research/spatial_probes/probe_aux_labels.py [--steps 1050] [--seeds 2]
"""
from __future__ import annotations

import argparse
import json
import math
import re
import struct

import numpy as np

from _common import MAPS, POLICIES, out_dir
from doom_env import make_env
from doom_env.wad import map_lumps, read_wad

DOOM_EXITS = {11, 51, 52, 124, 197, 198}
ZDOOM_EXITS = {243, 244}


def exit_points(wad, mp):
    lumps = map_lumps(read_wad(wad), mp)
    pts = []
    if "TEXTMAP" in lumps:
        txt = lumps["TEXTMAP"].decode("utf-8", "replace")
        ns = re.search(r'namespace\s*=\s*"([^"]*)"', txt).group(1).lower()
        exits = DOOM_EXITS if ns in ("doom", "heretic", "strife") else ZDOOM_EXITS
        verts = [(float(a), float(b)) for a, b in re.findall(r"vertex\s*\{\s*x\s*=\s*([-\d.]+);\s*y\s*=\s*([-\d.]+);", txt)]
        for body in re.findall(r"linedef\s*\{([^}]*)\}", txt):
            sp = re.search(r"special\s*=\s*(\d+)", body)
            if sp and int(sp.group(1)) in exits:
                v1 = int(re.search(r"v1\s*=\s*(\d+)", body).group(1))
                v2 = int(re.search(r"v2\s*=\s*(\d+)", body).group(1))
                pts.append(((verts[v1][0] + verts[v2][0]) / 2, (verts[v1][1] + verts[v2][1]) / 2))
    else:
        V = lumps["VERTEXES"]
        verts = [struct.unpack_from("<hh", V, 4 * i) for i in range(len(V) // 4)]
        L = lumps["LINEDEFS"]
        for i in range(len(L) // 14):
            v1, v2, _fl, sp, _tag, _f, _b = struct.unpack_from("<HHHHHHH", L, 14 * i)
            if sp in DOOM_EXITS:
                pts.append(((verts[v1][0] + verts[v2][0]) / 2, (verts[v1][1] + verts[v2][1]) / 2))
    return np.array(pts, np.float64).reshape(-1, 2)


def point_seg_dist(p, a, b):
    d = b - a
    t = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-9), 0, 1)
    return np.linalg.norm(a + t[:, None] * d - p, axis=1)


def run(group, pol_name, steps, seeds):
    acc = {k: [] for k in ("hidden_enemy_512", "hidden_enemy_any", "door_near_hidden", "door_near", "door_ray",
                           "locked_seen_before", "key_hidden_1024", "novel_cell64", "has_exit", "exit_dist")}
    fut_mse, fut_var, fut_mse_fov = [], [], []
    for wad, mp in MAPS[group]:
        ex = exit_points(wad, mp)
        for seed in range(seeds):
            env = make_env(wad, mp, episode_ticks=4 * steps + 8, render_buffers=False)
            obs, _ = env.reset(seed=seed)
            geo = env._geometry  # training-only internals (diagnostic)
            door_line = (geo.door_lock[geo.sec_a] >= 0) | ((geo.sec_b >= 0) & (geo.door_lock[np.maximum(geo.sec_b, 0)] >= 0))
            pol = POLICIES[pol_name](seed)
            visited = set()
            locked_seen = False
            ray_hist = []
            for t in range(steps):
                ps = env.privileged_state()
                p = np.array([ps["x"], ps["y"]])
                objs = env._game.get_state().objects
                ht = obs["rays"]["hit_type"]
                vis_types = obs["visible_entities"]["entity_type"][obs["visible_entities"]["mask"].astype(bool)]
                vis_enemy = int(np.isin(vis_types, [1, 2, 3]).sum())
                mons = [o for o in objs if o.category == "Monster" and o.name not in ("DeadZombieMan",)]
                alive_d = np.array([math.hypot(o.position_x - p[0], o.position_y - p[1]) for o in mons]) if mons else np.zeros(0)
                n512 = int((alive_d <= 512).sum())
                n1024 = int((alive_d <= 1024).sum())
                acc["hidden_enemy_512"].append(n512 > vis_enemy)
                acc["hidden_enemy_any"].append(n1024 > 0 and vis_enemy == 0 and not (ht == 4).any())
                dl = np.flatnonzero(door_line)
                dn = bool(len(dl)) and bool((point_seg_dist(p, geo.p1[dl], geo.p2[dl]) <= 256).any())
                dray = bool(np.isin(ht, [2, 3]).any())
                acc["door_near"].append(dn)
                acc["door_ray"].append(dray)
                acc["door_near_hidden"].append(dn and not dray)
                locked_seen = locked_seen or bool((ht == 3).any())
                acc["locked_seen_before"].append(locked_seen and not (ht == 3).any())
                keys = [o for o in objs if o.name in ("BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull")]
                kn = any(math.hypot(o.position_x - p[0], o.position_y - p[1]) <= 1024 for o in keys)
                acc["key_hidden_1024"].append(kn and not (vis_types == 9).any())
                cell = (int(p[0] // 64), int(p[1] // 64))
                acc["novel_cell64"].append(cell not in visited)
                visited.add(cell)
                acc["has_exit"].append(len(ex) > 0)
                if len(ex):
                    acc["exit_dist"].append(float(np.linalg.norm(ex - p, axis=1).min()))
                ray_hist.append(obs["rays"]["distance"].copy())
                obs, _, te, tr, _ = env.step(pol(obs))
                if te or tr:
                    break
            env.close()
            R = np.array(ray_hist)
            if len(R) > 8:
                fut_mse.append(float(((R[4:] - R[:-4]) ** 2).mean()))
                fut_var.append(float(R[4:].var(axis=0).mean()))
                fov = list(range(0, 9)) + list(range(56, 64))
                fut_mse_fov.append(float(((R[4:, fov] - R[:-4, fov]) ** 2).mean()))
    out = {"group": group, "policy": pol_name, "n_steps": len(acc["novel_cell64"])}
    for k, v in acc.items():
        if k == "exit_dist":
            if v:
                out["exit_dist_quantiles"] = np.round(np.quantile(v, [0.1, 0.5, 0.9]), 1).tolist()
        else:
            out[k] = float(np.mean(v)) if v else None
    out["rays_t+4_persistence_mse"] = float(np.mean(fut_mse))
    out["rays_t+4_target_var"] = float(np.mean(fut_var))
    out["rays_t+4_persistence_R2"] = 1 - float(np.mean(fut_mse)) / float(np.mean(fut_var))
    out["rays_t+4_fov_persistence_mse"] = float(np.mean(fut_mse_fov))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1050)
    ap.add_argument("--seeds", type=int, default=2)
    args = ap.parse_args()
    res = []
    for g in MAPS:
        for pol in POLICIES:
            r = run(g, pol, args.steps, args.seeds)
            res.append(r)
            print(json.dumps(r), flush=True)
    (out_dir() / "aux_labels.json").write_text(json.dumps(res, indent=1))
