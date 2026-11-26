"""TEST-ONLY scripted walker for generated maps.

It uses `env.privileged_state()` (absolute x/y/angle), the generator's cell grid and a BFS over it, plus
atan2 steering. All of that is forbidden for agents (docs/RULES.md). It exists only to prove, inside the test
suite, that generated maps really are completable in the engine. Never import this from training/ or
submission/.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np

FWD, BACK, LEFT, RIGHT, TURN_L, TURN_R, ATTACK, USE, NEXTW = range(9)


def _act(*buttons: int) -> np.ndarray:
    a = np.zeros(9, np.int8)
    for b in buttons:
        a[b] = 1
    return a


def _vertex_ok(lab: np.ndarray, vx: int, vy: int) -> bool:
    H, W = lab.shape
    return 1 <= vx < W and 1 <= vy < H and bool((lab[vy - 1 : vy + 1, vx - 1 : vx + 1] >= 0).all())


def _path(lab: np.ndarray, src: tuple[int, int], dst: tuple[int, int]) -> list[tuple[int, int]]:
    """Shortest vertex path, preferring vertices away from walls (cost only via BFS tie-breaking order)."""
    prev = {src: None}
    q = deque([src])
    while q:
        v = q.popleft()
        if v == dst:
            break
        x, y = v
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb not in prev and _vertex_ok(lab, *nb):
                prev[nb] = v
                q.append(nb)
    if dst not in prev:
        raise AssertionError(f"walker: no cell path {src} -> {dst}")
    out = [dst]
    while prev[out[-1]] is not None:
        out.append(prev[out[-1]])
    return out[::-1]


def _nearest_vertex(lab: np.ndarray, x: float, y: float, cell: int) -> tuple[int, int]:
    gx, gy = x / cell, y / cell
    best = None
    for vx in range(int(gx) - 2, int(gx) + 4):
        for vy in range(int(gy) - 2, int(gy) + 4):
            if _vertex_ok(lab, vx, vy):
                d = (vx - gx) ** 2 + (vy - gy) ** 2
                if best is None or d < best[0]:
                    best = (d, (vx, vy))
    assert best is not None, "walker: player not near any valid vertex"
    return best[1]


def walk_to_exit(env, gm, max_steps: int = 4000) -> dict:
    """Drive the player through every key (lock order) to the exit. Returns a summary dict."""
    lab = gm.geometry.labels
    meta = gm.meta
    cell = meta["cell_size"]
    door_secs = {d["sector"] for d in meta["doors"]}
    targets = [("key", tuple(k["cell"])) for k in meta["keys"]]
    ex = meta["exit"]
    targets.append(("exit", (int(round(ex["x"] / cell)), int(round(ex["y"] / cell)))))
    seen_hits: set[int] = set()
    steps = 0
    last_use = -99
    stuck = 0
    last_pos = None
    info = {"exit_reached": False}
    terminated = truncated = False
    st = env.privileged_state()
    keys_wanted = 0
    for kind, (tx, ty) in targets:
        keys_wanted += kind == "key"
        if kind == "key":
            goal = next(v for v in ((tx, ty), (tx + 1, ty), (tx, ty + 1), (tx + 1, ty + 1)) if _vertex_ok(lab, *v))
        elif ex["kind"] == "switch":
            fx, fy = ex["facing"]
            goal = (tx + fx, ty + fy)
        else:
            goal = (tx, ty)
        path = _path(lab, _nearest_vertex(lab, st["x"], st["y"], cell), goal)
        i = 0
        while True:
            st = env.privileged_state()
            px, py, ang = st["x"], st["y"], st["angle"]
            if kind == "key" and st["keys"].sum() >= keys_wanted:
                break
            while i < len(path) - 1 and math.hypot(path[i][0] * cell - px, path[i][1] * cell - py) < 20:
                i += 1
            j = min(i + 1, len(path) - 1)
            wx, wy = path[j][0] * cell, path[j][1] * cell
            at_goal = i >= len(path) - 1 and math.hypot(wx - px, wy - py) < 20
            if at_goal and kind == "exit" and ex["kind"] == "switch":
                want = math.degrees(math.atan2(-ex["facing"][1], -ex["facing"][0]))
            elif at_goal and kind == "key":
                break
            else:
                want = math.degrees(math.atan2(wy - py, wx - px))
            err = (want - ang + 180) % 360 - 180
            buttons = []
            if abs(err) > 8:
                buttons.append(TURN_L if err > 0 else TURN_R)
            if abs(err) < 40 and not at_goal:
                buttons.append(FWD)
            # Doors: if a closed door sector is within reach ahead and we are not moving, press USE once.
            moved = last_pos is None or math.hypot(px - last_pos[0], py - last_pos[1]) > 2
            stuck = 0 if moved else stuck + 1
            near_door = any(int(lab[cy, cx]) in door_secs and st["sector_ceiling"][lab[cy, cx]] - st["sector_floor"][lab[cy, cx]] < 56
                            for cx in range(int(px // cell) - 2, int(px // cell) + 3)
                            for cy in range(int(py // cell) - 2, int(py // cell) + 3)
                            if 0 <= cy < lab.shape[0] and 0 <= cx < lab.shape[1])
            if (stuck >= 2 and near_door and steps - last_use > 25) or (at_goal and kind == "exit" and abs(err) < 10 and steps - last_use > 10):
                buttons = [USE]
                last_use = steps
            elif stuck >= 4 and not near_door and not at_goal:
                # wedged on a wall corner: back off and side-step (alternating side), then resume steering
                buttons = [BACK, LEFT if (steps // 8) % 2 else RIGHT]
            last_pos = (px, py)
            obs, _r, terminated, truncated, info = env.step(_act(*buttons))
            seen_hits.update(int(h) for h in np.unique(obs["rays"]["hit_type"]))
            steps += 1
            if terminated or truncated or steps >= max_steps:
                return {"exit_reached": bool(info.get("exit_reached")), "terminated": terminated, "truncated": truncated,
                        "steps": steps, "keys": env.privileged_state()["keys"].tolist(), "hit_types": sorted(seen_hits)}
    return {"exit_reached": False, "terminated": False, "truncated": False, "steps": steps,
            "keys": st["keys"].tolist(), "hit_types": sorted(seen_hits)}
