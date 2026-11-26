"""PROPOSED CONTRACT: `map_inventory(env) -> dict`, static per-map facts for eval/reward code (training-only).

Requested by the evaluation researcher (docs/EVALUATION_PROTOCOL.md §9.1) to replace the private-attribute adapter
in `training/eval/inventory.py`. Call it once, right after `env.reset()`:

    {
      "num_sectors": int,
      "door_sectors": {sector: lock},   # lock 0 = ordinary door, > 0 = locked (env.mapinfo, static WAD parse)
      "doors_total": int, "locked_total": int,
      "monsters": int, "keys": int,     # live engine objects at reset (skill 3, after ACS OPEN scripts)
      "static_monsters": int, "static_keys": int,  # from the WAD THINGS (skill 3, single player); cross-check
      "sector_area": np.ndarray (S,) float64, map units^2 (16-unit raster of the static sector polygons),
      "cells_total": int,               # 64x64 cells whose sample points fall inside any sector
    }

Sources: `env.mapinfo` and `env.sector_polygons()` are public on DoomEnv. The live monster/key count needs the
engine's object list, which DoomEnv does not expose publicly; that single read (`env._game.get_state()`) is the only
wrapper-internal access and is isolated in `_live_objects`. Never imported by `submission/`.
"""

from __future__ import annotations

import struct

import numpy as np

CELL = 64.0  # exploration cell (docs/EVALUATION_PROTOCOL.md §4.3)
AREA_RES = 16.0

# Doom/Doom2 monster editor numbers (Freedoom uses the same numbers).
MONSTER_EDNUMS = frozenset({
    7, 9, 16, 58, 64, 65, 66, 67, 68, 69, 71, 72, 84, 3001, 3002, 3003, 3004, 3005, 3006,
})
KEY_EDNUMS = frozenset({5, 6, 13, 38, 39, 40})


def _live_objects(env):
    """ADAPTER: engine object list at the current state (wrapper internal)."""
    state = env._game.get_state()
    return [] if state is None else state.objects


def _count_live(env) -> tuple[int, int]:
    from doom_env.sensors import classify

    monsters = keys = 0
    for o in _live_objects(env):
        cat = getattr(o, "category", "")
        if cat == "Monster":
            monsters += 1
        elif classify(o.name, cat) == "key":
            keys += 1
    return monsters, keys


def static_things(wad: str, map_name: str, skill: int = 3) -> tuple[int, int]:
    """(monsters, keys) placed in the WAD for single player at `skill` (no ACS, no DEHACKED)."""
    from doom_env.mapinfo import parse_udmf
    from doom_env.wad import map_lumps, read_wad

    lumps = map_lumps(read_wad(wad), map_name)
    types: list[int] = []
    if "TEXTMAP" in lumps:
        for t in parse_udmf(lumps["TEXTMAP"].decode("utf-8", "replace")).get("thing", []):
            if t.get(f"skill{skill}") is True and t.get("single") is True:
                types.append(int(t.get("type", 0) or 0))
    else:
        data = lumps.get("THINGS", b"")
        hexen = "BEHAVIOR" in lumps
        size = 20 if hexen else 10
        skill_bit = 0x1 if skill <= 2 else (0x2 if skill == 3 else 0x4)
        for i in range(len(data) // size):
            if hexen:
                ttype, flags = struct.unpack_from("<HH", data, i * size + 10)
                single = bool(flags & 0x0100)
            else:
                ttype, flags = struct.unpack_from("<HH", data, i * size + 6)
                single = not flags & 0x0010
            if flags & skill_bit and single:
                types.append(ttype)
    return sum(t in MONSTER_EDNUMS for t in types), sum(t in KEY_EDNUMS for t in types)


def _even_odd(px: np.ndarray, py: np.ndarray, seg: np.ndarray) -> np.ndarray:
    x1, y1, x2, y2 = (seg[:, i][None, :] for i in range(4))
    pxx, pyy = px[:, None], py[:, None]
    cond = (y1 > pyy) != (y2 > pyy)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = x1 + (pyy - y1) * (x2 - x1) / (y2 - y1)
    return (cond & (pxx < xint)).sum(axis=1) % 2 == 1


def map_inventory(env) -> dict:
    mi = env.mapinfo
    n = mi.num_sectors
    door = {int(k): int(v) for k, v in sorted(mi.door_sectors.items())}
    monsters, keys = _count_live(env)
    s_mon, s_keys = static_things(env.wad, env.map_name)
    area = np.zeros(n, np.float64)
    cells: set[tuple[int, int]] = set()
    for sid, seg in env.sector_polygons():
        if not len(seg) or sid >= n:
            continue
        xs, ys = seg[:, [0, 2]], seg[:, [1, 3]]
        gx = np.arange(xs.min() + AREA_RES / 2, xs.max(), AREA_RES)
        gy = np.arange(ys.min() + AREA_RES / 2, ys.max(), AREA_RES)
        if not len(gx) or not len(gy):
            continue
        X, Y = np.meshgrid(gx, gy)
        X, Y = X.ravel(), Y.ravel()
        inside = np.zeros(len(X), bool)
        for k in range(0, len(X), 4096):  # bound memory: points x segments
            inside[k : k + 4096] = _even_odd(X[k : k + 4096], Y[k : k + 4096], seg)
        area[sid] = inside.sum() * AREA_RES**2
        cells.update(zip(np.floor(X[inside] / CELL).astype(int).tolist(), np.floor(Y[inside] / CELL).astype(int).tolist()))
    return {
        "num_sectors": n, "door_sectors": door,
        "doors_total": sum(1 for v in door.values() if v == 0), "locked_total": sum(1 for v in door.values() if v > 0),
        "monsters": monsters, "keys": keys, "static_monsters": s_mon, "static_keys": s_keys,
        "sector_area": area, "cells_total": len(cells),
    }
