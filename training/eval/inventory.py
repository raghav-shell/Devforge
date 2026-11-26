"""Static per-map facts the proxy scorer needs as denominators (eval-side only).

`privileged_state()` has no totals (monsters, keys, door count, map area), so this adapter reads them
once at reset. It touches wrapper internals (`env._game`, `env._sensors`) and is a THIN ADAPTER pending
the environment-engineer contract `map_inventory(env) -> MapInventory` (docs/EVALUATION_PROTOCOL.md §9).
Never imported by submission/.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

CELL = 64.0  # exploration cell size, map units (docs/EVALUATION_PROTOCOL.md §4.3)
AREA_RES = 16.0  # raster resolution for sector areas


def _point_in_segments(px: np.ndarray, py: np.ndarray, seg: np.ndarray) -> np.ndarray:
    """Even-odd test of many points against one sector's boundary segments (K,4)."""
    inside = np.zeros(px.shape, bool)
    for x1, y1, x2, y2 in seg:
        if y1 == y2:
            continue
        cond = (y1 > py) != (y2 > py)
        xint = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
        inside ^= cond & (px < xint)
    return inside


@dataclass
class MapInventory:
    num_sectors: int
    door_sectors: dict[int, int]  # sector id -> lock (0 = ordinary door, >0 = locked)
    keys_present: int  # key items in the level at reset
    monsters_present: int  # monsters in the level at reset (skill 3 spawn set)
    sector_area: np.ndarray  # (S,) approx area per sector (map units^2)
    cells_total: int  # 64x64 cells whose centre lies inside any sector
    seg_by_sector: dict[int, np.ndarray] = field(repr=False, default_factory=dict)
    bbox: dict[int, tuple[float, float, float, float]] = field(repr=False, default_factory=dict)

    @property
    def doors_total(self) -> int:
        return sum(1 for v in self.door_sectors.values() if v == 0)

    @property
    def locked_total(self) -> int:
        return sum(1 for v in self.door_sectors.values() if v > 0)

    def sector_at(self, x: float, y: float) -> int:
        """Sector containing (x, y) by even-odd test; -1 if none. Smallest-area match wins."""
        best, best_area = -1, np.inf
        px, py = np.array([x]), np.array([y])
        for sid, (x0, y0, x1, y1) in self.bbox.items():
            if x0 <= x <= x1 and y0 <= y <= y1 and _point_in_segments(px, py, self.seg_by_sector[sid])[0]:
                a = self.sector_area[sid] if sid < len(self.sector_area) else np.inf
                if a < best_area:
                    best, best_area = sid, a
        return best


def map_inventory(env) -> MapInventory:
    """Call right after env.reset(). Reads engine objects once (eval-side only)."""
    from doom_env.sensors import classify

    st = env.privileged_state()
    door_lock = st["door_lock"]
    doors = {int(s): int(v) for s, v in enumerate(door_lock) if v >= 0}
    state = env._game.get_state()  # ADAPTER: wrapper internal, see module docstring
    monsters = keys = 0
    for o in state.objects:
        cat = getattr(o, "category", "")
        t = classify(o.name, cat)
        if cat == "Monster":
            monsters += 1
        elif t == "key":
            keys += 1
    polys = env.sector_polygons()
    seg_by_sector = {sid: seg for sid, seg in polys if len(seg)}
    bbox = {}
    for sid, seg in seg_by_sector.items():
        xs, ys = seg[:, [0, 2]], seg[:, [1, 3]]
        bbox[sid] = (float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max()))
    n = len(door_lock)
    area = np.zeros(n)
    cells: set[tuple[int, int]] = set()
    for sid, (x0, y0, x1, y1) in bbox.items():
        gx = np.arange(x0 + AREA_RES / 2, x1, AREA_RES)
        gy = np.arange(y0 + AREA_RES / 2, y1, AREA_RES)
        if not len(gx) or not len(gy):
            continue
        X, Y = np.meshgrid(gx, gy)
        m = _point_in_segments(X.ravel(), Y.ravel(), seg_by_sector[sid])
        if sid < n:
            area[sid] = m.sum() * AREA_RES**2
        cx, cy = X.ravel()[m], Y.ravel()[m]
        # a 64-cell counts if any 16-unit sample inside a sector falls in it
        cells.update(zip(np.floor(cx / CELL).astype(int).tolist(), np.floor(cy / CELL).astype(int).tolist()))
    return MapInventory(n, doors, keys, monsters, area, len(cells), seg_by_sector, bbox)
