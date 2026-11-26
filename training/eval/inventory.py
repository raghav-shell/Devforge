"""Static per-map facts the proxy scorer needs as denominators (eval-side only, never imported by submission/).

D13 (docs/DECISIONS.md): the single `map_inventory` implementation is `training.envs.inventory.map_inventory`.
This module is a compatibility shim: it wraps that dict in the `MapInventory` dataclass the metric contract uses
(docs/MODULE_CONTRACTS.md §7, `MetricFn(EpisodeTrace, MapInventory)`), and adds the sector point lookup
(`sector_at`) that core metrics need, built from the public `env.sector_polygons()`. It reads no wrapper internals.

Field mapping (unchanged semantics vs the pre-D13 adapter, so proxy_v1 numbers stay comparable):
    keys_present     <- inv["keys"]      (live engine objects at reset)
    monsters_present <- inv["monsters"]  (live, skill 3, after ACS OPEN scripts; NOT static_monsters)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from training.envs.inventory import AREA_RES, CELL  # noqa: F401  re-exported (metrics.core imports CELL)
from training.envs.inventory import map_inventory as env_map_inventory


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
    keys_present: int  # key items in the level at reset (live)
    monsters_present: int  # monsters in the level at reset (live, skill 3)
    sector_area: np.ndarray  # (S,) area per sector (map units^2, 16-unit raster)
    cells_total: int  # 64x64 cells with a raster sample inside any sector
    seg_by_sector: dict[int, np.ndarray] = field(repr=False, default_factory=dict)
    bbox: dict[int, tuple[float, float, float, float]] = field(repr=False, default_factory=dict)
    static_monsters: int = 0  # WAD THINGS, skill 3 single player (cross-check only)
    static_keys: int = 0

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


def from_dict(inv: dict, polygons) -> MapInventory:
    seg_by_sector = {int(sid): seg for sid, seg in polygons if len(seg)}
    bbox = {sid: (float(seg[:, [0, 2]].min()), float(seg[:, [1, 3]].min()), float(seg[:, [0, 2]].max()),
                  float(seg[:, [1, 3]].max())) for sid, seg in seg_by_sector.items()}
    return MapInventory(
        num_sectors=int(inv["num_sectors"]), door_sectors=dict(inv["door_sectors"]), keys_present=int(inv["keys"]),
        monsters_present=int(inv["monsters"]), sector_area=np.asarray(inv["sector_area"], np.float64),
        cells_total=int(inv["cells_total"]), seg_by_sector=seg_by_sector, bbox=bbox,
        static_monsters=int(inv["static_monsters"]), static_keys=int(inv["static_keys"]))


def map_inventory(env) -> MapInventory:
    """Call right after env.reset(). Delegates to training.envs.inventory.map_inventory (D13)."""
    return from_dict(env_map_inventory(env), env.sector_polygons())
