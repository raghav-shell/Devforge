"""Observation extraction from engine state, all in the player's ego frame.

Conventions (see docs/CONTRACT.md):
- Angles: 0 = straight ahead, positive = counter-clockwise (to the player's left).
- Distances are divided by RAY_RANGE (1.0 = 1024 map units).
- Velocities are map units per tic, rotated into the player frame (forward, right, up).
Nothing returned here contains absolute map coordinates or absolute heading.
"""

from __future__ import annotations

import math
import warnings

import numpy as np

from . import spaces as S
from .mapinfo import MapInfo

PLAYER_HEIGHT = 56.0
MAX_STEP_UP = 24.0
ENTITY_MAX_DISTANCE = 4096.0
HALF_FOV = math.radians(45.0)  # ViZDoom default 90 degree horizontal FOV

RAY_NONE, RAY_WALL, RAY_DOOR, RAY_LOCKED, RAY_ENEMY, RAY_ITEM = range(6)
TYPE_ID = {name: i for i, name in enumerate(S.ENTITY_TYPES)}

HITSCAN = {"Zombieman", "ShotgunGuy", "ChaingunGuy", "WolfensteinSS", "SpiderMastermind"}
MELEE = {"Demon", "Spectre", "LostSoul"}
MONSTER_RADIUS = {
    "Demon": 30, "Spectre": 30, "LostSoul": 16, "Cacodemon": 31, "HellKnight": 24, "BaronOfHell": 24,
    "Arachnotron": 64, "PainElemental": 31, "Fatso": 48, "Cyberdemon": 40, "SpiderMastermind": 128,
}
DEFAULT_RADIUS = 20.0
KEY_NAMES = ("BlueCard", "YellowCard", "RedCard", "BlueSkull", "YellowSkull", "RedSkull")  # vitals order
ITEM_CATEGORIES = {"Health": "health", "Armor": "armor", "Ammo": "ammo", "Weapon": "weapon", "Key": "key", "Powerup": "other"}
PROJECTILE_SUFFIXES = ("Ball", "Shot", "Rocket", "Tracer", "Plasma", "Missile", "Spit", "Fireball")


def classify(name: str, category: str) -> str | None:
    """Map a ViZDoom object to one of spaces.ENTITY_TYPES, or None to ignore it."""
    if category == "Monster":
        if name in HITSCAN:
            return "enemy_hitscan"
        if name in MELEE:
            return "enemy_melee"
        return "enemy_projectile"
    if name in KEY_NAMES:
        return "key"
    if category in ("Projectile", "Missile") or name.endswith(PROJECTILE_SUFFIXES):
        return "projectile"
    if name == "ExplosiveBarrel":
        return "barrel"
    return ITEM_CATEGORIES.get(category)


class LevelGeometry:
    """Unique linedefs of the level (static) + per-step blocking computed from sector heights."""

    def __init__(self, sectors, mapinfo: MapInfo | None):
        num_sectors = (max(s.id for s in sectors) + 1) if sectors else 0
        wad = mapinfo.lines if mapinfo is not None and mapinfo.num_sectors == num_sectors else None
        if wad is not None and len(wad.p1):
            # Exact static geometry from the WAD (preferred): ViZDoom's runtime sector line lists
            # can be incomplete on large maps.
            self.source = "wad"
            keep = np.any(wad.p1 != wad.p2, axis=1)  # drop zero-length lines
            self.p1, self.p2 = wad.p1[keep].copy(), wad.p2[keep].copy()
            front, back = wad.front[keep], wad.back[keep]
            self.sec_a = np.where(front >= 0, front, back)
            self.sec_b = np.where((front >= 0) & (back >= 0), back, -1)
            # Two-sided lines with the same sector on both sides are treated like any two-sided line.
            self.flag_blocking = wad.impassable[keep].copy()
            valid = self.sec_a >= 0
            self.p1, self.p2, self.sec_a, self.sec_b, self.flag_blocking = (
                self.p1[valid], self.p2[valid], self.sec_a[valid], self.sec_b[valid], self.flag_blocking[valid])
            self.one_sided = self.sec_b < 0
            self.passable_same_sector = np.zeros(len(self.sec_a), bool)
        else:
            self.source = "engine"
            if mapinfo is not None and mapinfo.num_sectors > num_sectors:
                warnings.warn(
                    f"map has {mapinfo.num_sectors} sectors but ViZDoom only reports {num_sectors}: walls and doors "
                    "beyond that limit are missing from rays/visible_entities. Use maps with fewer than 2048 sectors.",
                    RuntimeWarning, stacklevel=3)
            lines: dict[tuple, list] = {}
            for sec in sectors:
                for ln in sec.lines:
                    a, b = (ln.x1, ln.y1), (ln.x2, ln.y2)
                    key = (min(a, b), max(a, b))
                    entry = lines.setdefault(key, [a, b, set(), False])
                    entry[2].add(sec.id)
                    entry[3] = entry[3] or bool(ln.is_blocking)
            n = len(lines)
            self.p1 = np.zeros((n, 2), np.float64)
            self.p2 = np.zeros((n, 2), np.float64)
            self.sec_a = np.zeros(n, np.int64)
            self.sec_b = np.full(n, -1, np.int64)
            self.flag_blocking = np.zeros(n, bool)
            for i, (a, b, secs, blocking) in enumerate(lines.values()):
                self.p1[i], self.p2[i] = a, b
                ids = sorted(secs)
                self.sec_a[i] = ids[0]
                if len(ids) > 1:
                    self.sec_b[i] = ids[1]
                self.flag_blocking[i] = blocking
            self.one_sided = self.sec_b < 0
            # A one-sector line that ViZDoom does not flag as blocking (e.g. a two-sided line with
            # the same sector on both sides) does not stop movement or sight.
            self.passable_same_sector = self.one_sided & ~self.flag_blocking
        self.ordered = [s.id for s in sectors] == list(range(num_sectors))
        self.door_lock = np.full(num_sectors, -1, np.int64)  # -1 = not a door
        if mapinfo is not None and mapinfo.num_sectors == num_sectors:
            for sid, lock in mapinfo.door_sectors.items():
                self.door_lock[sid] = lock
        self.lines_bb_min = np.minimum(self.p1, self.p2)
        self.lines_bb_max = np.maximum(self.p1, self.p2)

    def update(self, sectors) -> None:
        """Compute movement/sight blocking for the current sector heights."""
        if self.ordered:
            floor = np.array([s.floor_height for s in sectors], np.float64)
            ceil = np.array([s.ceiling_height for s in sectors], np.float64)
        else:
            floor = np.zeros(len(self.door_lock), np.float64)
            ceil = np.zeros(len(self.door_lock), np.float64)
            for s in sectors:
                floor[s.id], ceil[s.id] = s.floor_height, s.ceiling_height
        self.floor, self.ceil = floor, ceil
        b = np.where(self.one_sided, self.sec_a, self.sec_b)
        a = self.sec_a
        opening = np.minimum(ceil[a], ceil[b]) - np.maximum(floor[a], floor[b])
        step = np.abs(floor[a] - floor[b])
        self.move_block = (self.one_sided & ~self.passable_same_sector) | (
            ~self.one_sided & (self.flag_blocking | (opening < PLAYER_HEIGHT) | (step > MAX_STEP_UP))
        )
        self.sight_block = (self.one_sided & ~self.passable_same_sector) | (~self.one_sided & (opening <= 0))
        hit = np.full(len(a), RAY_WALL, np.int64)
        closed = ~self.one_sided & (opening < PLAYER_HEIGHT)
        lock_a = self.door_lock[a]
        lock_b = np.where(self.one_sided, -1, self.door_lock[b])
        lock = np.maximum(lock_a, lock_b)
        hit[closed & (lock == 0)] = RAY_DOOR
        hit[closed & (lock > 0)] = RAY_LOCKED
        self.hit_type = hit

    def nearby(self, x: float, y: float, radius: float) -> np.ndarray:
        return np.flatnonzero(
            (self.lines_bb_min[:, 0] <= x + radius)
            & (self.lines_bb_max[:, 0] >= x - radius)
            & (self.lines_bb_min[:, 1] <= y + radius)
            & (self.lines_bb_max[:, 1] >= y - radius)
        )


def _ray_line_t(origin: np.ndarray, dirs: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """Distance along each ray (R,2) to each segment (M,2). Returns (R, M), inf where no hit."""
    d = p2 - p1  # (M,2)
    w = p1 - origin  # (M,2)
    denom = dirs[:, None, 0] * d[None, :, 1] - dirs[:, None, 1] * d[None, :, 0]  # cross(r, d)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * d[None, :, 1] - w[None, :, 1] * d[None, :, 0]) / denom
        u = (w[None, :, 0] * dirs[:, None, 1] - w[None, :, 1] * dirs[:, None, 0]) / denom
    ok = (np.abs(denom) > 1e-9) & (t > 1e-6) & (u >= 0.0) & (u <= 1.0)
    return np.where(ok, t, np.inf)


def _segments_blocked(origin: np.ndarray, targets: np.ndarray, p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    """For each target point (K,2): does the segment origin->target cross any segment (M,2)?"""
    if len(targets) == 0 or len(p1) == 0:
        return np.zeros(len(targets), bool)
    r = targets - origin  # (K,2), not normalized: hit if 0 < t < 1
    d = p2 - p1
    w = p1 - origin
    denom = r[:, None, 0] * d[None, :, 1] - r[:, None, 1] * d[None, :, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (w[None, :, 0] * d[None, :, 1] - w[None, :, 1] * d[None, :, 0]) / denom
        u = (w[None, :, 0] * r[:, None, 1] - w[None, :, 1] * r[:, None, 0]) / denom
    hit = (np.abs(denom) > 1e-9) & (t > 1e-6) & (t < 1.0 - 1e-6) & (u >= 0.0) & (u <= 1.0)
    return hit.any(axis=1)


_CLASS_CACHE: dict[str, tuple[int, float]] = {}
_KEY_TYPE = TYPE_ID["key"]
_RAY_CODE_BY_TYPE = np.array(
    [RAY_ENEMY if t.startswith("enemy") else RAY_ITEM if t in ("health", "armor", "ammo", "weapon", "key", "other") else RAY_NONE
     for t in S.ENTITY_TYPES],
    np.int64,
)
_RAY_ANGLES = np.arange(S.NUM_RAYS) * (2.0 * np.pi / S.NUM_RAYS)


def ego_rotate(vx: np.ndarray | float, vy: np.ndarray | float, heading: float):
    """World vector -> (forward, right) in the player frame."""
    c, s = math.cos(heading), math.sin(heading)
    return vx * c + vy * s, vx * s - vy * c


class Sensors:
    def __init__(self, geometry: LevelGeometry):
        self.geo = geometry

    def compute(self, px: float, py: float, heading: float, pvel: tuple[float, float], objects) -> tuple[dict, dict]:
        geo = self.geo
        origin = np.array([px, py], np.float64)

        # ---- rays vs blocking lines ------------------------------------------------------
        idx = geo.nearby(px, py, S.RAY_RANGE)
        idx = idx[geo.move_block[idx]]
        angles = heading + _RAY_ANGLES
        dirs = np.stack([np.cos(angles), np.sin(angles)], axis=1)
        if len(idx):
            t = _ray_line_t(origin, dirs, geo.p1[idx], geo.p2[idx])
            best = np.argmin(t, axis=1)
            dist = t[np.arange(S.NUM_RAYS), best]
            hit = np.where(np.isfinite(dist), geo.hit_type[idx][best], RAY_NONE)
        else:
            dist = np.full(S.NUM_RAYS, np.inf)
            hit = np.full(S.NUM_RAYS, RAY_NONE)

        # ---- entities ------------------------------------------------------------------
        rows = []  # (type index, x, y, vx, vy, radius)
        keys: dict[int, str] = {}
        cache = _CLASS_CACHE
        for o in objects:
            name = o.name
            info = cache.get(name)
            if info is None:
                category = o.category
                kind = None if category == "Self" else classify(name, category)
                info = (TYPE_ID[kind], MONSTER_RADIUS.get(name, DEFAULT_RADIUS)) if kind else (0, 0.0)
                if category != "Monster":  # monsters change name when they die, others never change class
                    cache[name] = info
            if info[0]:
                if info[0] == _KEY_TYPE:
                    keys[o.id] = name
                rows.append((info[0], o.position_x, o.position_y, o.velocity_x, o.velocity_y, info[1]))
        self.present_keys = keys
        if rows:
            arr = np.array(rows, np.float64)
        else:
            arr = np.zeros((0, 6), np.float64)
        etypes = arr[:, 0].astype(np.int64)
        rel_xy = arr[:, 1:3] - origin
        edist = np.hypot(rel_xy[:, 0], rel_xy[:, 1])
        keep = (edist <= ENTITY_MAX_DISTANCE) & (edist > 1e-6)
        arr, etypes, rel_xy, edist = arr[keep], etypes[keep], rel_xy[keep], edist[keep]
        rel_angle = (np.arctan2(rel_xy[:, 1], rel_xy[:, 0]) - heading + np.pi) % (2 * np.pi) - np.pi
        radius = arr[:, 5]

        # Rays also stop at alive monsters (enemy) and pickups (item).
        ray_code = _RAY_CODE_BY_TYPE[etypes]
        solid = ray_code != RAY_NONE
        if solid.any():
            c = rel_xy[solid]  # (K,2)
            rad = radius[solid]
            tc = dirs @ c.T  # (R,K)
            d2 = (c * c).sum(axis=1)[None, :] - tc**2
            inside = (tc > 0) & (d2 <= rad[None, :] ** 2)
            t_hit = np.where(inside, np.maximum(tc - np.sqrt(np.maximum(rad[None, :] ** 2 - d2, 0.0)), 0.0), np.inf)
            k = np.argmin(t_hit, axis=1)
            t_best = t_hit[np.arange(S.NUM_RAYS), k]
            closer = t_best < dist
            dist = np.where(closer, t_best, dist)
            hit = np.where(closer, ray_code[solid][k], hit)

        in_range = dist <= S.RAY_RANGE
        rays = {
            "distance": np.where(in_range, dist / S.RAY_RANGE, 1.0).astype(np.float32),
            "hit_type": np.where(in_range, hit, RAY_NONE).astype(np.int64),
        }

        # ---- visible entities: field of view + line of sight -----------------------------
        vis = np.flatnonzero(np.abs(rel_angle) <= HALF_FOV + np.arctan2(radius, edist))
        if len(vis):
            # Only lines that can lie between the player and some target: within the farthest
            # target's distance and not entirely behind the player (targets are all in front).
            sidx = geo.nearby(px, py, float(edist[vis].max()))
            sidx = sidx[geo.sight_block[sidx]]
            fx, fy = math.cos(heading), math.sin(heading)
            f1 = (geo.p1[sidx, 0] - px) * fx + (geo.p1[sidx, 1] - py) * fy
            f2 = (geo.p2[sidx, 0] - px) * fx + (geo.p2[sidx, 1] - py) * fy
            sidx = sidx[np.maximum(f1, f2) > 0]
            blocked = _segments_blocked(origin, arr[vis, 1:3], geo.p1[sidx], geo.p2[sidx])
            vis = vis[~blocked]
        vis = vis[np.argsort(edist[vis], kind="stable")][: S.MAX_ENTITIES]
        n = len(vis)

        feats = np.zeros((S.MAX_ENTITIES, len(S.ENTITY_FEATURES)), np.float32)
        etype = np.zeros(S.MAX_ENTITIES, np.int64)
        mask = np.zeros(S.MAX_ENTITIES, np.int8)
        if n:
            fwd, right = ego_rotate(arr[vis, 3] - pvel[0], arr[vis, 4] - pvel[1], heading)
            ang = rel_angle[vis]
            feats[:n] = np.stack([edist[vis] / S.RAY_RANGE, np.sin(ang), np.cos(ang), fwd, right], axis=1)
            etype[:n] = etypes[vis]
            mask[:n] = 1
        entities = {"features": feats, "entity_type": etype, "mask": mask}
        return rays, entities


class KeyTracker:
    """Key flags: a key item that disappears from the level has been picked up (single player)."""

    def __init__(self):
        self.present: dict[int, str] = {}
        self.held = np.zeros(len(KEY_NAMES), np.float32)

    def reset(self, present: dict[int, str]) -> None:
        self.present = dict(present)
        self.held[:] = 0.0

    def update(self, present: dict[int, str]) -> None:
        """present: {object id: key class name} for key items currently in the level."""
        for oid, name in list(self.present.items()):
            if oid not in present:
                self.held[KEY_NAMES.index(name)] = 1.0
                del self.present[oid]
        for oid, name in present.items():  # keys spawned mid-level (rare)
            self.present.setdefault(oid, name)
