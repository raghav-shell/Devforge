"""Seeded, deterministic procedural training maps: (stage, family, seed) -> byte-identical WAD + metadata.

GENERATOR-INTERNAL, TRAINING-ONLY MAP CONSTRUCTION. Any graph search used here (room-graph BFS, cell-grid
flood fill) is map *authoring*: it places locks, keys and the exit, and certifies solvability. None of it is
reachable from `submission/` or given to the agent. The JSON sidecar (room graph, key/door positions) is
training/eval metadata and must never be fed to the policy.

Determinism: every random draw comes from `random.Random(<sha256 of the seed string>)`; nothing iterates over
hash-ordered containers of strings; floats are formatted with fixed precision. The generator version is part of
the seed string and of the `source_group`, so any change in output also changes the group.
"""

from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import things as T
from .config import DOOR_TEXTURES, EXIT_FLOOR, LOCKED_DOOR_TEXTURES, LOCKED_TRACK_TEXTURES, PALETTES, StageConfig, stage_config
from .layout import CELL, CORR_BASE, VOID, Layout, LayoutError, build_layout
from .progression import CellWorld, Progression, check_solvable, plan
from .udmf import DOOR_DELAY, DOOR_LOCKED_RAISE, DOOR_OPEN, DOOR_RAISE, DOOR_SPEED, MapGeometry, SectorDef, Thing, build_textmap, wad_bytes

GENERATOR = "doomsday-udmfgen"
GENERATOR_VERSION = 1
MAX_SECTORS = 1024  # hard cap, well under ViZDoom's 2,048-sector reporting limit
MAX_ATTEMPTS = 64
BRANCH_ATTEMPTS = 8  # first attempts insist on every key in a side branch (backtracking), then relax
MIN_HEADROOM = 96  # every walkable sector: ceiling - floor >= 96 (player height 56)


@dataclass
class GeneratedMap:
    stage: str
    family: str
    seed: int
    map_name: str
    wad: bytes
    meta: dict
    layout: Layout
    progression: Progression
    geometry: MapGeometry

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.wad).hexdigest()

    def write(self, out_dir: str | Path, name: str | None = None) -> tuple[Path, Path]:
        """Write <name>.wad and <name>.json into out_dir. Returns (wad_path, json_path)."""
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        name = name or default_name(self.stage, self.family, self.seed, self.meta.get("config_key", ""))
        wad_path, json_path = out / f"{name}.wad", out / f"{name}.json"
        wad_path.write_bytes(self.wad)
        json_path.write_text(json.dumps(self.meta, indent=1, sort_keys=True) + "\n")
        return wad_path, json_path


def default_name(stage: str, family: str, seed: int, config_key: str = "") -> str:
    """Overridden configs get their override hash appended, so they never overwrite the plain seed's file."""
    return f"{stage}-{family}-{seed:06d}" + (f"-{config_key}" if config_key else "")


def source_group(stage: str, family: str) -> str:
    return f"udmfgen:{stage}-{family}-v{GENERATOR_VERSION}"


def _rng(stage: str, family: str, seed: int, attempt: int, cfg_key: str) -> random.Random:
    s = f"{GENERATOR}:v{GENERATOR_VERSION}:{stage}:{family}:{seed}:{attempt}:{cfg_key}"
    return random.Random(int.from_bytes(hashlib.sha256(s.encode()).digest()[:8], "little"))


def generate(stage: str, seed: int, family: str = "a", map_name: str = "MAP01", **overrides) -> GeneratedMap:
    """Build one map. Same arguments -> byte-identical WAD. Raises RuntimeError after MAX_ATTEMPTS."""
    cfg = stage_config(stage, family, **overrides)
    cfg_key = "" if not overrides else hashlib.sha256(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:8]
    last = None
    for attempt in range(MAX_ATTEMPTS):
        rng = _rng(stage, family, seed, attempt, cfg_key)
        try:
            return _attempt(cfg, seed, attempt, map_name, rng, cfg_key)
        except LayoutError as e:
            last = e
    raise RuntimeError(f"mapgen {stage}/{family}/{seed}: no valid map in {MAX_ATTEMPTS} attempts ({last})")


# ------------------------------------------------------------------------------------------- build
def _attempt(cfg: StageConfig, seed: int, attempt: int, map_name: str, rng: random.Random, cfg_key: str) -> GeneratedMap:
    layout = build_layout(cfg, rng)
    prog = plan(layout, cfg, rng, require_branch=attempt < BRANCH_ATTEMPTS)
    pal = PALETTES[cfg.family]
    corr_by_id = {c.id: c for c in layout.corridors}
    labels = np.full((layout.height, layout.width), VOID, np.int64)
    sectors: list[SectorDef] = []

    def new_sector(sd: SectorDef) -> int:
        sectors.append(sd)
        return len(sectors) - 1

    room_sec: dict[int, int] = {}
    room_floor: dict[int, int] = {}
    for r in layout.rooms:
        f = rng.choice(cfg.floor_choices)
        h = rng.randrange(cfg.room_height[0], cfg.room_height[1] + 1, 8)
        room_floor[r.id] = f
        room_sec[r.id] = new_sector(SectorDef("room", f, f + h, _light(rng, cfg), rng.choice(pal["floors"]),
                                              rng.choice(pal["ceilings"]), rng.choice(pal["walls"])))
        for x, y in r.cells():
            labels[y, x] = room_sec[r.id]
    lock_of = {lk.corridor: lk for lk in prog.locks}
    corr_secs: dict[int, list[int]] = {}
    door_sector: dict[int, int] = {}
    corr_wall = rng.choice(pal["walls"])
    for c in layout.corridors:
        f = max(room_floor[c.a], room_floor[c.b])
        ch = f + rng.choice(cfg.corridor_height)
        mk = lambda: SectorDef("corridor", f, ch, _light(rng, cfg) - 16, rng.choice(pal["floors"]), rng.choice(pal["ceilings"]), corr_wall)
        has_door = c.id in prog.doors or c.id in lock_of
        if has_door:
            sa, sb = new_sector(mk()), new_sector(mk())
            for x, y in c.cells_a:
                labels[y, x] = sa
            for x, y in c.cells_b:
                labels[y, x] = sb
            if c.id in lock_of:
                lk = lock_of[c.id]
                sd = SectorDef("door", f, f, _light(rng, cfg), "FLAT20", "FLAT20", LOCKED_DOOR_TEXTURES[lk.color],
                               track=LOCKED_TRACK_TEXTURES[lk.color], door_special=DOOR_LOCKED_RAISE,
                               door_args=(0, DOOR_SPEED, 0, lk.lock), monsteruse=False)
            elif prog.doors[c.id]:
                sd = SectorDef("door", f, f, _light(rng, cfg), "FLAT20", "FLAT20", rng.choice(DOOR_TEXTURES),
                               door_special=DOOR_RAISE, door_args=(0, DOOR_SPEED, DOOR_DELAY))
            else:
                sd = SectorDef("door", f, f, _light(rng, cfg), "FLAT20", "FLAT20", rng.choice(DOOR_TEXTURES),
                               door_special=DOOR_OPEN, door_args=(0, DOOR_SPEED))
            ds = new_sector(sd)
            for x, y in c.slab:
                labels[y, x] = ds
            door_sector[c.id] = ds
            corr_secs[c.id] = [sa, ds, sb]
        else:
            s = new_sector(mk())
            for x, y in c.cells:
                labels[y, x] = s
            corr_secs[c.id] = [s]

    # ---- exit -------------------------------------------------------------------------------------------
    exit_room = layout.rooms[prog.exit_room]
    occupied: set[tuple[int, int]] = set()
    switch_edges: set[tuple[str, int, int]] = set()
    exit_info: dict = {"room": exit_room.id}
    exit_vertices: list[tuple[int, int]] = []
    kinds = ["switch", "pad"] if rng.random() < cfg.exit_switch_prob else ["pad", "switch"]
    for kind in kinds:
        if kind == "pad":
            spots = [(x, y) for x, y in exit_room.cells()
                     if all(exit_room.has(x + dx, y + dy) for dx in range(-1, 3) for dy in range(-1, 3))]
            if not spots:
                continue
            px, py = rng.choice(spots)
            pad = new_sector(SectorDef("exit", room_floor[exit_room.id], sectors[room_sec[exit_room.id]].ceil, 255,
                                       EXIT_FLOOR, sectors[room_sec[exit_room.id]].tex_ceil, sectors[room_sec[exit_room.id]].wall))
            for dx in (0, 1):
                for dy in (0, 1):
                    labels[py + dy, px + dx] = pad
                    occupied.add((px + dx, py + dy))
            exit_vertices = [(px + 1, py + 1)]
            exit_info.update(kind="pad", sector=pad, x=(px + 1) * CELL, y=(py + 1) * CELL)
            break
        spots = _switch_spots(exit_room, labels)
        if not spots:
            continue
        (cx, cy), (dx, dy), (tx, ty) = rng.choice(spots)
        cells = [(cx, cy), (cx + tx, cy + ty)]
        for x, y in cells:
            if dy:  # wall across a horizontal grid line
                switch_edges.add(("h", y + (1 if dy > 0 else 0), x))
            else:
                switch_edges.add(("v", x + (1 if dx > 0 else 0), y))
        # Centre of the 64-unit switch segment (vertex coords), and the vertex one cell (32 units) in front of
        # it: a player centred there is within USE range (64) of the switch.
        mx = (cx + (1 if dx > 0 else 0)) if dx else cx + 1
        my = (cy + (1 if dy > 0 else 0)) if dy else cy + 1
        inner = (mx - dx, my - dy)
        exit_vertices = [inner, (inner[0] - tx, inner[1] - ty), (inner[0] + tx, inner[1] + ty)]
        exit_info.update(kind="switch", x=mx * CELL, y=my * CELL, facing=[-dx, -dy])
        occupied.update(cells)
        break
    if "kind" not in exit_info:
        raise LayoutError("no room for an exit")

    # ---- player start -----------------------------------------------------------------------------------
    start_room = layout.rooms[prog.start_room]
    verts = [(x, y) for x, y in start_room.cells()
             if all(start_room.has(x + dx, y + dy) and (x + dx, y + dy) not in occupied for dx in (-1, 0) for dy in (-1, 0))]
    roomy = [v for v in verts if all(start_room.has(v[0] + dx, v[1] + dy) for dx in range(-2, 2) for dy in range(-2, 2))]
    pool = roomy or verts
    if not pool:
        raise LayoutError("no spawn spot")
    if prog.start_room == prog.exit_room:  # single-room map: spawn far from the exit
        ex, ey = exit_info["x"] / CELL, exit_info["y"] / CELL
        dmax = max(abs(v[0] - ex) + abs(v[1] - ey) for v in pool)
        pool = [v for v in pool if abs(v[0] - ex) + abs(v[1] - ey) >= 0.6 * dmax]
    sv = rng.choice(pool)
    jitter = (rng.randint(-8, 8), rng.randint(-8, 8)) if sv in roomy else (0, 0)
    start = {"room": start_room.id, "x": sv[0] * CELL + jitter[0], "y": sv[1] * CELL + jitter[1], "angle": rng.randrange(360)}
    for dx in (-2, -1, 0, 1):
        for dy in (-2, -1, 0, 1):
            occupied.add((sv[0] + dx, sv[1] + dy))
    things: list[Thing] = [Thing(start["x"], start["y"], start["angle"], T.PLAYER1_START)]

    def spot(room_id: int, min_spawn: int = 0, spacing: int = 1) -> tuple[int, int] | None:
        room = layout.rooms[room_id]
        cands = [(x, y) for x, y in room.cells()
                 if all(room.has(x + dx, y + dy) for dx in (-1, 0, 1) for dy in (-1, 0, 1))
                 and all((x + dx, y + dy) not in occupied for dx in range(-spacing + 1, spacing) for dy in range(-spacing + 1, spacing))
                 and max(abs(x - sv[0]), abs(y - sv[1])) >= min_spawn]
        return rng.choice(cands) if cands else None

    def centre(c: tuple[int, int]) -> tuple[float, float]:
        return (c[0] + 0.5) * CELL, (c[1] + 0.5) * CELL

    keys_meta = []
    for lk in prog.locks:
        c = spot(lk.key_room, min_spawn=3)
        if c is None:
            raise LayoutError("no key spot")
        occupied.add(c)
        card, card_cls, skull, skull_cls, lockno = T.KEYS[lk.color]
        use_skull = rng.random() < cfg.skull_prob
        x, y = centre(c)
        things.append(Thing(x, y, 0, skull if use_skull else card))
        keys_meta.append({"color": lk.color, "ednum": skull if use_skull else card, "class": skull_cls if use_skull else card_cls,
                          "room": lk.key_room, "x": x, "y": y, "cell": list(c), "lock": lockno, "in_branch": lk.key_in_branch,
                          "door_sector": door_sector[lk.corridor]})

    # ---- monsters -----------------------------------------------------------------------------------------
    monsters_meta = []
    n_mon = rng.randint(*cfg.monsters)
    roster = [m for lvl in T.ROSTER[: cfg.roster_level + 1] for m in lvl]
    weights = [len(T.ROSTER) - i for i, lvl in enumerate(T.ROSTER[: cfg.roster_level + 1]) for _ in lvl]
    # Monsters avoid a small start room; a big start room (arena) is fine, >= 6 cells from the spawn.
    mon_rooms = [r for r in layout.rooms if r.id != prog.start_room or r.area >= 100] or list(layout.rooms)
    for _ in range(n_mon):
        room = rng.choices(mon_rooms, weights=[r.area * (3 if r.arena else 1) for r in mon_rooms], k=1)[0]
        c = spot(room.id, min_spawn=6, spacing=2)
        if c is None:
            continue
        kind = rng.choices(roster, weights=weights, k=1)[0]
        ednum, cls, _radius = T.MONSTERS[kind]
        occupied.add(c)
        x, y = centre(c)
        amb = rng.random() < cfg.ambush_prob
        things.append(Thing(x, y, rng.randrange(0, 360, 45), ednum, ambush=amb))
        monsters_meta.append({"kind": kind, "class": cls, "ednum": ednum, "room": room.id, "x": x, "y": y, "ambush": amb})

    if len(monsters_meta) < cfg.monsters[0]:
        raise LayoutError(f"placed {len(monsters_meta)} monsters < {cfg.monsters[0]}")

    # ---- items ----------------------------------------------------------------------------------------------
    items_meta = []
    weapon_names = sorted(T.WEAPONS)
    chosen_weapons = [rng.choice(weapon_names) for _ in range(rng.randint(*cfg.weapons))]
    ammo_pref = [a for w in chosen_weapons for a in T.WEAPON_AMMO[w]] or ["clip", "shell"]
    plan_items = [("weapon", w) for w in chosen_weapons]
    plan_items += [("health", rng.choice(sorted(T.HEALTH))) for _ in range(rng.randint(*cfg.health_items))]
    plan_items += [("armor", rng.choice(sorted(T.ARMOR))) for _ in range(rng.randint(*cfg.armor_items))]
    plan_items += [("ammo", rng.choice(ammo_pref if rng.random() < 0.7 else sorted(T.AMMO))) for _ in range(rng.randint(*cfg.ammo_items))]
    tables = {"weapon": T.WEAPONS, "health": T.HEALTH, "armor": T.ARMOR, "ammo": T.AMMO}
    for cat, name in plan_items:
        order = list(layout.rooms)
        rng.shuffle(order)
        room, c = None, None
        for room in order:
            c = spot(room.id, min_spawn=2)
            if c is not None:
                break
        if c is None:
            if cat == "weapon":
                raise LayoutError("no spot for a weapon")
            continue
        occupied.add(c)
        ednum, cls = tables[cat][name]
        x, y = centre(c)
        things.append(Thing(x, y, 0, ednum))
        items_meta.append({"category": cat, "kind": name, "class": cls, "ednum": ednum, "room": room.id, "x": x, "y": y})

    if len(sectors) > MAX_SECTORS:
        raise LayoutError(f"{len(sectors)} sectors > cap {MAX_SECTORS}")
    for i, s in enumerate(sectors):
        if s.kind != "door" and s.ceil - s.floor < MIN_HEADROOM:
            raise LayoutError(f"sector {i} headroom {s.ceil - s.floor}")

    # ---- solvability on the emitted cell grid ---------------------------------------------------------------
    floor = np.array([s.floor for s in sectors], np.int64)
    ceil_open = np.array([s.ceil for s in sectors], np.int64)
    for c in layout.corridors:
        if c.id in door_sector:
            ds = door_sector[c.id]
            ceil_open[ds] = min(sectors[corr_secs[c.id][0]].ceil, sectors[corr_secs[c.id][2]].ceil) - 4
    world = CellWorld(labels, floor, ceil_open, {door_sector[lk.corridor]: lk.color for lk in prog.locks},
                      (sv[0], sv[1]), [(k["color"], tuple(k["cell"])) for k in keys_meta], exit_vertices)
    solv = check_solvable(world)
    if not solv["solvable"]:
        raise LayoutError(f"not solvable: {solv}")

    geo = MapGeometry(labels, sectors, things, switch_edges)
    textmap, stats = build_textmap(geo)
    wad = wad_bytes(map_name, textmap)
    on_path = set(prog.path_corridors)
    n_locks = len(prog.locks)
    tier = cfg.tier
    if cfg.stage in ("keys", "full"):
        tier = 4 if (n_locks >= 2 or (cfg.stage == "full" and len(monsters_meta) >= 6)) else 3
    meta = {
        "generator": GENERATOR, "generator_version": GENERATOR_VERSION, "stage": cfg.stage, "family": cfg.family,
        "seed": seed, "attempt": attempt, "map": map_name.upper(), "config_key": cfg_key,
        "source_group": source_group(cfg.stage, cfg.family) + (f"-{cfg_key}" if cfg_key else ""),
        "config": _jsonable(asdict(cfg)), "tier": tier, "episode_ticks": cfg.episode_ticks,
        "episode_ticks_range": list(cfg.episode_ticks_range), "cell_size": CELL, "grid": [layout.width, layout.height],
        **stats, "room_count": len(layout.rooms), "corridor_count": len(layout.corridors),
        "rooms": [{"id": r.id, "sector": room_sec[r.id], "bbox_cells": [r.x0, r.y0, r.x1, r.y1], "shape": r.shape,
                   "arena": r.arena, "floor": sectors[room_sec[r.id]].floor, "ceiling": sectors[room_sec[r.id]].ceil,
                   "area_cells": r.area} for r in layout.rooms],
        "corridors": [{"id": c.id, "a": c.a, "b": c.b, "width": c.width, "sectors": corr_secs[c.id], "tree": c.tree,
                       "on_path": c.id in on_path,
                       "door": "locked" if c.id in lock_of else ("normal" if c.id in prog.doors else "none"),
                       "door_sector": door_sector.get(c.id, -1)} for c in layout.corridors],
        "start": start, "exit": exit_info, "path_rooms": prog.path_rooms,
        "doors": [{"sector": door_sector[c], "corridor": c, "lock": lock_of[c].lock if c in lock_of else 0,
                   "closes": bool(prog.doors.get(c, False)), "on_path": c in on_path,
                   "x": float(np.mean([p[0] for p in corr_by_id[c].slab]) + 0.5) * CELL,
                   "y": float(np.mean([p[1] for p in corr_by_id[c].slab]) + 0.5) * CELL}
                  for c in sorted(door_sector)],
        "locked_doors": [{"sector": door_sector[lk.corridor], "corridor": lk.corridor, "color": lk.color, "lock": lk.lock,
                          "key_room": lk.key_room} for lk in prog.locks],
        "keys": keys_meta, "monsters": monsters_meta, "items": items_meta,
        "num_monsters": len(monsters_meta), "num_keys": len(keys_meta), "num_locked_doors": n_locks,
        "num_doors": sum(1 for c in door_sector if c not in lock_of),
        "solvable": solv["solvable"], "solvability": solv,
        "wad_sha256": hashlib.sha256(wad).hexdigest(),
    }
    return GeneratedMap(cfg.stage, cfg.family, seed, map_name.upper(), wad, meta, layout, prog, geo)


def _light(rng: random.Random, cfg: StageConfig) -> int:
    return rng.randrange(cfg.light[0], cfg.light[1] + 1, 16)


def _switch_spots(room, labels) -> list:
    """(cell, outward dir, along dir): two adjacent room cells on a straight wall facing void, 2 cells deep."""
    out = []
    H, W = labels.shape
    for x, y in room.cells():
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            tx, ty = (0, 1) if dx else (1, 0)
            a, b = (x, y), (x + tx, y + ty)
            if not room.has(*b):
                continue
            if any(labels[c[1] + dy, c[0] + dx] != VOID for c in (a, b)):
                continue
            # standing room in front of the switch (2 cells deep, 2 wide) so the player can reach it
            if not all(room.has(c[0] - k * dx, c[1] - k * dy) for c in (a, b) for k in (1, 2)):
                continue
            out.append((a, (dx, dy), (tx, ty)))
    return out


def _jsonable(d):
    if isinstance(d, dict):
        return {k: _jsonable(v) for k, v in d.items()}
    if isinstance(d, (list, tuple)):
        return [_jsonable(v) for v in d]
    return d
