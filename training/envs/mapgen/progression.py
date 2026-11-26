"""Progression design (start, exit, doors, key -> locked-door chains) and the solvability check.

GENERATOR-INTERNAL, TRAINING-ONLY MAP CONSTRUCTION. The breadth-first searches below run on the generator's
own room graph and cell grid while a map is being *built*, to place locks on bridges of the start->exit path
and keys in side branches, and to certify the result is solvable. They are never available to an agent:
nothing in this package is imported by `submission/`, and the agent never sees the room graph. This is map
authoring, not pathfinding for play (docs/RULES.md).
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import StageConfig
from .layout import Layout, LayoutError
from .things import KEY_COLORS, KEYS


@dataclass
class Lock:
    corridor: int
    color: str
    lock: int  # ZDoom lock number
    key_room: int = -1
    key_in_branch: bool = False


@dataclass
class Progression:
    start_room: int
    exit_room: int
    path_rooms: list[int]
    path_corridors: list[int]
    locks: list[Lock] = field(default_factory=list)
    doors: dict[int, bool] = field(default_factory=dict)  # normal-door corridor -> closes after delay


def _bfs(adj, start: int, blocked: set[int] = frozenset()) -> dict[int, tuple[int, int]]:
    """room -> (parent room, via corridor); start maps to (-1, -1)."""
    seen = {start: (-1, -1)}
    q = deque([start])
    while q:
        r = q.popleft()
        for nb, cid in adj[r]:
            if cid in blocked or nb in seen:
                continue
            seen[nb] = (r, cid)
            q.append(nb)
    return seen


def plan(layout: Layout, cfg: StageConfig, rng: random.Random, require_branch: bool = False) -> Progression:
    """require_branch: reject (LayoutError) unless every key sits in a side branch off the start->exit path."""
    adj = layout.neighbours()
    rooms = [r.id for r in layout.rooms]
    if cfg.arena_scale > 1.0 and len(rooms) > 1 and rng.random() < 0.6:
        start = rng.choice([r.id for r in layout.rooms if not r.arena])
    else:
        start = rng.choice(rooms)
    tree = _bfs(adj, start)
    dist = {}
    for r in rooms:
        d, cur = 0, r
        while tree[cur][0] != -1:
            cur, d = tree[cur][0], d + 1
        dist[r] = d
    far = max(dist.values())
    exits = [r for r in rooms if dist[r] >= max(far - (1 if far >= 3 else 0), 0) and (r != start or far == 0)]
    exit_room = rng.choice(sorted(exits))
    path_rooms, path_corr = [exit_room], []
    while tree[path_rooms[-1]][0] != -1:
        p, cid = tree[path_rooms[-1]]
        path_corr.append(cid)
        path_rooms.append(p)
    path_rooms.reverse()
    path_corr.reverse()
    prog = Progression(start, exit_room, path_rooms, path_corr)

    # ---- locks: bridges on the path that have a door slot, keys in the region they gate ----------------
    k = rng.randint(*cfg.n_locks)
    if k:
        corr = {c.id: c for c in layout.corridors}
        bridges = [cid for cid in path_corr if corr[cid].slab is not None and exit_room not in _bfs(adj, start, {cid})]
        # Every chosen lock is a bridge on the path, so the region it gates (locks i..k closed) is just the
        # start side of bridge i. Greedy: earliest bridge whose newly opened region has a side-branch room.
        side = {b: set(_bfs(adj, start, {b})) for b in bridges}
        k = min(k, len(bridges), len(KEY_COLORS))
        colors = rng.sample(KEY_COLORS, k=k)
        chosen: list[int] = []
        prev_region: set[int] = set()
        pos = 0
        for i in range(k):
            remaining = k - i - 1
            window = bridges[pos : len(bridges) - remaining]
            if not window:
                break
            pick = next((b for b in window if any(r not in path_rooms for r in side[b] - prev_region)), window[0])
            chosen.append(pick)
            prev_region = side[pick]
            pos = bridges.index(pick) + 1
        if not chosen:
            raise LayoutError("no lockable bridge on the exit path")
        prev: set[int] = set()
        for i, cid in enumerate(chosen):
            region = set(_bfs(adj, start, set(chosen[i:])))
            new = sorted(region - prev)
            branch = [r for r in new if r not in path_rooms]
            if branch:
                # Prefer rooms deep in the branch (more backtracking).
                depth = {r: _depth_off_path(adj, r, set(path_rooms)) for r in branch}
                key_room = rng.choices(branch, weights=[1 + depth[r] for r in branch], k=1)[0]
                in_branch = True
            else:
                if require_branch:
                    raise LayoutError("key region has no side branch")
                pool = [r for r in new if r != start] or new
                key_room, in_branch = rng.choice(pool), False
            color = colors[i]
            prog.locks.append(Lock(cid, color, KEYS[color][4], key_room, in_branch))
            prev = region

    # ---- normal doors ---------------------------------------------------------------------------------
    locked = {lk.corridor for lk in prog.locks}
    slots = [c.id for c in layout.corridors if c.slab is not None and c.id not in locked]
    for cid in slots:
        if rng.random() < cfg.door_prob:
            prog.doors[cid] = rng.random() < cfg.door_close_prob
    need = cfg.min_path_doors - sum(1 for c in path_corr if c in prog.doors)
    free = [c for c in path_corr if c in slots and c not in prog.doors]
    rng.shuffle(free)
    for cid in free[: max(need, 0)]:
        prog.doors[cid] = rng.random() < cfg.door_close_prob
    if cfg.min_path_doors and sum(1 for c in path_corr if c in prog.doors) < cfg.min_path_doors:
        raise LayoutError("not enough door slots on the exit path")
    return prog


def _depth_off_path(adj, room: int, path: set[int]) -> int:
    seen = {room: 0}
    q = deque([room])
    while q:
        r = q.popleft()
        if r in path:
            return seen[r]
        for nb, _ in adj[r]:
            if nb not in seen:
                seen[nb] = seen[r] + 1
                q.append(nb)
    return 0


# ---------------------------------------------------------------------------- solvability (cell grid)
@dataclass
class CellWorld:
    """What the checker needs, all derived from the *emitted* sector grid (not the room graph)."""

    labels: np.ndarray  # (H, W) sector id, -1 void
    floor: np.ndarray  # (S,) floor height
    ceil_open: np.ndarray  # (S,) ceiling height once doors are open
    lock_color: dict[int, str]  # locked door sector -> colour
    start_vertex: tuple[int, int]
    keys: list[tuple[str, tuple[int, int]]]  # (colour, cell)
    exit_vertices: list[tuple[int, int]]  # reaching any of these triggers the exit


PLAYER_HEIGHT, MAX_STEP = 56, 24


def _reach(world: CellWorld, colors: set[str]) -> set[tuple[int, int]]:
    """Vertices the player's centre (32x32 box = the 4 cells around a vertex) can reach with `colors`."""
    lab = world.labels
    H, W = lab.shape
    open_ok = np.zeros(len(world.floor) + 1, bool)
    open_ok[:-1] = True
    for sec, col in world.lock_color.items():
        open_ok[sec] = col in colors
    safe = np.where(lab >= 0, lab, len(world.floor))
    passable = open_ok[safe] & (lab >= 0)
    fl = np.where(lab >= 0, world.floor[np.maximum(lab, 0)], 0)
    ce = np.where(lab >= 0, world.ceil_open[np.maximum(lab, 0)], 0)

    def ok(vx: int, vy: int) -> bool:
        if not (1 <= vx < W and 1 <= vy < H):
            return False
        if not passable[vy - 1 : vy + 1, vx - 1 : vx + 1].all():
            return False
        f, c = fl[vy - 1 : vy + 1, vx - 1 : vx + 1], ce[vy - 1 : vy + 1, vx - 1 : vx + 1]
        return int(f.max() - f.min()) <= MAX_STEP and int(c.min() - f.max()) >= PLAYER_HEIGHT

    s = world.start_vertex
    if not ok(*s):
        return set()
    seen = {s}
    q = deque([s])
    while q:
        x, y = q.popleft()
        for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if nb not in seen and ok(*nb):
                seen.add(nb)
                q.append(nb)
    return seen


def _collect(world: CellWorld, withheld: str | None = None) -> tuple[set[str], set[tuple[int, int]]]:
    held: set[str] = set()
    while True:
        reach = _reach(world, held)
        got = {col for col, (cx, cy) in world.keys if col != withheld and
               any(v in reach for v in ((cx, cy), (cx + 1, cy), (cx, cy + 1), (cx + 1, cy + 1)))}
        if got <= held:
            return held, reach
        held |= got


def check_solvable(world: CellWorld) -> dict:
    """Key-state fixpoint flood fill on the emitted cell grid.

    Returns {"exit_reachable", "keys_collected", "keys_necessary": {colour: bool}, "solvable"}. `solvable`
    needs the exit reachable with collectable keys AND every lock to actually gate the exit (no bypass).
    """
    held, reach = _collect(world)
    exit_ok = any(v in reach for v in world.exit_vertices)
    necessary = {}
    for col in sorted(world.lock_color.values()):
        _, r = _collect(world, withheld=col)
        necessary[col] = not any(v in r for v in world.exit_vertices)
    return {
        "exit_reachable": bool(exit_ok),
        "keys_collected": sorted(held),
        "keys_necessary": necessary,
        "solvable": bool(exit_ok and all(necessary.values())),
        "reachable_vertices": len(reach),
        "method": "vertex flood fill on the emitted cell grid with key fixpoint (generator-internal)",
    }
