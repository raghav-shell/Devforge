"""Room-graph layout on a cell grid (generator-internal, training-only map construction).

The map is a grid of CELL x CELL map-unit cells. Rooms sit inside the slots of a jittered macro grid; corridors
connect rooms in neighbouring slots (straight, or with one jog at the slot boundary). Every cell is owned by
at most one room or corridor, and a corridor keeps a one-cell clearance from everything except its two
endpoint rooms, so regions never merge by accident (`progression.check_solvable` re-verifies this on the
final cell grid).

Nothing here is available to an agent: the room graph only drives where geometry is emitted.
"""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import StageConfig

CELL = 32  # map units per cell
MARGIN = 3  # min cells between a room and its slot edge -> >= 6 cells between rooms in adjacent slots
BORDER = 2  # void cells around the whole map
VOID = -1
CORR_BASE = 100_000  # owner codes: room r -> r, corridor c -> CORR_BASE + c


class LayoutError(RuntimeError):
    """This attempt cannot satisfy the constraints; the caller retries with the next attempt index."""


@dataclass
class Room:
    id: int
    x0: int
    y0: int
    x1: int  # exclusive
    y1: int  # exclusive
    shape: str
    mask: np.ndarray  # (h, w) bool, local coords [y - y0, x - x0]
    arena: bool = False

    def has(self, x: int, y: int) -> bool:
        return self.x0 <= x < self.x1 and self.y0 <= y < self.y1 and bool(self.mask[y - self.y0, x - self.x0])

    def cells(self) -> list[tuple[int, int]]:
        ys, xs = np.nonzero(self.mask)
        return [(int(x) + self.x0, int(y) + self.y0) for y, x in zip(ys, xs)]

    def extent(self, axis: str, v: int) -> tuple[int, int] | None:
        """(min_u, max_u) of room cells on line v; axis 'x' -> line is row y=v, 'y' -> column x=v."""
        if axis == "x":
            if not self.y0 <= v < self.y1:
                return None
            row = np.flatnonzero(self.mask[v - self.y0])
            return (int(row[0]) + self.x0, int(row[-1]) + self.x0) if len(row) else None
        if not self.x0 <= v < self.x1:
            return None
        col = np.flatnonzero(self.mask[:, v - self.x0])
        return (int(col[0]) + self.y0, int(col[-1]) + self.y0) if len(col) else None

    def span(self, axis: str) -> tuple[int, int, int, int]:
        """(u0, u1, v0, v1) bbox in corridor coordinates for `axis`."""
        return (self.x0, self.x1, self.y0, self.y1) if axis == "x" else (self.y0, self.y1, self.x0, self.x1)

    @property
    def area(self) -> int:
        return int(self.mask.sum())


@dataclass
class Corridor:
    id: int
    a: int  # room id on the low-u side
    b: int  # room id on the high-u side
    axis: str
    width: int
    cells: list[tuple[int, int]]
    # Door slot (a slab across the corridor) splitting the corridor into side a / side b; None if no slot.
    slab: list[tuple[int, int]] | None = None
    cells_a: list[tuple[int, int]] = field(default_factory=list)
    cells_b: list[tuple[int, int]] = field(default_factory=list)
    tree: bool = True  # spanning-tree edge (False = extra loop edge)


@dataclass
class Layout:
    width: int
    height: int
    rooms: list[Room]
    corridors: list[Corridor]
    owner: np.ndarray  # (H, W) int64 owner codes, VOID = -1

    def neighbours(self) -> dict[int, list[tuple[int, int]]]:
        adj: dict[int, list[tuple[int, int]]] = {r.id: [] for r in self.rooms}
        for c in self.corridors:
            adj[c.a].append((c.b, c.id))
            adj[c.b].append((c.a, c.id))
        return adj


# ---------------------------------------------------------------------------------------- shapes
def _connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if not len(cells):
        return False
    seen = np.zeros_like(mask)
    q = deque([tuple(cells[0])])
    seen[tuple(cells[0])] = True
    n = 1
    h, w = mask.shape
    while q:
        y, x = q.popleft()
        for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                seen[ny, nx] = True
                n += 1
                q.append((ny, nx))
    return n == int(mask.sum())


def make_mask(shape: str, w: int, h: int, rng: random.Random) -> tuple[str, np.ndarray]:
    """Room footprint. Every passage inside a room stays >= 2 cells (64 units) wide."""
    m = np.ones((h, w), bool)
    if shape == "l" and w >= 6 and h >= 6:
        cw, ch = rng.randint(max(2, w // 3), w - 3), rng.randint(max(2, h // 3), h - 3)
        corner = rng.randrange(4)
        ys = slice(0, ch) if corner in (0, 1) else slice(h - ch, h)
        xs = slice(0, cw) if corner in (0, 2) else slice(w - cw, w)
        m[ys, xs] = False
    elif shape == "cross" and w >= 7 and h >= 7:
        aw, ah = rng.randint(3, w - 4), rng.randint(3, h - 4)
        cx, cy = rng.randint(2, w - aw - 2), rng.randint(2, h - ah - 2)
        m[:] = False
        m[:, cx : cx + aw] = True
        m[cy : cy + ah, :] = True
    elif shape == "pillars" and w >= 7 and h >= 7:
        p = rng.choice((1, 2))
        step = p + rng.choice((2, 3))
        for y in range(2, h - 2 - p + 1, step):
            for x in range(2, w - 2 - p + 1, step):
                m[y : y + p, x : x + p] = False
    elif shape == "octagon" and w >= 6 and h >= 6:
        c = rng.randint(1, max(1, min(w, h) // 3))
        yy, xx = np.mgrid[0:h, 0:w]
        for cy, cx in ((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)):
            m &= (np.abs(yy - cy) + np.abs(xx - cx)) >= c
    elif shape == "ring" and w >= 9 and h >= 9:
        t = rng.randint(3, (min(w, h) - 3) // 2)
        m[t : h - t, t : w - t] = False
    else:
        shape = "rect"
    if not _connected(m):
        return "rect", np.ones((h, w), bool)
    return shape, m


def _pick(rng: random.Random, weighted: tuple[tuple[str, float], ...]) -> str:
    names = [n for n, _ in weighted]
    return rng.choices(names, weights=[wt for _, wt in weighted], k=1)[0]


# ---------------------------------------------------------------------------------------- layout
def _slot_set(rng: random.Random, n: int, cols: int, rows: int) -> list[tuple[int, int]]:
    """A random 4-connected set of n macro slots (so a spanning tree over slot adjacency exists)."""
    start = (rng.randrange(cols), rng.randrange(rows))
    chosen = [start]
    taken = {start}
    while len(chosen) < n:
        frontier = []
        for cx, cy in chosen:
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nb = (cx + dx, cy + dy)
                if 0 <= nb[0] < cols and 0 <= nb[1] < rows and nb not in taken and nb not in frontier:
                    frontier.append(nb)
        if not frontier:
            raise LayoutError("slot grid too small")
        nb = rng.choice(frontier)
        chosen.append(nb)
        taken.add(nb)
    return chosen


def build_layout(cfg: StageConfig, rng: random.Random) -> Layout:
    n = rng.randint(*cfg.n_rooms)
    total = n + rng.randint(0, max(1, n // 2))
    cols = max(1, int(round(total**0.5 + rng.uniform(-0.5, 0.8))))
    rows = max(1, -(-total // cols))
    slots = _slot_set(rng, n, cols, rows)
    arena_slot = rng.choice(slots) if cfg.arena_scale > 1.0 else None
    col_w = [rng.randint(*cfg.slot_size) for _ in range(cols)]
    row_h = [rng.randint(*cfg.slot_size) for _ in range(rows)]
    if arena_slot is not None:
        col_w[arena_slot[0]] = int(col_w[arena_slot[0]] * cfg.arena_scale)
        row_h[arena_slot[1]] = int(row_h[arena_slot[1]] * cfg.arena_scale)
    col_x = [BORDER + sum(col_w[:i]) for i in range(cols)]
    row_y = [BORDER + sum(row_h[:j]) for j in range(rows)]
    W, H = BORDER * 2 + sum(col_w), BORDER * 2 + sum(row_h)
    owner = np.full((H, W), VOID, np.int64)

    rooms: list[Room] = []
    slot_room: dict[tuple[int, int], int] = {}
    for sx, sy in slots:
        aw, ah = col_w[sx] - 2 * MARGIN, row_h[sy] - 2 * MARGIN
        lo, hi = cfg.room_size
        is_arena = (sx, sy) == arena_slot
        if is_arena:
            w = rng.randint(max(min(lo, aw), int(aw * 0.7)), aw)
            h = rng.randint(max(min(lo, ah), int(ah * 0.7)), ah)
        else:
            w = rng.randint(min(lo, aw), max(min(lo, aw), min(hi, aw)))
            h = rng.randint(min(lo, ah), max(min(lo, ah), min(hi, ah)))
        x0 = col_x[sx] + MARGIN + rng.randint(0, aw - w)
        y0 = row_y[sy] + MARGIN + rng.randint(0, ah - h)
        shape, mask = make_mask(_pick(rng, cfg.shapes), w, h, rng)
        room = Room(len(rooms), x0, y0, x0 + w, y0 + h, shape, mask, arena=is_arena)
        rooms.append(room)
        slot_room[(sx, sy)] = room.id
        for x, y in room.cells():
            owner[y, x] = room.id

    # Candidate edges: rooms in 4-adjacent slots. Order is shuffled, then Kruskal + optional loops.
    cand = []
    for (sx, sy), rid in slot_room.items():
        for dx, dy, axis in ((1, 0, "x"), (0, 1, "y")):
            other = slot_room.get((sx + dx, sy + dy))
            if other is not None:
                cand.append((rid, other, axis))
    cand.sort()
    rng.shuffle(cand)
    parent = list(range(len(rooms)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    corridors: list[Corridor] = []
    rest = []
    for a, b, axis in cand:
        ra, rb = find(a), find(b)
        if ra == rb:
            rest.append((a, b, axis))
            continue
        c = _carve(rooms[a], rooms[b], axis, rng.choice(cfg.corridor_width), owner, len(corridors), cfg, rng)
        if c is not None:
            parent[ra] = rb
            corridors.append(c)
    if len({find(r.id) for r in rooms}) != 1:
        raise LayoutError("rooms not connected")
    for a, b, axis in rest:
        if rng.random() < cfg.loop_prob:
            c = _carve(rooms[a], rooms[b], axis, rng.choice(cfg.corridor_width), owner, len(corridors), cfg, rng)
            if c is not None:
                c.tree = False
                corridors.append(c)
    return Layout(W, H, rooms, corridors, owner)


def _bands(room: Room, axis: str, width: int) -> list[int]:
    _, _, v0, v1 = room.span(axis)
    ok = [room.extent(axis, v) is not None for v in range(v0, v1)]
    out = [v0 + i for i in range(len(ok) - width + 1) if all(ok[i : i + width])]
    inner = [v for v in out if v > v0 and v + width < v1]  # avoid corner bands: fewer clearance clashes
    return inner or out


def _carve(A: Room, B: Room, axis: str, width: int, owner: np.ndarray, cid: int, cfg: StageConfig,
           rng: random.Random) -> Corridor | None:
    """Carve a corridor from A (low u) to B (high u). Returns None (nothing written) if it does not fit."""
    if A.span(axis)[0] > B.span(axis)[0]:
        A, B = B, A
    xy = (lambda u, v: (u, v)) if axis == "x" else (lambda u, v: (v, u))
    ba, bb = _bands(A, axis, width), _bands(B, axis, width)
    if not ba or not bb:
        return None
    common = sorted(set(ba) & set(bb))
    if common and rng.random() < cfg.straight_prob:
        va = vb = rng.choice(common)
    else:
        va, vb = rng.choice(ba), rng.choice(bb)
    a_u1, b_u0 = A.span(axis)[1], B.span(axis)[0]
    lo, hi = a_u1 + 1, b_u0 - width - 1
    if lo > hi:
        return None
    uj = rng.randint(lo, hi)
    cells: dict[tuple[int, int], None] = {}
    for v in range(va, va + width):
        for u in range(A.extent(axis, v)[1] + 1, uj + width):
            cells[xy(u, v)] = None
    for u in range(uj, uj + width):
        for v in range(min(va, vb), max(va, vb) + width):
            cells[xy(u, v)] = None
    for v in range(vb, vb + width):
        for u in range(uj, B.extent(axis, v)[0]):
            cells[xy(u, v)] = None
    H, W = owner.shape
    allowed = {VOID, A.id, B.id}
    for x, y in cells:
        if not (BORDER <= x < W - BORDER and BORDER <= y < H - BORDER) or owner[y, x] != VOID:
            return None
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if (x + dx, y + dy) not in cells and int(owner[y + dy, x + dx]) not in allowed:
                    return None
    corr = Corridor(cid, A.id, B.id, axis, width, list(cells))
    _door_slot(corr, A, B, va, vb, axis, xy, owner, rng)
    for x, y in cells:
        owner[y, x] = CORR_BASE + cid
    return corr


def _door_slot(corr: Corridor, A: Room, B: Room, va: int, vb: int, axis: str, xy, owner: np.ndarray,
               rng: random.Random) -> None:
    """Find a slab across the corridor that splits it into an A side and a B side (door candidate)."""
    cells = set(corr.cells)
    w = corr.width
    cands = []
    for v0 in sorted({va, vb}):
        band = range(v0, v0 + w)
        us = sorted({(p[0] if axis == "x" else p[1]) for p in cells})
        for u in us:
            if not all(xy(u, v) in cells for v in band):
                continue
            slab = [xy(u, v) for v in band]
            lat = [xy(u, v0 - 1), xy(u, v0 + w)]
            if any(p in cells or owner[p[1], p[0]] != VOID for p in lat):
                continue
            if not all(xy(u - 1, v) in cells and xy(u + 1, v) in cells for v in band):
                continue
            cands.append(slab)
    rng.shuffle(cands)
    for slab in cands:
        rest = cells - set(slab)
        comps = _components(rest)
        if len(comps) != 2:
            continue
        side = []
        for comp in comps:
            touch_a = any(A.has(x + dx, y + dy) for x, y in comp for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            touch_b = any(B.has(x + dx, y + dy) for x, y in comp for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)))
            side.append((touch_a, touch_b))
        if side == [(True, False), (False, True)]:
            ca, cb = comps
        elif side == [(False, True), (True, False)]:
            cb, ca = comps
        else:
            continue
        order = {p: i for i, p in enumerate(corr.cells)}
        corr.slab = slab
        corr.cells_a = sorted(ca, key=order.__getitem__)
        corr.cells_b = sorted(cb, key=order.__getitem__)
        return


def _components(cells: set[tuple[int, int]]) -> list[list[tuple[int, int]]]:
    left = set(cells)
    comps = []
    for start in sorted(cells):
        if start not in left:
            continue
        left.discard(start)
        comp, q = [start], deque([start])
        while q:
            x, y = q.popleft()
            for nb in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if nb in left:
                    left.discard(nb)
                    comp.append(nb)
                    q.append(nb)
        comps.append(comp)
    return comps
