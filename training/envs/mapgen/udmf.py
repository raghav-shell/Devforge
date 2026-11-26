"""Cell grid -> UDMF TEXTMAP (namespace "zdoom", node-less like maps/two_rooms.wad) -> WAD bytes.

Linedefs are emitted on every cell edge whose two cells belong to different sectors (or sector/void), and
collinear edges are merged while the (side, side, tag) key stays the same. A run therefore always ends at a
vertex where any other line touches it, so there are no T-junctions or overlaps, and every sector boundary
is closed. GZDoom builds the BSP nodes itself on load.

Orientation (Doom: the front side is on the right of v1->v2):
- one-sided: front = the non-void cell;
- door lines: front = room/corridor, back = door sector (Hexen specials with arg0 = 0 act on the back sector,
  which is also how env/doom_env/mapinfo.py resolves them);
- exit-pad lines: front = room, back = pad.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import numpy as np

from .config import DOOR_TRACK, EXIT_SWITCH
from .layout import CELL

EXIT_NORMAL = 243  # Exit_Normal
DOOR_RAISE, DOOR_OPEN, DOOR_LOCKED_RAISE = 11, 12, 13
DOOR_SPEED = 16  # 2 map units / tic, the Doom default
DOOR_DELAY = 150  # tics a Door_Raise stays open


@dataclass
class SectorDef:
    kind: str  # room | corridor | door | exit
    floor: int
    ceil: int
    light: int
    tex_floor: str
    tex_ceil: str
    wall: str  # wall texture for lines facing this sector (door face texture for doors)
    track: str = DOOR_TRACK  # door sectors: one-sided track texture
    door_special: int = 0
    door_args: tuple[int, ...] = ()
    monsteruse: bool = True


@dataclass
class Thing:
    x: float
    y: float
    angle: int
    type: int
    ambush: bool = False


@dataclass
class MapGeometry:
    labels: np.ndarray  # (H, W) sector id per cell, -1 void
    sectors: list[SectorDef]
    things: list[Thing]
    switch_edges: set[tuple[str, int, int]] = field(default_factory=set)  # ('h', y, x) / ('v', x, y)


def _f(v: float) -> str:
    return f"{float(v):.1f}"


def _runs(line_cells):
    """Merge consecutive positions with identical keys. line_cells: list of key or None. Yields (start, end, key)."""
    start, cur = 0, None
    for i, key in enumerate(list(line_cells) + [None]):
        if key != cur:
            if cur is not None:
                yield start, i, cur
            start, cur = i, key
    return


def build_textmap(geo: MapGeometry) -> tuple[str, dict]:
    lab = geo.labels
    H, W = lab.shape
    secs = geo.sectors
    verts: dict[tuple[int, int], int] = {}
    vlines: list[str] = []
    sides: list[str] = []
    lines: list[str] = []

    def vid(gx: int, gy: int) -> int:
        k = (gx, gy)
        if k not in verts:
            verts[k] = len(verts)
            vlines.append(f"vertex {{ x = {_f(gx * CELL)}; y = {_f(gy * CELL)}; }}")
        return verts[k]

    def side(sec: int, top: str, mid: str, bot: str) -> int:
        sides.append(f'sidedef {{ sector = {sec}; texturetop = "{top}"; texturemiddle = "{mid}"; texturebottom = "{bot}"; }}')
        return len(sides) - 1

    def emit(p1: tuple[int, int], p2: tuple[int, int], s_a: int, s_b: int, tag) -> None:
        """s_a is on the right of p1->p2 when walking the line 'forwards' (see callers)."""
        # choose front/back
        if s_a < 0 or s_b < 0:
            front, back = (s_b, -1) if s_a < 0 else (s_a, -1)
        else:
            ka, kb = secs[s_a].kind, secs[s_b].kind
            if kb in ("door", "exit") and ka not in ("door", "exit"):
                front, back = s_a, s_b
            elif ka in ("door", "exit") and kb not in ("door", "exit"):
                front, back = s_b, s_a
            else:
                front, back = min(s_a, s_b), max(s_a, s_b)
        v1, v2 = (p1, p2) if front == s_a else (p2, p1)
        props = [f"v1 = {vid(*v1)}", f"v2 = {vid(*v2)}"]
        fs = secs[front]
        if back < 0:
            mid = fs.track if fs.kind == "door" else (EXIT_SWITCH if tag == "exit_switch" else fs.wall)
            props.append(f"sidefront = {side(front, '-', mid, '-')}")
            props.append("blocking = true")
            if fs.kind == "door":
                props.append("dontpegbottom = true")
            if tag == "exit_switch":
                props += [f"special = {EXIT_NORMAL}", "playeruse = true"]
        else:
            bs = secs[back]
            top = bs.wall if bs.kind == "door" else fs.wall
            props.append(f"sidefront = {side(front, top, '-', fs.wall)}")
            props.append(f"sideback = {side(back, bs.wall, '-', bs.wall)}")
            props.append("twosided = true")
            if bs.kind == "door":
                props.append(f"special = {bs.door_special}")
                props += [f"arg{i} = {a}" for i, a in enumerate(bs.door_args)]
                props += ["playeruse = true", "repeatspecial = true", f"monsteruse = {'true' if bs.monsteruse else 'false'}"]
            elif bs.kind == "exit":
                props += [f"special = {EXIT_NORMAL}", "playercross = true"]
        lines.append("linedef { " + "; ".join(props) + "; }")

    # horizontal grid lines y = j: s_a = cell below (right of +x), s_b = cell above
    for j in range(H + 1):
        keys = []
        for i in range(W):
            below = int(lab[j - 1, i]) if j > 0 else -1
            above = int(lab[j, i]) if j < H else -1
            tag = "exit_switch" if ("h", j, i) in geo.switch_edges else None
            keys.append(None if below == above else (below, above, tag))
        for s, e, (below, above, tag) in _runs(keys):
            emit((s, j), (e, j), below, above, tag)
    # vertical grid lines x = i: walking +y, the right side is the cell at x >= i
    for i in range(W + 1):
        keys = []
        for j in range(H):
            left = int(lab[j, i - 1]) if i > 0 else -1
            right = int(lab[j, i]) if i < W else -1
            tag = "exit_switch" if ("v", i, j) in geo.switch_edges else None
            keys.append(None if left == right else (right, left, tag))
        for s, e, (right, left, tag) in _runs(keys):
            emit((i, s), (i, e), right, left, tag)

    out = ['namespace = "zdoom";']
    out += vlines
    for s in secs:
        out.append(f'sector {{ texturefloor = "{s.tex_floor}"; textureceiling = "{s.tex_ceil}"; heightfloor = {s.floor}; '
                   f"heightceiling = {s.ceil}; lightlevel = {s.light}; }}")
    out += sides
    out += lines
    for t in geo.things:
        extra = " ambush = true;" if t.ambush else ""
        out.append(f"thing {{ x = {_f(t.x)}; y = {_f(t.y)}; angle = {t.angle}; type = {t.type}; skill1 = true; skill2 = true; "
                   f"skill3 = true; skill4 = true; skill5 = true; single = true;{extra} }}")
    stats = {"num_vertices": len(verts), "num_linedefs": len(lines), "num_sidedefs": len(sides),
             "num_sectors": len(secs), "num_things": len(geo.things)}
    return "\n".join(out) + "\n", stats


def wad_bytes(map_name: str, textmap: str) -> bytes:
    """PWAD with MAP marker, TEXTMAP, ENDMAP (same layout as maps/two_rooms.wad). Deterministic."""
    lumps = [(map_name.upper(), b""), ("TEXTMAP", textmap.encode("ascii")), ("ENDMAP", b"")]
    body, directory, offset = b"", b"", 12
    for name, data in lumps:
        directory += struct.pack("<ii8s", offset, len(data), name.encode("ascii").ljust(8, b"\0"))
        body += data
        offset += len(data)
    return b"PWAD" + struct.pack("<ii", len(lumps), offset) + body + directory
