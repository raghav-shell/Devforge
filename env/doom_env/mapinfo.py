"""Static map metadata parsed from WAD lumps.

ViZDoom's Line API only exposes geometry and `is_blocking`, so we parse the map's linedef
specials to find which sectors are doors and which of those are locked. Supported formats:
UDMF (TEXTMAP), Hexen binary and Doom binary (including Boom generalized doors).

Sector indices here equal ViZDoom `Sector.id` (both follow map order); `DoomEnv` double-checks
that by comparing sector counts.
"""

from __future__ import annotations

import re
import struct

import numpy as np
from dataclasses import dataclass, field
from pathlib import Path

from .wad import map_lumps, read_wad

# Lock numbers we report (anything > 0 means "needs a key"): 1-3 a colour (card or skull),
# 100 any key, 101 every key colour (ZDoom lock numbers).
LOCK_RED, LOCK_BLUE, LOCK_YELLOW, LOCK_ANY, LOCK_ALL = 1, 2, 3, 100, 101
# Boom generalized locked door key field (bits 6-8): any, R/B/Y card, R/B/Y skull, all keys.
BOOM_KEY_LOCK = (LOCK_ANY, LOCK_RED, LOCK_BLUE, LOCK_YELLOW, LOCK_RED, LOCK_BLUE, LOCK_YELLOW, LOCK_ALL)

# --- Doom-format line specials -------------------------------------------------------------
DOOM_MANUAL_DOORS = {1: 0, 31: 0, 117: 0, 118: 0, 26: LOCK_BLUE, 27: LOCK_YELLOW, 28: LOCK_RED, 32: LOCK_BLUE, 33: LOCK_RED, 34: LOCK_YELLOW}
# Tagged doors that open (raise); pure "close" specials (3, 42, 50, 75, 107, ...) are not listed.
DOOM_REMOTE_DOORS = {
    2: 0, 4: 0, 16: 0, 29: 0, 46: 0, 61: 0, 63: 0, 76: 0, 86: 0, 90: 0, 103: 0, 105: 0, 106: 0, 108: 0, 109: 0,
    111: 0, 112: 0, 114: 0, 115: 0,
    99: LOCK_BLUE, 133: LOCK_BLUE, 134: LOCK_RED, 135: LOCK_RED, 136: LOCK_YELLOW, 137: LOCK_YELLOW,
}
# --- Hexen / UDMF (ZDoom) action specials: special -> index of the lock argument (None = no lock)
HEXEN_DOORS = {11: None, 12: None, 13: 3, 14: 3, 202: 4}


@dataclass
class LineGeometry:
    """Static linedef geometry straight from the WAD (exact; ViZDoom's runtime sector line lists can
    be incomplete on large maps). Arrays of length N."""

    p1: np.ndarray  # (N, 2) float64
    p2: np.ndarray  # (N, 2)
    front: np.ndarray  # (N,) sector index of the front side
    back: np.ndarray  # (N,) sector index of the back side, -1 = one-sided
    impassable: np.ndarray  # (N,) bool, the "blocks everything" flag


@dataclass
class MapInfo:
    num_sectors: int
    # sector index -> lock number (0 = ordinary door, >0 = locked door)
    door_sectors: dict[int, int] = field(default_factory=dict)
    format: str = ""
    lines: LineGeometry | None = None

    def sector_lines(self) -> dict[int, np.ndarray]:
        """sector index -> (K, 4) [x1, y1, x2, y2] boundary segments."""
        out: dict[int, list] = {}
        g = self.lines
        if g is None:
            return {}
        seg = np.concatenate([g.p1, g.p2], axis=1)
        for i in range(len(seg)):
            if g.front[i] == g.back[i]:
                continue  # same sector on both sides: not a boundary
            for sec in (int(g.front[i]), int(g.back[i])):
                if sec >= 0:
                    out.setdefault(sec, []).append(seg[i])
        return {k: np.array(v) for k, v in out.items()}

    def is_door(self, sector: int) -> bool:
        return sector in self.door_sectors

    def lock(self, sector: int) -> int:
        return self.door_sectors.get(sector, 0)


def load_mapinfo(wad_path: str | Path, map_name: str) -> MapInfo:
    lumps = map_lumps(read_wad(wad_path), map_name)
    if "TEXTMAP" in lumps:
        text = lumps["TEXTMAP"].decode("utf-8", "replace")
        if udmf_namespace(text) in DOOM_NAMESPACES:
            return _from_udmf_doom(text)
        return _from_udmf(text)
    if "BEHAVIOR" in lumps:
        return _from_hexen(lumps)
    return _from_doom(lumps)


def _add(info: MapInfo, sector: int, lock: int) -> None:
    if 0 <= sector < info.num_sectors:
        # If a sector is reachable both with and without a key, the key is not required.
        info.door_sectors[sector] = min(info.door_sectors.get(sector, lock), lock)


# --- UDMF ---------------------------------------------------------------------------------
_COMMENTS = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_BLOCK = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*\{([^{}]*)\}", re.S)
_ASSIGN = re.compile(r"([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(\"(?:[^\"\\]|\\.)*\"|[^;]+?)\s*;", re.S)


def _udmf_value(raw: str):
    raw = raw.strip()
    if raw.startswith('"'):
        return raw[1:-1]
    low = raw.lower()
    if low in ("true", "false"):
        return low == "true"
    try:
        return int(raw, 0)
    except ValueError:
        try:
            return float(raw)
        except ValueError:
            return raw


def parse_udmf(text: str) -> dict[str, list[dict]]:
    blocks: dict[str, list[dict]] = {}
    for kind, body in _BLOCK.findall(_COMMENTS.sub("", text)):
        props = {k.lower(): _udmf_value(v) for k, v in _ASSIGN.findall(body)}
        blocks.setdefault(kind.lower(), []).append(props)
    return blocks


# UDMF namespaces whose line specials use Doom/Boom numbering (the others use Hexen/ZDoom numbering).
DOOM_NAMESPACES = {"doom", "heretic", "strife"}
_NAMESPACE = re.compile(r"^\s*namespace\s*=\s*\"([^\"]*)\"\s*;", re.M | re.I)


def udmf_namespace(text: str) -> str:
    m = _NAMESPACE.search(_COMMENTS.sub("", text))
    return m.group(1).lower() if m else ""


def _from_udmf_doom(text: str) -> MapInfo:
    """UDMF in a Doom-like namespace: Doom specials; the line's `id` is the sector tag."""
    info = _from_udmf(text, doors=False)
    info.format = "udmf-doom"
    b = parse_udmf(text)
    sides = b.get("sidedef", [])
    tags: dict[int, list[int]] = {}
    for i, sec in enumerate(b.get("sector", [])):
        if isinstance(sec.get("id"), int) and sec["id"] != 0:
            tags.setdefault(sec["id"], []).append(i)
    for line in b.get("linedef", []):
        special = int(line.get("special", 0) or 0)
        back = line.get("sideback", -1)
        back_sec = int(sides[back].get("sector", -1)) if isinstance(back, int) and 0 <= back < len(sides) else -1
        tag = int(line.get("id", 0) or 0)
        for sec in _doom_door_targets(special, tag, back_sec, tags):
            _add(info, sec[0], sec[1])
    return info


def _from_udmf(text: str, doors: bool = True) -> MapInfo:
    b = parse_udmf(text)
    sectors = b.get("sector", [])
    sides = b.get("sidedef", [])
    info = MapInfo(num_sectors=len(sectors), format="udmf")
    tags: dict[int, list[int]] = {}
    for i, sec in enumerate(sectors):
        for key in ("id",):
            if isinstance(sec.get(key), int) and sec[key] != 0:
                tags.setdefault(sec[key], []).append(i)
        for extra in str(sec.get("moreids", "")).split():
            if extra.lstrip("-").isdigit():
                tags.setdefault(int(extra), []).append(i)

    def side_sector(idx) -> int:
        if isinstance(idx, int) and 0 <= idx < len(sides):
            return int(sides[idx].get("sector", -1))
        return -1

    verts = [(float(v.get("x", 0.0)), float(v.get("y", 0.0))) for v in b.get("vertex", [])]
    geo = []
    for line in b.get("linedef", []):
        try:
            a, c = verts[int(line["v1"])], verts[int(line["v2"])]
        except (KeyError, IndexError, ValueError, TypeError):
            continue
        geo.append((a, c, side_sector(line.get("sidefront", -1)), side_sector(line.get("sideback", -1)),
                    bool(line.get("blocking", False))))
    info.lines = _geometry(geo)
    for line in (b.get("linedef", []) if doors else []):
        special = int(line.get("special", 0) or 0)
        if special not in HEXEN_DOORS:
            continue
        args = [int(line.get(f"arg{i}", 0) or 0) for i in range(5)]
        lock_arg = HEXEN_DOORS[special]
        lock = args[lock_arg] if lock_arg is not None else 0
        lock = lock or int(line.get("locknumber", 0) or 0)
        targets = tags.get(args[0], []) if args[0] else [side_sector(line.get("sideback", -1))]
        for sec in targets:
            _add(info, sec, lock)
    return info


# --- Binary formats -----------------------------------------------------------------------
def _geometry(rows) -> LineGeometry:
    n = len(rows)
    g = LineGeometry(np.zeros((n, 2)), np.zeros((n, 2)), np.full(n, -1, np.int64), np.full(n, -1, np.int64), np.zeros(n, bool))
    for i, (a, c, front, back, imp) in enumerate(rows):
        g.p1[i], g.p2[i], g.front[i], g.back[i], g.impassable[i] = a, c, front, back, imp
    return g


def _vertices(data: bytes) -> list[tuple[float, float]]:
    return [tuple(float(x) for x in struct.unpack_from("<hh", data, i * 4)) for i in range(len(data) // 4)]


def _binary_geometry(lumps: dict[str, bytes], sides: list[int], hexen: bool) -> LineGeometry:
    verts = _vertices(lumps.get("VERTEXES", b""))
    data = lumps.get("LINEDEFS", b"")
    size = 16 if hexen else 14
    rows = []
    for i in range(len(data) // size):
        v1, v2, flags = struct.unpack_from("<HHH", data, i * size)
        front, back = struct.unpack_from("<HH", data, i * size + size - 4)
        if v1 >= len(verts) or v2 >= len(verts):
            continue
        fs = sides[front] if front != 0xFFFF and front < len(sides) else -1
        bs = sides[back] if back != 0xFFFF and back < len(sides) else -1
        rows.append((verts[v1], verts[v2], fs, bs, bool(flags & 0x0001)))
    return _geometry(rows)


def _sides(data: bytes) -> list[int]:
    return [struct.unpack_from("<h", data, i * 30 + 28)[0] for i in range(len(data) // 30)]


def _sector_tags(data: bytes) -> tuple[int, dict[int, list[int]]]:
    n = len(data) // 26
    tags: dict[int, list[int]] = {}
    for i in range(n):
        tag = struct.unpack_from("<h", data, i * 26 + 24)[0]
        if tag:
            tags.setdefault(tag, []).append(i)
    return n, tags


def _from_hexen(lumps: dict[str, bytes]) -> MapInfo:
    n, tags = _sector_tags(lumps.get("SECTORS", b""))
    sides = _sides(lumps.get("SIDEDEFS", b""))
    info = MapInfo(num_sectors=n, format="hexen")
    info.lines = _binary_geometry(lumps, sides, hexen=True)
    data = lumps.get("LINEDEFS", b"")
    for i in range(len(data) // 16):
        _v1, _v2, _flags, special, a0, a1, a2, a3, a4, _front, back = struct.unpack_from("<HHHB5BHH", data, i * 16)
        if special not in HEXEN_DOORS:
            continue
        args = (a0, a1, a2, a3, a4)
        lock_arg = HEXEN_DOORS[special]
        lock = args[lock_arg] if lock_arg is not None else 0
        back_sec = sides[back] if back != 0xFFFF and back < len(sides) else -1
        for sec in (tags.get(a0, []) if a0 else [back_sec]):
            _add(info, sec, lock)
    return info


def _doom_door_targets(special: int, tag: int, back_sec: int, tags: dict[int, list[int]]) -> list[tuple[int, int]]:
    """(sector, lock) pairs a Doom/Boom line special opens as a door."""
    manual = None
    lock = 0
    if special in DOOM_MANUAL_DOORS:
        manual, lock = True, DOOM_MANUAL_DOORS[special]
    elif special in DOOM_REMOTE_DOORS:
        manual, lock = False, DOOM_REMOTE_DOORS[special]
    elif 0x3C00 <= special <= 0x3FFF:  # Boom generalized door: kind (bits 5-6) 0/1 open, 2/3 close
        if (special >> 5) & 0x3 in (2, 3):
            return []
        manual = (special & 0x7) in (6, 7)
    elif 0x3800 <= special <= 0x3BFF:  # Boom generalized locked door: key in bits 6-8
        manual, lock = (special & 0x7) in (6, 7), BOOM_KEY_LOCK[(special >> 6) & 0x7]
    if manual is None:
        return []
    return [(sec, lock) for sec in ([back_sec] if manual else tags.get(tag, []))]


def _from_doom(lumps: dict[str, bytes]) -> MapInfo:
    n, tags = _sector_tags(lumps.get("SECTORS", b""))
    sides = _sides(lumps.get("SIDEDEFS", b""))
    info = MapInfo(num_sectors=n, format="doom")
    info.lines = _binary_geometry(lumps, sides, hexen=False)
    data = lumps.get("LINEDEFS", b"")
    for i in range(len(data) // 14):
        _v1, _v2, _flags, special, tag, _front, back = struct.unpack_from("<HHHHhHH", data, i * 14)
        back_sec = sides[back] if back != 0xFFFF and back < len(sides) else -1
        for sec, lock in _doom_door_targets(special, tag, back_sec, tags):
            _add(info, sec, lock)
    return info
