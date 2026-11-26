"""Minimal WAD file reader/writer (no external dependencies)."""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Lump:
    name: str
    data: bytes


def read_wad(path: str | Path) -> list[Lump]:
    raw = Path(path).read_bytes()
    if len(raw) < 12 or raw[:4] not in (b"IWAD", b"PWAD"):
        raise ValueError(f"{path}: not a WAD file")
    count, dir_offset = struct.unpack_from("<ii", raw, 4)
    if count < 0 or dir_offset < 12 or dir_offset + 16 * count > len(raw):
        raise ValueError(f"{path}: corrupt WAD directory")
    lumps = []
    for i in range(count):
        offset, size, name = struct.unpack_from("<ii8s", raw, dir_offset + 16 * i)
        if offset < 0 or size < 0 or offset + size > len(raw):
            raise ValueError(f"{path}: lump {i} out of bounds")
        lumps.append(Lump(name.rstrip(b"\0").decode("ascii", "replace").upper(), raw[offset : offset + size]))
    return lumps


def write_wad(path: str | Path, lumps: list[Lump], kind: bytes = b"PWAD") -> None:
    body = b""
    directory = b""
    offset = 12
    for lump in lumps:
        name = lump.name.upper().encode("ascii")
        if len(name) > 8:
            raise ValueError(f"lump name too long: {lump.name}")
        directory += struct.pack("<ii8s", offset, len(lump.data), name.ljust(8, b"\0"))
        body += lump.data
        offset += len(lump.data)
    header = kind + struct.pack("<ii", len(lumps), offset)
    Path(path).write_bytes(header + body + directory)


MAP_LUMPS_DOOM = ("THINGS", "LINEDEFS", "SIDEDEFS", "VERTEXES", "SEGS", "SSECTORS", "NODES", "SECTORS", "REJECT", "BLOCKMAP", "BEHAVIOR", "SCRIPTS")


def map_lumps(lumps: list[Lump], map_name: str) -> dict[str, bytes]:
    """Return the lumps belonging to `map_name` (marker excluded), keyed by name.

    If a WAD contains the same map twice, the engine uses the last one, and so do we."""
    map_name = map_name.upper()
    for i in range(len(lumps) - 1, -1, -1):
        lump = lumps[i]
        if lump.name != map_name:
            continue
        out: dict[str, bytes] = {}
        if i + 1 < len(lumps) and lumps[i + 1].name == "TEXTMAP":
            for nxt in lumps[i + 1 :]:
                if nxt.name == "ENDMAP":
                    break
                out[nxt.name] = nxt.data
            return out
        for nxt in lumps[i + 1 :]:
            if nxt.name not in MAP_LUMPS_DOOM:
                break
            out[nxt.name] = nxt.data
        return out
    raise KeyError(f"map {map_name} not found")


def list_maps(lumps: list[Lump]) -> list[str]:
    names = []
    for i, lump in enumerate(lumps[:-1]):
        if lumps[i + 1].name in ("TEXTMAP", "THINGS"):
            names.append(lump.name)
    return names
