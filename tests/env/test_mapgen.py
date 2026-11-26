"""Generator unit tests (no engine): determinism, caps, stage semantics, door/lock detection, solvability."""

from __future__ import annotations

import hashlib
import os
import struct
import subprocess
import sys
from pathlib import Path

import pytest
import vizdoom as vd

from doom_env.mapinfo import load_mapinfo, parse_udmf
from doom_env.wad import read_wad
from training.envs.mapgen import MAX_SECTORS, STAGES, generate
from training.envs.mapgen.__main__ import main as cli_main
from training.envs.mapgen.progression import CellWorld, check_solvable

ROOT = Path(__file__).resolve().parents[2]
SEEDS = range(12)


@pytest.fixture(scope="module")
def maps():
    return {(st, fam, s): generate(st, s, family=fam) for st in STAGES for fam in ("a", "b") for s in SEEDS}


def test_same_seed_same_bytes():
    for st in STAGES:
        a, b = generate(st, 7), generate(st, 7)
        assert a.wad == b.wad and a.meta == b.meta
    assert generate("keys", 1).wad != generate("keys", 2).wad
    assert generate("keys", 1, family="a").wad != generate("keys", 1, family="b").wad


def test_bytes_stable_across_processes_and_hash_seeds():
    code = ("from training.envs.mapgen import generate, STAGES\n"
            "print(','.join(generate(s, 31).sha256 for s in STAGES))")
    here = ",".join(generate(s, 31).sha256 for s in STAGES)
    for hs in ("1", "12345"):
        env = dict(os.environ, PYTHONHASHSEED=hs, PYTHONPATH=str(ROOT))
        out = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, capture_output=True, text=True, timeout=120)
        assert out.returncode == 0, out.stderr
        assert out.stdout.strip() == here


def test_write_roundtrip_and_sidecar(tmp_path):
    m = generate("keys", 3)
    wad, js = m.write(tmp_path)
    assert wad.read_bytes() == m.wad
    assert hashlib.sha256(wad.read_bytes()).hexdigest() == m.meta["wad_sha256"]
    assert js.read_text().startswith("{")
    names = [lump.name for lump in read_wad(wad)]
    assert names == ["MAP01", "TEXTMAP", "ENDMAP"]


def test_override_maps_get_distinct_names(tmp_path):
    plain, over = generate("combat", 4), generate("combat", 4, monsters=(9, 9))
    assert plain.wad != over.wad and over.meta["config_key"]
    (pw, _), (ow, _) = plain.write(tmp_path), over.write(tmp_path)
    assert pw != ow and pw.read_bytes() == plain.wad
    assert over.meta["source_group"] != plain.meta["source_group"]


def test_cli(tmp_path, capsys):
    assert cli_main(["--stage", "doors", "--seed", "5", "--count", "2", "--out", str(tmp_path)]) == 0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["doors-a-000005.json", "doors-a-000005.wad",
                                                          "doors-a-000006.json", "doors-a-000006.wad"]
    assert (tmp_path / "doors-a-000005.wad").read_bytes() == generate("doors", 5).wad
    assert cli_main(["--stage", "keys", "--seed", "9", "--out", str(tmp_path / "x" / "k.wad")]) == 0
    assert (tmp_path / "x" / "k.wad").read_bytes() == generate("keys", 9).wad


def test_sector_cap_headroom_single_start(maps):
    for (st, fam, s), m in maps.items():
        b = parse_udmf(_textmap(m.wad))
        assert len(b["sector"]) == m.meta["num_sectors"] <= MAX_SECTORS < 2048
        door_secs = {d["sector"] for d in m.meta["doors"]}
        for i, sec in enumerate(b["sector"]):
            if i not in door_secs:
                assert sec["heightceiling"] - sec["heightfloor"] >= 96, (st, fam, s, i)
        starts = [t for t in b["thing"] if t["type"] == 1]
        assert len(starts) == 1, (st, fam, s)
        assert all(t["skill3"] is True and t["single"] is True for t in b["thing"])


def _textmap(wad: bytes) -> str:
    count, diroff = struct.unpack_from("<ii", wad, 4)
    for i in range(count):
        off, size, name = struct.unpack_from("<ii8s", wad, diroff + 16 * i)
        if name.rstrip(b"\0") == b"TEXTMAP":
            return wad[off : off + size].decode()
    raise AssertionError("no TEXTMAP")


def test_mapinfo_detects_generated_doors_and_locks(maps, tmp_path):
    seen_locked = seen_door = 0
    for (st, fam, s), m in maps.items():
        wad, _ = m.write(tmp_path, f"{st}-{fam}-{s}")
        mi = load_mapinfo(wad, "MAP01")
        assert mi.num_sectors == m.meta["num_sectors"]
        assert mi.door_sectors == {d["sector"]: d["lock"] for d in m.meta["doors"]}, (st, fam, s)
        for ld in m.meta["locked_doors"]:
            assert mi.lock(ld["sector"]) > 0
        seen_locked += len(m.meta["locked_doors"])
        seen_door += m.meta["num_doors"]
    assert seen_locked > 0 and seen_door > 0


def test_stage_semantics(maps):
    for (st, fam, s), m in maps.items():
        md = m.meta
        assert md["solvable"] is True and md["solvability"]["exit_reachable"]
        assert all(md["solvability"]["keys_necessary"].values())
        assert md["num_keys"] == md["num_locked_doors"]
        assert sorted(k["color"] for k in md["keys"]) == sorted(d["color"] for d in md["locked_doors"])
        if st == "locomotion":
            assert md["num_doors"] == md["num_locked_doors"] == md["num_monsters"] == 0 and md["room_count"] >= 2
        if st == "doors":
            assert md["num_locked_doors"] == 0 and any(d["on_path"] for d in md["doors"])
        if st in ("keys", "full"):
            assert 1 <= md["num_locked_doors"] <= 3
            assert md["tier"] in (3, 4)
        if st == "keys":
            assert md["num_monsters"] == 0
        if st in ("combat", "full"):
            assert md["num_monsters"] >= 3
        if st == "combat":
            assert md["num_locked_doors"] == 0 and md["tier"] == 1
            assert any(i["category"] == "weapon" for i in md["items"])
    keys_maps = [m for (st, _f, _s), m in maps.items() if st == "keys"]
    in_branch = sum(k["in_branch"] for m in keys_maps for k in m.meta["keys"])
    total = sum(len(m.meta["keys"]) for m in keys_maps)
    assert in_branch / total >= 0.9  # backtracking: keys sit off the start->exit path
    assert {m.meta["exit"]["kind"] for m in maps.values()} == {"pad", "switch"}


def test_key_not_at_spawn(maps):
    for m in maps.values():
        sx, sy = m.meta["start"]["x"], m.meta["start"]["y"]
        for k in m.meta["keys"]:
            assert max(abs(k["x"] - sx), abs(k["y"] - sy)) >= 64


def test_locks_really_gate_the_exit(maps):
    """Independent re-check: with every key withheld, the exit is unreachable on the emitted grid."""
    m = next(m for (st, _f, _s), m in maps.items() if st == "keys" and m.meta["num_locked_doors"] >= 2)
    lab = m.geometry.labels
    import numpy as np

    secs = m.geometry.sectors
    floor = np.array([s.floor for s in secs])
    ceil = np.array([s.ceil if s.kind != "door" else s.floor + 120 for s in secs])
    ex = m.meta["exit"]
    exit_v = [(round(ex["x"] / 32) + ex["facing"][0], round(ex["y"] / 32) + ex["facing"][1])] if ex["kind"] == "switch" \
        else [(round(ex["x"] / 32), round(ex["y"] / 32))]
    start_v = (round(m.meta["start"]["x"] / 32), round(m.meta["start"]["y"] / 32))
    locks = {d["sector"]: d["color"] for d in m.meta["locked_doors"]}
    no_keys = CellWorld(lab, floor, ceil, locks, start_v, [], exit_v)
    assert check_solvable(no_keys)["exit_reachable"] is False
    with_keys = CellWorld(lab, floor, ceil, locks, start_v, [(k["color"], tuple(k["cell"])) for k in m.meta["keys"]], exit_v)
    assert check_solvable(with_keys)["solvable"] is True


def test_textures_exist_in_freedoom2(maps):
    lumps = read_wad(os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad"))
    names = [lump.name for lump in lumps]
    tex = set()
    for lump in lumps:
        if lump.name in ("TEXTURE1", "TEXTURE2"):
            n = struct.unpack_from("<i", lump.data, 0)[0]
            for off in struct.unpack_from(f"<{n}i", lump.data, 4):
                tex.add(lump.data[off : off + 8].rstrip(b"\0").decode())
    flats = set(names[names.index("F_START") + 1 : names.index("F_END")])
    used_t, used_f = set(), set()
    for m in maps.values():
        b = parse_udmf(_textmap(m.wad))
        for sd in b["sidedef"]:
            used_t |= {sd[k] for k in ("texturetop", "texturemiddle", "texturebottom")} - {"-"}
        for sec in b["sector"]:
            used_f |= {sec["texturefloor"], sec["textureceiling"]}
    assert used_t <= tex, used_t - tex
    assert used_f <= flats, used_f - flats
