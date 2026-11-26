"""Smoke-load tool and the PROPOSED map_inventory(env) contract."""

from __future__ import annotations

from dataclasses import asdict

import numpy as np

from doom_env import make_env
from training.envs import maps as R
from training.envs.inventory import map_inventory, static_things
from training.envs.mapgen import generate
from training.envs.smoke_load import smoke_one, table


def test_smoke_one_generated_and_bundled(tmp_path):
    generate("keys", 4).write(tmp_path / "keys")
    gen = R.generated_entries(tmp_path)[0]
    fd2 = next(e for e in R.freedoom2_entries() if e.map == "MAP01")
    rows = [smoke_one(asdict(e), steps=30) for e in (gen, fd2)]
    for r in rows:
        assert r["loads"] and not r["error"] and r["geometry"] == "wad" and r["wad_sectors"] == r["engine_sectors"]
        assert r["steps"] == 30
    assert rows[0]["meta_doors_match"] and rows[0]["meta_spawn_match"] and rows[0]["meta_things_match"]
    assert rows[0]["locked_lines"] == 2 * rows[0]["locked"] > 0
    assert "| yes |" in table(rows)


def test_smoke_one_reports_load_failure():
    bad = next(e for e in R.all_entries(["vizdoom"]) if e.map == "MAP01")
    r = smoke_one(dict(asdict(bad), map="MAP99"), steps=1)
    assert not r["loads"] and r["error"]


def test_map_inventory_matches_generator(tmp_path):
    gm = generate("full", 2)
    wad, _ = gm.write(tmp_path)
    env = make_env(str(wad), render_buffers=False)
    env.reset(seed=0)
    try:
        inv = map_inventory(env)
    finally:
        env.close()
    md = gm.meta
    assert inv["num_sectors"] == md["num_sectors"]
    assert inv["monsters"] == inv["static_monsters"] == md["num_monsters"]
    assert inv["keys"] == inv["static_keys"] == md["num_keys"]
    assert inv["doors_total"] == md["num_doors"] and inv["locked_total"] == md["num_locked_doors"]
    assert inv["sector_area"].shape == (md["num_sectors"],) and inv["sector_area"].dtype == np.float64
    pad_cells = 4 if md["exit"]["kind"] == "pad" else 0  # the 2x2 exit pad is its own sector
    room_area = (sum(r["area_cells"] for r in md["rooms"]) - pad_cells) * 32 * 32
    assert inv["sector_area"][[r["sector"] for r in md["rooms"]]].sum() == room_area
    assert inv["cells_total"] > 0


def test_static_things_on_freedoom():
    path = R.resolve_wad("@vizdoom/freedoom2.wad")
    mon, keys = static_things(str(path), "MAP01")
    assert mon > 0 and keys == 0
