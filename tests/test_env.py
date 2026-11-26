"""Tests for the official wrapper (doom_env). Run: uv run pytest -q"""

import math

import numpy as np
import pytest

from doom_env import CONTRACT_VERSION, make_env
from doom_env import spaces as S
from fixtures import fixture_wad
from doom_env.sensors import RAY_DOOR, RAY_ENEMY, RAY_ITEM, RAY_LOCKED, RAY_NONE, RAY_WALL

FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
USE = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0], np.int8)
NOOP = np.zeros(9, np.int8)
TURN_L = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0], np.int8)


def wad(tmp_path, grid, name="m.wad"):
    return fixture_wad(grid)


def ray_at(obs, angle_deg):
    k = round(angle_deg / 360 * S.NUM_RAYS) % S.NUM_RAYS
    return float(obs["rays"]["distance"][k]) * S.RAY_RANGE, int(obs["rays"]["hit_type"][k])


def test_contract_version():
    assert CONTRACT_VERSION


def test_observation_matches_space(tmp_path):
    env = make_env(wad(tmp_path, "#####\n#P.z#\n#####"))
    obs, info = env.reset(seed=1)
    assert env.observation_space.contains(obs)
    for _ in range(5):
        obs, *_ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
    assert obs["rgb"].shape == (120, 160, 3)
    env.close()


def test_rays_walls_and_doors(tmp_path):
    # Player at x=96 (cell 1), facing east. Door cell at col 4 -> x in [256, 320].
    env = make_env(wad(tmp_path, "##########\n#P..D...E#\n##########"))
    obs, _ = env.reset()
    d, t = ray_at(obs, 0)
    assert t == RAY_DOOR and d == pytest.approx(256 - 96, abs=1)
    d, t = ray_at(obs, 90)  # north wall: cell top at y=128, player y=96
    assert t == RAY_WALL and d == pytest.approx(32, abs=1)
    d, t = ray_at(obs, 180)  # west wall at x=64
    assert t == RAY_WALL and d == pytest.approx(32, abs=1)
    env.close()


def test_locked_door_ray_type(tmp_path):
    env = make_env(wad(tmp_path, "#######\n#P..B.E#\n#######"))
    obs, _ = env.reset()
    assert ray_at(obs, 0)[1] == RAY_LOCKED
    env.close()


def test_enemy_and_item_rays_and_entities(tmp_path):
    env = make_env(wad(tmp_path, "#########\n#P..h..z#\n#########"))
    obs, _ = env.reset()
    d, t = ray_at(obs, 0)
    assert t == RAY_ITEM  # stimpack in front of the zombieman
    ents = obs["visible_entities"]
    assert ents["mask"].sum() == 2
    first = ents["features"][0]
    assert first[0] * S.RAY_RANGE == pytest.approx(288 - 96, abs=1)
    assert first[2] == pytest.approx(1.0, abs=1e-3)  # cos(rel angle) = 1 -> straight ahead
    assert S.ENTITY_TYPES[ents["entity_type"][0]] == "health"
    assert S.ENTITY_TYPES[ents["entity_type"][1]] == "enemy_hitscan"
    env.close()


def test_line_of_sight_and_fov(tmp_path):
    # Zombieman behind a closed door is not visible; one behind the player is outside the FOV.
    env = make_env(wad(tmp_path, "###########\n#z.P..D..z#\n###########"))
    obs, _ = env.reset()
    assert obs["visible_entities"]["mask"].sum() == 0
    d, t = ray_at(obs, 180)
    assert t == RAY_ENEMY  # rays are 360 degrees, so the one behind is still sensed by rays
    env.close()


def test_key_pickup_and_locked_door_exit(tmp_path):
    env = make_env(wad(tmp_path, "############\n#P.r..R...E#\n############"), episode_ticks=1000)
    obs, _ = env.reset()
    assert obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 0
    picked = False
    last_x = None
    for i in range(240):
        st = env.privileged_state()
        stuck = last_x is not None and abs(st["x"] - last_x) < 0.5
        last_x = st["x"]
        action = USE if (stuck and i % 6 == 0) else FWD
        obs, r, term, trunc, info = env.step(action)
        picked = picked or obs["vitals"][S.VITALS_INDEX["key_red_card"]] == 1
        if term or trunc:
            break
    assert picked
    assert term and info["exit_reached"] and r >= 1.0
    env.close()


def test_truncation_at_tick_cap(tmp_path):
    env = make_env(wad(tmp_path, "#####\n#P..#\n#####"), episode_ticks=40)
    env.reset()
    steps = 0
    while True:
        steps += 1
        _, _, term, trunc, info = env.step(NOOP)
        if term or trunc:
            break
    assert trunc and not term and not info["exit_reached"]
    assert steps == 10
    env.close()


def test_translation_invariance(tmp_path):
    """The same layout placed elsewhere on the map must yield identical non-pixel observations."""
    grid = "#########\n#P.....z#\n#..D....#\n#########"
    shifted = "\n".join(["#" * 13] * 3 + ["####" + row for row in grid.splitlines()])
    a = make_env(wad(tmp_path, grid, "a.wad"))
    b = make_env(wad(tmp_path, shifted, "b.wad"))
    oa, _ = a.reset(seed=3)
    ob, _ = b.reset(seed=3)
    for step in range(12):
        act = TURN_L if step % 3 else FWD
        for k in ("vitals",):
            np.testing.assert_allclose(oa[k], ob[k], atol=1e-4)
        np.testing.assert_allclose(oa["rays"]["distance"], ob["rays"]["distance"], atol=1e-4)
        np.testing.assert_array_equal(oa["rays"]["hit_type"], ob["rays"]["hit_type"])
        np.testing.assert_allclose(oa["visible_entities"]["features"], ob["visible_entities"]["features"], atol=1e-3)
        oa, *_ = a.step(act)
        ob, *_ = b.step(act)
    assert a.privileged_state()["x"] != b.privileged_state()["x"]
    a.close()
    b.close()


def test_invalid_actions_rejected(tmp_path):
    env = make_env(wad(tmp_path, "####\n#P.#\n####"))
    env.reset()
    with pytest.raises(ValueError):
        env.step(np.zeros(8))
    with pytest.raises(ValueError):
        env.step(np.full(9, 2))
    env.close()


def test_freedoom_map_runs():
    """Real Doom-format map (Freedoom MAP01): many sectors, doors, monsters."""
    import os
    import vizdoom
    env = make_env(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"), "MAP01", episode_ticks=400)
    obs, _ = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    for _ in range(50):
        obs, _, term, trunc, _ = env.step(env.action_space.sample())
        assert env.observation_space.contains(obs)
        if term or trunc:
            break
    env.close()


def test_projectiles_are_classified(tmp_path):
    """Imp fireballs have ViZDoom category 'Explosive' (like bullet puffs); they must show up as projectiles."""
    env = make_env(wad(tmp_path, "##########\n#P......i#\n##########"), episode_ticks=700)
    obs, _ = env.reset(seed=1)
    seen = set()
    for _ in range(170):
        obs, _, term, trunc, _ = env.step(NOOP)
        m = obs["visible_entities"]["mask"].astype(bool)
        seen |= {S.ENTITY_TYPES[t] for t in obs["visible_entities"]["entity_type"][m]}
        if term or trunc:
            break
    assert {"enemy_projectile", "projectile"} <= seen
    env.close()


def test_final_step_variables_are_refreshed(tmp_path):
    """The killing blow / last kill happen on the terminal step and must still be visible to scoring."""
    env = make_env(wad(tmp_path, "##########\n#P......c#\n##########"), episode_ticks=2000)
    env.reset(seed=0)
    while True:
        _, _, term, trunc, _ = env.step(NOOP)
        if term or trunc:
            break
    st = env.privileged_state()
    assert st["dead"] and st["health"] <= 0 and st["damage_taken"] >= 100
    env.close()


def test_static_geometry_comes_from_the_wad():
    """ViZDoom's runtime sector line lists can be incomplete on big maps; doom_env reads exact WAD geometry."""
    import os
    import vizdoom
    env = make_env(os.path.join(os.path.dirname(vizdoom.__file__), "freedoom2.wad"), "MAP01", episode_ticks=100)
    env.reset()
    assert env._geometry.source == "wad"
    assert len(env._geometry.p1) == len(env.mapinfo.lines.p1)
    env.close()


def test_out_of_order_calls_raise_clear_errors(tmp_path):
    env = make_env(fixture_wad("#####\n#P..#\n#####"), "MAP01", episode_ticks=100)
    with pytest.raises(RuntimeError, match="reset"):
        env.step(np.zeros(9, np.int8))
    env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset()


def test_boom_generalized_lock_keys_and_doom_udmf():
    from doom_env import mapinfo as M
    # Boom generalized locked doors: trigger in bits 0-2 (6 = manual), key in bits 6-8
    assert M._doom_door_targets(0x3800 | (1 << 6) | 6, 0, 5, {}) == [(5, M.LOCK_RED)]
    assert M._doom_door_targets(0x3800 | (7 << 6) | 6, 0, 5, {}) == [(5, M.LOCK_ALL)]
    assert M._doom_door_targets(0x3800 | (5 << 6), 3, 5, {3: [9]}) == [(9, M.LOCK_BLUE)]
    assert M._doom_door_targets(0x3C00 | (2 << 5) | 6, 0, 5, {}) == []  # close-type generalized door
    assert M._doom_door_targets(106, 4, 5, {4: [7]}) == [(7, 0)]
    udmf = '''namespace = "doom";
    vertex { x = 0.0; y = 0.0; } vertex { x = 64.0; y = 0.0; }
    sector { heightfloor = 0; heightceiling = 128; } sector { heightfloor = 0; heightceiling = 0; }
    sidedef { sector = 0; } sidedef { sector = 1; }
    linedef { v1 = 0; v2 = 1; sidefront = 0; sideback = 1; special = 1; }
    linedef { v1 = 1; v2 = 0; sidefront = 0; special = 11; }'''
    info = M._from_udmf_doom(udmf)
    assert M.udmf_namespace(udmf) == "doom" and info.door_sectors == {1: 0}  # DR door, not the exit switch


def test_duplicate_map_uses_last_lump():
    from doom_env.wad import Lump, map_lumps
    lumps = [Lump("MAP01", b""), Lump("THINGS", b"old"), Lump("MAP01", b""), Lump("THINGS", b"new")]
    assert map_lumps(lumps, "MAP01")["THINGS"] == b"new"
