"""Generated maps through the official doom_env.make_env: load, mapinfo, things, rays, completable exits.

The scripted walker (env_walker.py) is TEST-ONLY: it uses privileged_state and BFS to prove the maps can be
completed in the engine. It is never available to training or agent code.
"""

from __future__ import annotations

import collections

import numpy as np
import pytest

from doom_env import make_env
from doom_env.sensors import RAY_DOOR, RAY_LOCKED
from env_walker import walk_to_exit
from training.envs.mapgen import STAGES, generate


def _env(gm, tmp_path, ticks: int = 30000):
    wad, _ = gm.write(tmp_path)
    env = make_env(str(wad), gm.map_name, episode_ticks=ticks, render_buffers=False)
    env.reset(seed=0)
    return env


@pytest.mark.parametrize("stage", STAGES)
def test_generated_maps_load_and_match_metadata(stage, tmp_path):
    for seed, fam in ((0, "a"), (1, "b")):
        gm = generate(stage, seed, family=fam)
        env = _env(gm, tmp_path / f"{seed}", ticks=2100)
        try:
            st = env.privileged_state()
            state = env._game.get_state()
            assert env._geometry.source == "wad"  # engine sector count == WAD sector count
            assert len(state.sectors) == gm.meta["num_sectors"]
            assert env.mapinfo.door_sectors == {d["sector"]: d["lock"] for d in gm.meta["doors"]}
            assert abs(st["x"] - gm.meta["start"]["x"]) < 1 and abs(st["y"] - gm.meta["start"]["y"]) < 1
            got = collections.Counter(o.name for o in state.objects if o.category != "Self")
            want = collections.Counter([m["class"] for m in gm.meta["monsters"]] + [k["class"] for k in gm.meta["keys"]]
                                       + [i["class"] for i in gm.meta["items"]])
            assert got == want
            cats = {o.name: o.category for o in state.objects}
            for m in gm.meta["monsters"]:
                assert cats[m["class"]] == "Monster"
            ht = env._geometry.hit_type  # static per-line ray classification at reset (doors closed)
            assert int((ht == RAY_DOOR).sum()) == 2 * gm.meta["num_doors"]  # two door faces per door
            assert int((ht == RAY_LOCKED).sum()) == 2 * gm.meta["num_locked_doors"]
            env.action_space.seed(seed)
            for _ in range(50):
                obs, _r, term, trunc, _i = env.step(env.action_space.sample().astype(np.int8))
                assert env.observation_space.contains(obs)
                if term or trunc:
                    env.reset(seed=1)
        finally:
            env.close()


def test_every_monster_and_key_class_spawns(tmp_path):
    """Probe the full thing roster once: every monster/key/item class the generator can place exists in the engine."""
    from training.envs.mapgen import things as T

    names = collections.Counter()
    for seed in range(8):
        gm = generate("full", seed, monsters=(14, 14), roster_level=3, weapons=(3, 3), skull_prob=0.5)
        env = _env(gm, tmp_path / str(seed), ticks=2100)
        names.update(o.name for o in env._game.get_state().objects)
        env.close()
    known = {cls for _e, cls, _r in T.MONSTERS.values()} | {c for t in (T.HEALTH, T.ARMOR, T.AMMO, T.WEAPONS) for _e, c in t.values()}
    known |= {v[1] for v in T.KEYS.values()} | {v[3] for v in T.KEYS.values()}
    unknown = set(names) - known - {"DoomPlayer"}
    assert not unknown, unknown
    assert len(set(names) & {cls for _e, cls, _r in T.MONSTERS.values()}) >= 8


@pytest.mark.parametrize("stage,seed", [("locomotion", 0), ("locomotion", 2), ("doors", 0), ("doors", 2), ("doors", 12),
                                        ("keys", 0), ("keys", 3), ("keys", 11)])
def test_exit_reachable_by_scripted_walk(stage, seed, tmp_path):
    gm = generate(stage, seed)
    env = _env(gm, tmp_path)
    try:
        r = walk_to_exit(env, gm)
    finally:
        env.close()
    assert r["exit_reached"] and r["terminated"], r
    assert sum(r["keys"]) == gm.meta["num_keys"]
    if gm.meta["num_locked_doors"]:
        assert RAY_LOCKED in r["hit_types"]  # the walker faced a closed locked door before unlocking it


def test_switch_and_pad_exits_both_work(tmp_path):
    kinds = {}
    for seed in range(20):
        gm = generate("locomotion", seed)
        k = gm.meta["exit"]["kind"]
        if k in kinds:
            continue
        env = _env(gm, tmp_path / str(seed))
        try:
            kinds[k] = walk_to_exit(env, gm)["exit_reached"]
        finally:
            env.close()
        if len(kinds) == 2:
            break
    assert kinds == {"pad": True, "switch": True}


def test_door_ray_seen_on_doors_walk(tmp_path):
    gm = generate("doors", 0)
    env = _env(gm, tmp_path)
    try:
        r = walk_to_exit(env, gm)
    finally:
        env.close()
    assert r["exit_reached"] and RAY_DOOR in r["hit_types"]


def test_locked_door_blocks_without_key(tmp_path):
    """Engine-level negative control: skipping the keys, the walker cannot finish the map."""
    gm = generate("keys", 0)
    assert gm.meta["num_locked_doors"] >= 1
    gm.meta = dict(gm.meta, keys=[])  # walker goes straight for the exit
    env = _env(gm, tmp_path)
    try:
        r = walk_to_exit(env, gm, max_steps=700)
    finally:
        env.close()
    assert not r["exit_reached"]
    assert RAY_LOCKED in r["hit_types"]
