"""TrainEnv / make_train_env (docs/MODULE_CONTRACTS.md §5.1): reset/step, info layout, done steps, ticks, parity."""

from __future__ import annotations

import json

import numpy as np
import pytest

from doom_agent.obs import FEATURE_SPECS, ObsConfig, ObsPreprocessor
from doom_env import make_env
from training.envs.factory import PRIVILEGED_KEYS, EngineCrash, EnvSpec, make_train_env
from training.envs.maps import ROOT, MapSampler, resolve_wad

SMOKE = {m["id"]: m for m in json.loads((ROOT / "maps/splits/smoke.json").read_text())["maps"]}
FWD = np.array([1, 0, 0, 0, 0, 0, 0, 0, 0], np.int8)
NOOP = np.zeros(9, np.int8)
TERMS = {"terms": [{"name": "env_default", "weight": 1.0}, {"name": "test_full_priv", "weight": 0.5},
                   {"name": "test_step_index", "weight": 0.0}], "modules": ["env_test_terms"]}
STEP_KEYS = {"map_id", "tic", "exit_reached", "reward_terms", "env_reward"}


def spec(map_id: str, ticks: int = 2100) -> EnvSpec:
    m = SMOKE[map_id]
    return EnvSpec(map_id=map_id, wad=m["wad"], map=m["map"], episode_ticks=ticks)


def run_until_done(env, action, cap=2000):
    for _ in range(cap):
        out = env.step(action)
        if out[2] or out[3]:
            return out
    raise AssertionError("episode did not end")


def test_reset_step_layout_and_features():
    env = make_train_env(spec("two_rooms"), seed=5, reward_cfg=TERMS)
    try:
        feats, info = env.reset()
        assert set(feats) == set(FEATURE_SPECS) and "privileged" not in feats
        assert set(info) == {"map_id", "tic", "exit_reached", "reward_terms"}
        f, r, term, trunc, info = env.step(FWD)
        assert set(f) == set(FEATURE_SPECS)
        assert set(info) == STEP_KEYS and "privileged" not in info
        assert info["reward_terms"] == {"env_default": 0.0, "test_full_priv": 1.0, "test_step_index": 0.0}
        assert r == pytest.approx(0.5)
        assert env.step(FWD)[4]["reward_terms"]["test_step_index"] == 1.0
        for k, s in FEATURE_SPECS.items():
            assert f[k].shape == s.shape and f[k].dtype == s.np_dtype
        assert env.vanilla
    finally:
        env.close()


def test_return_privileged_is_opt_in_and_slim():
    env = make_train_env(spec("two_rooms"), seed=5, return_privileged=True)
    try:
        feats, info = env.reset()
        assert set(info["privileged"]) == set(PRIVILEGED_KEYS)
        _f, _r, _te, _tr, info = env.step(FWD)
        assert set(info["privileged"]) == set(PRIVILEGED_KEYS)
        assert not any(k in info["privileged"] for k in ("sector_floor", "sector_ceiling", "door_lock"))
        assert set(_f) == set(FEATURE_SPECS)
    finally:
        env.close()


@pytest.mark.parametrize("map_id,action,ticks,outcome,term", [
    ("fx_exit_only", FWD, 2100, "exit", True),
    ("fx_lethal_c", NOOP, 2100, "death", True),
    ("fx_corridor_empty", NOOP, 40, "timeout", False),
])
def test_done_steps_carry_final_feats_and_episode(map_id, action, ticks, outcome, term):
    env = make_train_env(spec(map_id, ticks), seed=1, reward_cfg=TERMS)
    try:
        env.reset()
        f, r, te, tr, info = run_until_done(env, action)
        assert te is term and tr is (not term)
        assert set(info["final_feats"]) == set(FEATURE_SPECS)
        assert all(np.array_equal(info["final_feats"][k], f[k]) for k in f)
        ep = info["episode"]
        assert set(ep) >= {"map_id", "seed", "length", "return_shaped", "return_env", "outcome", "cells", "keys", "kills",
                           "use_edges", "weapon_edges"}
        assert ep["outcome"] == outcome and ep["map_id"] == map_id and ep["seed"] == 1
        assert ep["return_env"] == pytest.approx({"exit": 1.0, "death": -1.0, "timeout": 0.0}[outcome])
        if outcome == "timeout":
            assert ep["length"] == 10  # 40 tics / frame skip 4
        assert ep["reward_terms"]["test_full_priv"] == ep["length"]
    finally:
        env.close()


def test_episode_ticks_changed_in_place_without_reopen():
    env = make_train_env(spec("fx_corridor_empty", 40), seed=0)
    try:
        env.reset()
        assert run_until_done(env, NOOP)[4]["episode"]["length"] == 10
        env.reset(seed=1, spec=spec("fx_corridor_empty", 80))
        out = run_until_done(env, NOOP)
        assert out[3] and out[4]["episode"]["length"] == 20 and out[4]["episode"]["episode_ticks"] == 80
        assert env.engine_opens == 1
        env.reset(seed=2, spec=spec("fx_tiny_room", 40))
        assert env.engine_opens == 2 and env.spec.map_id == "fx_tiny_room"
    finally:
        env.close()


def test_training_seeds_below_10000():
    with pytest.raises(ValueError):
        make_train_env(spec("two_rooms"), seed=10_000)
    env = make_train_env(spec("two_rooms"), seed=0)
    try:
        with pytest.raises(ValueError):
            env.reset(seed=12_345)
    finally:
        env.close()


def test_engine_crash_raises_and_recovers():
    env = make_train_env(spec("two_rooms"), seed=0)
    try:
        env.reset()
        env._inject_crash()
        with pytest.raises(EngineCrash):
            env.step(FWD)
        assert env.crash_count == 1
        env.recover()
        feats, _ = env.reset(seed=3)
        assert set(feats) == set(FEATURE_SPECS)
        env.step(FWD)
    finally:
        env.close()


def test_nan_observation_fails_loudly():
    env = make_train_env(spec("two_rooms"), seed=0)
    try:
        env.reset()
        real = env.env.step

        def bad(a):
            obs, r, te, tr, info = real(a)
            obs = dict(obs, rays=dict(obs["rays"], distance=np.full(64, np.nan, np.float32)))
            return obs, r, te, tr, info

        env.env.step = bad
        with pytest.raises(ValueError):
            env.step(FWD)
    finally:
        env.close()


def test_bad_action_rejected():
    env = make_train_env(spec("two_rooms"), seed=0)
    try:
        env.reset()
        with pytest.raises(ValueError):
            env.step(np.full(9, 2, np.int8))
        with pytest.raises(ValueError):
            env.step(np.zeros(8, np.int8))
    finally:
        env.close()


def test_features_match_eval_path():
    """Train/eval parity: TrainEnv features == ObsPreprocessor on the raw official env (same seed, same actions)."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 2, (60, 9), dtype=np.int8)
    tenv = make_train_env(spec("two_rooms"), seed=7)
    raw = make_env(str(resolve_wad(SMOKE["two_rooms"]["wad"])), "MAP01", episode_ticks=2100, render_buffers=False)
    pre = ObsPreprocessor(ObsConfig())
    try:
        f_train, _ = tenv.reset()
        obs, _ = raw.reset(seed=7)
        f_eval = pre.process(obs)
        for a in actions:
            assert all(np.array_equal(f_train[k], f_eval[k]) for k in FEATURE_SPECS)
            f_train, _r, te, tr, _i = tenv.step(a)
            pre.commit_action(a)
            obs, _r2, te2, tr2, _ = raw.step(a)
            f_eval = pre.process(obs)
            assert (te, tr) == (te2, tr2)
            if te or tr:
                break
        assert all(np.array_equal(f_train[k], f_eval[k]) for k in FEATURE_SPECS)
    finally:
        tenv.close()
        raw.close()


def test_mods_not_implemented():
    env = make_train_env(EnvSpec("two_rooms", "maps/two_rooms.wad", "MAP01", 2100, mods=("fast_monsters",)), seed=0)
    try:
        assert not env.vanilla
        with pytest.raises(NotImplementedError):
            env.reset()
    finally:
        env.close()


def test_sampler_specs_feed_factory():
    s = MapSampler(["maps/splits/smoke.json"], rng=np.random.default_rng(1))
    env = make_train_env(s.sample(), seed=0)
    try:
        for _ in range(3):
            feats, info = env.reset(seed=int(s.rng.integers(10_000)), spec=s.sample())
            assert info["map_id"] == env.spec.map_id and 2100 <= env.spec.episode_ticks <= 4200
    finally:
        env.close()


@pytest.mark.parametrize("seq,reversals,edges", [
    ([7] + [None] * 20, 0, 1),            # one press: the door opens, no reversal
    ([7, None, 7] + [None] * 20, 1, 2),   # second press while opening reverses it
    ([7, None] * 10, 9, 10),              # USE spam keeps bouncing the door (D11 logging)
])
def test_door_reversals_and_use_edges(seq, reversals, edges):
    env = make_train_env(spec("fx_door_exit", 400), seed=0)
    try:
        env.reset()
        for _ in range(8):  # walk up to the door (grid '#P..D...E#')
            env.step(FWD)
        out = None
        for b in seq:
            out = env.step(np.eye(9, dtype=np.int8)[b] if b is not None else NOOP)
        while not (out[2] or out[3]):
            out = env.step(NOOP)
        ep = out[4]["episode"]
        assert ep["door_reversals"] == reversals and ep["use_edges"] == edges
    finally:
        env.close()
