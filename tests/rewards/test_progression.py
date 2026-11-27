"""Progression reward terms (training/rewards/progression.py): unit tests plus a real-engine walk.

The real-engine test drives a generated keys map with the TEST-ONLY scripted walker
(tests/env/env_walker.py) and checks that each event pays exactly once (docs/DECISIONS.md D10).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))  # env_walker (test-only)

from training.rewards.base import RewardComposer, StepContext  # noqa: E402
from training.rewards.progression import (  # noqa: E402
    PLAYER_HEIGHT,
    Death,
    DoorOpen,
    ExitReached,
    KeyPickup,
    LockedDoorOpen,
)

NOOP = np.zeros(9, np.int8)


def _priv(keys=(0,) * 6, floor=(0.0, 0.0, 0.0), ceil=(128.0, 0.0, 0.0), dead=False, exit_reached=False):
    return {"keys": np.asarray(keys, np.float32), "sector_floor": np.asarray(floor, np.float64),
            "sector_ceiling": np.asarray(ceil, np.float64), "dead": dead, "exit_reached": exit_reached}


def _ctx(prev, cur, *, terminated=False, step=1):
    return StepContext(priv_prev=prev, priv=cur, obs={}, action_mb=NOOP, terminated=terminated, truncated=False,
                       env_reward=0.0, info={}, step=step)


INV = {"door_sectors": {1: 0, 2: 129}}  # sector 1 normal door, sector 2 locked door


def test_key_pays_once_and_not_for_spawn_keys():
    t = KeyPickup()
    p0 = _priv(keys=(1, 0, 0, 0, 0, 0))  # blue card held at spawn
    t.reset(p0, {}, INV, None)
    assert t(_ctx(p0, p0)) == 0.0
    p1 = _priv(keys=(1, 1, 0, 0, 0, 0))
    assert t(_ctx(p0, p1)) == 1.0
    assert t(_ctx(p1, p1)) == 0.0
    p2 = _priv(keys=(1, 0, 0, 0, 0, 0))  # flag drops (should not happen, but must not re-pay)
    assert t(_ctx(p1, p2)) == 0.0
    assert t(_ctx(p2, p1)) == 0.0
    t.reset(p0, {}, INV, None)  # next episode: ledger cleared
    assert t(_ctx(p0, p1)) == 1.0


@pytest.mark.parametrize("cls,sector", [(DoorOpen, 1), (LockedDoorOpen, 2)])
def test_door_pays_once_per_kind_even_if_it_recloses(cls, sector):
    t = cls()
    closed = _priv()
    t.reset(closed, {}, INV, None)
    ceil_open = [128.0, 0.0, 0.0]
    ceil_open[sector] = PLAYER_HEIGHT
    opened = _priv(ceil=ceil_open)
    other = 2 if sector == 1 else 1
    ceil_other = [128.0, 0.0, 0.0]
    ceil_other[other] = 100.0
    assert t(_ctx(closed, _priv(ceil=ceil_other))) == 0.0  # the other kind of door does not count
    assert t(_ctx(closed, _priv(ceil=[128.0, 40.0, 40.0]))) == 0.0  # half open
    assert t(_ctx(closed, opened)) == 1.0
    assert t(_ctx(opened, closed)) == 0.0  # re-closes
    assert t(_ctx(closed, opened)) == 0.0  # re-opens: no second payment


def test_door_already_open_at_spawn_never_pays():
    t = DoorOpen()
    open0 = _priv(ceil=[128.0, 64.0, 0.0])
    t.reset(open0, {}, INV, None)
    assert t(_ctx(open0, open0, step=0)) == 0.0


def test_exit_and_death_pay_once_and_timeout_is_not_death():
    e, d = ExitReached(), Death()
    p = _priv()
    for t in (e, d):
        t.reset(p, {}, INV, None)
    assert e(_ctx(p, _priv(exit_reached=True), terminated=True)) == 1.0
    assert e(_ctx(p, _priv(exit_reached=True), terminated=True)) == 0.0
    assert d(_ctx(p, _priv(dead=False), terminated=False)) == 0.0  # timeout/truncation is not death
    assert d(_ctx(p, _priv(dead=True), terminated=True)) == 1.0
    assert d(_ctx(p, _priv(dead=True), terminated=True)) == 0.0


def test_terms_load_through_the_composer():
    comp = RewardComposer.from_config({"terms": [
        {"name": "key_pickup", "weight": 3.0}, {"name": "door_open", "weight": 1.0},
        {"name": "locked_door_open", "weight": 3.0}, {"name": "exit_reached", "weight": 10.0},
        {"name": "death", "weight": -3.0}]})
    assert set(comp.names) == {"key_pickup", "door_open", "locked_door_open", "exit_reached", "death"}


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_real_walk_on_generated_keys_map_pays_each_event_once(seed, tmp_path):
    from doom_env import make_env
    from env_walker import walk_to_exit
    from training.envs.inventory import map_inventory
    from training.envs.mapgen import generate

    gm = generate("keys", seed)
    wad, _ = gm.write(tmp_path)
    env = make_env(str(wad), gm.map_name, episode_ticks=30000, render_buffers=False)
    try:
        env.reset(seed=0)
        inv = map_inventory(env)
        terms = [KeyPickup(), DoorOpen(), LockedDoorOpen(), ExitReached(), Death()]
        p0 = env.privileged_state()
        for t in terms:
            t.reset(p0, {}, inv, None)
        totals = {t.name: 0.0 for t in terms}

        class Recorder:  # proxies the env, scoring every step the walker takes
            def __init__(self, inner):
                self._inner, self._prev, self._n = inner, p0, 0

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def step(self, action):
                out = self._inner.step(action)
                cur = self._inner.privileged_state()
                ctx = StepContext(priv_prev=self._prev, priv=cur, obs=out[0], action_mb=np.asarray(action, np.int8),
                                  terminated=bool(out[2]), truncated=bool(out[3]), env_reward=float(out[1]),
                                  info=out[4], step=self._n)
                for t in terms:
                    totals[t.name] += t(ctx)
                self._prev, self._n = cur, self._n + 1
                return out

        r = walk_to_exit(Recorder(env), gm)
    finally:
        env.close()
    assert r["exit_reached"], r
    assert totals["key_pickup"] == gm.meta["num_keys"]
    assert totals["locked_door_open"] == gm.meta["num_locked_doors"]
    assert totals["door_open"] <= gm.meta["num_doors"]
    assert totals["exit_reached"] == 1.0
    assert totals["death"] == 0.0


def test_door_reversal_counts_use_edge_on_moving_door_only():
    from training.rewards.progression import DoorReversal

    t = DoorReversal()
    inv = {"door_sectors": {1: 0}}
    use, noop = np.zeros(9, np.int8), np.zeros(9, np.int8)
    use[7] = 1

    def st(c1):
        return _priv(ceil=(128.0, c1, 0.0))

    def step(prev, cur, a):
        return t(StepContext(priv_prev=prev, priv=cur, obs={}, action_mb=a, terminated=False, truncated=False,
                             env_reward=0.0, info={}, step=1))

    t.reset(st(0.0), {}, inv, None)
    assert step(st(0.0), st(8.0), noop) == 0.0    # door starts rising by itself
    assert step(st(8.0), st(4.0), use) == 1.0     # USE edge while rising -> it now falls: reversal
    assert step(st(4.0), st(0.0), use) == 0.0     # USE held (no edge): nothing
    assert step(st(0.0), st(0.0), noop) == 0.0
    assert step(st(0.0), st(8.0), use) == 0.0     # edge on a door at rest: opening it is fine
    assert step(st(8.0), st(16.0), noop) == 0.0
    assert step(st(16.0), st(16.0), use) == 0.0   # edge while rising, pauses this step...
    assert step(st(16.0), st(8.0), noop) == 1.0   # ...and flips one step late: counted
    t.reset(st(0.0), {}, {"door_sectors": {}}, None)
    assert step(st(0.0), st(8.0), use) == 0.0     # no doors on the map
