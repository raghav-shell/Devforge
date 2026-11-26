"""Combat reward terms (training/rewards/combat.py): unit tests with fake privileged dicts."""

from __future__ import annotations

import numpy as np
import pytest

from training.rewards.base import REWARD_TERMS, RewardComposer, StepContext
from training.rewards.combat import AmmoSpent, DamageTaken, Kill

NOOP = np.zeros(9, np.int8)


def _priv(kills=0, dmg=0.0, health=100.0, dead=False):
    return {"kills": kills, "damage_taken": dmg, "health": health, "dead": dead}


def _ctx(prev, cur, *, obs=None, terminated=False, truncated=False, step=1):
    return StepContext(priv_prev=prev, priv=cur, obs=obs or {}, action_mb=NOOP, terminated=terminated,
                       truncated=truncated, env_reward=0.0, info={}, step=step)


def _run(term, rows):
    """Feed priv rows 1.. through `term` after reset on row 0; return per-step values."""
    term.reset(rows[0], {}, {}, None)
    return [term(_ctx(a, b)) for a, b in zip(rows[:-1], rows[1:])]


def test_names_match_h1_config_and_are_registered():
    assert REWARD_TERMS["kill"] is Kill and REWARD_TERMS["damage_taken"] is DamageTaken
    assert REWARD_TERMS["ammo_spent"] is AmmoSpent
    assert "death" in REWARD_TERMS and REWARD_TERMS["death"].__module__ == "training.rewards.progression"


def test_kill_pays_per_increment_including_multi_kill_steps():
    rows = [_priv(kills=k) for k in (0, 0, 1, 1, 3, 3)]
    assert _run(Kill(), rows) == [0.0, 1.0, 0.0, 2.0, 0.0]


def test_kill_spawn_count_is_baseline_not_paid():
    rows = [_priv(kills=k) for k in (2, 2, 3)]
    assert _run(Kill(), rows) == [0.0, 1.0]


def test_kill_robust_to_exit_row_reset_and_cannot_repay():
    # post-exit terminal row reads 0; a later re-rise (not possible in one episode, but must not repay) pays only
    # above the old maximum
    rows = [_priv(kills=k) for k in (0, 2, 0, 1, 2, 3)]
    vals = _run(Kill(), rows)
    assert vals == [2.0, 0.0, 0.0, 0.0, 1.0]
    assert sum(vals) == max(r["kills"] for r in rows) - rows[0]["kills"]
    assert all(v >= 0 for v in vals)


def test_damage_taken_is_positive_hp_and_robust_to_reset_row():
    rows = [_priv(dmg=d) for d in (0.0, 0.0, 42.0, 46.0, 46.0, 0.0, 110.0)]
    vals = _run(DamageTaken(), rows)
    assert vals == [0.0, 42.0, 4.0, 0.0, 0.0, 64.0]
    assert sum(vals) == 110.0


def test_reset_clears_high_water_mark_between_episodes():
    k, d = Kill(), DamageTaken()
    assert _run(k, [_priv(kills=0), _priv(kills=5)]) == [5.0]
    assert _run(k, [_priv(kills=0), _priv(kills=1)]) == [1.0]  # new episode: baseline is priv0, not 5
    assert _run(d, [_priv(dmg=0.0), _priv(dmg=30.0)]) == [30.0]
    assert _run(d, [_priv(dmg=0.0), _priv(dmg=10.0)]) == [10.0]


def test_terms_take_no_params():
    with pytest.raises(TypeError):
        Kill(scale=2.0)


def _obs(bullets, shells=0, rockets=0, cells=0):
    v = np.zeros(17, np.float32)
    v[4:8] = (bullets, shells, rockets, cells)
    return {"vitals": v}


def test_ammo_spent_counts_decreases_only_and_ignores_stale_terminal_obs():
    t = AmmoSpent(type_weights=(1.0, 2.0, 5.0, 0.5))
    p = _priv()
    t.reset(p, _obs(50, 8), {}, None)
    assert t(_ctx(p, p, obs=_obs(49, 8))) == 1.0
    assert t(_ctx(p, p, obs=_obs(49, 7))) == 2.0
    assert t(_ctx(p, p, obs=_obs(69, 11))) == 0.0  # pickup
    assert t(_ctx(p, p, obs=_obs(67, 11, 0, 0))) == 2.0
    assert t(_ctx(p, p, obs=_obs(0, 0), terminated=True)) == 0.0  # stale terminal obs
    with pytest.raises(ValueError):
        AmmoSpent(type_weights=(1.0, -1.0, 1.0, 1.0))


def test_composer_total_is_weighted_sum_with_h1_weights():
    comp = RewardComposer.from_config({"terms": [{"name": "kill", "weight": 0.5},
                                                 {"name": "damage_taken", "weight": -0.01},
                                                 {"name": "death", "weight": -3.0}]})
    p0 = _priv()
    comp.reset(p0, {}, {}, None)
    p1 = _priv(kills=1, dmg=20.0, health=80.0)
    total, per = comp(_ctx(p0, p1))
    assert per == {"kill": 1.0, "damage_taken": 20.0, "death": 0.0}
    assert total == pytest.approx(0.5 - 0.2)
    p2 = _priv(kills=1, dmg=105.0, health=-5.0, dead=True)
    total, per = comp(_ctx(p1, p2, terminated=True))
    assert per == {"kill": 0.0, "damage_taken": 85.0, "death": 1.0}
    assert total == pytest.approx(-0.85 - 3.0)
    st = comp.episode_stats()
    assert st.raw == {"kill": 1.0, "damage_taken": 105.0, "death": 1.0}


def test_kill_capped_at_monsters_present_at_reset():
    """Spawned monsters (pain elemental lost souls) cannot pay beyond the monsters present at reset."""
    t = Kill()
    rows = [_priv(kills=k) for k in (0, 1, 3, 4, 6)]
    t.reset(rows[0], {}, {"monsters": 3}, None)
    vals = [t(_ctx(a, b)) for a, b in zip(rows[:-1], rows[1:])]
    assert vals == [1.0, 2.0, 0.0, 0.0] and sum(vals) == 3.0
    t.reset(rows[0], {}, {"monsters": 3}, None)  # the cap budget resets per episode
    assert t(_ctx(rows[0], rows[2])) == 3.0
    off = Kill(cap_at_monsters=False)
    off.reset(rows[0], {}, {"monsters": 3}, None)
    assert [off(_ctx(a, b)) for a, b in zip(rows[:-1], rows[1:])] == [1.0, 2.0, 1.0, 2.0]
    t.reset(rows[0], {}, {"monsters": 0}, None)  # nothing to kill at reset: evaluator credits 0, so do we
    assert t(_ctx(rows[0], rows[1])) == 0.0
