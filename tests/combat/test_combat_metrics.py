"""Combat metrics plug-in (training/eval/metrics/combat.py): unit tests on hand-built traces."""

from __future__ import annotations

import json
import math

import numpy as np

from training.eval import metrics as M
from training.eval.inventory import MapInventory
from training.eval.metrics.combat import METRICS, combat_metrics
from training.eval.metrics.core import EpisodeTrace


def _inv(monsters=4):
    return MapInventory(num_sectors=1, door_sectors={}, keys_present=0, monsters_present=monsters,
                        sector_area=np.ones(1), cells_total=4)


def _trace(kills, dmg, health, ammo, actions, outcome, ticks=2100):
    tr = EpisodeTrace(episode_ticks=ticks)
    n = len(kills)
    tr.x = [0.0] * n; tr.y = [0.0] * n; tr.angle = [0.0] * n
    tr.kills, tr.damage_taken, tr.health, tr.ammo = list(kills), list(dmg), list(health), [list(a) for a in ammo]
    tr.keys = [[0.0] * 6] * n
    tr.door_opening = [[]] * n
    tr.actions = [[]] + [list(a) for a in actions]
    tr.env_reward = [0.0] * n
    tr.outcome, tr.steps = outcome, n - 1
    return tr


def _act(attack=0, switch=0):
    a = [0] * 9
    a[6], a[8] = attack, switch
    return a


def test_registered_with_prefix_and_json_scalars():
    assert M.METRICS_BY_MODULE["combat"] == METRICS == [combat_metrics]
    tr = _trace([0, 0], [0.0, 0.0], [100.0, 100.0], [[50, 0, 0, 0]] * 2, [_act()], "timeout")
    m = combat_metrics(tr, _inv())
    assert all(k.startswith("combat.") for k in m)
    assert all(type(v) in (int, float, str, bool) for v in m.values()), {k: type(v) for k, v in m.items()}
    json.dumps(m)
    assert M.run_all(tr, _inv())["combat.kills"] == 0  # no collisions with core / exploration keys


def test_death_episode():
    kills = [0, 0, 1, 1, 2]
    dmg = [0.0, 10.0, 10.0, 50.0, 110.0]
    health = [100.0, 90.0, 115.0, 75.0, -10.0]  # +25 medikit at row 2
    ammo = [[50, 0, 0, 0], [49, 0, 0, 0], [47, 0, 0, 0], [47, 8, 0, 0], [46, 8, 0, 0]]
    acts = [_act(1), _act(1, 1), _act(0, 1), _act(1)]
    m = combat_metrics(_trace(kills, dmg, health, ammo, acts, "death"), _inv(4))
    assert m["combat.kills"] == 2 and m["combat.kill_frac"] == 0.5 and m["combat.cleared"] == 0
    assert m["combat.damage_taken"] == 110.0
    assert m["combat.first_kill_step"] == 2 and m["combat.first_damage_step"] == 1
    assert m["combat.health_end"] == 0.0 and m["combat.health_min"] == 0.0
    assert m["combat.health_gained"] == 25.0
    assert m["combat.died"] == 1 and m["combat.survived"] == 0
    assert m["combat.death_time_frac"] == 4 * 4 / 2100
    assert m["combat.ammo_spent"] == 4.0 and m["combat.ammo_spent_bullets"] == 4.0
    assert m["combat.ammo_spent_shells"] == 0.0  # pickup is not negative spending
    assert m["combat.kills_per_100_ammo"] == 50.0
    assert m["combat.attack_steps"] == 3 and m["combat.attack_rate"] == 0.75
    assert m["combat.kills_per_100_attacks"] == 100.0 * 2 / 3
    assert m["combat.weapon_switch_presses"] == 1  # held over two steps = one edge
    assert m["combat.kills_per_min"] == 2 / (16 / 35 / 60)


def test_exit_row_reset_is_ignored():
    # terminal row after an exit reads post-level-change values (kills/damage 0, health 100)
    kills, dmg = [0, 3, 3, 0], [0.0, 20.0, 30.0, 0.0]
    health = [100.0, 80.0, 70.0, 100.0]
    ammo = [[50, 0, 0, 0], [45, 0, 0, 0], [44, 0, 0, 0], [44, 0, 0, 0]]
    m = combat_metrics(_trace(kills, dmg, health, ammo, [_act(1)] * 3, "exit"), _inv(3))
    assert m["combat.kills"] == 3 and m["combat.cleared"] == 1 and m["combat.kill_frac"] == 1.0
    assert m["combat.damage_taken"] == 30.0
    assert m["combat.health_end"] == 70.0 and m["combat.health_min"] == 70.0 and m["combat.health_gained"] == 0.0
    assert m["combat.survived"] == 1 and math.isnan(m["combat.death_time_frac"])


def test_degenerate_traces_no_monsters_no_ammo_empty():
    tr = _trace([0], [0.0], [100.0], [[0, 0, 0, 0]], [], "timeout")
    m = combat_metrics(tr, _inv(0))
    assert m["combat.kills"] == 0 and math.isnan(m["combat.kill_frac"]) and m["combat.cleared"] == 0
    assert math.isnan(m["combat.kills_per_100_ammo"]) and math.isnan(m["combat.kills_per_100_attacks"])
    assert math.isnan(m["combat.kills_per_min"])  # zero steps
    assert m["combat.no_ammo_steps_frac"] == 1.0 and m["combat.weapon_switch_presses"] == 0
    m = combat_metrics(EpisodeTrace(episode_ticks=10), _inv(0))  # completely empty trace
    assert m["combat.kills"] == 0 and m["combat.damage_taken"] == 0.0 and m["combat.health_end"] == 0.0
