"""Exploration metrics plug-in (training/eval/metrics/exploration.py) on synthetic trajectories."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from training.eval import metrics as M
from training.eval.inventory import MapInventory
from training.eval.metrics.core import EpisodeTrace, core_metrics
from training.eval.metrics.exploration import (
    METRICS,
    coverage_metrics,
    objective_metrics,
    repetition_metrics,
)


def _inv(cells_total=20, doors=None, keys_present=0):
    return MapInventory(num_sectors=1, door_sectors=dict(doors or {}), keys_present=keys_present, monsters_present=0,
                        sector_area=np.zeros(1), cells_total=cells_total)


def _trace(xs, ys, angles=None, keys=None, doors=None, outcome="timeout"):
    n = len(xs)
    tr = EpisodeTrace(episode_ticks=4 * max(n, 1))
    tr.x, tr.y = [float(v) for v in xs], [float(v) for v in ys]
    tr.angle = [float(a) for a in (angles if angles is not None else [0.0] * n)]
    tr.health, tr.kills, tr.damage_taken = [100.0] * n, [0] * n, [0.0] * n
    tr.keys = keys if keys is not None else [[0.0] * 6 for _ in range(n)]
    tr.ammo = [[50.0, 0.0, 0.0, 0.0] for _ in range(n)]
    tr.door_opening = doors if doors is not None else [[] for _ in range(n)]
    tr.actions = [[]] + [[0] * 9 for _ in range(n - 1)]
    tr.env_reward = [0.0] * n
    tr.outcome, tr.steps = outcome, n - 1
    return tr


def test_straight_walk_matches_core_coverage():
    xs = [32 + 64 * t for t in range(5)]  # cells 0..4
    tr, inv = _trace(xs, [32] * 5), _inv(cells_total=9)
    m, c = coverage_metrics(tr, inv), core_metrics(tr, inv)
    assert m["expl.coverage_cells"] == c["core.cells_new"] == 4
    assert m["expl.coverage_frac"] == pytest.approx(c["core.cell_coverage"]) == pytest.approx(0.5)
    assert m["expl.cell_entries"] == 4 and m["expl.revisit_ratio"] == 0.0
    assert m["expl.cells_per_100_steps"] == pytest.approx(100.0)
    assert m["expl.coverage_auc"] == pytest.approx(np.mean([1, 2, 3, 4]) / 8)


def test_back_and_forth_is_revisits_not_coverage():
    xs = [32, 96, 32, 96, 32]  # cells 0,1,0,1,0
    m = coverage_metrics(_trace(xs, [32] * 5), _inv())
    assert m["expl.coverage_cells"] == 1
    assert m["expl.cell_entries"] == 4
    assert m["expl.revisit_ratio"] == pytest.approx(0.75)  # 3 of 4 entries go into an already-seen cell
    assert m["expl.long_revisit_ratio"] == 0.0  # each return is 2 steps after leaving: local wobble


def test_long_revisit_counts_returns_after_a_long_absence():
    # spawn cell 0, walk 1..4 slowly (10 steps per cell), then straight back to cell 0
    xs = [32.0] + [32 + 64 * c for c in (1, 2, 3, 4) for _ in range(10)] + [32 + 64 * c for c in (3, 2, 1, 0)]
    m = coverage_metrics(_trace(xs, [32] * len(xs)), _inv())
    assert m["expl.cell_entries"] == 8 and m["expl.revisit_ratio"] == pytest.approx(4 / 8)
    # cells 0..4 last occupied at rows 0, 10, 20, 30; re-entered at rows 41..44, i.e. 11, 22, 33, 44 steps
    # after leaving them: the returns into cell 1 and cell 0 (> 32) are long
    assert m["expl.long_revisit_ratio"] == pytest.approx(2 / 8)


def test_spinning_in_place():
    n = 41
    angles = [(45.0 * i) % 360 for i in range(n)]  # one octant per step
    tr = _trace([100.0] * n, [100.0] * n, angles=angles)
    cov, rep = coverage_metrics(tr, _inv()), repetition_metrics(tr, _inv())
    assert cov["expl.coverage_cells"] == 0 and cov["expl.coverage_frac"] == 0.0
    assert math.isnan(cov["expl.revisit_ratio"])  # never changed cell
    assert rep["expl.still_steps_frac"] == 1.0 and rep["expl.spin_steps_frac"] == 1.0
    assert rep["expl.repeated_state_frac"] == pytest.approx((40 - 7) / 40)  # 40 state changes, 8 states, then repeats
    assert rep["expl.stuck_steps"] == 9 and rep["expl.longest_stuck_run"] == 9  # steps 32..40 (W = 32)


def test_standing_still_is_still_not_spin_and_moving_is_neither():
    rep = repetition_metrics(_trace([5.0] * 10, [5.0] * 10), _inv())
    assert rep["expl.still_steps_frac"] == 1.0 and rep["expl.spin_steps_frac"] == 0.0
    assert math.isnan(rep["expl.repeated_state_frac"])  # no state change at all: dwell is still/stuck, not repetition
    rep = repetition_metrics(_trace([10.0 * t for t in range(10)], [0.0] * 10), _inv())
    assert rep["expl.still_steps_frac"] == 0.0 and rep["expl.stuck_steps"] == 0
    assert rep["expl.repeated_state_frac"] == 0.0  # straight line: every state change is new


def test_slow_purposeful_walk_is_not_repetition():
    xs = [32 + 64 * c for c in range(6) for _ in range(10)]  # 10 steps of dwell in each cell
    rep = repetition_metrics(_trace(xs, [32] * len(xs)), _inv())
    assert rep["expl.repeated_state_frac"] == 0.0


def test_door_bounce_reversal_keys_exit_and_objectives():
    # one ordinary door (sector 3): bounced at 48 (never passable), then opened fully and closed again
    o = [0, 16, 32, 48, 32, 16, 0, 16, 64, 128, 128, 64, 0]
    n = len(o)
    keys = [[0.0] * 6 for _ in range(n)]
    for t in range(3, n):
        keys[t][1] = 1.0  # one key picked at step 3
    tr = _trace([32.0] * n, [32.0] * n, keys=keys, doors=[[float(v)] for v in o], outcome="exit")
    inv = _inv(doors={3: 0}, keys_present=2)
    m = objective_metrics(tr, inv)
    assert m["expl.door_reversals"] == 2 and m["expl.door_bounces"] == 1
    assert m["expl.first_door_step"] == 8.0 and m["expl.doors_reached_frac"] == 1.0
    assert m["expl.first_key_step"] == 3.0 and m["expl.keys_reached_frac"] == 0.5
    assert m["expl.exit_reached"] == 1 and m["expl.exit_step"] == float(n - 1)
    assert (m["expl.objectives_reached"], m["expl.objectives_total"]) == (3, 4)
    assert m["expl.objectives_frac"] == pytest.approx(0.75)
    assert core_metrics(tr, inv)["core.doors_opened"] == 1  # same "opened" rule as core


def test_spawn_keys_and_open_doors_do_not_count():
    n = 4
    keys = [[1.0, 0, 0, 0, 0, 0] for _ in range(n)]  # key held at spawn
    tr = _trace([0.0] * n, [0.0] * n, keys=keys, doors=[[128.0]] * n)  # door open at spawn
    m = objective_metrics(tr, _inv(doors={1: 0}, keys_present=1))
    assert math.isnan(m["expl.first_key_step"]) and m["expl.keys_reached_frac"] == 0.0
    assert math.isnan(m["expl.doors_reached_frac"]) and math.isnan(m["expl.first_door_step"])
    assert m["expl.objectives_total"] == 2 and m["expl.exit_reached"] == 0 and math.isnan(m["expl.exit_step"])


@pytest.mark.parametrize("n", [0, 1, 2])
def test_degenerate_traces_do_not_raise(n):
    tr = _trace([0.0] * n, [0.0] * n) if n else EpisodeTrace(episode_ticks=10)
    for fn in METRICS:
        out = fn(tr, _inv(cells_total=1))
        assert all(k.startswith("expl.") for k in out)


def test_registered_with_run_all_and_values_are_json_scalars():
    assert M.METRICS_BY_MODULE["exploration"] == list(METRICS)
    xs = [32 + 64 * (t % 5) for t in range(60)]
    tr = _trace(xs, [32] * 60, angles=[(7.0 * t) % 360 for t in range(60)], doors=[[0.0]] * 60)
    out = M.run_all(tr, _inv(doors={2: 129}, keys_present=1))
    expl = {k: v for k, v in out.items() if k.startswith("expl.")}
    assert len(expl) == 23
    for k, v in expl.items():
        assert type(v) in (int, float), (k, type(v))
    json.dumps(out)
