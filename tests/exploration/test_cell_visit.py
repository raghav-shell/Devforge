"""`cell_visit` reward term (training/rewards/exploration.py): unit tests plus a real-engine walk.

The real-engine test drives generated maps with the TEST-ONLY scripted walker (tests/env/env_walker.py) and
checks that the episode sum of the term == the independently counted distinct non-spawn cells == the eval
metrics `core.cells_new` and `expl.coverage_cells` on the same trajectory (shaping and scoring agree).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "env"))  # env_walker (test-only)

from training.rewards.base import REWARD_TERMS, RewardComposer, StepContext  # noqa: E402
from training.rewards.exploration import CELL, CellVisit, cell_of  # noqa: E402

NOOP = np.zeros(9, np.int8)


def _p(x, y, angle=0.0):
    return {"x": float(x), "y": float(y), "angle": float(angle)}


def _ctx(prev, cur, step=1):
    return StepContext(priv_prev=prev, priv=cur, obs={}, action_mb=NOOP, terminated=False, truncated=False,
                       env_reward=0.0, info={}, step=step)


def _run(term, path):
    """Reset at path[0], step through path[1:]; return the per-step values."""
    term.reset(path[0], {}, {}, None)
    out = []
    for i in range(1, len(path)):
        out.append(term(_ctx(path[i - 1], path[i], step=i - 1)))
    return out


def test_cell_size_matches_eval_metric():
    from training.envs.inventory import CELL as ENV_CELL
    from training.eval.inventory import CELL as EVAL_CELL

    assert CELL == ENV_CELL == EVAL_CELL == 64.0
    assert CellVisit().cell == CELL


def test_cell_of_uses_floor_across_zero():
    assert cell_of(-1.0, -1.0) == (-1, -1)  # int() would give (0, 0) and merge four cells
    assert cell_of(0.0, 63.99) == (0, 0)
    assert cell_of(64.0, -64.0) == (1, -1)


def test_pays_once_per_new_cell_and_spawn_excluded():
    t = CellVisit()
    path = [_p(10, 10), _p(20, 10), _p(70, 10), _p(130, 10), _p(70, 10), _p(10, 10), _p(130, 10), _p(200, 10)]
    #       spawn(0,0)  same      (1,0) new   (2,0) new   (1,0) again (0,0)=spawn (2,0) again (3,0) new
    assert _run(t, path) == [0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    assert t.cells_visited == 4  # spawn + 3


def test_border_jitter_pays_once_each_side():
    t = CellVisit()
    path = [_p(63.5, 10)] + [_p(64.5 if i % 2 == 0 else 63.5, 10) for i in range(20)]
    vals = _run(t, path)
    assert sum(vals) == 1.0 and vals[0] == 1.0  # only the first crossing into (1,0)


def test_negative_coordinates():
    t = CellVisit()
    vals = _run(t, [_p(5, 5), _p(-5, 5), _p(-5, -5), _p(5, -5), _p(5, 5)])
    assert vals == [1.0, 1.0, 1.0, 0.0]


def test_spinning_and_standing_pay_zero():
    t = CellVisit()
    spin = [_p(100.0, 100.0, angle=(30.0 * i) % 360) for i in range(200)]
    assert sum(_run(t, spin)) == 0.0
    still = [_p(100.0, 100.0)] * 50
    assert sum(_run(t, still)) == 0.0


def test_resets_per_episode():
    t = CellVisit()
    path = [_p(10, 10), _p(70, 10), _p(130, 10)]
    assert sum(_run(t, path)) == 2.0
    assert sum(_run(t, path)) == 2.0  # same cells pay again in the next episode
    # a new spawn cell is excluded and the old spawn cell now pays
    assert _run(t, [_p(70, 10), _p(10, 10), _p(70, 10)]) == [1.0, 0.0]


def test_non_finite_position_pays_zero():
    t = CellVisit()
    assert _run(t, [_p(10, 10), _p(math.nan, 10), _p(math.inf, 10)]) == [0.0, 0.0]


def test_params_validated():
    assert CellVisit(cell=32).cell == 32.0
    with pytest.raises(TypeError):
        CellVisit(bogus=1)
    for bad in (0, -64, math.nan, math.inf):
        with pytest.raises(ValueError):
            CellVisit(cell=bad)
    with pytest.raises(TypeError):
        CellVisit(cell="big")


def test_registered_and_contract_config_loads_through_composer():
    comp = RewardComposer.from_config({"terms": [{"name": "cell_visit", "weight": 0.05, "params": {"cell": 64}}]})
    assert REWARD_TERMS["cell_visit"] is CellVisit
    assert comp.names == ["cell_visit"] and comp.weights == {"cell_visit": 0.05}
    path = [_p(10, 10), _p(70, 10), _p(70, 10), _p(130, 10)]
    comp.reset(path[0], {}, {}, None)
    totals = [comp(_ctx(path[i - 1], path[i]))[0] for i in range(1, len(path))]
    assert totals == pytest.approx([0.05, 0.0, 0.05])
    st = comp.episode_stats()
    assert st.raw == {"cell_visit": 2.0} and st.nonzero_steps == {"cell_visit": 2}
    assert st.total == pytest.approx(0.1)


@pytest.mark.parametrize("stage,seed", [("locomotion", 0), ("keys", 3)])
def test_real_walk_cell_reward_equals_distinct_cells_and_eval_metric(stage, seed, tmp_path):
    from doom_env import make_env
    from env_walker import walk_to_exit
    from training.envs.mapgen import generate
    from training.eval.inventory import map_inventory
    from training.eval.metrics.core import EpisodeTrace, core_metrics
    from training.eval.metrics.exploration import coverage_metrics

    gm = generate(stage, seed)
    wad, _ = gm.write(tmp_path)
    ticks = 30000
    env = make_env(str(wad), gm.map_name, episode_ticks=ticks, render_buffers=False)
    term = CellVisit()
    trace = EpisodeTrace(episode_ticks=ticks)
    try:
        obs0, _ = env.reset(seed=0)
        inv = map_inventory(env)
        door_ids = sorted(inv.door_sectors)
        p0 = env.privileged_state()
        term.reset(p0, obs0, {}, None)
        trace.record(p0, obs0, door_ids, None, 0.0)
        state = {"prev": p0, "n": 0, "sum": 0.0, "xy": [(p0["x"], p0["y"])], "done": False}

        class Recorder:  # proxies the env, scoring every step the walker takes
            def __init__(self, inner):
                self._inner = inner

            def __getattr__(self, name):
                return getattr(self._inner, name)

            def step(self, action):
                out = self._inner.step(action)
                cur = self._inner.privileged_state()
                ctx = StepContext(priv_prev=state["prev"], priv=cur, obs=out[0], action_mb=np.asarray(action, np.int8),
                                  terminated=bool(out[2]), truncated=bool(out[3]), env_reward=float(out[1]),
                                  info=out[4], step=state["n"])
                state["sum"] += term(ctx)
                state["xy"].append((cur["x"], cur["y"]))
                trace.steps += 1
                trace.record(cur, out[0], door_ids, action, float(out[1]))
                if out[2] or out[3]:
                    trace.outcome = "exit" if out[4]["exit_reached"] else ("death" if out[2] else "timeout")
                state["prev"], state["n"] = cur, state["n"] + 1
                return out

        r = walk_to_exit(Recorder(env), gm)
    finally:
        env.close()
    assert r["exit_reached"], r
    distinct = {(math.floor(x / 64.0), math.floor(y / 64.0)) for x, y in state["xy"]}
    spawn = (math.floor(state["xy"][0][0] / 64.0), math.floor(state["xy"][0][1] / 64.0))
    expected = len(distinct - {spawn})
    assert expected > 5  # the walk really explored
    m_core = core_metrics(trace, inv)
    m_expl = coverage_metrics(trace, inv)
    assert state["sum"] == expected == m_core["core.cells_new"] == m_expl["expl.coverage_cells"]
    assert m_expl["expl.coverage_frac"] == pytest.approx(m_core["core.cell_coverage"])
