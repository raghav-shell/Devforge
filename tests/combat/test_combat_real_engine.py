"""Real engine: combat terms and metrics on generated combat-stage maps agree with privileged deltas.

Policy: seeded random buttons with ATTACK held about 70% of the time (TEST-ONLY driver, not an agent).
Checks, per episode:
- sum of `kill` values == max_t kills_t - kills_0, sum of `damage_taken` == max_t dmg_t - dmg_0;
- per-step `damage_taken` value == health lost on every non-terminal step with no health gain (MEASURED:
  armor-absorbed damage is not counted);
- the combat metrics computed from an EpisodeTrace recorded like training/eval/harness.py agree.

PINNED to the current generator output (training/envs/mapgen GENERATOR_VERSION 1): `damage > 0` on these
seeds, the kill on seed 0, and no same-step health pickup coinciding with damage (which would break the
damage == HP-lost equality). After a generator bump, re-check these facts before reading a failure as a
reward bug.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.eval.metrics.combat import combat_metrics
from training.eval.metrics.core import EpisodeTrace
from training.rewards.base import RewardComposer, StepContext

TERMS = [{"name": "kill", "weight": 0.5}, {"name": "damage_taken", "weight": -0.01}, {"name": "death", "weight": -3.0}]


def _episode(map_seed: int, tmp_path, policy_seed: int):
    from doom_env import make_env

    from training.envs.mapgen import generate
    from training.eval.inventory import map_inventory

    gm = generate("combat", map_seed)
    wad, _ = gm.write(tmp_path)
    env = make_env(str(wad), gm.map_name, episode_ticks=2100, render_buffers=False)
    rng = np.random.default_rng(policy_seed)
    comp = RewardComposer.from_config({"terms": TERMS})
    rows, vals = [], []
    try:
        obs, _ = env.reset(seed=map_seed)
        inv = map_inventory(env)
        door_ids = sorted(inv.door_sectors)
        p = env.privileged_state()
        comp.reset(p, obs, {"monsters": inv.monsters_present}, None)
        tr = EpisodeTrace(episode_ticks=2100)
        tr.record(p, obs, door_ids, None, 0.0)
        rows.append(p)
        n = 0
        while True:
            a = (rng.random(9) < 0.3).astype(np.int8)
            a[6] = int(rng.random() < 0.7)
            obs, r, term, trunc, info = env.step(a)
            q = env.privileged_state()
            total, per = comp(StepContext(priv_prev=p, priv=q, obs=obs, action_mb=a, terminated=bool(term),
                                          truncated=bool(trunc), env_reward=float(r), info=info, step=n))
            vals.append((total, per, bool(term or trunc)))
            tr.steps += 1
            tr.record(q, obs, door_ids, a, r)
            rows.append(q)
            p, n = q, n + 1
            if term or trunc:
                tr.outcome = "exit" if info["exit_reached"] else ("death" if term else "timeout")
                break
    finally:
        env.close()
    return rows, vals, tr, inv, comp.episode_stats()


@pytest.mark.parametrize("map_seed", [0, 1, 2])
def test_terms_and_metrics_equal_privileged_deltas(map_seed, tmp_path):
    rows, vals, tr, inv, st = _episode(map_seed, tmp_path, policy_seed=map_seed)
    kills = np.array([r["kills"] for r in rows], float)
    dmg = np.array([r["damage_taken"] for r in rows], float)
    hp = np.array([r["health"] for r in rows], float)
    want_k, want_d = kills.max() - kills[0], dmg.max() - dmg[0]

    assert st.raw["kill"] == want_k and st.raw["damage_taken"] == want_d
    assert st.raw["death"] == float(rows[-1]["dead"])
    assert want_d > 0  # every one of these seeds takes damage (combat maps, 3..10 monsters)
    assert all(v[1]["kill"] >= 0 and v[1]["damage_taken"] >= 0 for v in vals)
    assert st.total == pytest.approx(0.5 * want_k - 0.01 * want_d - 3.0 * st.raw["death"])
    for i, (_, per, _final) in enumerate(vals):  # damage value == HP lost when no health was picked up that step
        lost = hp[i] - hp[i + 1]
        if per["damage_taken"] > 0 and lost >= 0:
            assert per["damage_taken"] == pytest.approx(lost), (i, per, hp[i], hp[i + 1])

    m = combat_metrics(tr, inv)
    assert m["combat.kills"] == int(want_k) and m["combat.damage_taken"] == want_d
    assert m["combat.died"] == int(rows[-1]["dead"]) and m["combat.monsters_present"] == inv.monsters_present > 0
    assert m["combat.attack_steps"] == sum(int(a[6]) for a in tr.actions if a)


def test_seed0_gets_a_kill_deterministically(tmp_path):
    """MEASURED: map seed 0 with policy seed 0 kills one monster; two runs must agree exactly."""
    a = _episode(0, tmp_path / "a", policy_seed=0)
    b = _episode(0, tmp_path / "b", policy_seed=0)
    assert a[4].raw == b[4].raw
    assert a[4].raw["kill"] >= 1
