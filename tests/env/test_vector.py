"""SubprocVecEnv / StepResult (docs/MODULE_CONTRACTS.md §5.2): auto-reset, alignment, final_feats, determinism,
crash/restart, refusal, info layout, no privileged data in obs."""

from __future__ import annotations

import json

import numpy as np
import pytest

from doom_agent.obs import FEATURE_SPECS
from training.envs.maps import ROOT
from training.envs.vector import StepResult, SubprocVecEnv, WorkerError

SMOKE = {m["id"]: m for m in json.loads((ROOT / "maps/splits/smoke.json").read_text())["maps"]}
TERMS = {"terms": [{"name": "env_default", "weight": 1.0}, {"name": "test_full_priv", "weight": 0.5},
                   {"name": "test_step_index", "weight": 0.0}], "modules": ["env_test_terms"]}


def short_cfg(ticks=40, ids=("fx_corridor_empty", "fx_tiny_room", "two_rooms"), **kw):
    """Fixed short episodes: 40 tics = 10 decisions, so done steps happen often."""
    return {"entries": [dict(SMOKE[i], episode_ticks=ticks) for i in ids], "randomize_ticks": False, **kw}


def actions(n, rng):
    return rng.integers(0, 2, (n, 9), dtype=np.int8)


def test_reset_shapes_and_step_result():
    with SubprocVecEnv(short_cfg(), num_workers=2, envs_per_worker=2, seed_base=1, reward_cfg=TERMS) as v:
        assert v.num_envs == 4
        obs = v.reset()
        assert set(obs) == set(FEATURE_SPECS)
        for k, s in FEATURE_SPECS.items():
            assert obs[k].shape == (4, *s.shape) and obs[k].dtype == s.np_dtype
        res = v.step(np.zeros((4, 9), np.int8))
        assert isinstance(res, StepResult)
        assert res.reward.dtype == np.float32 and res.reward.shape == (4,)
        assert res.terminated.dtype == bool and res.truncated.dtype == bool and res.episode_start.dtype == bool
        assert len(res.infos) == 4
        for info in res.infos:
            assert set(info) == {"map_id", "tic", "exit_reached", "reward_terms", "env_reward"}
            assert set(info["reward_terms"]) == {"env_default", "test_full_priv", "test_step_index"}
        assert np.allclose(res.reward, 0.5)  # env_default 0 + 0.5 * test_full_priv (full privileged dict seen)


def test_auto_reset_ordering_and_episode_start_alignment():
    with SubprocVecEnv(short_cfg(), num_workers=1, envs_per_worker=3, seed_base=2, reward_cfg=TERMS) as v:
        v.reset()
        rng = np.random.default_rng(0)
        for t in range(1, 31):
            res = v.step(actions(3, rng))
            assert np.array_equal(res.episode_start, res.terminated | res.truncated)
            for i, info in enumerate(res.infos):
                done = bool(res.episode_start[i])
                assert ("final_feats" in info) == done and ("episode" in info) == done
                # StepContext.step = decision index within the episode: 0..9, restarting after auto-reset
                assert info["reward_terms"]["test_step_index"] == (t - 1) % 10
                if done:
                    ff = info["final_feats"]
                    assert set(ff) == set(FEATURE_SPECS)
                    # obs is the fresh episode: no previous action, time 0; final_feats is the terminal one
                    assert not res.obs["prev_action"][i].any() and res.obs["act_aux"][i][3] == 0.0
                    assert ff["act_aux"][3] > 0.0
                    assert info["episode"]["length"] == 10 and info["episode"]["outcome"] == "timeout"
            assert res.truncated.all() == (t % 10 == 0)


def test_no_privileged_in_obs_and_opt_in_info():
    with SubprocVecEnv(short_cfg(return_privileged=True), num_workers=1, envs_per_worker=2, seed_base=3) as v:
        obs = v.reset()
        res = v.step(np.zeros((2, 9), np.int8))
        for o in (obs, res.obs):
            assert set(o) == set(FEATURE_SPECS)
        for info in res.infos:
            assert set(info["privileged"]) == {"x", "y", "angle", "health", "kills", "damage_taken", "keys", "dead",
                                               "exit_reached"}
    with SubprocVecEnv(short_cfg(), num_workers=1, envs_per_worker=2, seed_base=3) as v:
        v.reset()
        assert all("privileged" not in i for i in v.step(np.zeros((2, 9), np.int8)).infos)


def _rollout(num_workers, epw, seed_base, steps=35, threads=True):
    rng = np.random.default_rng(42)
    out = []
    with SubprocVecEnv(short_cfg(step_threads=threads), num_workers=num_workers, envs_per_worker=epw, seed_base=seed_base, reward_cfg=TERMS) as v:
        out.append(v.reset())
        for _ in range(steps):
            r = v.step(actions(num_workers * epw, rng))
            out.append((r.obs, r.reward, r.truncated, [i.get("episode", {}).get("seed") for i in r.infos],
                        [i["map_id"] for i in r.infos]))
    return out


def _same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(_same(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)):
        return len(a) == len(b) and all(_same(x, y) for x, y in zip(a, b))
    if isinstance(a, np.ndarray):
        return np.array_equal(a, b)
    return a == b


def test_determinism_from_run_seed_and_worker_split():
    a = _rollout(2, 2, seed_base=11)
    b = _rollout(2, 2, seed_base=11)
    c = _rollout(1, 4, seed_base=11)  # same slots, different process split -> identical
    d = _rollout(2, 2, seed_base=12)
    e = _rollout(1, 4, seed_base=11, threads=False)  # threaded slot stepping == sequential, bit for bit
    assert _same(a, b) and _same(a, c) and _same(a, e)
    assert not _same(a, d)
    seeds = [s for step in a[1:] for s in step[3] if s is not None]
    assert seeds and all(0 <= s < 10_000 for s in seeds)


def test_engine_crash_is_flagged_counted_and_recovered():
    with SubprocVecEnv(short_cfg(ticks=400), num_workers=2, envs_per_worker=2, seed_base=4, reward_cfg=TERMS) as v:
        v.reset()
        v.step(np.zeros((4, 9), np.int8))
        v._debug_inject_crash(1)
        res = v.step(np.zeros((4, 9), np.int8))
        info = res.infos[1]
        assert info["crashed"] is True and "final_feats" not in info
        assert res.truncated[1] and not res.terminated[1] and res.episode_start[1] and res.reward[1] == 0.0
        assert not res.episode_start[[0, 2, 3]].any()
        assert set(res.obs) == set(FEATURE_SPECS) and not res.obs["prev_action"][1].any()
        for _ in range(3):
            res = v.step(np.zeros((4, 9), np.int8))
            assert not any(i.get("crashed") for i in res.infos)
        st = v.stats()
        assert st["crashes"] == 1 and st["crashed_steps"] == 1 and st["worker_restarts"] == 0


def _engine_alive(pgid: int) -> bool:
    import subprocess

    return subprocess.run(["pgrep", "-g", str(pgid)], capture_output=True).returncode == 0


def test_dead_worker_is_restarted_and_its_slots_flagged():
    with SubprocVecEnv(short_cfg(ticks=400), num_workers=2, envs_per_worker=2, seed_base=5, timeout=60) as v:
        v.reset()
        old_pid = v._procs[1].pid
        v._debug_kill_worker(1)
        res = v.step(np.zeros((4, 9), np.int8))
        assert [bool(i.get("crashed")) for i in res.infos] == [False, False, True, True]
        assert all(i.get("worker_restarted") for i in res.infos[2:])
        assert res.truncated[2:].all() and not res.terminated.any() and "final_feats" not in res.infos[2]
        res = v.step(np.zeros((4, 9), np.int8))
        assert not any(i.get("crashed") for i in res.infos)
        assert v.stats()["worker_restarts"] == 1
        assert not _engine_alive(old_pid)  # the dead worker's orphaned ViZDoom engines were reaped


def test_refuses_val_test_and_bad_config():
    with pytest.raises(ValueError, match="refusing"):
        SubprocVecEnv({"manifests": ["maps/splits/val.json"]}, num_workers=1, envs_per_worker=1)
    with pytest.raises(ValueError, match="refusing"):
        SubprocVecEnv({"manifests": ["maps/splits/test.json"]}, num_workers=1, envs_per_worker=1)
    with pytest.raises(ValueError):
        SubprocVecEnv(short_cfg(), num_workers=1, envs_per_worker=1, seed_base=10_000)
    with pytest.raises(ValueError, match="unknown sampler_cfg"):
        SubprocVecEnv(dict(short_cfg(), bogus=1), num_workers=1, envs_per_worker=1)


def test_worker_exception_fails_loudly():
    with pytest.raises(WorkerError, match="no_such_term"):
        SubprocVecEnv(short_cfg(), num_workers=1, envs_per_worker=1, reward_cfg=[{"name": "no_such_term", "weight": 1.0}])
    with SubprocVecEnv(short_cfg(), num_workers=1, envs_per_worker=1) as v:
        v.reset()
        with pytest.raises(ValueError):
            v.step(np.full((1, 9), 3, np.int8))


def test_episodes_per_map_reuses_engine():
    cfg = short_cfg(episodes_per_map=4, ids=("fx_corridor_empty", "fx_tiny_room"))
    with SubprocVecEnv(cfg, num_workers=1, envs_per_worker=1, seed_base=7) as v:
        v.reset()
        maps = []
        for _ in range(80):
            r = v.step(np.zeros((1, 9), np.int8))
            if r.episode_start[0]:
                maps.append(r.infos[0]["episode"]["map_id"])
        assert len(maps) == 8
        assert all(len(set(maps[i : i + 4])) == 1 for i in (0, 4))
        assert v.stats()["engine_opens"] <= 3


def test_determinism_with_real_h1_reward_terms_under_threads():
    """Threads must not expose shared state in the real reward terms: threaded == sequential, bit for bit."""
    from training.rewards.base import REWARD_TERMS, load_term_modules

    load_term_modules()
    terms = json.loads((ROOT / "training" / "configs" / "h1.json").read_text())["reward"]["terms"]
    if any(t["name"] not in REWARD_TERMS for t in terms):
        pytest.skip("not every H1 reward term is registered")

    def roll(nw, epw, threads):
        rng = np.random.default_rng(3)
        out = []
        with SubprocVecEnv({"manifests": ["maps/splits/smoke.json"], "step_threads": threads}, num_workers=nw,
                           envs_per_worker=epw, seed_base=21, reward_cfg=terms) as v:
            out.append(v.reset())
            for _ in range(60):
                r = v.step(actions(nw * epw, rng))
                out.append((r.obs, r.reward, r.truncated, [sorted(i["reward_terms"].items()) for i in r.infos]))
        return out

    assert _same(roll(1, 4, True), roll(2, 2, False))


def test_parent_sigkill_leaves_no_workers_or_engines(tmp_path):
    """A killed trainer (SIGKILL, no cleanup) must not leave vec-env workers or ViZDoom engines behind."""
    import os
    import signal
    import subprocess
    import sys
    import time

    script = tmp_path / "parent.py"
    script.write_text(
        "import json, sys, time\n"
        f"sys.path[:0] = [{str(ROOT / 'submission')!r}, {str(ROOT)!r}, {str(ROOT / 'tests' / 'env')!r}]\n"
        "from training.envs.vector import SubprocVecEnv\n"
        "from test_vector import short_cfg\n"
        "if __name__ == '__main__':\n"
        "    v = SubprocVecEnv(short_cfg(ticks=400), num_workers=2, envs_per_worker=2, seed_base=1)\n"
        "    v.reset()\n"
        "    print(' '.join(str(p.pid) for p in v._procs), flush=True)\n"
        "    time.sleep(60)\n")
    proc = subprocess.Popen([sys.executable, str(script)], cwd=ROOT, stdout=subprocess.PIPE, text=True)
    groups = [int(x) for x in proc.stdout.readline().split()]
    assert len(groups) == 2 and all(_engine_alive(g) for g in groups)
    os.kill(proc.pid, signal.SIGKILL)
    proc.wait(timeout=10)
    deadline = time.time() + 20
    while time.time() < deadline and any(_engine_alive(g) for g in groups):
        time.sleep(0.2)
    leftover = [g for g in groups if _engine_alive(g)]
    for g in leftover:  # never leak into the machine even if the assertion fails
        os.killpg(g, signal.SIGKILL)
    assert not leftover, f"worker groups survived a parent SIGKILL: {leftover}"
