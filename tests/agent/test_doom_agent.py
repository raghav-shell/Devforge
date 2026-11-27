"""DoomAgent (submission/agent.py): the evaluated entry point.

Covers D12 (catch and keep playing), D12a (preprocessor stays consistent after an error), strict mode,
reset semantics, and a real in-process episode with zero caught errors.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
MAP = str(ROOT / "maps" / "two_rooms.wad")


def _load_agent_module():
    spec = importlib.util.spec_from_file_location("submission_agent_under_test", ROOT / "submission" / "agent.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def agent_mod():
    return _load_agent_module()


@pytest.fixture
def env():
    from doom_env import make_env

    e = make_env(MAP, "MAP01", episode_ticks=2100, render_buffers=True)
    yield e
    e.close()


def _make(agent_mod, env, monkeypatch, strict: bool, **envvars):
    monkeypatch.setenv("DOOM_AGENT_STRICT", "1" if strict else "0")
    for k, v in envvars.items():
        monkeypatch.setenv(k, v)
    return agent_mod.DoomAgent(env.observation_space, env.action_space, "cpu")


def _valid(a) -> bool:
    a = np.asarray(a)
    return a.shape == (9,) and a.dtype == np.int8 and set(np.unique(a)) <= {0, 1}


def test_real_episode_strict_no_errors(agent_mod, env, monkeypatch):
    agent = _make(agent_mod, env, monkeypatch, strict=True)
    agent.reset()
    obs, _ = env.reset(seed=0)
    steps = 0
    while True:
        a = agent.act(obs)
        assert _valid(a)
        obs, _r, term, trunc, _ = env.step(a)
        steps += 1
        if term or trunc:
            break
    assert agent.errors == 0
    assert steps > 0


def test_strict_mode_raises_on_bad_observation(agent_mod, env, monkeypatch):
    agent = _make(agent_mod, env, monkeypatch, strict=True)
    agent.reset()
    obs, _ = env.reset(seed=0)
    bad = {k: v for k, v in obs.items() if k != "rays"}
    with pytest.raises(Exception):
        agent.act(bad)


def test_non_strict_returns_noop_and_recovers(agent_mod, env, monkeypatch, capsys):
    agent = _make(agent_mod, env, monkeypatch, strict=False)
    agent.reset()
    obs, _ = env.reset(seed=0)
    bad = {k: v for k, v in obs.items() if k != "rays"}
    a = agent.act(bad)
    assert _valid(a) and not a.any()
    assert agent.errors == 1
    assert "act() failed" in capsys.readouterr().err
    # D12a: the very next good observation must work (preprocessor not stuck "awaiting action")
    for _ in range(20):
        a = agent.act(obs)
        assert _valid(a)
        obs, *_ = env.step(a)
    assert agent.errors == 1


def test_error_after_process_keeps_preprocessor_consistent(agent_mod, env, monkeypatch):
    """Fail *inside* the policy step (after pre.process): D12a must commit a pending action."""
    agent = _make(agent_mod, env, monkeypatch, strict=False)
    agent.reset()
    obs, _ = env.reset(seed=0)
    real_step = agent.policy.step
    calls = {"n": 0}

    def flaky(*args, **kw):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("injected")
        return real_step(*args, **kw)

    monkeypatch.setattr(agent.policy, "step", flaky)
    for _ in range(10):
        a = agent.act(obs)
        assert _valid(a)
        obs, *_ = env.step(a)
    assert agent.errors == 1


def test_reset_gives_reproducible_per_map_streams(agent_mod, env, monkeypatch):
    """Same seed + same observations -> same actions, for the first map of two fresh agents."""
    seqs = []
    for _ in range(2):
        agent = _make(agent_mod, env, monkeypatch, strict=True, DOOM_AGENT_SEED="7")
        agent.reset()
        obs, _ = env.reset(seed=3)
        acts = []
        for _ in range(60):
            a = agent.act(obs)
            acts.append(a.copy())
            obs, _r, term, trunc, _ = env.step(a)
            if term or trunc:  # a trained agent can finish two_rooms inside 60 steps
                break
        seqs.append(np.stack(acts))
    assert np.array_equal(seqs[0], seqs[1])


def test_reset_clears_recurrent_state(agent_mod, env, monkeypatch):
    agent = _make(agent_mod, env, monkeypatch, strict=True)
    agent.reset()
    obs, _ = env.reset(seed=0)
    for _ in range(10):
        obs, *_ = env.step(agent.act(obs))
    assert float(agent.state.abs().sum()) > 0
    agent.reset()
    assert float(agent.state.abs().sum()) == 0.0
    assert agent.start is True


def test_invalid_inference_env_var_falls_back(agent_mod, env, monkeypatch, capsys):
    agent = _make(agent_mod, env, monkeypatch, strict=True, DOOM_AGENT_INFERENCE="bogus")
    assert agent.mode in ("sample", "greedy")
    assert "invalid DOOM_AGENT_INFERENCE" in capsys.readouterr().err


def test_agent_does_not_import_training_or_privileged(agent_mod):
    src = (ROOT / "submission" / "agent.py").read_text()
    assert "privileged" not in src.replace("privileged game state", "")
    assert "import training" not in src and "from training" not in src
