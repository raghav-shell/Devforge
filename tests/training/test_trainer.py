"""Trainer-level tests on the fake policy / fake vec env.

- epoch-1 ratio == 1 before the first optimiser step (MODULE_CONTRACTS 4.3), with negative controls;
- recurrent reset at episode_start inside minibatch sequences (whole-rollout and chunked);
- trunc_value wiring: V(final_feats, h_t_post, episode_start=False), checked against an independent replay;
- constant-reward fake env with truncation: advantage equals the analytic value;
- crashed envs are counted (never silently dropped) and never bootstrapped.
"""

import copy

import numpy as np
import pytest
import torch
from conftest import build_trainer, make_cfg

from training.ppo.trainer import RatioCheckError, features_to_torch

ENV = dict(episode_len=(5, 9, 3, 11), end_kind="alternate")


def _logp_from_update_path(tr, env_idx):
    b = tr._minibatch_tensors(np.asarray(env_idx))
    with torch.no_grad():
        out, _ = tr.policy.forward_sequence(b["obs"], b["state0"], b["episode_start"])
        lp = tr.api.dist_cls(out["logits"]).log_prob(b["action_idx"])
    return lp, b


@pytest.mark.parametrize("chunk", [0, 8, 4])
def test_epoch1_ratio_is_one(chunk):
    tr = build_trainer(make_cfg(num_envs=4, T=16, chunk=chunk), env_kwargs=ENV)
    for _ in range(3):  # several iterations so state0 is non-zero and resets fall inside chunks
        tr.collect()
        assert tr.buffer.episode_start[1:].any(), "test needs resets inside the rollout"
        s = tr.update()  # ratio_check_fail=True: raises if violated
        assert s["ppo/ratio_check_max_dev"] < 1e-5
    assert np.abs(tr.buffer.state0).sum() > 0, "state0 should carry state across rollouts"


def test_ratio_negative_controls():
    tr = build_trainer(make_cfg(num_envs=4, T=16), env_kwargs=ENV)
    tr.collect()
    tr.update()
    tr.collect()
    lp, b = _logp_from_update_path(tr, range(4))
    seq_old = b["logp"]
    assert (lp - seq_old).abs().max() < 1e-5
    buf = tr.buffer
    # (a) corrupt state0
    saved = buf.state0.copy()
    buf.state0 += 0.5
    lp_a, _ = _logp_from_update_path(tr, range(4))
    buf.state0[:] = saved
    assert (lp_a - seq_old).abs().max() > 1e-3
    # (b) shift episode_start by one step (the classic leak)
    es = buf.episode_start.copy()
    buf.episode_start[1:] = es[:-1]
    buf.episode_start[0] = False
    lp_b, _ = _logp_from_update_path(tr, range(4))
    buf.episode_start[:] = es
    assert (lp_b - seq_old).abs().max() > 1e-3
    # and the trainer's gate fires
    buf.state0 += 0.5
    with pytest.raises(RatioCheckError):
        tr.update()


def test_recurrent_reset_inside_minibatch_sequence():
    tr = build_trainer(make_cfg(num_envs=4, T=16), env_kwargs=dict(episode_len=(7,), end_kind="terminated"))
    tr.collect()
    tr.update()
    tr.collect()
    buf = tr.buffer
    e = 0
    starts = np.flatnonzero(buf.episode_start[1:, e]) + 1
    assert starts.size, "need a mid-rollout reset in env 0"
    s = int(starts[0])
    lp0, _ = _logp_from_update_path(tr, [e])
    saved_obs = {k: v.copy() for k, v in buf.obs.items()}
    saved_s0 = buf.state0.copy()
    rng = np.random.default_rng(1)
    for k, v in buf.obs.items():
        if v.dtype == np.float32:
            v[:s, e] = rng.random(v[:s, e].shape).astype(np.float32)
    buf.state0[:, e] += 1.0
    lp1, _ = _logp_from_update_path(tr, [e])
    for k in buf.obs:
        buf.obs[k][:] = saved_obs[k]
    buf.state0[:] = saved_s0
    assert torch.equal(lp0[s:], lp1[s:]), "outputs after episode_start must not depend on the past"
    assert (lp0[:s] - lp1[:s]).abs().max() > 1e-4, "outputs before the reset should change"


def test_rollout_state_carry_and_pre_mask_state0():
    tr = build_trainer(make_cfg(num_envs=4, T=16), env_kwargs=ENV)
    tr.collect()
    nxt = tr.buffer.next_state.copy()
    nxt_es = tr.buffer.next_episode_start.copy()
    tr.update()
    tr.collect()
    np.testing.assert_array_equal(tr.buffer.state0[0], nxt)  # PRE-mask, even where episode_start[0]
    np.testing.assert_array_equal(tr.buffer.episode_start[0], nxt_es)


class _Recorder:
    """Wraps a vec env and records raw StepResults."""

    def __init__(self, env):
        self.env, self.num_envs, self.results = env, env.num_envs, []

    def reset(self):
        return self.env.reset()

    def step(self, a):
        r = self.env.step(a)
        self.results.append(copy.deepcopy(r))
        return r

    def close(self):
        self.env.close()


def test_trunc_value_and_constant_reward_analytic():
    cfg = make_cfg(num_envs=2, T=12, gamma=0.9, gae_lambda=1.0, return_norm=False)
    tr = build_trainer(cfg, env_kwargs=dict(episode_len=(5,), end_kind="truncated", const_reward=1.0,
                                            forward_reward=0.0))
    tr.env = _Recorder(tr.env)
    tr.collect()
    buf = tr.buffer
    assert buf.truncated.any() and not buf.terminated.any()
    # independent replay of the rollout, one step at a time, from state0
    pol = tr.policy
    h = torch.from_numpy(buf.state0[0].copy())
    with torch.no_grad():
        for t in range(cfg.ppo.rollout_len):
            x = features_to_torch({k: v[t] for k, v in buf.obs.items()}, "cpu")
            out, h = pol.step(x, h, torch.from_numpy(buf.episode_start[t].copy()))
            np.testing.assert_allclose(out["value"].numpy(), buf.value[t], atol=1e-6)
            for i in range(cfg.ppo.num_envs):
                if buf.truncated[t, i]:
                    ff = tr.env.results[t].infos[i]["final_feats"]
                    fx = features_to_torch({k: v[None] for k, v in ff.items()}, "cpu")
                    o2, _ = pol.step(fx, h[i:i + 1], torch.zeros(1, dtype=torch.bool))
                    assert buf.trunc_value[t, i] == pytest.approx(float(o2["value"][0]), abs=1e-6)
                    assert buf.trunc_value[t, i] != 0.0
                else:
                    assert buf.trunc_value[t, i] == 0.0
    # analytic lambda=1 advantage: A_t = sum_{j=t..k} g^(j-t) * 1 + g^(k-t+1) * V_final - V_t
    g = cfg.ppo.gamma
    for i in range(cfg.ppo.num_envs):
        ends = list(np.flatnonzero(buf.truncated[:, i]))
        for t in range(cfg.ppo.rollout_len):
            k = next((e for e in ends if e >= t), None)
            if k is None:
                k_end, boot = cfg.ppo.rollout_len - 1, buf.last_value[i]
            else:
                k_end, boot = k, buf.trunc_value[k, i]
            expect = sum(g ** (j - t) for j in range(t, k_end + 1)) + g ** (k_end - t + 1) * boot - buf.value[t, i]
            assert buf.advantages[t, i] == pytest.approx(expect, rel=1e-4, abs=1e-5)


def test_crashed_envs_counted_not_dropped():
    crash = {(1, 3), (2, 5), (1, 20)}
    tr = build_trainer(make_cfg(num_envs=4, T=16), env_kwargs=dict(episode_len=(6,), crash_at=crash))
    s1 = tr.collect()
    assert s1["env/crashed"] == 2 and tr.counters["crashed"] == 2
    buf = tr.buffer
    assert buf.crashed[3, 1] and buf.crashed[5, 2] and buf.crashed.sum() == 2
    assert buf.truncated[3, 1] and buf.truncated[5, 2]
    assert buf.trunc_value[3, 1] == 0.0 and buf.trunc_value[5, 2] == 0.0
    assert buf.episode_start[4, 1] and buf.episode_start[6, 2]
    tr.update()
    s2 = tr.collect()
    assert s2["env/crashed"] == 1 and s2["env/crashed_total"] == 3 and tr.counters["crashed"] == 3


class _BadEnv(_Recorder):
    def __init__(self, env, mode):
        super().__init__(env)
        self.mode = mode

    def step(self, a):
        r = self.env.step(a)
        if r.truncated.any() or r.terminated.any():
            i = int(np.flatnonzero(r.truncated | r.terminated)[0])
            if self.mode == "drop_final":
                r.infos[i].pop("final_feats", None)
            else:
                r.episode_start[i] = False
        return r


@pytest.mark.parametrize("mode,match", [("drop_final", "final_feats"), ("bad_start", "episode_start")])
def test_contract_violations_raise(mode, match):
    tr = build_trainer(make_cfg(num_envs=2, T=16), env_kwargs=dict(episode_len=(4,), end_kind="truncated"))
    tr.env = _BadEnv(tr.env, mode)
    with pytest.raises(RuntimeError, match=match):
        tr.collect()


def test_learning_signal_smoke():
    """The fake env pays 1 for MOVE_FORWARD; PPO should raise p(forward) within a few updates."""
    cfg = make_cfg(num_envs=8, T=32, minibatches=2, epochs=3, lr=3e-3, ent_coef=0.0, ent_coef_final=0.0)
    tr = build_trainer(cfg, env_kwargs=dict(episode_len=(50,), end_kind="truncated"))
    first = tr.collect()["reward/raw_mean"]
    tr.update()
    for _ in range(14):
        s = tr.train_iteration()
    assert s["reward/raw_mean"] > first + 0.15, (first, s["reward/raw_mean"])
    assert np.isfinite(s["loss/total"])
