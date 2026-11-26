"""Checkpoint save -> load -> bit-identical next update (MODULE_CONTRACTS 8.1, ckpt-1)."""

import numpy as np
import pytest
import torch
from conftest import build_trainer, make_cfg

from training.configs import TrainConfig
from training.ppo.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
from training.train import run

ENV = dict(episode_len=(5, 9, 3, 11), end_kind="alternate", crash_at={(2, 30)})


def _ckpt(tr):
    return build_checkpoint(tr, policy_config={"hidden_size": 16}, obs_config={}, action_space_version="1.0.0",
                            train_manifest_ids=["fake"])


def _assert_same(a, b):
    sa, sb = a.policy.state_dict(), b.policy.state_dict()
    assert sa.keys() == sb.keys()
    for k in sa:
        assert torch.equal(sa[k], sb[k]), k
    oa, ob = a.optimizer.state_dict()["state"], b.optimizer.state_dict()["state"]
    for i in oa:
        for k in oa[i]:
            assert torch.equal(torch.as_tensor(oa[i][k]), torch.as_tensor(ob[i][k])), (i, k)


def test_save_load_bit_identical_next_update(tmp_path):
    cfg = make_cfg(num_envs=4, T=16, lr_decay_frac=0.5)
    a = build_trainer(cfg, policy_seed=0, env_kwargs=ENV)
    for _ in range(2):
        a.train_iteration()
    path = save_checkpoint(tmp_path / "ckpt_000002.pt", _ckpt(a))
    sa = a.train_iteration()

    # fresh trainer, policy initialised from a DIFFERENT seed, fresh env object, polluted global RNGs
    b = build_trainer(cfg, policy_seed=123, env_kwargs=ENV)
    torch.rand(7)
    np.random.rand(3)
    ck = load_checkpoint(path)
    assert ck["format_version"] == "ckpt-1"
    for key in ("policy_state", "optimizer_state", "scheduler_state", "rng_states", "return_norm_state",
                "update", "global_step", "obs_pipeline_hash", "obs_schema_version", "train_manifest_ids",
                "train_seeds", "ppo_config", "reward_config", "env_config", "git_commit", "git_dirty",
                "policy_config", "obs_config", "action_space_version", "aux_state", "privileged_critic_state"):
        assert key in ck, key
    b.load_state_dict(ck)
    assert b.update_idx == 2 and b.global_step == 2 * 4 * 16
    sb = b.train_iteration()
    _assert_same(a, b)
    for k in ("loss/total", "ppo/approx_kl", "reward/raw_mean", "sched/lr", "sched/ent_coef", "env/crashed_total"):
        assert sa[k] == sb[k], k
    np.testing.assert_array_equal(a.buffer.advantages, b.buffer.advantages)


def test_resume_without_env_snapshot_restarts_episodes(tmp_path):
    cfg = make_cfg(num_envs=4, T=16)
    a = build_trainer(cfg, env_kwargs=ENV)
    a.train_iteration()
    ck = _ckpt(a)
    ck["env_carry"]["env_state"] = None  # like a real SubprocVecEnv
    b = build_trainer(cfg, policy_seed=5, env_kwargs=ENV)
    b.load_state_dict(ck)
    assert b.obs is None
    b.collect()
    assert b.buffer.episode_start[0].all()
    for k, v in a.policy.state_dict().items():
        assert torch.equal(v, ck["policy_state"][k])


def test_seed_guard():
    with pytest.raises(ValueError):
        make_cfg(seed=10000)


def test_cli_resume_matches_uninterrupted(tmp_path):
    def cfg():
        c = TrainConfig()
        c.ppo.num_envs, c.ppo.rollout_len, c.ppo.num_minibatches, c.ppo.epochs = 4, 16, 2, 2
        c.env.num_workers, c.env.envs_per_worker = 1, 4
        c.run.tensorboard, c.run.ckpt_every = False, 1000
        c.run.torch_threads = 1
        return c

    straight = run(cfg(), fake=True, max_updates=3, run_dir=tmp_path / "a")
    part = run(cfg(), fake=True, max_updates=2, run_dir=tmp_path / "b")
    run(cfg(), fake=True, resume=str(part), max_updates=1)
    ca = load_checkpoint(straight / "checkpoints" / "ckpt_000003.pt")
    cb = load_checkpoint(part / "checkpoints" / "ckpt_000003.pt")
    assert ca["update"] == cb["update"] == 3
    for k in ca["policy_state"]:
        assert torch.equal(ca["policy_state"][k], cb["policy_state"][k]), k
