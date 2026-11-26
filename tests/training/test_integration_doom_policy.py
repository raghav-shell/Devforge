"""PPO engine on the REAL DoomPolicy (B2) with the fake vector env (A3 not landed).

Checks the epoch-1 ratio gate (whole-rollout and chunk-32), a real update, checkpoint resume, and the
export through the default DoomPolicy factory with a fresh-process strict reload.
"""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import ROOT, make_cfg

pytestmark = pytest.mark.skipif(importlib.util.find_spec("doom_agent.models.policy") is None,
                                reason="B2 doom_agent.models not landed")


def _build(cfg, policy_seed=0):
    from doom_agent.models.config import PolicyConfig
    from doom_agent.models.policy import DoomPolicy

    from training.ppo.adapters import get_action_api
    from training.ppo.testing import FakeVecEnv, obs_specs
    from training.ppo.trainer import PPOTrainer, seed_everything

    seed_everything(policy_seed)
    pcfg = PolicyConfig()
    pol = DoomPolicy(pcfg)
    env = FakeVecEnv(cfg.ppo.num_envs, episode_len=(20, 45, 9, 70), end_kind="alternate", crash_at={(1, 40)})
    seed_everything(cfg.run.seed)
    return PPOTrainer(pol, env, cfg, get_action_api(pcfg.action_head), obs_specs=obs_specs()), pcfg


@pytest.mark.parametrize("chunk", [0, 32])
def test_real_policy_ratio_gate_and_update(chunk):
    cfg = make_cfg(num_envs=4, T=64, chunk=chunk, minibatches=2, epochs=2)
    tr, _ = _build(cfg)
    assert tr.api.source == "doom_agent.models"
    for _ in range(2):
        s = tr.train_iteration()  # ratio_check_fail=True
        assert s["ppo/ratio_check_max_dev"] < 1e-5
        assert np.isfinite(s["loss/total"]) and s["ppo/optimizer_steps"] >= 1
    assert tr.counters["crashed"] == 1


def test_real_policy_checkpoint_and_export(tmp_path):
    from doom_agent.obs import dummy_features

    from training.ppo.checkpoint import build_checkpoint, load_checkpoint, save_checkpoint
    from training.ppo.export import export_policy
    from training.ppo.trainer import features_to_torch

    cfg = make_cfg(num_envs=4, T=32, minibatches=2, epochs=1)
    a, pcfg = _build(cfg)
    a.train_iteration()
    ck = build_checkpoint(a, policy_config=pcfg.to_dict(), obs_config={}, action_space_version=pcfg.action_space_version,
                          train_manifest_ids=["fake"])
    path = save_checkpoint(tmp_path / "ckpt_000001.pt", ck)
    a.train_iteration()
    b, _ = _build(cfg, policy_seed=99)
    b.load_state_dict(load_checkpoint(path))
    b.train_iteration()
    for k, v in a.policy.state_dict().items():
        assert torch.equal(v, b.policy.state_dict()[k]), k

    out = tmp_path / "export"
    meta = export_policy(path, out)  # default factory = real DoomPolicy
    assert meta["policy_config"] == pcfg.to_dict()
    feats = dummy_features((2,), np.random.default_rng(0))
    np.savez(tmp_path / "f.npz", **feats)
    ref_pol, _ = _build(cfg)
    ref_pol.policy.load_state_dict(load_checkpoint(path)["policy_state"])
    with torch.no_grad():
        ref, _ = ref_pol.policy.step(features_to_torch(feats, "cpu"), ref_pol.policy.initial_state(2),
                                     torch.ones(2, dtype=torch.bool))
    code = f"""
import sys, json, numpy as np, torch
sys.path[:0] = [{str(ROOT / 'submission')!r}]
from doom_agent.models.config import PolicyConfig
from doom_agent.models.policy import DoomPolicy
meta = json.load(open({str(out / 'policy.json')!r}))
p = DoomPolicy(PolicyConfig.from_dict(meta['policy_config'])).eval()
p.load_state_dict(torch.load({str(out / 'policy.pt')!r}, weights_only=True), strict=True)
f = dict(np.load({str(tmp_path / 'f.npz')!r}))
with torch.no_grad():
    o, _ = p.step({{k: torch.from_numpy(v) for k, v in f.items()}}, p.initial_state(2), torch.ones(2, dtype=torch.bool))
np.save({str(tmp_path / 'l.npy')!r}, o['logits'].numpy())
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
    np.testing.assert_allclose(np.load(tmp_path / "l.npy"), ref["logits"].numpy(), atol=1e-6)
    assert json.loads((out / "policy.json").read_text())["format_version"] == "export-1"
