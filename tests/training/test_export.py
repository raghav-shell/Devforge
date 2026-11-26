"""export-1 round trip (MODULE_CONTRACTS 8.2) on the fake policy."""

import json
import subprocess
import sys

import numpy as np
import pytest
import torch
from conftest import ROOT, build_trainer, make_cfg

from doom_agent.obs import dummy_features, obs_pipeline_hash
from training.ppo.checkpoint import build_checkpoint, save_checkpoint
from training.ppo.export import ExportError, export_policy
from training.ppo.testing import fake_policy_factory
from training.ppo.trainer import features_to_torch


@pytest.fixture
def ckpt_file(tmp_path):
    tr = build_trainer(make_cfg(num_envs=2, T=8), env_kwargs=dict(episode_len=(5,)))
    tr.train_iteration()
    ck = build_checkpoint(tr, policy_config={"hidden_size": 16, "num_logits": 15}, obs_config={},
                          action_space_version="1.0.0", train_manifest_ids=["fake"],
                          aux_state={"aux.fut.weight": torch.ones(3, 16)},
                          privileged_critic_state={"priv.v.weight": torch.ones(1, 16)})
    return save_checkpoint(tmp_path / "ckpt_000001.pt", ck), tr


def test_export_round_trip_fresh_process(ckpt_file, tmp_path):
    path, tr = ckpt_file
    out = tmp_path / "export"
    meta = export_policy(path, out, policy_factory=fake_policy_factory)
    js = json.loads((out / "policy.json").read_text())
    assert js == meta
    for k in ("format_version", "policy_config", "obs_schema_version", "obs_pipeline_hash", "obs_config",
              "action_space_version", "default_inference", "inference_temperature", "source_ckpt_sha256",
              "git_commit", "global_step"):
        assert k in js, k
    assert js["format_version"] == "export-1" and js["obs_pipeline_hash"] == obs_pipeline_hash()
    sd = torch.load(out / "policy.pt", weights_only=True)
    assert set(sd) == set(tr.policy.state_dict())
    assert not [k for k in sd if "aux" in k or "priv" in k]
    assert all(v.dtype == torch.float32 for v in sd.values() if v.is_floating_point())

    feats = dummy_features((3,), np.random.default_rng(0))
    np.savez(tmp_path / "feats.npz", **feats)
    with torch.no_grad():
        out_ref, _ = tr.policy.step(features_to_torch(feats, "cpu"), tr.policy.initial_state(3),
                                    torch.ones(3, dtype=torch.bool))
    code = f"""
import sys, numpy as np, torch
sys.path[:0] = [{str(ROOT / 'submission')!r}, {str(ROOT)!r}]
from training.ppo.testing import fake_policy_factory
import json
meta = json.load(open({str(out / 'policy.json')!r}))
p = fake_policy_factory(meta['policy_config']).eval()
p.load_state_dict(torch.load({str(out / 'policy.pt')!r}, weights_only=True), strict=True)
f = dict(np.load({str(tmp_path / 'feats.npz')!r}))
x = {{k: torch.from_numpy(v) for k, v in f.items()}}
with torch.no_grad():
    o, _ = p.step(x, p.initial_state(3), torch.ones(3, dtype=torch.bool))
np.save({str(tmp_path / 'logits.npy')!r}, o['logits'].numpy())
"""
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
    np.testing.assert_array_equal(np.load(tmp_path / "logits.npy"), out_ref["logits"].numpy())


def test_export_refuses_hash_mismatch_and_logs_override(ckpt_file, tmp_path, capsys):
    path, _ = ckpt_file
    ck = torch.load(path, weights_only=False)
    ck["obs_pipeline_hash"] = "0" * 64
    bad = save_checkpoint(tmp_path / "bad.pt", ck)
    with pytest.raises(ExportError, match="obs_pipeline_hash"):
        export_policy(bad, tmp_path / "x", policy_factory=fake_policy_factory)
    meta = export_policy(bad, tmp_path / "y", policy_factory=fake_policy_factory, allow_hash_mismatch=True)
    assert "export_overrides" in meta
    assert (tmp_path / "y" / "export_overrides.log").read_text().count("OVERRIDE") == 1
    assert "obs_pipeline_hash mismatch" in capsys.readouterr().err


def test_export_refuses_key_mismatch_and_flag_violation(ckpt_file, tmp_path):
    path, _ = ckpt_file
    ck = torch.load(path, weights_only=False)
    ck2 = dict(ck, policy_config={"hidden_size": 8})  # shapes/keys same names, but strict load must fail
    p2 = save_checkpoint(tmp_path / "k.pt", ck2)
    with pytest.raises(Exception):
        export_policy(p2, tmp_path / "k", policy_factory=fake_policy_factory)
    ck3 = dict(ck)
    ck3["policy_state"] = dict(ck["policy_state"], **{"aux_head.weight": torch.zeros(1)})
    with pytest.raises(ExportError, match="keys"):
        export_policy(save_checkpoint(tmp_path / "a.pt", ck3), tmp_path / "a", policy_factory=fake_policy_factory)
    ck4 = dict(ck, obs_config={"enable_depth_band": True})
    with pytest.raises(ExportError, match="flag authority"):
        export_policy(save_checkpoint(tmp_path / "f.pt", ck4), tmp_path / "f", policy_factory=fake_policy_factory)


def test_export_cli(ckpt_file, tmp_path):
    path, _ = ckpt_file
    r = subprocess.run([sys.executable, "-m", "training.ppo.export", "--ckpt", str(path), "--out", str(tmp_path / "c"),
                        "--policy-factory", "training.ppo.testing:fake_policy_factory"],
                       cwd=ROOT, capture_output=True, text=True, timeout=120,
                       env={"PYTHONPATH": f"{ROOT / 'submission'}:{ROOT}", "PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "c" / "policy.pt").exists() and (tmp_path / "c" / "policy.json").exists()
