"""Config validation / serialisation, adapter contract, CLI wiring and logging."""

import importlib.util
import json
import math
import sys
import warnings

import pytest
import torch
from conftest import ROOT

from training.configs import TrainConfig
from training.ppo.adapters import get_action_api
from training.train import main

H1 = ROOT / "training" / "configs" / "h1.json"
REAL_MODELS = importlib.util.find_spec("doom_agent.models.policy") is not None


def test_h1_json_equals_defaults_and_round_trips():
    c = TrainConfig.load(H1)
    assert c.to_dict() == TrainConfig().to_dict()
    p = c.ppo
    assert (p.gamma, p.gae_lambda, p.clip_eps, p.lr, p.adam_eps, p.max_grad_norm) == (0.995, 0.95, 0.2, 3e-4, 1e-5, 0.5)
    assert (p.num_envs, p.rollout_len, p.num_minibatches, p.epochs, p.vf_coef) == (64, 128, 4, 3, 0.5)
    assert (p.ent_coef, p.ent_coef_final, p.target_kl, p.value_clip, p.return_norm) == (0.005, 0.001, 0.03, False, True)
    assert TrainConfig.from_dict(json.loads(json.dumps(c.to_dict()))).to_dict() == c.to_dict()


@pytest.mark.parametrize("ov", [["env.seed_base=10000"], ["run.seed=12345"], ["ppo.num_minibatches=5"],
                                ["ppo.bptt_chunk=48"], ["ppo.nope=1"], ["nogroup.x=1"]])
def test_bad_configs_rejected(ov):
    with pytest.raises(ValueError):
        TrainConfig().with_overrides(ov)


def test_local_fallback_distribution_contract():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api = get_action_api("factored", allow_fallback=True)
    d = api.dist_cls(torch.zeros(4, 15))
    assert torch.allclose(d.entropy(), torch.full((4,), 3 * math.log(3) + 3 * math.log(2)), atol=1e-5)
    assert d.entropy_per_head().shape == (4, 6)
    idx = d.sample(torch.Generator().manual_seed(0))
    assert idx.shape == (4, 6) and torch.isfinite(d.log_prob(idx)).all()
    import itertools

    import numpy as np
    all_idx = np.array(list(itertools.product(range(3), range(3), range(3), range(2), range(2), range(2))))
    mb = api.to_buttons(all_idx)
    assert mb.dtype == np.int8 and mb.shape == (216, 9) and len({tuple(r) for r in mb}) == 216
    assert not (mb[:, 0] & mb[:, 1]).any() and not (mb[:, 4] & mb[:, 5]).any()
    with pytest.raises(ValueError):
        api.to_buttons(np.array([[3, 0, 0, 0, 0, 0]]))


@pytest.mark.skipif(not REAL_MODELS, reason="B2 doom_agent.models not landed")
def test_fallback_matches_real_distribution():
    from training.ppo import adapters

    real = get_action_api("factored")
    logits = torch.randn(5, 15)
    a, b = real.dist_cls(logits), adapters._LocalFactoredCategorical(logits)
    idx = b.sample(torch.Generator().manual_seed(1))
    assert torch.allclose(a.log_prob(idx), b.log_prob(idx), atol=1e-6)
    assert torch.allclose(a.entropy_per_head(), b.entropy_per_head(), atol=1e-6)


def test_real_action_api_refuses_fallback(monkeypatch):
    for m in ("doom_agent.models", "doom_agent.models.action_codec", "doom_agent.models.distributions"):
        monkeypatch.setitem(sys.modules, m, None)  # import of a None entry raises ImportError
    with pytest.raises(ImportError, match="model-architect"):
        get_action_api("factored", allow_fallback=False)


def test_cli_real_mode_rejects_unregistered_reward_terms(capsys):
    """Real mode validates reward terms in the parent (no workers start) and exits 2 with a clear message."""
    rc = main(["--updates", "1", "--set", "run.tensorboard=false",
               "--set", 'reward.terms=[{"name": "no_such_term", "weight": 1.0}]'])
    assert rc == 2
    assert "unregistered term" in capsys.readouterr().err


def test_tensorboard_logging(tmp_path):
    pytest.importorskip("tensorboard")
    from training.ppo.logger import MetricsLogger

    lg = MetricsLogger(tmp_path, tensorboard=True)
    assert lg.tb_status == "enabled"
    lg.log(1, {"a": 1.0, "b": float("nan"), "s": "x"})
    lg.close()
    assert list((tmp_path / "tb").glob("events.*"))


def test_cli_fake_smoke_writes_logs_and_checkpoint(tmp_path):
    rd = tmp_path / "rid"
    from training.train import run
    c = TrainConfig().with_overrides(["ppo.num_envs=4", "ppo.rollout_len=16", "ppo.num_minibatches=2",
                                      "env.num_workers=1", "env.envs_per_worker=4", "run.tensorboard=false",
                                      "run.torch_threads=1"])
    run(c, fake=True, max_updates=2, run_dir=rd)
    meta = json.loads((rd / "run.json").read_text())
    assert meta["fake"] is True
    assert meta["action_api_source"] == ("doom_agent.models" if REAL_MODELS else "local-fallback")
    assert meta["obs_pipeline_hash"] and meta["git_commit"]
    lines = [json.loads(x) for x in (rd / "metrics.jsonl").read_text().splitlines()]
    assert len(lines) == 2
    need = ["loss/policy", "loss/value", "loss/entropy", "ppo/approx_kl", "ppo/clip_frac", "ppo/explained_variance",
            "episode/reward_mean", "episode/length_mean", "episode/completion_rate", "reward/intrinsic_mean",
            "reward/extrinsic_mean", "sps/total", "sps/env_only", "sps/learner_only", "time/env_s", "time/learner_s",
            "ppo/ratio_check_max_dev", "env/crashed_total", "grad/norm_preclip", "entropy_head/use",
            "entropy_head/move", "episode/use_edges_mean", "episode/door_reversals_mean", "value/at_episode_start",
            "sched/ent_coef", "sched/lr"]
    for k in need:
        assert k in lines[-1], k
    assert (rd / "checkpoints" / "ckpt_000002.pt").exists()
