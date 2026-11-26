"""tests/training fixtures: build a PPOTrainer on the FakePolicy / FakeVecEnv test doubles."""

import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for p in (ROOT / "submission", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import pytest  # noqa: E402
import torch  # noqa: E402

from training.configs import TrainConfig  # noqa: E402
from training.ppo.adapters import get_action_api  # noqa: E402
from training.ppo.testing import FakePolicy, FakeVecEnv, obs_specs  # noqa: E402
from training.ppo.trainer import PPOTrainer, seed_everything  # noqa: E402


def make_cfg(num_envs=4, T=16, chunk=0, minibatches=2, epochs=2, seed=0, **ppo_over):
    cfg = TrainConfig()
    cfg.ppo.num_envs = num_envs
    cfg.ppo.rollout_len = T
    cfg.ppo.bptt_chunk = chunk
    cfg.ppo.num_minibatches = minibatches
    cfg.ppo.epochs = epochs
    cfg.ppo.total_steps = num_envs * T * 100
    cfg.ppo.ratio_check_fail = True
    cfg.run.seed = seed
    cfg.run.tensorboard = False
    for k, v in ppo_over.items():
        setattr(cfg.ppo, k, v)
    cfg.validate()
    return cfg


def build_trainer(cfg, *, policy_seed=0, env_kwargs=None, hidden=16):
    seed_everything(policy_seed)
    policy = FakePolicy(hidden_size=hidden)
    env = FakeVecEnv(cfg.ppo.num_envs, **(env_kwargs or {}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        api = get_action_api("factored", allow_fallback=True)
    seed_everything(cfg.run.seed)
    return PPOTrainer(policy, env, cfg, api, obs_specs=obs_specs(), device="cpu")


@pytest.fixture(autouse=True)
def _threads():
    n = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(n)
