"""Test doubles for the PPO engine (NOT used by real runs).

- `FakePolicy`: a small, genuinely recurrent nn.Module implementing the DoomPolicy API of
  docs/MODULE_CONTRACTS.md section 3.4 (initial_state / step / forward_sequence, the
  `episode_start` mask applied BEFORE consuming obs_t, out keys logits/value/values/core).
  `ProxyPolicy` is the same API at roughly the real v1 size (GRU-256, ~0.5M params) for benchmarks.
- `FakeVecEnv`: implements the SubprocVecEnv/StepResult contract of section 5.2 (auto-reset,
  episode_start = terminated | truncated, info["final_feats"] on non-crash done steps,
  info["episode"], info["reward_terms"], info["crashed"] on injected crashes). It supports
  get_state/set_state so the checkpoint test can check a bit-identical full iteration.

Rewards: 1.0 when MOVE_FORWARD (button 0) is pressed, else 0 (plus `const_reward`), so a smoke run
has a learnable signal. This is a test fixture, not a training reward.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from doom_agent.obs import FEATURE_SPECS, dummy_features


@dataclass
class StepResult:  # mirrors training/envs/vector.py StepResult (section 5.2)
    obs: dict
    reward: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_start: np.ndarray
    infos: list = field(default_factory=list)


def obs_specs() -> dict:
    return {k: (tuple(s.shape), s.dtype) for k, s in FEATURE_SPECS.items()}


class FakePolicy(nn.Module):
    """Tiny recurrent policy with the DoomPolicy API. Reads ray_dist, vitals_feat, prev_action, act_aux."""

    _KEYS = ("ray_dist", "vitals_feat", "prev_action", "act_aux")

    def __init__(self, hidden_size: int = 16, num_logits: int = 15, enc: int = 32, extra_value_heads=()):
        super().__init__()
        in_dim = sum(FEATURE_SPECS[k].size for k in self._KEYS)
        self.state_size = hidden_size
        self.enc = nn.Sequential(nn.Linear(in_dim, enc), nn.ReLU())
        self.core = nn.GRUCell(enc, hidden_size)
        self.actor = nn.Linear(hidden_size, num_logits)
        self.critic = nn.Linear(hidden_size, 1)
        self.extra = nn.ModuleDict({n: nn.Linear(hidden_size, 1) for n in extra_value_heads})

    def initial_state(self, batch_size: int, device="cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.state_size, device=device)

    def _embed(self, obs: dict) -> torch.Tensor:
        lead = obs["ray_dist"].shape[:-1]
        x = torch.cat([obs[k].reshape(*lead, -1).float() for k in self._KEYS], dim=-1)
        return self.enc(x)

    def _heads(self, core: torch.Tensor) -> dict:
        return {
            "logits": self.actor(core),
            "value": self.critic(core).squeeze(-1),
            "values": {n: h(core).squeeze(-1) for n, h in self.extra.items()},
            "core": core,
        }

    def step(self, obs: dict, state: torch.Tensor, episode_start: torch.Tensor):
        z = self._embed(obs)
        state = state * (~episode_start.bool()).to(state.dtype)[:, None]
        h = self.core(z, state)
        return self._heads(h), h

    def forward_sequence(self, obs: dict, state0: torch.Tensor, episode_start: torch.Tensor):
        z = self._embed(obs)  # (T, B, E), batched over T*B
        T = z.shape[0]
        h = state0
        cores = []
        keep = (~episode_start.bool()).to(state0.dtype)
        for t in range(T):
            h = h * keep[t][:, None]
            h = self.core(z[t], h)
            cores.append(h)
        core = torch.stack(cores, 0)
        return self._heads(core), h


class ProxyPolicy(FakePolicy):
    """Benchmark proxy at roughly the v1 size: all features flattened, 2x256 MLP, GRU-256."""

    _KEYS = tuple(FEATURE_SPECS)

    def __init__(self, hidden_size: int = 256, num_logits: int = 15):
        super().__init__(hidden_size=hidden_size, num_logits=num_logits, enc=256)
        in_dim = sum(FEATURE_SPECS[k].size for k in self._KEYS)
        self.enc = nn.Sequential(nn.Linear(in_dim, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU())


class FakeVecEnv:
    """Deterministic fake vector env.

    episode_len: per-env list cycled over episodes; an episode of length L ends at its L-th step.
    end_kind:    "truncated" | "terminated" | "alternate" (per episode, starting with terminated).
    crash_at:    set of (env_index, global_step_index) where the env reports a crash instead.
    """

    def __init__(self, num_envs: int, *, seed: int = 0, episode_len=(7,), end_kind: str = "alternate",
                 crash_at: set | None = None, const_reward: float = 0.0, forward_reward: float = 1.0):
        self.num_envs = num_envs
        self.seed = seed
        self.episode_len = [list(episode_len)] * num_envs if np.ndim(episode_len[0]) == 0 else \
            [list(x) for x in episode_len]
        self.end_kind = end_kind
        self.crash_at = set(crash_at or ())
        self.const_reward = const_reward
        self.forward_reward = forward_reward
        self.rng = np.random.default_rng(seed)
        self._t = 0
        self._ep_step = np.zeros(num_envs, dtype=np.int64)
        self._ep_idx = np.zeros(num_envs, dtype=np.int64)
        self._ep_ret = np.zeros(num_envs, dtype=np.float64)
        self._obs: dict | None = None
        self.closed = False

    def _feats(self, n: int) -> dict:
        return dummy_features((n,), self.rng)

    def _len(self, i: int) -> int:
        lens = self.episode_len[i]
        return int(lens[self._ep_idx[i] % len(lens)])

    def _kind(self, i: int) -> str:
        if self.end_kind == "alternate":
            return "terminated" if self._ep_idx[i] % 2 == 0 else "truncated"
        return self.end_kind

    def reset(self) -> dict:
        self._obs = self._feats(self.num_envs)
        self._ep_step[:] = 0
        self._ep_ret[:] = 0
        return copy.deepcopy(self._obs)

    def step(self, actions_mb: np.ndarray) -> StepResult:
        a = np.asarray(actions_mb)
        if a.shape != (self.num_envs, 9) or a.dtype != np.int8:
            raise ValueError(f"actions must be ({self.num_envs}, 9) int8, got {a.shape} {a.dtype}")
        N = self.num_envs
        nxt = self._feats(N)
        reward = (self.const_reward + self.forward_reward * (a[:, 0] == 1)).astype(np.float32)
        term = np.zeros(N, dtype=bool)
        trunc = np.zeros(N, dtype=bool)
        infos = [{"reward_terms": {"forward": float(a[i, 0] == 1)}, "map_id": "fake", "tic": int(self._ep_step[i] * 4)}
                 for i in range(N)]
        self._ep_step += 1
        self._ep_ret += reward
        fresh = None
        for i in range(N):
            crashed = (i, self._t) in self.crash_at
            ended = self._ep_step[i] >= self._len(i)
            if crashed:
                trunc[i] = True
                infos[i]["crashed"] = True
            elif ended:
                if self._kind(i) == "terminated":
                    term[i] = True
                else:
                    trunc[i] = True
                infos[i]["final_feats"] = {k: v[i].copy() for k, v in nxt.items()}
                outcome = "exit" if term[i] else "timeout"
                infos[i]["episode"] = {
                    "map_id": "fake", "seed": int(self._ep_idx[i]), "length": int(self._ep_step[i]),
                    "return_shaped": float(self._ep_ret[i]), "return_env": 0.0, "outcome": outcome,
                    "cells": 0, "keys": 0, "kills": 0, "use_edges": int(a[i, 7]), "weapon_edges": 0,
                    "door_reversals": 0,
                }
            if crashed or ended:
                if fresh is None:
                    fresh = self._feats(N)
                for k in nxt:
                    nxt[k][i] = fresh[k][i]
                self._ep_step[i] = 0
                self._ep_ret[i] = 0
                self._ep_idx[i] += 1
        self._t += 1
        self._obs = nxt
        done = term | trunc
        return StepResult(copy.deepcopy(nxt), reward, term, trunc, done.copy(), infos)

    def get_state(self) -> dict:
        return {"rng": self.rng.bit_generator.state, "t": self._t, "ep_step": self._ep_step.tolist(),
                "ep_idx": self._ep_idx.tolist(), "ep_ret": self._ep_ret.tolist()}

    def set_state(self, s: dict) -> None:
        self.rng.bit_generator.state = s["rng"]
        self._t = int(s["t"])
        self._ep_step = np.asarray(s["ep_step"], dtype=np.int64)
        self._ep_idx = np.asarray(s["ep_idx"], dtype=np.int64)
        self._ep_ret = np.asarray(s["ep_ret"], dtype=np.float64)

    def close(self) -> None:
        self.closed = True


__all__ = ["FakePolicy", "FakeVecEnv", "ProxyPolicy", "StepResult", "obs_specs"]


def fake_policy_factory(policy_config: dict) -> FakePolicy:
    """Export-test factory: builds FakePolicy from its config dict (hidden_size, num_logits)."""
    return FakePolicy(hidden_size=int(policy_config.get("hidden_size", 16)),
                      num_logits=int(policy_config.get("num_logits", 15)))
