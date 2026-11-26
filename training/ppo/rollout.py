"""Time-major rollout storage (docs/MODULE_CONTRACTS.md section 4.1).

All arrays are numpy, CPU, time-major (T, N, ...). One buffer holds one rollout of T decisions
from N envs. Field meanings:

  obs[k]            (T, N, ...)  features of obs_t
  episode_start     (T, N) bool  obs_t is the first obs of an episode (episode_start[t] = done[t-1])
  state0            (C, N, S)    hidden state passed into `step` at each chunk start t = c*chunk,
                                 BEFORE masking by episode_start[t]. C = T // chunk (C = 1 in v1)
  action_idx        (T, N, A)    int64 factor indices (A = 6 factored, 9 bernoulli)
  action_mb         (T, N, 9)    int8 buttons actually sent to the env
  logp, value       (T, N)       behaviour log-prob (joint) and V_ext(obs_t), at rollout time
  reward            (T, N)       raw total shaped training reward (sum of weighted terms)
  reward_scaled     (T, N)       reward after return-std scaling + clip (what GAE uses)
  terminated, truncated (T, N)   flags for transition t -> t+1, from the env
  crashed           (T, N) bool  env reported info["crashed"] (also truncated, trunc_value = 0)
  trunc_value       (T, N)       V_ext(final_obs_t, h_t_post) where truncated & ~terminated & ~crashed, else 0
  next_obs, next_episode_start, next_state   end-of-rollout bootstrap inputs; next_state is the
                                 next rollout's state0[0]
  advantages, returns (T, N)     filled by `compute_advantages`
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from training.ppo.gae import compute_gae


@dataclass
class RolloutBuffer:
    T: int
    N: int
    state_size: int
    obs_specs: dict  # key -> (shape tuple, numpy dtype)
    action_dim: int = 6
    chunk: int = 0  # 0 -> T
    extra_value_heads: tuple = ()

    obs: dict = field(init=False)
    t: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if self.chunk in (0, None):
            self.chunk = self.T
        if self.T % self.chunk:
            raise ValueError(f"T={self.T} not divisible by chunk={self.chunk}")
        T, N = self.T, self.N
        self.obs = {k: np.zeros((T, N) + tuple(shape), dtype=dt) for k, (shape, dt) in self.obs_specs.items()}
        self.episode_start = np.zeros((T, N), dtype=bool)
        self.state0 = np.zeros((T // self.chunk, N, self.state_size), dtype=np.float32)
        self.action_idx = np.zeros((T, N, self.action_dim), dtype=np.int64)
        self.action_mb = np.zeros((T, N, 9), dtype=np.int8)
        self.logp = np.zeros((T, N), dtype=np.float32)
        self.value = np.zeros((T, N), dtype=np.float32)
        self.values_extra = {name: np.zeros((T, N), dtype=np.float32) for name in self.extra_value_heads}
        self.reward = np.zeros((T, N), dtype=np.float32)
        self.reward_scaled = np.zeros((T, N), dtype=np.float32)
        self.terminated = np.zeros((T, N), dtype=bool)
        self.truncated = np.zeros((T, N), dtype=bool)
        self.crashed = np.zeros((T, N), dtype=bool)
        self.trunc_value = np.zeros((T, N), dtype=np.float32)
        self.next_obs: dict = {}
        self.next_episode_start = np.zeros(N, dtype=bool)
        self.next_state = np.zeros((N, self.state_size), dtype=np.float32)
        self.last_value = np.zeros(N, dtype=np.float32)
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns = np.zeros((T, N), dtype=np.float32)

    @property
    def num_chunks(self) -> int:
        return self.T // self.chunk

    def reset(self) -> None:
        self.t = 0
        self.trunc_value[:] = 0.0
        self.crashed[:] = False

    def add_pre(self, obs: dict, episode_start: np.ndarray, state_in: np.ndarray) -> None:
        """Record obs_t / episode_start_t, and the pre-mask state when t starts a chunk."""
        t = self.t
        if t >= self.T:
            raise IndexError("rollout buffer full")
        missing = set(self.obs) - set(obs)
        if missing:
            raise KeyError(f"obs is missing feature keys {sorted(missing)}")
        for k in self.obs:
            self.obs[k][t] = obs[k]
        self.episode_start[t] = episode_start
        if t % self.chunk == 0:
            self.state0[t // self.chunk] = state_in

    def add_post(self, *, action_idx, action_mb, logp, value, reward, reward_scaled, terminated, truncated,
                 crashed, trunc_value, values_extra: dict | None = None) -> None:
        t = self.t
        self.action_idx[t] = action_idx
        self.action_mb[t] = action_mb
        self.logp[t] = logp
        self.value[t] = value
        for name, v in (values_extra or {}).items():
            self.values_extra[name][t] = v
        self.reward[t] = reward
        self.reward_scaled[t] = reward_scaled
        self.terminated[t] = terminated
        self.truncated[t] = truncated
        self.crashed[t] = crashed
        self.trunc_value[t] = trunc_value
        self.t += 1

    def set_next(self, next_obs: dict, next_episode_start: np.ndarray, next_state: np.ndarray,
                 last_value: np.ndarray) -> None:
        if self.t != self.T:
            raise RuntimeError(f"rollout incomplete: t={self.t} of {self.T}")
        self.next_obs = {k: np.array(v, copy=True) for k, v in next_obs.items()}
        self.next_episode_start = np.array(next_episode_start, dtype=bool, copy=True)
        self.next_state = np.array(next_state, dtype=np.float32, copy=True)
        self.last_value = np.array(last_value, dtype=np.float32, copy=True)

    def compute_advantages(self, gamma: float, gae_lambda: float) -> None:
        self.advantages, self.returns = compute_gae(
            self.reward_scaled, self.value, self.terminated, self.truncated, self.trunc_value,
            self.last_value, gamma, gae_lambda,
        )

    def env_minibatch(self, env_idx: np.ndarray) -> dict:
        """Whole-env sequences for a set of envs: arrays (T, M, ...), state0 (C, M, S)."""
        e = np.asarray(env_idx)
        return {
            "obs": {k: v[:, e] for k, v in self.obs.items()},
            "episode_start": self.episode_start[:, e],
            "state0": self.state0[:, e],
            "action_idx": self.action_idx[:, e],
            "logp": self.logp[:, e],
            "value": self.value[:, e],
            "advantages": self.advantages[:, e],
            "returns": self.returns[:, e],
        }


__all__ = ["RolloutBuffer"]
