"""PPO losses and reward / return normalisation.

- Clipped surrogate policy loss.
- Value loss: 0.5 * (V - R)^2, optionally clipped around the behaviour value (off by default,
  research/RL_RESEARCH.md section 3: value clipping hurts).
- Entropy: the joint entropy of the factorised policy = SUM over factor heads (docs/ACTION_SPACE.md
  section 2.2). The coefficient multiplies the summed entropy.
- Per-minibatch advantage normalisation behind `norm_adv`.
- ReturnNormalizer: rewards divided by the running std of the discounted return (the "reward
  scaling" detail), then clipped to +-reward_clip. The value head therefore learns returns on the
  normalised scale, which is what `out["value"]` is documented as (section 3.4 "normalised-return scale").
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


class RunningMeanStd:
    """Parallel-variance running mean/std (Chan et al.), float64. Checkpointable."""

    def __init__(self, shape: tuple[int, ...] = (), epsilon: float = 1e-4):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = float(epsilon)

    def update(self, x: np.ndarray) -> None:
        x = np.asarray(x, dtype=np.float64)
        if x.size == 0:
            return
        b_mean, b_var, b_count = x.mean(axis=0), x.var(axis=0), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + b_count
        new_mean = self.mean + delta * b_count / tot
        m2 = self.var * self.count + b_var * b_count + delta**2 * self.count * b_count / tot
        self.mean, self.var, self.count = new_mean, m2 / tot, tot

    def state_dict(self) -> dict:
        return {"mean": self.mean.tolist(), "var": self.var.tolist(), "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = np.asarray(d["mean"], dtype=np.float64)
        self.var = np.asarray(d["var"], dtype=np.float64)
        self.count = float(d["count"])


class ReturnNormalizer:
    """Per-env discounted return accumulator R <- gamma * R * (1 - done_prev) + r; scale r / std(R)."""

    def __init__(self, num_envs: int, gamma: float, clip: float = 10.0, enabled: bool = True, eps: float = 1e-8):
        self.enabled = enabled
        self.gamma = gamma
        self.clip = clip
        self.eps = eps
        self.rms = RunningMeanStd(())
        self.ret = np.zeros(num_envs, dtype=np.float64)

    def __call__(self, reward: np.ndarray, done: np.ndarray) -> np.ndarray:
        """`reward` / `done` for this transition. Returns scaled rewards (N,) float32."""
        r = np.asarray(reward, dtype=np.float64)
        if not self.enabled:
            return r.astype(np.float32)
        self.ret = self.ret * self.gamma + r
        self.rms.update(self.ret)
        self.ret[np.asarray(done, dtype=bool)] = 0.0
        scaled = r / np.sqrt(self.rms.var + self.eps)
        return np.clip(scaled, -self.clip, self.clip).astype(np.float32)

    @property
    def std(self) -> float:
        return float(np.sqrt(self.rms.var + self.eps))

    def state_dict(self) -> dict:
        return {"rms": self.rms.state_dict(), "ret": self.ret.tolist(), "enabled": self.enabled,
                "gamma": self.gamma, "clip": self.clip}

    def load_state_dict(self, d: dict) -> None:
        self.rms.load_state_dict(d["rms"])
        self.ret = np.asarray(d["ret"], dtype=np.float64)
        self.enabled = bool(d["enabled"])
        self.gamma = float(d["gamma"])
        self.clip = float(d["clip"])


@dataclass
class LossOut:
    loss: torch.Tensor
    pg_loss: torch.Tensor
    v_loss: torch.Tensor
    entropy: torch.Tensor  # mean summed entropy
    entropy_per_head: torch.Tensor  # (A,) mean per head
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor
    ratio_max_dev: torch.Tensor  # max |ratio - 1|


def ppo_loss(
    new_logp: torch.Tensor,
    old_logp: torch.Tensor,
    advantages: torch.Tensor,
    new_value: torch.Tensor,
    old_value: torch.Tensor,
    returns: torch.Tensor,
    entropy_per_head: torch.Tensor,  # (..., A)
    *,
    clip_eps: float,
    vf_coef: float,
    ent_coef: float,
    norm_adv: bool = True,
    value_clip: bool = False,
    value_clip_eps: float = 0.2,
) -> LossOut:
    """All tensors flattened or same-shaped (...,). Returns LossOut (loss is the scalar to minimise)."""
    adv = advantages
    if norm_adv:
        adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    log_ratio = new_logp - old_logp
    ratio = log_ratio.exp()
    pg1 = -adv * ratio
    pg2 = -adv * torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps)
    pg_loss = torch.max(pg1, pg2).mean()

    if value_clip:
        v_clipped = old_value + torch.clamp(new_value - old_value, -value_clip_eps, value_clip_eps)
        v_loss = 0.5 * torch.max((new_value - returns) ** 2, (v_clipped - returns) ** 2).mean()
    else:
        v_loss = 0.5 * ((new_value - returns) ** 2).mean()

    ent_head = entropy_per_head.reshape(-1, entropy_per_head.shape[-1]).mean(0)
    entropy = entropy_per_head.sum(-1).mean()
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy

    with torch.no_grad():
        approx_kl = ((ratio - 1.0) - log_ratio).mean()  # k3 estimator, >= 0
        clip_frac = ((ratio - 1.0).abs() > clip_eps).float().mean()
        ratio_max_dev = (ratio - 1.0).abs().max()
    return LossOut(loss, pg_loss, v_loss, entropy, ent_head, approx_kl, clip_frac, ratio_max_dev)


def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    var_y = float(np.var(y_true))
    if var_y == 0.0:
        return float("nan")
    return float(1.0 - np.var(y_true - y_pred) / var_y)


__all__ = ["LossOut", "ReturnNormalizer", "RunningMeanStd", "explained_variance", "ppo_loss"]
