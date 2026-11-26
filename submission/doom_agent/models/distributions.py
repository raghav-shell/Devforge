"""Action distributions (torch, for training) and the shared numpy action selector (eval + eval-mode
rollouts). Contract: docs/ACTION_SPACE.md 2.2 and 4.

* `FactoredCategorical`: 6 independent categorical heads over the (..., 15) logits (D2).
* `BernoulliHeads`: the H2 control arm, 9 independent Bernoulli logits; actions are the buttons.
* `select_action`: numpy-only, float64, fixed draw order, so the eval RNG stream does not depend on
  torch or the device.

Torch sampling inverts the CDF with uniforms drawn on the generator's device (works with a CPU
generator for MPS logits too); `torch.multinomial` is not used.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor

from doom_agent.models.action_codec import (
    HEAD_SIZES,
    LOGIT_SLICES,
    NUM_BUTTONS,
    NUM_HEADS,
    NUM_LOGITS,
    factors_to_buttons,
)

INFERENCE_MODES = ("greedy", "sample")
ACTION_HEADS = ("factored", "bernoulli")


def _check_temperature(temperature: float) -> float:
    t = float(temperature)
    if not (t > 0.0 and np.isfinite(t)):
        raise ValueError(f"temperature must be a finite positive number, got {temperature!r}")
    return t


def _uniform(shape: tuple[int, ...], like: Tensor, generator: torch.Generator | None) -> Tensor:
    dev = generator.device if generator is not None else like.device
    u = torch.rand(shape, generator=generator, device=dev, dtype=torch.float32)
    return u.to(like.device)


class FactoredCategorical:
    """Six independent categoricals. logits: (..., 15). Actions: (..., 6) int64 factor indices.

    `log_prob` and `entropy` are sums over heads (the PPO entropy coefficient applies to the sum).
    """

    action_dim = NUM_HEADS

    def __init__(self, logits: Tensor, temperature: float = 1.0):
        if logits.shape[-1] != NUM_LOGITS:
            raise ValueError(f"FactoredCategorical expects (..., {NUM_LOGITS}) logits, got {tuple(logits.shape)}")
        self.temperature = _check_temperature(temperature)
        self.logits = logits if self.temperature == 1.0 else logits / self.temperature
        self._logp = [torch.log_softmax(self.logits[..., s], dim=-1) for s in LOGIT_SLICES]

    @property
    def batch_shape(self) -> torch.Size:
        return self.logits.shape[:-1]

    def head_log_probs(self) -> list[Tensor]:
        """Per-head normalised log-probabilities, each (..., n_k)."""
        return self._logp

    def probs(self) -> list[Tensor]:
        return [lp.exp() for lp in self._logp]

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        u = _uniform(tuple(self.batch_shape) + (NUM_HEADS,), self.logits, generator)
        out = []
        for k, lp in enumerate(self._logp):
            cdf = torch.cumsum(lp.exp(), dim=-1)
            idx = (cdf <= u[..., k : k + 1]).sum(dim=-1)  # == searchsorted(cdf, u, side="right")
            out.append(idx.clamp_(max=HEAD_SIZES[k] - 1))
        return torch.stack(out, dim=-1).to(torch.int64)

    def mode(self) -> Tensor:
        """Per-head argmax; ties go to the lowest index (torch.argmax returns the first maximum)."""
        return torch.stack([torch.argmax(lp, dim=-1) for lp in self._logp], dim=-1).to(torch.int64)

    def log_prob(self, idx: Tensor) -> Tensor:
        if idx.shape[-1] != NUM_HEADS:
            raise ValueError(f"expected (..., {NUM_HEADS}) action indices, got {tuple(idx.shape)}")
        idx = idx.long()
        parts = [lp.gather(-1, idx[..., k : k + 1]).squeeze(-1) for k, lp in enumerate(self._logp)]
        return torch.stack(parts, dim=-1).sum(dim=-1)

    def entropy_per_head(self) -> Tensor:
        return torch.stack([-(lp.exp() * lp).sum(dim=-1) for lp in self._logp], dim=-1)

    def entropy(self) -> Tensor:
        return self.entropy_per_head().sum(dim=-1)


class BernoulliHeads:
    """Nine independent Bernoulli buttons (H2 arm). logits: (..., 9). Actions: (..., 9) int64 in {0,1};
    the codec is the identity."""

    action_dim = NUM_BUTTONS

    def __init__(self, logits: Tensor, temperature: float = 1.0):
        if logits.shape[-1] != NUM_BUTTONS:
            raise ValueError(f"BernoulliHeads expects (..., {NUM_BUTTONS}) logits, got {tuple(logits.shape)}")
        self.temperature = _check_temperature(temperature)
        self.logits = logits if self.temperature == 1.0 else logits / self.temperature
        self._logp1 = torch.nn.functional.logsigmoid(self.logits)
        self._logp0 = torch.nn.functional.logsigmoid(-self.logits)

    @property
    def batch_shape(self) -> torch.Size:
        return self.logits.shape[:-1]

    def probs(self) -> Tensor:
        return torch.sigmoid(self.logits)

    def sample(self, generator: torch.Generator | None = None) -> Tensor:
        u = _uniform(tuple(self.logits.shape), self.logits, generator)
        return (u < torch.sigmoid(self.logits)).to(torch.int64)

    def mode(self) -> Tensor:
        return (self.logits > 0).to(torch.int64)  # p = 0.5 exactly -> 0 (off)

    def log_prob(self, idx: Tensor) -> Tensor:
        if idx.shape[-1] != NUM_BUTTONS:
            raise ValueError(f"expected (..., {NUM_BUTTONS}) actions, got {tuple(idx.shape)}")
        a = idx.to(self.logits.dtype)
        return (a * self._logp1 + (1.0 - a) * self._logp0).sum(dim=-1)

    def entropy_per_head(self) -> Tensor:
        return -(self._logp1.exp() * self._logp1 + self._logp0.exp() * self._logp0)

    def entropy(self) -> Tensor:
        return self.entropy_per_head().sum(dim=-1)


def make_distribution(
    logits: Tensor, action_head: str = "factored", temperature: float = 1.0
) -> FactoredCategorical | BernoulliHeads:
    if action_head == "factored":
        return FactoredCategorical(logits, temperature)
    if action_head == "bernoulli":
        return BernoulliHeads(logits, temperature)
    raise ValueError(f"unknown action_head {action_head!r}; expected one of {ACTION_HEADS}")


def actions_to_buttons(action_idx: np.ndarray, action_head: str = "factored") -> np.ndarray:
    """Distribution actions -> MultiBinary (..., 9) int8 (numpy). Identity for the Bernoulli arm."""
    a = np.asarray(action_idx)
    if action_head == "factored":
        return factors_to_buttons(a)
    if action_head == "bernoulli":
        if a.shape[-1:] != (NUM_BUTTONS,) or not np.all((a == 0) | (a == 1)):
            raise ValueError("bernoulli actions must be (..., 9) in {0, 1}")
        return a.astype(np.int8)
    raise ValueError(f"unknown action_head {action_head!r}")


def _softmax64(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / e.sum()


def select_action(
    logits: np.ndarray,
    mode: str,
    temperature: float,
    rng: np.random.Generator | None,
    action_head: str = "factored",
) -> np.ndarray:
    """One agent's logits ((15,) factored or (9,) bernoulli, no batch dim) -> MultiBinary (9,) int8.

    factored : greedy = per-head argmax (ties -> lowest index); sample = for heads 0..5 in order,
               u = rng.random(), choice = searchsorted(cumsum(softmax64(l_k / T)), u, "right"),
               clipped to n_k - 1. Then factors_to_buttons.
    bernoulli: greedy = (logits > 0); sample = rng.random(9) < sigmoid(logits / T), index order 0..8.
    """
    if mode not in INFERENCE_MODES:
        raise ValueError(f"mode must be one of {INFERENCE_MODES}, got {mode!r}")
    x = np.asarray(logits, dtype=np.float64)
    if mode == "sample":
        if rng is None:
            raise ValueError("rng is required for mode='sample'")
        t = _check_temperature(temperature)
    if action_head == "factored":
        if x.shape != (NUM_LOGITS,):
            raise ValueError(f"factored select_action expects ({NUM_LOGITS},) logits, got {x.shape}")
        idx = np.zeros(NUM_HEADS, dtype=np.int64)
        for k, s in enumerate(LOGIT_SLICES):
            lk = x[s]
            if mode == "greedy":
                idx[k] = int(np.argmax(lk))  # first maximum
            else:
                cdf = np.cumsum(_softmax64(lk / t))
                u = rng.random()
                idx[k] = min(int(np.searchsorted(cdf, u, side="right")), HEAD_SIZES[k] - 1)
        return factors_to_buttons(idx)
    if action_head == "bernoulli":
        if x.shape != (NUM_BUTTONS,):
            raise ValueError(f"bernoulli select_action expects ({NUM_BUTTONS},) logits, got {x.shape}")
        if mode == "greedy":
            return (x > 0).astype(np.int8)
        p = 1.0 / (1.0 + np.exp(-x / t))
        return (rng.random(NUM_BUTTONS) < p).astype(np.int8)
    raise ValueError(f"unknown action_head {action_head!r}; expected one of {ACTION_HEADS}")
