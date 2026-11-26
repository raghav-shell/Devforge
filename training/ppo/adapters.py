"""Thin adapter over the model-architect's action API (docs/ACTION_SPACE.md section 2).

The PPO engine needs, from `doom_agent.models` (B1/B2):
  - `distributions.FactoredCategorical(logits)` with sample(generator) / log_prob / entropy / entropy_per_head
  - `distributions.BernoulliHeads(logits)` (H2 arm, same method names)
  - `action_codec.factors_to_buttons(idx (..., 6) int64) -> (..., 9) int8`, `action_codec.HEADS`

Until B1 lands, `get_action_api(allow_fallback=True)` returns a LOCAL implementation of the same
contract (`_LocalFactoredCategorical` etc.). The fallback is only permitted for fake / unit-test runs:
`training/train.py` calls `get_action_api(allow_fallback=False)` for real runs, which raises a clear
error if `doom_agent.models` is missing. Once B1 lands, tests/training/test_adapters.py checks that
the fallback and the real implementation agree numerically.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

# Frozen layout (docs/ACTION_SPACE.md section 2.1).
HEADS: tuple[tuple[str, int], ...] = (
    ("move", 3), ("strafe", 3), ("turn", 3), ("attack", 2), ("use", 2), ("next_weapon", 2),
)
NUM_LOGITS = 15
# _TABLE[k][choice] -> button index or -1 (none)
_TABLE = ((-1, 0, 1), (-1, 2, 3), (-1, 4, 5), (-1, 6), (-1, 7), (-1, 8))


def _slices() -> tuple[slice, ...]:
    out, o = [], 0
    for _, n in HEADS:
        out.append(slice(o, o + n))
        o += n
    return tuple(out)


LOGIT_SLICES = _slices()


class _LocalFactoredCategorical:
    """Contract copy of distributions.FactoredCategorical (fallback only)."""

    def __init__(self, logits: torch.Tensor):
        if logits.shape[-1] != NUM_LOGITS:
            raise ValueError(f"factored logits must have last dim {NUM_LOGITS}, got {tuple(logits.shape)}")
        self.logits = logits
        self._logp = [torch.log_softmax(logits[..., s], dim=-1) for s in LOGIT_SLICES]

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        outs = []
        for lp in self._logp:
            p = lp.exp().reshape(-1, lp.shape[-1])
            idx = torch.multinomial(p, 1, generator=generator).reshape(lp.shape[:-1])
            outs.append(idx)
        return torch.stack(outs, dim=-1).to(torch.int64)

    def mode(self) -> torch.Tensor:
        return torch.stack([lp.argmax(dim=-1) for lp in self._logp], dim=-1)

    def log_prob(self, idx: torch.Tensor) -> torch.Tensor:
        parts = [lp.gather(-1, idx[..., k : k + 1].long()).squeeze(-1) for k, lp in enumerate(self._logp)]
        return torch.stack(parts, dim=-1).sum(-1)

    def entropy_per_head(self) -> torch.Tensor:
        return torch.stack([-(lp.exp() * lp).sum(-1) for lp in self._logp], dim=-1)

    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head().sum(-1)


class _LocalBernoulliHeads:
    """Contract copy of distributions.BernoulliHeads (fallback only)."""

    def __init__(self, logits: torch.Tensor):
        self.logits = logits
        self._d = torch.distributions.Bernoulli(logits=logits)

    def sample(self, generator: torch.Generator | None = None) -> torch.Tensor:
        u = torch.rand(self.logits.shape, generator=generator, device=self.logits.device)
        return (u < torch.sigmoid(self.logits)).to(torch.int64)

    def mode(self) -> torch.Tensor:
        return (self.logits > 0).to(torch.int64)

    def log_prob(self, idx: torch.Tensor) -> torch.Tensor:
        return self._d.log_prob(idx.to(self.logits.dtype)).sum(-1)

    def entropy_per_head(self) -> torch.Tensor:
        return self._d.entropy()

    def entropy(self) -> torch.Tensor:
        return self.entropy_per_head().sum(-1)


def _local_factors_to_buttons(idx: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx)
    if idx.shape[-1] != len(HEADS):
        raise ValueError(f"factor idx must have last dim {len(HEADS)}, got {idx.shape}")
    out = np.zeros(idx.shape[:-1] + (9,), dtype=np.int8)
    for k, (_, n) in enumerate(HEADS):
        col = idx[..., k]
        if np.any((col < 0) | (col >= n)):
            raise ValueError(f"factor index out of range for head {HEADS[k][0]}")
        for choice in range(1, n):
            out[..., _TABLE[k][choice]] |= (col == choice).astype(np.int8)
    return out


@dataclass(frozen=True)
class ActionAPI:
    dist_cls: Callable[[torch.Tensor], Any]
    to_buttons: Callable[[np.ndarray], np.ndarray]  # (..., A) int64 -> (..., 9) int8
    head_names: tuple[str, ...]
    action_dim: int  # 6 factored, 9 bernoulli
    source: str  # "doom_agent.models" or "local-fallback"


def get_action_api(action_head: str = "factored", *, allow_fallback: bool = False) -> ActionAPI:
    try:
        from doom_agent.models import action_codec, distributions  # type: ignore

        real = True
    except ImportError as e:  # B1 not landed
        if not allow_fallback:
            raise ImportError(
                "doom_agent.models (action_codec, distributions) is not importable: B1/B2 from the "
                "model-architect has not landed. Real training runs need it; use --fake for the "
                f"engine smoke path. Original error: {e}"
            ) from e
        warnings.warn("using LOCAL fallback action distribution/codec (fake runs only)", stacklevel=2)
        real = False

    if action_head == "factored":
        if real:
            return ActionAPI(distributions.FactoredCategorical, action_codec.factors_to_buttons,
                             tuple(n for n, _ in action_codec.HEADS), len(action_codec.HEADS),
                             "doom_agent.models")
        return ActionAPI(_LocalFactoredCategorical, _local_factors_to_buttons,
                         tuple(n for n, _ in HEADS), len(HEADS), "local-fallback")
    if action_head == "bernoulli":
        names = tuple(f"b{i}" for i in range(9))

        def ident(a: np.ndarray) -> np.ndarray:
            return np.asarray(a).astype(np.int8)

        cls = distributions.BernoulliHeads if real else _LocalBernoulliHeads
        return ActionAPI(cls, ident, names, 9, "doom_agent.models" if real else "local-fallback")
    raise ValueError(f"unknown action_head {action_head!r}")


def _local_features_to_torch(feats: dict, device="cpu") -> dict:
    """Fallback copy of doom_agent.models.io.features_to_torch (keeps leading dims and dtypes)."""
    return {k: torch.from_numpy(np.ascontiguousarray(v)).to(device) for k, v in feats.items()}


try:  # one conversion shared with submission/agent.py (MODULE_CONTRACTS 3.1)
    from doom_agent.models.io import features_to_torch  # type: ignore
except ImportError:  # B2 not landed: fake runs only
    features_to_torch = _local_features_to_torch


__all__ = ["HEADS", "LOGIT_SLICES", "NUM_LOGITS", "ActionAPI", "features_to_torch", "get_action_api"]
