"""Per-modality encoders (docs/MODULE_CONTRACTS.md 3.3, docs/SPATIAL_OBSERVATION_DESIGN.md 2).

All sizes are derived from `doom_agent.obs.schema.FEATURE_SPECS`; nothing about the observation is
hardcoded here. No BatchNorm / Dropout / anything mode- or batch-dependent: step and sequence
forwards must agree exactly, and PPO's ratio check needs train == eval behaviour.

FiLM hook: every encoder accepts an optional `film=(gamma, beta)` and exposes `film_dim`, the width
of the features it modulates (used only when PolicyConfig.fusion == "film").
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from doom_agent.obs.schema import FEATURE_SPECS

RAY_FLOAT_KEYS = ("ray_dist", "ray_near", "ray_ddist")
VITALS_KEYS = ("vitals_feat", "prev_action", "act_aux")


def _vocab(key: str) -> int:
    return int(FEATURE_SPECS[key].high_array().max()) + 1


NUM_RAYS = FEATURE_SPECS["ray_dist"].shape[0]
RAY_HIT_VOCAB = _vocab("ray_hit")
MAX_ENTITIES, ENT_FEAT_DIM = FEATURE_SPECS["ent_feat"].shape
ENT_TYPE_VOCAB = _vocab("ent_type")
VITALS_IN_DIM = sum(FEATURE_SPECS[k].size for k in VITALS_KEYS)

RAY_HIT_EMB = 4
ENT_TYPE_EMB = 8


def init_linear(layer: nn.Module, gain: float = math.sqrt(2.0)) -> nn.Module:
    """Orthogonal weight with `gain`, zero bias (research/RL_RESEARCH.md: hidden layers sqrt(2))."""
    nn.init.orthogonal_(layer.weight, gain=gain)
    if getattr(layer, "bias", None) is not None:
        nn.init.zeros_(layer.bias)
    return layer


def _apply_film(x: Tensor, film: tuple[Tensor, Tensor] | None, channel_dim: int) -> Tensor:
    if film is None:
        return x
    gamma, beta = film
    if channel_dim == 1 and x.dim() == 3:  # (B, C, L)
        gamma, beta = gamma.unsqueeze(-1), beta.unsqueeze(-1)
    return x * gamma + beta


class CircularConv1d(nn.Conv1d):
    """Conv1d with circular padding, computed as circular pad + unfold + matmul.

    Same parameters and state_dict keys as nn.Conv1d(padding_mode="circular"), numerically equal to it
    (tested). Reason: on the macOS arm64 torch 2.15 CPU build (no mkldnn) the native conv1d kernel took
    124 ms for a 2048 x 7 x 64 batch vs 8.4 ms for unfold+matmul (MEASURED, docs/MODEL.md), which made
    the ray encoder ~90 % of a 16 x 128 PPO minibatch.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__(in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2,
                         padding_mode="circular")

    # One code path for every batch size, so batch-1 acting, batch-N rollouts and T*B training replay
    # share numerics (PPO ratio gate, MODULE_CONTRACTS 4.3). At B=1 unfold costs ~5 us/layer more
    # than the native kernel; at B>=16 it is 3-4x faster (MEASURED, docs/MODEL.md).

    def forward(self, x: Tensor) -> Tensor:  # (B, C, L) -> (B, O, L')
        p = self.padding[0]
        k = self.kernel_size[0]
        if p:
            x = torch.cat([x[..., -p:], x, x[..., :p]], dim=-1)
        cols = x.unfold(-1, k, self.stride[0])  # (B, C, L', k)
        cols = cols.permute(0, 2, 1, 3).reshape(x.shape[0], cols.shape[2], -1)  # (B, L', C*k)
        out = torch.matmul(cols, self.weight.reshape(self.out_channels, -1).t()) + self.bias
        return out.transpose(1, 2)


class RayEncoder(nn.Module):
    """64 rays x (dist, near, ddist, emb4(hit)) -> z_ray (B, out_dim).

    kind="circconv" (v1): circular Conv1d 7-32 k5 s1, 32-32 k5 s2, 32-64 k3 s2, flatten, Linear, ReLU.
        Circular padding: ray 63 neighbours ray 0. The stack is equivariant to input shifts by
        multiples of the total stride (4); the flatten then keeps the ego direction on purpose.
    kind="mlp" (ablation, SPATIAL_OBSERVATION_DESIGN Proposal A): flatten 448 -> 256 -> out_dim.
    """

    def __init__(self, kind: str = "circconv", out_dim: int = 128):
        super().__init__()
        self.kind = kind
        self.out_dim = out_dim
        self.hit_emb = nn.Embedding(RAY_HIT_VOCAB, RAY_HIT_EMB)
        in_ch = len(RAY_FLOAT_KEYS) + RAY_HIT_EMB
        if kind == "circconv":
            self.convs = nn.ModuleList(
                [
                    CircularConv1d(in_ch, 32, 5, stride=1),
                    CircularConv1d(32, 32, 5, stride=2),
                    CircularConv1d(32, 64, 3, stride=2),
                ]
            )
            for c in self.convs:
                init_linear(c)
            self.film_dim = self.convs[-1].out_channels
            with torch.no_grad():
                flat = self.convs_out(torch.zeros(1, in_ch, NUM_RAYS)).flatten(1).shape[1]
            self.proj = init_linear(nn.Linear(flat, out_dim))
            self.total_stride = 1
            for c in self.convs:
                self.total_stride *= c.stride[0]
        elif kind == "mlp":
            self.hidden = init_linear(nn.Linear(in_ch * NUM_RAYS, 256))
            self.film_dim = 256
            self.proj = init_linear(nn.Linear(256, out_dim))
        else:
            raise ValueError(f"unknown ray_encoder {kind!r}")

    def channels(self, obs: dict[str, Tensor]) -> Tensor:
        """(B, 7, NUM_RAYS) input channels."""
        x = torch.stack([obs[k] for k in RAY_FLOAT_KEYS], dim=1)
        h = self.hit_emb(obs["ray_hit"]).transpose(1, 2)
        return torch.cat([x, h], dim=1)

    def convs_out(self, x: Tensor) -> Tensor:
        for c in self.convs:
            x = F.relu(c(x))
        return x

    def features(self, obs: dict[str, Tensor]) -> Tensor:
        """Pre-flatten conv features (B, 64, NUM_RAYS / total_stride). circconv only (for tests)."""
        return self.convs_out(self.channels(obs))

    def forward(self, obs: dict[str, Tensor], film: tuple[Tensor, Tensor] | None = None) -> Tensor:
        x = self.channels(obs)
        if self.kind == "circconv":
            x = _apply_film(self.convs_out(x), film, channel_dim=1)
            return F.relu(self.proj(x.flatten(1)))
        x = _apply_film(F.relu(self.hidden(x.flatten(1))), film, channel_dim=-1)
        return F.relu(self.proj(x))


class EntityEncoder(nn.Module):
    """Masked set encoder over 32 entity slots -> z_ent (B, out_dim). Permutation-invariant.

    phi: (feat 7 || emb8(type)) -> 64 -> 64, ReLU, shared per slot.
    kind="deepsets" (v1): masked mean || masked max || count/32 -> rho -> out_dim.
    kind="attnpool" (ablation 1): attention pooling with a learned, always-valid null slot replaces
        the mean (so n = 0 never produces a softmax over an empty row).

    Masked slots are sanitised with torch.where BEFORE phi (features and type index), so garbage,
    even NaN/inf, in padded slots cannot reach the outputs or the gradients.
    """

    def __init__(self, kind: str = "deepsets", out_dim: int = 64, hidden: int = 64):
        super().__init__()
        if kind not in ("deepsets", "attnpool"):
            raise ValueError(f"unknown entity_encoder {kind!r}")
        self.kind = kind
        self.out_dim = out_dim
        self.film_dim = out_dim
        self.type_emb = nn.Embedding(ENT_TYPE_VOCAB, ENT_TYPE_EMB)
        self.phi1 = init_linear(nn.Linear(ENT_FEAT_DIM + ENT_TYPE_EMB, hidden))
        self.phi2 = init_linear(nn.Linear(hidden, hidden))
        if kind == "attnpool":
            self.score = init_linear(nn.Linear(hidden, 1), gain=0.01)
            self.null_value = nn.Parameter(torch.zeros(hidden))
            self.null_score = nn.Parameter(torch.zeros(()))
        self.rho = init_linear(nn.Linear(2 * hidden + 1, out_dim))

    def forward(self, obs: dict[str, Tensor], film: tuple[Tensor, Tensor] | None = None) -> Tensor:
        mask = obs["ent_mask"].bool()  # (B, N)
        m3 = mask.unsqueeze(-1)
        feat = torch.where(m3, obs["ent_feat"], torch.zeros((), dtype=obs["ent_feat"].dtype, device=mask.device))
        etype = torch.where(mask, obs["ent_type"], torch.zeros((), dtype=obs["ent_type"].dtype, device=mask.device))
        e = torch.cat([feat, self.type_emb(etype)], dim=-1)
        e = F.relu(self.phi2(F.relu(self.phi1(e))))  # (B, N, hidden), >= 0
        e = torch.where(m3, e, torch.zeros((), dtype=e.dtype, device=e.device))
        n = mask.sum(dim=-1, keepdim=True).to(e.dtype)  # (B, 1)
        # e >= 0 after ReLU and masked slots are 0, so this max is the exact masked max when n > 0
        # and exactly 0 when n == 0 (no -inf, no NaN).
        mx = e.amax(dim=1)
        if self.kind == "deepsets":
            pooled = e.sum(dim=1) / n.clamp(min=1.0)
        else:
            s = self.score(e).squeeze(-1).masked_fill(~mask, float("-inf"))  # (B, N)
            s = torch.cat([s, self.null_score.expand(s.shape[0], 1)], dim=1)  # null slot always valid
            w = torch.softmax(s, dim=1)
            v = torch.cat([e, self.null_value.expand(e.shape[0], 1, -1)], dim=1)
            pooled = (w.unsqueeze(-1) * v).sum(dim=1)
        z = self.rho(torch.cat([pooled, mx, n / float(mask.shape[-1])], dim=-1))
        return F.relu(_apply_film(z, film, channel_dim=-1))


class VitalsEncoder(nn.Module):
    """vitals_feat || prev_action || act_aux (42) -> Linear -> ReLU -> z_vit (B, out_dim)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.out_dim = out_dim
        self.fc = init_linear(nn.Linear(VITALS_IN_DIM, out_dim))

    def forward(self, obs: dict[str, Tensor]) -> Tensor:
        x = torch.cat([obs[k] for k in VITALS_KEYS], dim=-1)
        return F.relu(self.fc(x))


class FiLMGenerator(nn.Module):
    """z_vit -> (gamma, beta) per modulated encoder. Zero-initialised, so it starts as the identity
    (gamma = 1, beta = 0) and the film arm begins exactly like the concat arm's encoders."""

    def __init__(self, in_dim: int, dims: tuple[int, ...]):
        super().__init__()
        self.dims = dims
        self.fc = nn.Linear(in_dim, 2 * sum(dims))
        nn.init.zeros_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, z: Tensor) -> list[tuple[Tensor, Tensor]]:
        out, parts = [], torch.split(self.fc(z), [d for d in self.dims for _ in (0, 1)], dim=-1)
        for i in range(len(self.dims)):
            out.append((1.0 + parts[2 * i], parts[2 * i + 1]))
        return out
