"""DoomPolicy: encoders -> fusion -> recurrent core -> actor / critic heads.

Contract: docs/MODULE_CONTRACTS.md 3.3-3.4. Hidden state is an explicit (B, S) tensor, never module
state. Done-mask convention: `episode_start[t]` is True iff obs[t] is the first obs of an episode; the
state is zeroed BEFORE obs[t] is consumed, identically in `step` and `forward_sequence` (both call
`_core_step`), so a rollout replays exactly and no future obs can reach the state.

Aux heads and the privileged critic are NOT part of this module (D9): they live in `training/` and
attach to `out["core"]`, so `state_dict()` never contains them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from doom_agent.models.action_codec import (
    BUTTON_OF_BINARY_HEAD,
    HEADS,
    LOGIT_SLICES,
    NUM_BUTTONS,
    NUM_LOGITS,
)
from doom_agent.models.config import CHOICES, PolicyConfig
from doom_agent.models.distributions import make_distribution
from doom_agent.models.encoders import (
    EntityEncoder,
    FiLMGenerator,
    RayEncoder,
    VitalsEncoder,
    init_linear,
)
from doom_agent.obs.schema import FEATURE_SPECS

REQUIRED_KEYS: tuple[str, ...] = tuple(FEATURE_SPECS.keys())

Out = dict[str, "Tensor | dict[str, Tensor]"]


class DoomPolicy(nn.Module):
    def __init__(self, cfg: PolicyConfig | None = None):
        super().__init__()
        cfg = PolicyConfig() if cfg is None else cfg
        cfg.validate()
        for key, allowed in CHOICES.items():
            v = getattr(cfg, key)
            if not allowed[v]:
                raise NotImplementedError(f"PolicyConfig.{key}={v!r} is reserved and not built in v1")
        if cfg.use_egomap or cfg.use_depth_band:
            raise NotImplementedError("use_egomap / use_depth_band are v2 slots (D7); not built in v1")
        self.cfg = cfg
        H = cfg.hidden_size
        self.hidden_size = H
        self.state_size = H  # GRU: S = H

        self.ray_enc = RayEncoder(cfg.ray_encoder, out_dim=128)
        self.ent_enc = EntityEncoder(cfg.entity_encoder, out_dim=64)
        self.vit_enc = VitalsEncoder(out_dim=64)
        self.film = (
            FiLMGenerator(self.vit_enc.out_dim, (self.ray_enc.film_dim, self.ent_enc.film_dim))
            if cfg.fusion == "film"
            else None
        )
        fused_in = self.ray_enc.out_dim + self.ent_enc.out_dim + self.vit_enc.out_dim
        self.fuse = init_linear(nn.Linear(fused_in, H))
        self.core = nn.GRUCell(H, H)
        self._init_gru(self.core)

        self.num_logits = NUM_LOGITS if cfg.action_head == "factored" else NUM_BUTTONS
        self.actor = init_linear(nn.Linear(H, self.num_logits), gain=0.01)
        self.critic = init_linear(nn.Linear(H, 1), gain=1.0)
        self.extra_values = nn.ModuleDict(
            {name: init_linear(nn.Linear(H, 1), gain=1.0) for name in cfg.extra_value_heads}
        )
        self._init_head_bias(cfg.head_bias_init)

    # ------------------------------------------------------------------ init
    @staticmethod
    def _init_gru(cell: nn.GRUCell) -> None:
        H = cell.hidden_size
        for w in (cell.weight_ih, cell.weight_hh):
            for g in range(3):  # r, z, n gate blocks, each orthogonal
                nn.init.orthogonal_(w.data[g * H : (g + 1) * H], gain=1.0)
        nn.init.zeros_(cell.bias_ih)
        nn.init.zeros_(cell.bias_hh)

    def _init_head_bias(self, bias: dict[str, float]) -> None:
        sizes = dict(HEADS)
        k_of = {name: k for k, (name, _) in enumerate(HEADS)}
        with torch.no_grad():
            for name, value in bias.items():
                if name not in sizes:
                    raise ValueError(f"head_bias_init: unknown head {name!r}")
                if sizes[name] != 2:
                    raise ValueError(f"head_bias_init: {name!r} is not a binary head")
                if self.cfg.action_head == "factored":
                    # class 0 stays 0, class 1 gets `value`: p(1) = sigmoid(value)
                    self.actor.bias[LOGIT_SLICES[k_of[name]].start + 1] = float(value)
                else:
                    self.actor.bias[BUTTON_OF_BINARY_HEAD[name]] = float(value)

    # ------------------------------------------------------------------ API
    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> Tensor:
        return torch.zeros(batch_size, self.state_size, dtype=torch.float32, device=device)

    def distribution(self, logits: Tensor, temperature: float = 1.0):
        """FactoredCategorical or BernoulliHeads for this config's action head."""
        return make_distribution(logits, self.cfg.action_head, temperature)

    def encode(self, obs: dict[str, Tensor]) -> Tensor:
        """Feature dict with one leading batch dim -> fused z (B, H). Pure function of obs[t]."""
        missing = [k for k in REQUIRED_KEYS if k not in obs]
        if missing:
            raise KeyError(f"DoomPolicy: missing feature keys {missing}")
        z_vit = self.vit_enc(obs)
        if self.film is not None:
            film_ray, film_ent = self.film(z_vit)
        else:
            film_ray = film_ent = None
        z_ray = self.ray_enc(obs, film_ray)
        z_ent = self.ent_enc(obs, film_ent)
        return F.relu(self.fuse(torch.cat([z_ray, z_ent, z_vit], dim=-1)))

    def _core_step(self, z: Tensor, state: Tensor, episode_start: Tensor) -> Tensor:
        start = episode_start.to(device=state.device).bool().unsqueeze(-1)
        state = torch.where(start, torch.zeros((), dtype=state.dtype, device=state.device), state)
        return self.core(z, state)

    def _heads(self, core: Tensor) -> Out:
        return {
            "logits": self.actor(core),
            "value": self.critic(core).squeeze(-1),
            "values": {name: head(core).squeeze(-1) for name, head in self.extra_values.items()},
            "core": core,
        }

    def step(self, obs: dict[str, Tensor], state: Tensor, episode_start: Tensor) -> tuple[Out, Tensor]:
        """obs: each (B, ...). state: (B, S). episode_start: (B,) bool. Returns (out, new_state)."""
        new_state = self._core_step(self.encode(obs), state, episode_start)
        return self._heads(new_state), new_state

    def forward_sequence(
        self, obs: dict[str, Tensor], state0: Tensor, episode_start: Tensor
    ) -> tuple[Out, Tensor]:
        """obs: each (T, B, ...). state0: (B, S) before step 0. episode_start: (T, B) bool.
        Encoders run batched over T*B; the GRUCell loops over T with the same masking as `step`."""
        T, B = episode_start.shape[:2]
        flat = {k: v.reshape((T * B,) + tuple(v.shape[2:])) for k, v in obs.items() if k in REQUIRED_KEYS}
        z = self.encode(flat).reshape(T, B, -1)
        s = state0
        cores = []
        for t in range(T):
            s = self._core_step(z[t], s, episode_start[t])
            cores.append(s)
        core = torch.stack(cores, dim=0)
        return self._heads(core), s

    def forward(self, obs: dict[str, Tensor], state: Tensor, episode_start: Tensor) -> tuple[Out, Tensor]:
        """Alias of `step` (nn.Module call convention)."""
        return self.step(obs, state, episode_start)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


__all__ = ["DoomPolicy", "REQUIRED_KEYS"]
