"""PolicyConfig (docs/MODULE_CONTRACTS.md 3.2). JSON-serialisable via to_dict() / from_dict().

Every optional block has a flag here so it can be ablated. Values that are reserved in the contract
but not built in v1 are accepted by the dataclass but raise NotImplementedError in DoomPolicy.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any

from doom_agent.models.action_codec import ACTION_SPACE_VERSION

# value -> implemented in v1?
CHOICES: dict[str, dict[str, bool]] = {
    "core": {"gru": True, "lstm": False},
    "ray_encoder": {"circconv": True, "mlp": True},
    "entity_encoder": {"deepsets": True, "attnpool": True, "transformer": False},
    "fusion": {"concat": True, "film": True},
    "action_head": {"factored": True, "bernoulli": True},
}


def _default_bias() -> dict[str, float]:
    return {"use": -2.0, "next_weapon": -2.0}


@dataclass
class PolicyConfig:
    hidden_size: int = 256
    core: str = "gru"
    ray_encoder: str = "circconv"
    entity_encoder: str = "deepsets"
    fusion: str = "concat"
    action_head: str = "factored"
    head_bias_init: dict[str, float] = field(default_factory=_default_bias)
    extra_value_heads: list[str] = field(default_factory=list)
    use_egomap: bool = False
    use_depth_band: bool = False
    obs_schema_version: str = "1.0.0"
    action_space_version: str = ACTION_SPACE_VERSION

    def validate(self) -> None:
        """Structural validation: unknown enum values, bad types. Does not reject reserved values."""
        for key, allowed in CHOICES.items():
            v = getattr(self, key)
            if v not in allowed:
                raise ValueError(f"PolicyConfig.{key}={v!r}; expected one of {sorted(allowed)}")
        if not isinstance(self.hidden_size, int) or self.hidden_size <= 0:
            raise ValueError(f"hidden_size must be a positive int, got {self.hidden_size!r}")
        if len(set(self.extra_value_heads)) != len(self.extra_value_heads):
            raise ValueError(f"duplicate extra_value_heads: {self.extra_value_heads}")
        for name in self.extra_value_heads:
            if not isinstance(name, str) or not name.isidentifier():
                raise ValueError(f"extra value head name must be an identifier, got {name!r}")

    def to_dict(self) -> dict[str, Any]:
        d = dataclasses.asdict(self)
        d["head_bias_init"] = {str(k): float(v) for k, v in self.head_bias_init.items()}
        d["extra_value_heads"] = [str(n) for n in self.extra_value_heads]
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PolicyConfig":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown PolicyConfig keys: {sorted(unknown)}")
        kw = dict(d)
        if "head_bias_init" in kw:
            kw["head_bias_init"] = {str(k): float(v) for k, v in kw["head_bias_init"].items()}
        if "extra_value_heads" in kw:
            kw["extra_value_heads"] = list(kw["extra_value_heads"])
        cfg = cls(**kw)
        cfg.validate()
        return cfg
