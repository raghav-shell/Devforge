"""Run configuration for PPO training (owner: rl-training-engineer).

Key names follow docs/MODULE_CONTRACTS.md section 4.4 (frozen); default values follow
research/RL_RESEARCH.md section 3 (H1 baseline). Every run serialises its full config to
`runs/rl-training-engineer/<run-id>/run.json` and into every checkpoint.

Groups:
  ppo     PPO / GAE / optimiser / schedule keys (section 4.4)
  aux     auxiliary-head slot (empty in v1, section 4.5)
  env     env factory / vector env keys (section 5.1)
  reward  reward-term list (section 6.3); weights are config, not contract (DECISIONS D10)
  obs     ObsConfig kwargs (passed through untouched)
  model   PolicyConfig overrides (passed through to PolicyConfig.from_dict)
  run     run bookkeeping (seed, device, threads, logging, checkpoint cadence)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any

MAX_TRAIN_SEED = 10000  # seeds >= 10000 are reserved for val/test (docs/EVALUATION_PROTOCOL.md section 5)


@dataclass
class PPOConfig:
    gamma: float = 0.995
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    lr: float = 3e-4
    lr_final: float = 1e-4  # linear decay target over the final `lr_decay_frac` of training
    lr_decay_frac: float = 0.0  # 0 = constant lr (curriculum stages); set >0 for the final stage
    adam_eps: float = 1e-5
    max_grad_norm: float = 0.5
    num_envs: int = 64
    rollout_len: int = 128
    bptt_chunk: int = 0  # 0 = whole rollout (v1, BPTT = T); H9 arm uses 32
    num_minibatches: int = 4  # minibatches are formed over envs (section 4.3)
    epochs: int = 3
    vf_coef: float = 0.5
    ent_coef: float = 0.005  # start; applied to the entropy SUMMED over factor heads
    ent_coef_final: float = 0.001  # linear decay over total_updates
    target_kl: float = 0.03  # early stop; <= 0 disables
    value_clip: bool = False
    value_clip_eps: float = 0.2
    return_norm: bool = True  # divide rewards by running std of discounted returns
    reward_clip: float = 10.0  # applied after return-std scaling
    norm_adv: bool = True  # per-minibatch advantage normalisation
    total_steps: int = 10_000_000  # decisions; total_updates = total_steps // (num_envs * rollout_len)
    ratio_check_tol: float = 1e-5  # epoch-1 / minibatch-1 |ratio - 1| gate (section 4.3)
    ratio_check_fail: bool = True  # DECISIONS D17: raise on violation (a sequence/mask/state bug); False: warn + count

    @property
    def chunk(self) -> int:
        return self.rollout_len if self.bptt_chunk in (0, None) else int(self.bptt_chunk)

    @property
    def batch_size(self) -> int:
        return self.num_envs * self.rollout_len

    @property
    def total_updates(self) -> int:
        return max(1, self.total_steps // self.batch_size)


@dataclass
class AuxConfig:
    weight: float = 0.25
    heads: list = field(default_factory=list)  # empty in v1


@dataclass
class EnvConfig:
    manifests: list = field(default_factory=lambda: ["maps/splits/smoke.json"])
    randomize_ticks: bool = True
    render_buffers: bool = False
    mods: list = field(default_factory=list)
    return_privileged: bool = False
    num_workers: int = 8
    envs_per_worker: int = 8
    seed_base: int = 0  # must be < 10000
    episodes_per_map: int = 4  # A3 sampler: new map every k episodes (engine re-init ~0.3 s; docs/ENVIRONMENT.md)
    step_threads: bool = True  # A3: step a worker's slots concurrently (bit-identical to sequential)


@dataclass
class RewardConfig:
    # Start weights from docs/MODULE_CONTRACTS.md section 6.2 (config values, DECISIONS D10). Names follow the
    # registered terms: training/rewards/progression.py implements the contract's "exit" as "exit_reached",
    # and locked doors as a separate "locked_door_open" term (disjoint sector sets). cell_visit (exploration)
    # and kill / damage_taken (combat) must be registered by their owners before H1 can start.
    terms: list = field(
        default_factory=lambda: [
            {"name": "cell_visit", "weight": 0.05, "params": {"cell": 64}},
            {"name": "exit_reached", "weight": 10.0},
            {"name": "key_pickup", "weight": 3.0},
            {"name": "door_open", "weight": 1.0},
            {"name": "locked_door_open", "weight": 3.0},
            {"name": "kill", "weight": 0.5},
            {"name": "damage_taken", "weight": -0.01},
            {"name": "death", "weight": -3.0},
        ]
    )


@dataclass
class RunConfig:
    seed: int = 0  # torch / numpy / python seed for the learner; must be < 10000
    device: str = "cpu"  # learner + rollout inference device; MPS only where measured faster
    torch_threads: int = 2  # learner threads (RL_RESEARCH section 2: leave cores to env workers)
    out_root: str = "runs/rl-training-engineer"
    run_id: str = ""  # auto-generated if empty
    log_every: int = 1  # updates
    ckpt_every: int = 50  # updates
    tensorboard: bool = True  # silently impossible if tensorboard is missing -> logged loudly


@dataclass
class TrainConfig:
    ppo: PPOConfig = field(default_factory=PPOConfig)
    aux: AuxConfig = field(default_factory=AuxConfig)
    env: EnvConfig = field(default_factory=EnvConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    obs: dict = field(default_factory=dict)
    model: dict = field(default_factory=dict)
    run: RunConfig = field(default_factory=RunConfig)

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        p = self.ppo
        if not (0 <= self.env.seed_base < MAX_TRAIN_SEED):
            raise ValueError(f"env.seed_base={self.env.seed_base} must be in [0, {MAX_TRAIN_SEED})")
        if not (0 <= self.run.seed < MAX_TRAIN_SEED):
            raise ValueError(f"run.seed={self.run.seed} must be in [0, {MAX_TRAIN_SEED})")
        if p.num_envs % p.num_minibatches != 0:
            raise ValueError(f"ppo.num_envs={p.num_envs} not divisible by num_minibatches={p.num_minibatches}")
        if p.rollout_len % p.chunk != 0:
            raise ValueError(f"ppo.rollout_len={p.rollout_len} not divisible by bptt_chunk={p.chunk}")
        if not (0.0 <= p.gamma <= 1.0 and 0.0 <= p.gae_lambda <= 1.0):
            raise ValueError("gamma and gae_lambda must be in [0, 1]")
        if p.epochs < 1 or p.rollout_len < 1:
            raise ValueError("epochs and rollout_len must be >= 1")

    # ---- serialisation ----
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TrainConfig":
        d = dict(d)
        kw: dict[str, Any] = {}
        sub = {"ppo": PPOConfig, "aux": AuxConfig, "env": EnvConfig, "reward": RewardConfig, "run": RunConfig}
        for name, typ in sub.items():
            if name in d:
                kw[name] = _build(typ, d.pop(name), name)
        for name in ("obs", "model"):
            if name in d:
                kw[name] = dict(d.pop(name))
        if d:
            raise ValueError(f"unknown config groups: {sorted(d)}")
        return cls(**kw)

    @classmethod
    def load(cls, path: str | Path) -> "TrainConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def with_overrides(self, overrides: list[str]) -> "TrainConfig":
        """Apply `group.key=value` overrides (value parsed as JSON, else string)."""
        d = self.to_dict()
        for ov in overrides:
            if "=" not in ov:
                raise ValueError(f"override {ov!r} must be group.key=value")
            path, raw = ov.split("=", 1)
            try:
                val = json.loads(raw)
            except json.JSONDecodeError:
                val = raw
            parts = path.split(".")
            node = d
            for p in parts[:-1]:
                if p not in node or not isinstance(node[p], dict):
                    raise ValueError(f"override {ov!r}: unknown group {p!r}")
                node = node[p]
            if parts[0] not in ("obs", "model") and parts[-1] not in node:
                raise ValueError(f"override {ov!r}: unknown key {parts[-1]!r}")
            node[parts[-1]] = val
        return TrainConfig.from_dict(d)


def _build(typ, value: dict, name: str):
    if is_dataclass(value):
        return value
    known = {f.name for f in fields(typ)}
    unknown = set(value) - known
    if unknown:
        raise ValueError(f"unknown keys in config group {name!r}: {sorted(unknown)}")
    return typ(**value)


__all__ = [
    "MAX_TRAIN_SEED",
    "AuxConfig",
    "EnvConfig",
    "PPOConfig",
    "RewardConfig",
    "RunConfig",
    "TrainConfig",
]
