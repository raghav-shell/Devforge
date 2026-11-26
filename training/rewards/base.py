"""Composable reward-term framework (docs/MODULE_CONTRACTS.md section 6). Training-only.

Terms see training-only ground truth (`env.privileged_state()`, the map inventory) through
`StepContext` and `reset(...)` only. Nothing here touches the policy's observation path, and nothing
under `submission/` may import this module.

    composer = RewardComposer([{"name": "cell_visit", "weight": 0.05, "params": {"cell": 64}}, ...])
    composer.reset(priv0, obs0, inventory, spec)          # once per episode, after env.reset()
    total, per_term = composer(ctx)                       # once per env.step(); per_term is UNWEIGHTED

Rulings honoured:
- D10 (docs/DECISIONS.md): cells, keys, doors and locked doors pay only on first occurrence per
  episode. `FirstOccurrence` is the shared ledger for that; `BaseRewardTerm` clears every ledger it
  owns on each `reset`, so a term cannot forget to.
- Reward weights are config values, not contract.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:  # avoid a runtime dependency on training/envs (another module, built in parallel)
    from training.envs.factory import EnvSpec
else:
    EnvSpec = Any


@dataclass(frozen=True)
class StepContext:
    """Everything a term may read for one transition. Training-only."""

    priv_prev: dict  # privileged_state() before the step
    priv: dict  # privileged_state() after the step
    obs: dict  # raw official obs after the step (stale on the terminal step, wrapper.py:142)
    action_mb: np.ndarray  # (9,) int8 committed action
    terminated: bool
    truncated: bool
    env_reward: float  # wrapper default reward
    info: dict  # wrapper info (tic, exit_reached)
    step: int  # decision index in episode

    def __post_init__(self) -> None:
        a = np.asarray(self.action_mb)
        if a.shape != (9,):
            raise ValueError(f"StepContext.action_mb shape {a.shape} != (9,)")


@runtime_checkable
class RewardTerm(Protocol):
    name: str  # unique, snake_case; key in info["reward_terms"]

    def reset(self, priv0: dict, obs0: dict, inventory: dict, spec: EnvSpec) -> None: ...

    def __call__(self, ctx: StepContext) -> float: ...  # UNWEIGHTED value


REWARD_TERMS: dict[str, type] = {}

# Modules that define reward terms (each registers its classes on import). Owners: exploration-engineer,
# main session, combat-and-survival-engineer. A module that does not exist yet is skipped; any other
# import error propagates.
TERM_MODULES: tuple[str, ...] = (
    "training.rewards.exploration",
    "training.rewards.progression",
    "training.rewards.combat",
)


def register_term(cls: type) -> type:
    """Class decorator: add `cls` to REWARD_TERMS under `cls.name`. Duplicate names raise."""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum() or name != name.lower():
        raise ValueError(f"reward term {cls!r} needs a snake_case class attribute `name`, got {name!r}")
    existing = REWARD_TERMS.get(name)
    if existing is not None and existing is not cls:
        if (existing.__module__, existing.__qualname__) != (cls.__module__, cls.__qualname__):
            raise ValueError(f"reward term name {name!r} already registered by {existing!r}")
    REWARD_TERMS[name] = cls  # same class re-imported (e.g. module reload) replaces itself
    return cls


def load_term_modules(modules: Iterable[str] = TERM_MODULES) -> list[str]:
    """Import the term modules so their @register_term decorators run. Returns the modules loaded.

    Only a module that is itself missing is skipped. An ImportError raised *inside* an existing
    module (a broken dependency, a typo) propagates.
    """
    loaded = []
    for mod in modules:
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name is not None and (mod == e.name or mod.startswith(e.name + ".")):
                continue
            raise
        loaded.append(mod)
    return loaded


class FirstOccurrence:
    """Per-episode ledger of keys already paid for (D10: first occurrence per episode only)."""

    def __init__(self) -> None:
        self._seen: set[Hashable] = set()

    def clear(self) -> None:
        self._seen.clear()

    def first(self, key: Hashable) -> bool:
        """True exactly once per episode per key, and records it."""
        if key in self._seen:
            return False
        self._seen.add(key)
        return True

    def update(self, keys: Iterable[Hashable]) -> list[Hashable]:
        """Record `keys`; return those never seen this episode, in input order, without duplicates."""
        new = []
        for k in keys:
            if self.first(k):
                new.append(k)
        return new

    def __contains__(self, key: object) -> bool:
        return key in self._seen

    def __len__(self) -> int:
        return len(self._seen)


def new_set_bits(cur: np.ndarray | Iterable[float], ledger: FirstOccurrence) -> list[int]:
    """Indices i with cur[i] != 0 that were never set before this episode (e.g. `priv["keys"]`)."""
    flags = np.asarray(cur).reshape(-1)
    return [int(i) for i in ledger.update(int(i) for i in np.flatnonzero(flags))]


class BaseRewardTerm:
    """Optional base class for terms. Subclasses set `name`, implement `value(ctx)`, and may override
    `on_reset(...)`. Every FirstOccurrence returned by `ledger()` is cleared on each `reset`.

    Override `on_reset`, not `reset`.
    """

    name: ClassVar[str] = ""

    def __init__(self, **params: Any) -> None:
        if params:
            raise TypeError(f"{type(self).__name__} got unexpected params {sorted(params)}")
        self._ledgers: list[FirstOccurrence] = []

    def ledger(self) -> FirstOccurrence:
        led = FirstOccurrence()
        if not hasattr(self, "_ledgers"):
            self._ledgers = []
        self._ledgers.append(led)
        return led

    def reset(self, priv0: dict, obs0: dict, inventory: dict, spec: EnvSpec) -> None:
        for led in getattr(self, "_ledgers", ()):
            led.clear()
        self.on_reset(priv0, obs0, inventory, spec)

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: EnvSpec) -> None:
        pass

    def __call__(self, ctx: StepContext) -> float:
        return self.value(ctx)

    def value(self, ctx: StepContext) -> float:
        raise NotImplementedError


@dataclass
class _Entry:
    name: str
    weight: float
    term: Any
    ep_raw: float = 0.0
    ep_weighted: float = 0.0
    ep_nonzero: int = 0


_CFG_KEYS = frozenset({"name", "weight", "params"})


@dataclass
class EpisodeRewardStats:
    """Per-episode sums, for info["episode"] and logging."""

    steps: int = 0
    total: float = 0.0
    raw: dict[str, float] = field(default_factory=dict)  # sum of unweighted values per term
    weighted: dict[str, float] = field(default_factory=dict)  # sum of weight * value per term
    nonzero_steps: dict[str, int] = field(default_factory=dict)


class RewardComposer:
    """Weighted sum of reward terms built from config `[{"name", "weight", "params"}, ...]`."""

    def __init__(self, cfg: list[dict], *, registry: Mapping[str, type] | None = None) -> None:
        if not isinstance(cfg, list):
            raise TypeError(f"reward cfg must be a list of term dicts, got {type(cfg).__name__}")
        reg = REWARD_TERMS if registry is None else registry
        if registry is None and any(isinstance(c, Mapping) and c.get("name") not in reg for c in cfg):
            load_term_modules()
        self._entries: list[_Entry] = []
        for i, c in enumerate(cfg):
            if not isinstance(c, Mapping):
                raise TypeError(f"reward term cfg #{i} must be a dict, got {type(c).__name__}")
            unknown = set(c) - _CFG_KEYS
            if unknown or "name" not in c or "weight" not in c:
                raise ValueError(f"reward term cfg #{i} needs keys name, weight[, params]; got {sorted(c)}")
            name = c["name"]
            if name not in reg:
                raise KeyError(f"unknown reward term {name!r}; registered: {sorted(reg)}")
            if any(e.name == name for e in self._entries):
                raise ValueError(f"reward term {name!r} configured twice")
            weight = float(c["weight"])
            if not math.isfinite(weight):
                raise ValueError(f"reward term {name!r} weight must be finite, got {weight}")
            params = dict(c.get("params") or {})
            term = reg[name](**params)
            if getattr(term, "name", None) != name:
                raise ValueError(f"reward term class for {name!r} reports name {getattr(term, 'name', None)!r}")
            self._entries.append(_Entry(name, weight, term))
        self._ready = False
        self._steps = 0

    @classmethod
    def from_config(cls, reward_cfg: Mapping | list, **kw: Any) -> RewardComposer:
        """Accepts either the `reward:` block (`{"terms": [...]}`) or the term list itself."""
        if isinstance(reward_cfg, Mapping):
            unknown = set(reward_cfg) - {"terms"}
            if unknown:
                raise ValueError(f"unknown reward config keys {sorted(unknown)}")
            reward_cfg = list(reward_cfg.get("terms") or [])
        return cls(reward_cfg, **kw)

    @property
    def names(self) -> list[str]:
        return [e.name for e in self._entries]

    @property
    def weights(self) -> dict[str, float]:
        return {e.name: e.weight for e in self._entries}

    def reset(self, priv0: dict, obs0: dict, inventory: dict, spec: EnvSpec) -> None:
        """Start an episode: reset every term and the per-episode sums."""
        for e in self._entries:
            e.term.reset(priv0, obs0, inventory, spec)
            e.ep_raw = e.ep_weighted = 0.0
            e.ep_nonzero = 0
        self._steps = 0
        self._ready = True

    def __call__(self, ctx: StepContext) -> tuple[float, dict[str, float]]:
        """(weighted total, unweighted value per configured term). Every configured term is present."""
        if not self._ready:
            raise RuntimeError("RewardComposer called before reset()")
        per_term: dict[str, float] = {}
        total = 0.0
        for e in self._entries:
            raw = e.term(ctx)
            try:
                val = float(raw)
            except (TypeError, ValueError) as exc:
                raise TypeError(f"reward term {e.name!r} returned non-numeric {raw!r}") from exc
            if not math.isfinite(val):
                raise ValueError(f"reward term {e.name!r} returned non-finite value {val}")
            per_term[e.name] = val
            total += e.weight * val
            e.ep_raw += val
            e.ep_weighted += e.weight * val
            e.ep_nonzero += val != 0.0
        self._steps += 1
        return total, per_term

    def episode_stats(self) -> EpisodeRewardStats:
        """Sums since the last reset()."""
        return EpisodeRewardStats(
            steps=self._steps,
            total=sum(e.ep_weighted for e in self._entries),
            raw={e.name: e.ep_raw for e in self._entries},
            weighted={e.name: e.ep_weighted for e in self._entries},
            nonzero_steps={e.name: e.ep_nonzero for e in self._entries},
        )
