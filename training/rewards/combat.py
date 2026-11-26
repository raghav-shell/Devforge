"""Combat reward terms: kills, damage taken and (optional, off by default) ammo spent. TRAINING-ONLY.

These terms read `privileged_state()` through `StepContext`. That is ground truth the agent never sees
at evaluation, and nothing under `submission/` may import this module (docs/OWNERSHIP.md rule 3).

Every term returns an UNWEIGHTED value; weights live in the run config (docs/DECISIONS.md D10).
H1 start weights (training/configs/h1.json): kill +0.5, damage_taken -0.01 per HP.

Robustness (docs/EVALUATION_PROTOCOL.md section 4.1, docs/MODULE_CONTRACTS.md section 6.1):
`kills` and `damage_taken` are monotone in the engine, but whether the post-exit terminal row keeps
them or reads 0 after the level change is OPEN. Both terms therefore pay against a per-episode
HIGH-WATER MARK seeded from `priv0`: value = max(0, current - highest seen so far). A row that drops
(to 0 or anything else) pays 0, and the next rise only pays above the old maximum, so the episode total
always equals max_t(var_t) - var_0 and nothing can pay twice. `kill` is further capped at the monsters
present at reset (see Kill), so monsters spawned mid-episode cannot be farmed.

`death` is NOT defined here: `training/rewards/progression.py` already registers it (and the registry
rejects duplicate names). docs/MODULE_CONTRACTS.md section 6.2 lists it under combat.py; see docs/COMBAT.md.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRewardTerm, StepContext, register_term

AMMO_SLICE = slice(4, 8)  # obs vitals: bullets, shells, rockets, cells (docs/CONTRACT.md section 2.1)


class _HighWaterDelta(BaseRewardTerm):
    """Pays max(0, priv[key] - running max of priv[key] this episode). Subclasses set `key`."""

    key: str = ""

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._high = float(priv0[self.key])

    def value(self, ctx: StepContext) -> float:
        cur = float(ctx.priv[self.key])
        if cur <= self._high:
            return 0.0
        gain = cur - self._high
        self._high = cur
        return gain


@register_term
class Kill(_HighWaterDelta):
    """+1 per new kill this step (privileged KILLCOUNT above its episode maximum). Use a positive weight.

    Anti-farming cap (default on): the episode total is capped at `inventory["monsters"]`, the live
    monster count at reset, when the inventory provides it. Monsters spawned mid-episode (Pain
    Elemental lost souls, roster level 3 in the mapgen `full` stage) then cannot pay beyond what the
    evaluator credits, which is min(1, kills / monsters_at_reset) (docs/EVALUATION_PROTOCOL.md). With no
    `monsters` key in the inventory the cap is off. `cap_at_monsters=False` disables it.

    KILLCOUNT is the engine kill counter. Whether infighting kills count toward it is UNVERIFIED here
    (docs/COMBAT.md section 2.3).
    """

    name = "kill"
    key = "kills"

    def __init__(self, cap_at_monsters: bool = True) -> None:
        super().__init__()
        self._cap_on = bool(cap_at_monsters)

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        super().on_reset(priv0, obs0, inventory, spec)
        m = (inventory or {}).get("monsters") if self._cap_on else None
        self._left = None if m is None else max(0.0, float(m))

    def value(self, ctx: StepContext) -> float:
        gain = super().value(ctx)
        if self._left is None or gain == 0.0:
            return gain
        paid = min(gain, self._left)
        self._left -= paid
        return paid


@register_term
class DamageTaken(_HighWaterDelta):
    """HP of damage taken this step (positive). Use a negative weight (H1: -0.01 per HP).

    MEASURED (docs/COMBAT.md section 2.2): the per-step delta of privileged `damage_taken` equals the
    health lost. Points absorbed by armor are NOT included. Overkill past 0 HP IS included: final health
    went as low as -20 over 10 seeds. Health pickups did not lower it, since it is monotone in every
    measured episode. All damage sources count; that is per docs/EVALUATION_PROTOCOL.md and was not
    measured here, and it matches the evaluator's damage figure.
    """

    name = "damage_taken"
    key = "damage_taken"


@register_term
class AmmoSpent(BaseRewardTerm):
    """OFF BY DEFAULT (not in any config). Ammo units spent this step, from obs vitals[4:8].

    Counts only decreases per ammo type (pickups raise the counts and are ignored), times an optional
    per-type weight (bullets, shells, rockets, cells). The terminal obs is stale (wrapper.py:142), so
    shots fired on the very last step are not counted. This penalises ALL shooting, not only misses: a
    true "wasted ammo" signal needs hit or damage-dealt ground truth that privileged_state does not
    expose (docs/COMBAT.md section 3). Use a small negative weight only together with `kill`.
    """

    name = "ammo_spent"

    def __init__(self, type_weights: tuple[float, float, float, float] | list[float] = (1.0, 1.0, 1.0, 1.0)) -> None:
        super().__init__()
        w = np.asarray(type_weights, np.float64).reshape(-1)
        if w.shape != (4,) or not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError(f"ammo_spent type_weights must be 4 finite non-negative floats, got {type_weights!r}")
        self._w = w

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._prev = np.asarray(obs0["vitals"][AMMO_SLICE], np.float64).copy()

    def value(self, ctx: StepContext) -> float:
        if ctx.terminated or ctx.truncated:  # stale obs: nothing measurable
            return 0.0
        cur = np.asarray(ctx.obs["vitals"][AMMO_SLICE], np.float64)
        spent = np.clip(self._prev - cur, 0.0, None)
        self._prev = cur.copy()
        return float(spent @ self._w)
