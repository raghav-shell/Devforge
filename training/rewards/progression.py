"""Progression reward terms: keys, doors, locked doors, exit and death. TRAINING-ONLY.

These terms read `privileged_state()` through `StepContext` and the map inventory at reset. Neither
is available to the agent at evaluation, and nothing under `submission/` may import this module.

Every term returns an UNWEIGHTED value. Weights live in the run config (docs/DECISIONS.md D10:
weights are config values, not contract). Each event pays **once per episode** (D10): a key dropped
and picked up again, or a door that closes and reopens, does not pay twice.

Suggested starting weights (from docs/RESEARCH_NOTES.md §6, to be tuned): key 3, door 1,
locked door 3, exit 10, death -3.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .base import BaseRewardTerm, StepContext, new_set_bits, register_term

# A door sector counts as open once its ceiling-to-floor gap fits the player.
# Same definition as the evaluation metrics (training/eval/metrics/core.py, doom_env PLAYER_HEIGHT).
PLAYER_HEIGHT = 56.0


@register_term
class KeyPickup(BaseRewardTerm):
    """+1 for each key (card or skull) that becomes held for the first time this episode."""

    name = "key_pickup"

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._held = self.ledger()
        new_set_bits(priv0["keys"], self._held)  # keys held at spawn never pay

    def value(self, ctx: StepContext) -> float:
        return float(len(new_set_bits(ctx.priv["keys"], self._held)))


class _DoorOpenBase(BaseRewardTerm):
    """+1 the first time each door sector of the selected kind opens to player height this episode.

    Only doors that start closed at reset count, so a door sector that is already open at spawn never
    pays on step 0.
    """

    locked: bool = False

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._opened = self.ledger()
        doors = inventory.get("door_sectors") or {}
        floor, ceil = priv0.get("sector_floor"), priv0.get("sector_ceiling")
        self._sectors: list[int] = []
        if floor is None or ceil is None:
            return
        for sector, lock in doors.items():
            s = int(sector)
            if bool(lock) != self.locked:
                continue
            if float(ceil[s] - floor[s]) < PLAYER_HEIGHT:  # starts closed
                self._sectors.append(s)
        self._sectors.sort()

    def value(self, ctx: StepContext) -> float:
        if not self._sectors:
            return 0.0
        floor, ceil = ctx.priv.get("sector_floor"), ctx.priv.get("sector_ceiling")
        if floor is None or ceil is None:
            return 0.0
        idx = np.asarray(self._sectors)
        now_open = idx[(np.asarray(ceil)[idx] - np.asarray(floor)[idx]) >= PLAYER_HEIGHT]
        return float(len(self._opened.update(int(s) for s in now_open)))


@register_term
class DoorOpen(_DoorOpenBase):
    name = "door_open"
    locked = False


@register_term
class LockedDoorOpen(_DoorOpenBase):
    name = "locked_door_open"
    locked = True


@register_term
class ExitReached(BaseRewardTerm):
    """+1 on the step the exit is reached (at most once per episode)."""

    name = "exit_reached"

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._paid = self.ledger()

    def value(self, ctx: StepContext) -> float:
        if ctx.priv.get("exit_reached") and self._paid.first("exit"):
            return 1.0
        return 0.0


@register_term
class Death(BaseRewardTerm):
    """+1 on the step the player dies (use a negative weight). Timeouts are not deaths."""

    name = "death"

    def on_reset(self, priv0: dict, obs0: dict, inventory: dict, spec: Any) -> None:
        self._paid = self.ledger()

    def value(self, ctx: StepContext) -> float:
        if ctx.terminated and ctx.priv.get("dead") and self._paid.first("dead"):
            return 1.0
        return 0.0
