"""Combat and survival metrics plug-in (prefix `combat.`). Eval-side only; pure function of the trace.

Contract: docs/MODULE_CONTRACTS.md section 7 (`METRICS: list[MetricFn]`, `MetricFn(EpisodeTrace,
MapInventory) -> dict`, JSON scalars, keys prefixed `combat.`). Definitions: docs/COMBAT.md section 4.

Quirks handled exactly as training/eval/metrics/core.py does:
- kills and damage_taken are monotone, so take the max over the trace (robust to a post-exit row reset);
- on an exit the terminal row is read after the level change, so health uses rows up to N-1;
- the terminal obs is stale (wrapper.py:142), so ammo spent on the last step is not counted.
Undefined ratios are NaN (as core.kills_per_100_ammo); the harness aggregates with nanmean.
"""

from __future__ import annotations

import numpy as np

from ..inventory import MapInventory
from .core import FRAME_SKIP, EpisodeTrace, MetricFn

TICRATE = 35  # Doom tics per second
ATTACK, SELECT_NEXT_WEAPON = 6, 8  # docs/CONTRACT.md action layout
AMMO_NAMES = ("bullets", "shells", "rockets", "cells")
NAN = float("nan")


def _first_rise(v: np.ndarray) -> int:
    """First row index i >= 1 where v rises above v[0]; -1 if never."""
    if len(v) < 2:
        return -1
    idx = np.flatnonzero(v[1:] > v[0])
    return int(idx[0]) + 1 if len(idx) else -1


def combat_metrics(tr: EpisodeTrace, inv: MapInventory) -> dict:
    n_rows = len(tr.kills)
    steps = int(tr.steps)
    t_end = min(steps * FRAME_SKIP, int(tr.episode_ticks))
    minutes = t_end / (TICRATE * 60.0)
    dead = tr.outcome == "death"
    present = int(inv.monsters_present)

    kills_v = np.asarray(tr.kills, np.float64)
    dmg_v = np.asarray(tr.damage_taken, np.float64)
    kills = int(kills_v.max() - kills_v[0]) if n_rows else 0
    dmg = float(dmg_v.max() - dmg_v[0]) if n_rows else 0.0

    # health rows that are valid: drop the post-level-change terminal row on an exit
    hp = np.asarray(tr.health, np.float64)
    if tr.outcome == "exit" and len(hp) > 1:
        hp = hp[:-1]
    if dead:
        hp_end = 0.0
    else:
        hp_end = float(max(0.0, hp[-1])) if len(hp) else 0.0
    hp_min = 0.0 if dead else (float(max(0.0, hp.min())) if len(hp) else 0.0)
    hp_gained = float(np.clip(np.diff(hp), 0.0, None).sum()) if len(hp) > 1 else 0.0

    am = np.asarray(tr.ammo, np.float64).reshape(-1, 4) if tr.ammo else np.zeros((0, 4))
    spent_by = np.clip(am[:-1] - am[1:], 0.0, None).sum(0) if len(am) > 1 else np.zeros(4)
    spent = float(spent_by.sum())
    no_ammo_frac = float(np.mean(am.sum(1) <= 0)) if len(am) else 0.0

    acts = [a for a in tr.actions if a]
    acts_a = np.asarray(acts, np.int64).reshape(-1, 9) if acts else np.zeros((0, 9), np.int64)
    attack_steps = int(acts_a[:, ATTACK].sum())
    sw = acts_a[:, SELECT_NEXT_WEAPON]
    switch_edges = int(np.sum((sw[1:] == 1) & (sw[:-1] == 0)) + (sw[0] == 1)) if len(sw) else 0

    out = {
        "combat.kills": kills,
        "combat.monsters_present": present,
        "combat.kill_frac": min(1.0, kills / present) if present > 0 else NAN,
        "combat.cleared": int(present > 0 and kills >= present),
        "combat.kills_per_min": kills / minutes if minutes > 0 else NAN,
        "combat.first_kill_step": _first_rise(kills_v),
        "combat.damage_taken": dmg,
        "combat.damage_per_min": dmg / minutes if minutes > 0 else NAN,
        "combat.first_damage_step": _first_rise(dmg_v),
        "combat.health_end": hp_end,
        "combat.health_min": hp_min,
        "combat.health_gained": hp_gained,
        "combat.died": int(dead),
        "combat.survived": int(not dead),
        "combat.death_time_frac": t_end / tr.episode_ticks if dead and tr.episode_ticks > 0 else NAN,
        "combat.ammo_spent": spent,
        "combat.kills_per_100_ammo": 100.0 * kills / spent if spent > 0 else NAN,
        "combat.no_ammo_steps_frac": no_ammo_frac,
        "combat.attack_steps": attack_steps,
        "combat.attack_rate": attack_steps / len(acts_a) if len(acts_a) else 0.0,
        "combat.kills_per_100_attacks": 100.0 * kills / attack_steps if attack_steps > 0 else NAN,
        "combat.weapon_switch_presses": switch_edges,
    }
    for name, v in zip(AMMO_NAMES, spent_by):
        out[f"combat.ammo_spent_{name}"] = float(v)
    return out


# Plug-in registry entry (docs/MODULE_CONTRACTS.md section 7).
METRICS: list[MetricFn] = [combat_metrics]
