"""Core per-episode metrics computed from a recorded trajectory (eval-side only).

Shared metric interface (PROPOSED CONTRACT, docs/EVALUATION_PROTOCOL.md §9.3):

    MetricFn = Callable[[EpisodeTrace, MapInventory], dict[str, float | int | str]]

Every plug-in (core, exploration.py, combat.py) is a pure function of the trace and the inventory,
returns flat JSON-serialisable values, and prefixes its keys with its namespace (`core.`, `expl.`,
`combat.`). Formulas: docs/EVALUATION_PROTOCOL.md §4.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..inventory import CELL, MapInventory

PLAYER_HEIGHT = 56.0  # doom_env.sensors.PLAYER_HEIGHT: a door counts as open at this opening
STUCK_WINDOW = 32  # decision steps (~3.7 s)
STUCK_RADIUS = 48.0  # map units
NO_NEW_CELL_STEPS = 100
FRAME_SKIP = 4
AMMO_SLICE = slice(4, 8)  # vitals: bullets, shells, rockets, cells


@dataclass
class EpisodeTrace:
    """One row per decision step, recorded after env.step(). Row 0 = state right after reset."""

    episode_ticks: int
    x: list[float] = field(default_factory=list)
    y: list[float] = field(default_factory=list)
    angle: list[float] = field(default_factory=list)
    health: list[float] = field(default_factory=list)
    kills: list[int] = field(default_factory=list)
    damage_taken: list[float] = field(default_factory=list)
    keys: list[list[float]] = field(default_factory=list)
    ammo: list[list[float]] = field(default_factory=list)  # obs vitals[4:8]
    door_opening: list[list[float]] = field(default_factory=list)  # ceil - floor per door sector (sorted ids)
    actions: list[list[int]] = field(default_factory=list)  # action applied to reach row i (row 0: none)
    env_reward: list[float] = field(default_factory=list)
    outcome: str = ""  # exit | death | timeout
    agent_failure: str | None = None  # None or "<kind>@step<k>"
    failure_step: int | None = None
    outcome_consistent: bool = True  # outcome agrees with privileged dead/timeout/exit_reached/finished
    steps: int = 0

    def record(self, st: dict, obs: dict, door_ids: list[int], action, reward: float) -> None:
        self.x.append(st["x"]); self.y.append(st["y"]); self.angle.append(st["angle"])
        self.health.append(st["health"]); self.kills.append(st["kills"])
        self.damage_taken.append(st["damage_taken"]); self.keys.append(st["keys"].tolist())
        self.ammo.append(np.asarray(obs["vitals"][AMMO_SLICE], float).tolist())
        f, c = st["sector_floor"], st["sector_ceiling"]
        self.door_opening.append([float(c[d] - f[d]) for d in door_ids])
        self.actions.append([] if action is None else [int(a) for a in action])
        self.env_reward.append(float(reward))


MetricFn = Callable[[EpisodeTrace, MapInventory], dict]


def _stuck_mask(xy: np.ndarray) -> np.ndarray:
    n = len(xy)
    m = np.zeros(n, bool)
    for t in range(STUCK_WINDOW, n):
        w = xy[t - STUCK_WINDOW:t + 1]
        m[t] = np.max(np.hypot(*(w - xy[t]).T)) < STUCK_RADIUS
    return m


def core_metrics(tr: EpisodeTrace, inv: MapInventory) -> dict:
    xy = np.stack([tr.x, tr.y], 1)
    steps = tr.steps
    tic_end = min(steps * FRAME_SKIP, tr.episode_ticks)  # engine tic reads 0 after an exit (§4.1)
    # coverage: 64-unit cells and sector area
    cells = np.floor(xy / CELL).astype(int)
    seen, first_new = set(), []
    for i, c in enumerate(map(tuple, cells)):
        if c not in seen:
            seen.add(c); first_new.append(i)
    # spawn cell and spawn sector are excluded from numerator AND denominator: standing still earns 0
    spawn_cell = tuple(cells[0])
    spawn_sec = inv.sector_at(float(xy[0, 0]), float(xy[0, 1]))
    uniq = np.unique(np.round(xy / 4.0) * 4.0, axis=0)  # 4-unit dedupe keeps the lookup cheap
    sectors = {inv.sector_at(float(px), float(py)) for px, py in uniq} - {-1}
    sa = inv.sector_area
    denom_area = float(sa.sum() - (sa[spawn_sec] if 0 <= spawn_sec < len(sa) else 0.0))
    num_area = float(sum(sa[s] for s in sectors if s < len(sa) and s != spawn_sec))
    area_cov = num_area / denom_area if denom_area > 0 else 0.0
    cells_new = len(seen - {spawn_cell})
    cells_denom = inv.cells_total - 1
    # no-new-cell streak
    gaps = np.diff(first_new + [len(xy) - 1]) if first_new else np.array([0])
    # stuck
    sm = _stuck_mask(xy)
    events = int(np.sum(sm[1:] & ~sm[:-1])) + int(sm[0]) if len(sm) else 0
    # doors (latched once per sector)
    door_ids = sorted(inv.door_sectors)
    do = np.array(tr.door_opening) if tr.door_opening and door_ids else np.zeros((len(xy), 0))
    init_closed = do[0] < PLAYER_HEIGHT if do.shape[1] else np.zeros(0, bool)
    opened = init_closed & np.any(do >= PLAYER_HEIGHT, axis=0) if do.shape[1] else np.zeros(0, bool)
    locked = np.array([inv.door_sectors[d] > 0 for d in door_ids], bool)
    # keys
    keys_end = np.array(tr.keys[-1]) if tr.keys else np.zeros(6)
    # ammo
    am = np.array(tr.ammo)
    spent = float(np.clip(am[:-1] - am[1:], 0, None).sum()) if len(am) > 1 else 0.0
    acts = np.array([a for a in tr.actions if a]) if any(tr.actions) else np.zeros((0, 9))
    # Terminal row after an exit is read after the level change (tic reads 0, MEASURED); kills and
    # damage are monotone so take the max over the trace, and on exit take health from the last
    # pre-terminal row.
    kills = int(max(tr.kills)) if tr.kills else 0
    dmg = float(max(tr.damage_taken)) if tr.damage_taken else 0.0
    dead = tr.outcome == "death"
    if dead:
        hp_end = 0.0
    elif tr.outcome == "exit" and len(tr.health) > 1:
        hp_end = float(max(0.0, tr.health[-2]))
    else:
        hp_end = float(max(0.0, tr.health[-1]))
    return {
        "core.outcome": tr.outcome,
        "core.completed": int(tr.outcome == "exit"),
        "core.survived": int(not dead),
        "core.agent_failure": tr.agent_failure or "",
        "core.outcome_consistent": int(tr.outcome_consistent),
        "core.steps": steps,
        "core.tic_end": tic_end,
        "core.time_frac": tic_end / tr.episode_ticks,
        "core.health_end": hp_end,
        "core.damage_taken": dmg,
        "core.kills": kills,
        "core.monsters_present": inv.monsters_present,
        "core.cells_visited": len(seen),
        "core.cells_total": inv.cells_total,
        "core.cells_new": cells_new,
        "core.cell_coverage": min(1.0, cells_new / cells_denom) if cells_denom > 0 else 0.0,
        "core.sectors_visited": len(sectors),
        "core.area_coverage": area_cov,
        "core.max_no_new_cell_steps": int(gaps.max()) if len(gaps) else 0,
        "core.stuck_steps_frac": float(sm.mean()) if len(sm) else 0.0,
        "core.stuck_events_per_1k": 1000.0 * events / max(1, steps),
        "core.keys_picked": int(keys_end.sum()),
        "core.keys_present": inv.keys_present,
        "core.doors_opened": int((opened & ~locked).sum()),
        "core.doors_total": int((~locked).sum()),
        "core.locked_opened": int((opened & locked).sum()),
        "core.locked_total": int(locked.sum()),
        "core.ammo_spent": spent,
        "core.kills_per_100_ammo": 100.0 * kills / spent if spent > 0 else float("nan"),
        "core.attack_rate": float(acts[:, 6].mean()) if len(acts) else 0.0,
        "core.use_rate": float(acts[:, 7].mean()) if len(acts) else 0.0,
        "core.env_return": float(np.sum(tr.env_reward)),
    }


def classify_failure(m: dict) -> str:
    """Failure taxonomy, docs/EVALUATION_PROTOCOL.md §5. Exploration/combat plug-ins may refine it."""
    if m["core.outcome"] == "exit":
        return "success" if not m["core.agent_failure"] else "success_after_agent_failure"
    if m["core.agent_failure"]:
        return "agent_failure:" + m["core.agent_failure"].split("@")[0]
    if m["core.outcome"] == "death":
        return "death_early" if m["core.time_frac"] < 0.25 else "death_late"
    # timeout
    if m["core.stuck_steps_frac"] >= 0.5:
        return "timeout_stuck"
    if m["core.locked_total"] > m["core.locked_opened"] and m["core.keys_picked"] < m["core.keys_present"]:
        return "timeout_missing_key"
    if m["core.max_no_new_cell_steps"] >= 0.5 * m["core.steps"]:
        return "timeout_looping"
    return "timeout_exploring"


# Plug-in registry entry (docs/MODULE_CONTRACTS.md §7): every metric module exports METRICS: list[MetricFn].
METRICS: list[MetricFn] = [core_metrics]
