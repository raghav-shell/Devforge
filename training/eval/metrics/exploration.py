"""Exploration metrics plug-in (`expl.` prefix). Eval-side only, pure functions of (EpisodeTrace, MapInventory).

Contract: docs/MODULE_CONTRACTS.md §7. Each MetricFn returns flat JSON scalars (Python int/float/str); a value
that is undefined for the episode (e.g. revisit ratio with no cell changes, or the step of a key that was never
picked) is float("nan"), matching core's `kills_per_100_ammo`. Definitions are in docs/EXPLORATION.md §3.

Shared definitions are imported from core so the two can never drift: the 64-unit cell, spawn-cell exclusion,
the stuck window (W = 32 steps, R = 48 units) and the door passability height (56). These are metrics only. They
never feed back into an action (no stuck override exists anywhere in the agent).
"""

from __future__ import annotations

import numpy as np

from ..inventory import CELL, MapInventory
from .core import PLAYER_HEIGHT, EpisodeTrace, MetricFn, _stuck_mask

STILL_EPS = 1.0  # map units per decision step: below this the player did not move
SPIN_DEG = 5.0  # |heading change| per step that counts as turning while still
HEADING_BINS = 8  # repeated-state key = (cell, 45-degree heading octant)
REVISIT_GAP = 32  # steps (= core STUCK_WINDOW, ~3.7 s): a return after a longer absence is a "long" revisit
NAN = float("nan")


def _cells(tr: EpisodeTrace) -> np.ndarray:
    xy = np.stack([np.asarray(tr.x, float), np.asarray(tr.y, float)], 1)
    return np.floor(xy / CELL).astype(np.int64)


def _runs(mask: np.ndarray) -> int:
    """Longest run of True."""
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def coverage_metrics(tr: EpisodeTrace, inv: MapInventory) -> dict:
    """Coverage (same cell and spawn rule as core) plus how efficiently it was gained."""
    n = len(tr.x)
    if n == 0:
        return {"expl.coverage_cells": 0, "expl.coverage_frac": 0.0, "expl.coverage_auc": 0.0,
                "expl.cells_per_100_steps": 0.0, "expl.cell_entries": 0, "expl.revisit_ratio": NAN,
                "expl.long_revisit_ratio": NAN}
    cells = [tuple(int(v) for v in c) for c in _cells(tr)]
    spawn = cells[0]
    denom = inv.cells_total - 1
    last_at = {spawn: 0}  # cell -> last row it was occupied
    new_by_t = np.zeros(n, np.int64)  # non-spawn distinct cells seen up to row t
    entries = revisits = long_revisits = 0
    for t in range(1, n):
        c = cells[t]
        if c != cells[t - 1]:
            entries += 1
            if c in last_at:
                revisits += 1
                long_revisits += int(t - last_at[c] > REVISIT_GAP)
        last_at[c] = t
        new_by_t[t] = len(last_at) - 1
    cells_new = int(new_by_t[-1])
    frac_t = np.minimum(1.0, new_by_t / denom) if denom > 0 else np.zeros(n)
    steps = max(1, n - 1)
    return {
        "expl.coverage_cells": cells_new,  # == core.cells_new == episode sum of the cell_visit reward term
        "expl.coverage_frac": float(frac_t[-1]),  # == core.cell_coverage
        "expl.coverage_auc": float(frac_t[1:].mean()) if n > 1 else 0.0,  # rewards finding cells early
        "expl.cells_per_100_steps": 100.0 * cells_new / steps,
        "expl.cell_entries": int(entries),  # steps whose cell differs from the previous step's
        "expl.revisit_ratio": float(revisits / entries) if entries else NAN,  # share of entries into a seen cell
        # entries into a cell left more than REVISIT_GAP steps ago: coming back to explored ground (loops,
        # backtracking out of dead ends), as opposed to local wobble between neighbouring cells
        "expl.long_revisit_ratio": float(long_revisits / entries) if entries else NAN,
    }


def repetition_metrics(tr: EpisodeTrace, inv: MapInventory) -> dict:
    """Spinning, standing still, stuck and repeated (cell, heading) states. Diagnostics only."""
    n = len(tr.x)
    if n < 2:
        return {"expl.repeated_state_frac": NAN, "expl.still_steps_frac": 0.0, "expl.spin_steps_frac": 0.0,
                "expl.stuck_steps": 0, "expl.longest_stuck_run": 0}
    xy = np.stack([np.asarray(tr.x, float), np.asarray(tr.y, float)], 1)
    cells = _cells(tr)
    ang = np.asarray(tr.angle, float) % 360.0
    octant = np.floor(ang / (360.0 / HEADING_BINS)).astype(np.int64) % HEADING_BINS
    # Consecutive identical states are collapsed (dwell is not repetition; still/stuck measure dwell). The
    # metric is the share of state CHANGES that land in a state already occupied earlier this episode.
    states = [(int(cells[t, 0]), int(cells[t, 1]), int(octant[t])) for t in range(n)]
    seen = {states[0]}
    changes = repeats = 0
    for t in range(1, n):
        s = states[t]
        if s != states[t - 1]:
            changes += 1
            repeats += int(s in seen)
        seen.add(s)
    disp = np.hypot(*np.diff(xy, axis=0).T)
    dang = np.abs((np.diff(ang) + 180.0) % 360.0 - 180.0)
    still = disp < STILL_EPS
    spin = still & (dang >= SPIN_DEG)
    sm = _stuck_mask(xy)  # identical rule to core.stuck_steps_frac (W = 32, R = 48)
    return {
        "expl.repeated_state_frac": repeats / changes if changes else NAN,
        "expl.still_steps_frac": float(still.mean()),
        "expl.spin_steps_frac": float(spin.mean()),  # still AND turning >= 5 deg: spinning in place
        "expl.stuck_steps": int(sm.sum()),
        "expl.longest_stuck_run": int(_runs(sm)),
    }


def _door_reversals(o: np.ndarray) -> tuple[int, int]:
    """(reversals, bounces) for one door's opening series. A reversal is a rise followed by a fall; a bounce is
    a reversal whose rising phase peaked below player height (never passable), i.e. USE re-pressed mid-motion."""
    rev = bounce = 0
    direction = 0
    for i in range(1, len(o)):
        d = o[i] - o[i - 1]
        if d == 0:
            continue
        s = 1 if d > 0 else -1
        if direction == 1 and s == -1:
            rev += 1
            bounce += int(o[i - 1] < PLAYER_HEIGHT)
        direction = s
    return rev, bounce


def objective_metrics(tr: EpisodeTrace, inv: MapInventory) -> dict:
    """Objective discovery as REACHED (keys picked, doors opened, exit). 'Seen' needs trace fields that
    EpisodeTrace does not record yet (requested from the eval researcher; docs/EXPLORATION.md §3.4)."""
    n = len(tr.x)
    keys = np.asarray(tr.keys, float).reshape(len(tr.keys), -1) if tr.keys else np.zeros((n, 6))
    k0 = keys[0] > 0 if len(keys) else np.zeros(6, bool)
    gained = (keys > 0) & ~k0 if len(keys) else np.zeros((0, 6), bool)
    first_key = NAN
    if len(gained) and gained.any():
        first_key = float(int(np.argmax(gained.any(axis=1))))
    keys_new = int(gained.any(axis=0).sum()) if len(gained) else 0
    door_ids = sorted(inv.door_sectors)
    do = np.asarray(tr.door_opening, float) if (tr.door_opening and door_ids) else np.zeros((n, 0))
    first_door = NAN
    opened = 0
    rev = bounce = 0
    closed0 = do[0] < PLAYER_HEIGHT if do.shape[0] and do.shape[1] else np.zeros(0, bool)
    if do.shape[0] and do.shape[1]:
        open_now = (do >= PLAYER_HEIGHT) & closed0
        if open_now.any():
            first_door = float(int(np.argmax(open_now.any(axis=1))))
        opened = int(open_now.any(axis=0).sum())
        for j in range(do.shape[1]):
            r, b = _door_reversals(do[:, j])
            rev += r
            bounce += b
    exit_reached = int(tr.outcome == "exit")
    total = int(inv.keys_present) + int(closed0.sum()) + 1
    reached = min(keys_new, int(inv.keys_present)) + opened + exit_reached
    return {
        "expl.keys_reached_frac": min(1.0, keys_new / inv.keys_present) if inv.keys_present > 0 else NAN,
        "expl.first_key_step": first_key,
        "expl.doors_reached_frac": opened / int(closed0.sum()) if closed0.sum() > 0 else NAN,  # normal + locked
        "expl.first_door_step": first_door,
        "expl.exit_reached": exit_reached,
        "expl.exit_step": float(tr.steps) if exit_reached else NAN,
        "expl.objectives_reached": int(reached),
        "expl.objectives_total": int(total),
        "expl.objectives_frac": reached / total,
        "expl.door_reversals": int(rev),
        "expl.door_bounces": int(bounce),
    }


# Plug-in registry entry (docs/MODULE_CONTRACTS.md §7).
METRICS: list[MetricFn] = [coverage_metrics, repetition_metrics, objective_metrics]
