"""Proxy for the hidden official score (APPROXIMATION; docs/EVALUATION_PROTOCOL.md §3).

The official weights are hidden (docs/EVALUATION.md). These weights are provisional and frozen per
PROXY_VERSION: never compare numbers produced by different versions. Raw components are stored per
episode so any future version can be recomputed offline from saved metrics.
"""

from __future__ import annotations

PROXY_VERSION = "proxy_v1"  # v0 = pre-freeze drafts (area-only, then no spawn exclusion); not comparable

W = {
    "exit": 50.0,  # largest category (EVALUATION.md: "Exit reached (largest)")
    "exit_health": 5.0,  # x health_end/100 (clip 0..1), only on exit
    "exit_time": 5.0,  # x (1 - tic_end/episode_ticks), only on exit
    "progression": 15.0,  # x fraction of available progression units
    "unit_key": 1.0, "unit_door": 1.0, "unit_locked": 2.0,
    "exploration": 15.0,  # x (0.5 sector-area coverage + 0.5 64-cell coverage)
    "combat": 10.0,  # x min(1, kills / monsters_present)
    "damage_per_hp": 0.02, "damage_cap": 5.0,  # all damage (no self-damage split available)
    "early_death": 10.0,  # x (1 - tic_end/episode_ticks), only on death
}
TIER_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}


def proxy_components(m: dict) -> dict:
    exit_ = m["core.completed"] == 1
    units_avail = (W["unit_key"] * m["core.keys_present"] + W["unit_door"] * m["core.doors_total"]
                   + W["unit_locked"] * m["core.locked_total"])
    units_got = (W["unit_key"] * min(m["core.keys_picked"], m["core.keys_present"]) + W["unit_door"] * m["core.doors_opened"]
                 + W["unit_locked"] * m["core.locked_opened"])
    mons = m["core.monsters_present"]
    c = {
        "proxy.exit": W["exit"] * exit_,
        "proxy.exit_eff": (W["exit_health"] * min(1.0, m["core.health_end"] / 100.0)
                           + W["exit_time"] * (1.0 - m["core.time_frac"])) if exit_ else 0.0,
        "proxy.progression": W["progression"] * (units_got / units_avail) if units_avail > 0 else 0.0,
        "proxy.exploration": W["exploration"] * (0.5 * m["core.area_coverage"] + 0.5 * m["core.cell_coverage"]),
        "proxy.combat": W["combat"] * min(1.0, m["core.kills"] / mons) if mons > 0 else 0.0,
        "proxy.damage_pen": -min(W["damage_cap"], W["damage_per_hp"] * m["core.damage_taken"]),
        "proxy.death_pen": -W["early_death"] * (1.0 - m["core.time_frac"]) if m["core.outcome"] == "death" else 0.0,
    }
    c["proxy.score"] = sum(c.values())
    c["proxy.version"] = PROXY_VERSION
    return c
