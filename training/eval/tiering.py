"""Static, reproducible tier rule for maps without a generator tier (docs/EVALUATION_PROTOCOL.md §1.2).

Rule `static_rule_v1`. It reads only static WAD data: `doom_env.mapinfo.load_mapinfo` for door/lock sectors and
`training.envs.inventory.static_things` for skill-3 single-player monsters. No engine, no replays, no outcomes.
It mirrors the generator's own tier assignment (`training/envs/mapgen/generator.py:317-319`, `config.py` stage
tiers), so one rule covers generated and authored maps:

    L = number of distinct key colours that lock some door (lock codes normalised: 1/4/129 red, 2/5/130 blue,
        3/6/131 yellow; 100 "any key" counts as one colour; 101 "all keys" as red+blue+yellow)
    M = static monsters (skill 3, single player), D = ordinary doors, S = sectors
    tier 4  if L >= 2, or (L >= 1 and M >= 6)          # multiple locks, or a lock plus a monster gauntlet
    tier 3  if L == 1                                    # key -> locked door, no meaningful combat
    tier 2  if L == 0 and (D >= 1 or (M == 0 and S >= 2))  # doors / multi-room exploration
    tier 1  if L == 0 and D == 0 and M >= 1              # combat without door or key mechanics
    null    otherwise (single sector, nothing to do)

Monster density is deliberately NOT used for tier 1 vs 2: EVALUATION.md lists doors as a tier-2 skill, and the
density of generator combat-a maps (MEASURED 0.88-4.0 monsters / 100 cells) overlaps door-heavy Freedoom levels.

Known limits: the rule cannot tell whether a lock is on the exit path (authored maps may have optional locks),
and ACS-spawned monsters (ViZDoom scenarios) are invisible to it.

    uv run --no-sync python -m training.eval.tiering            # agreement with generator tiers + all rule tiers
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

RULE_VERSION = "static_rule_v1"
_COLOUR = {1: "red", 4: "red", 129: "red", 2: "blue", 5: "blue", 130: "blue", 3: "yellow", 6: "yellow", 131: "yellow",
           100: "any"}


def lock_colours(codes) -> set[str]:
    out: set[str] = set()
    for c in codes:
        c = int(c)
        if c <= 0:
            continue
        if c == 101:
            out |= {"red", "blue", "yellow"}
        else:
            out.add(_COLOUR.get(c, f"code{c}"))
    return out


def static_inputs(wad: str | Path, map_name: str) -> dict:
    from doom_env.mapinfo import load_mapinfo

    from training.envs.inventory import static_things

    mi = load_mapinfo(Path(wad), map_name)
    locks = [v for v in mi.door_sectors.values() if v > 0]
    monsters, keys = static_things(str(wad), map_name)
    return {"sectors": int(mi.num_sectors), "doors": sum(1 for v in mi.door_sectors.values() if v == 0),
            "locked_sectors": len(locks), "lock_colours": sorted(lock_colours(locks)), "static_monsters": int(monsters),
            "static_keys": int(keys)}


def tier_from_inputs(x: dict) -> int | None:
    L, M, D, S = len(x["lock_colours"]), x["static_monsters"], x["doors"], x["sectors"]
    if L >= 2 or (L >= 1 and M >= 6):
        return 4
    if L == 1:
        return 3
    if D >= 1 or (M == 0 and S >= 2):
        return 2
    if M >= 1:
        return 1
    return None


def static_tier(wad: str | Path, map_name: str) -> tuple[int | None, dict]:
    x = static_inputs(wad, map_name)
    return tier_from_inputs(x), x


def describe(x: dict) -> str:
    return (f"{RULE_VERSION}: lock_colours={','.join(x['lock_colours']) or '-'} locked_sectors={x['locked_sectors']} "
            f"doors={x['doors']} static_monsters={x['static_monsters']} sectors={x['sectors']}")


def main() -> int:
    from training.envs import map_registry as R

    agree = total = 0
    rows = []
    for e in R.all_entries():
        t, x = static_tier(e.path, e.map)
        rows.append({"id": e.id, "source_group": e.source_group, "registry_tier": e.tier, "rule_tier": t, **x})
        if e.tier is not None:
            total += 1
            agree += int(t == e.tier)
            if t != e.tier:
                print(f"DISAGREE {e.id}: generator={e.tier} rule={t} ({describe(x)})")
    if "--json" in sys.argv:
        print(json.dumps(rows, indent=1))
    else:
        for r in rows:
            if r["registry_tier"] is None:
                print(f"{r['id']:34s} {r['source_group']:28s} rule_tier={r['rule_tier']}")
    print(f"rule vs generator tier agreement: {agree}/{total}")
    return 0 if agree == total else 1


if __name__ == "__main__":
    sys.exit(main())
