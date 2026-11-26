"""CLI: generate training maps.

    uv run --no-sync python -m training.envs.mapgen --stage keys --seed 123 --out maps/generated/keys
    uv run --no-sync python -m training.envs.mapgen --stage all --seed 0 --count 10 --out maps/generated
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .config import FAMILIES, STAGES
from .generator import generate


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m training.envs.mapgen", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=(*STAGES, "all"))
    p.add_argument("--seed", type=int, default=0, help="first seed")
    p.add_argument("--count", type=int, default=1, help="number of consecutive seeds")
    p.add_argument("--family", default="a", choices=FAMILIES)
    p.add_argument("--map", default="MAP01")
    p.add_argument("--out", required=True,
                   help="output dir (with --stage all, maps go to <out>/<stage>/), or a .wad path for a single map")
    a = p.parse_args(argv)
    stages = STAGES if a.stage == "all" else (a.stage,)
    out = Path(a.out)
    for stage in stages:
        for seed in range(a.seed, a.seed + a.count):
            m = generate(stage, seed, family=a.family, map_name=a.map)
            if out.suffix.lower() == ".wad":
                if len(stages) * a.count != 1:
                    p.error("a .wad --out needs exactly one map")
                wad, js = m.write(out.parent, out.stem)
            else:
                wad, js = m.write(out / stage if a.stage == "all" else out)
            md = m.meta
            print(f"{wad}  sectors={md['num_sectors']} rooms={md['room_count']} doors={md['num_doors']} "
                  f"locks={md['num_locked_doors']} monsters={md['num_monsters']} exit={md['exit']['kind']} "
                  f"solvable={md['solvable']} attempt={md['attempt']} sha256={md['wad_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
