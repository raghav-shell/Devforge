"""Build a run-local TRAIN manifest from generated maps (owner: rl-training-engineer).

    uv run --no-sync python -m training.envs.mapgen --stage all --family a --seed 0 --count 200 --out runs/maps/gen-a-v1
    uv run --no-sync python -m training.configs.make_manifest --root runs/maps/gen-a-v1 --family a \
        --out runs/maps/gen-a-v1/train_gen_a.json

Uses the A3 registry (`training.envs.maps.generated_entries`, sha256-checked against each sidecar) and its
held-out guard (`check_trainable`: id, source_group and (sha256, map) vs maps/splits/val.json + test.json).
Additionally refuses any entry whose family is not in --family, or whose seed is >= 1000 (family b's
held-out seed range). The output has split "train", so MapSampler accepts it. Entries are uniform within a
manifest, so equal counts per stage give a uniform stage mix (no curriculum).
Generated maps for training go under runs/ (gitignored), never maps/generated/ (eval researcher's splits check).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for _p in (REPO / "submission", REPO):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from training.envs.maps import check_trainable, generated_entries, manifest_entries  # noqa: E402


def build(root: Path, families: tuple[str, ...] = ("a",), max_seed: int = 1000) -> dict:
    entries = generated_entries(root)
    if not entries:
        raise ValueError(f"no generated maps under {root}")
    bad = []
    for e in entries:
        meta = json.loads((REPO / e.meta_path).read_text()) if not Path(e.meta_path).is_absolute() else \
            json.loads(Path(e.meta_path).read_text())
        if meta["family"] not in families or int(meta["seed"]) >= max_seed:
            bad.append(e.id)
    if bad:
        raise ValueError(f"refusing {len(bad)} entries outside families={families} / seed<{max_seed}: {bad[:5]}")
    check_trainable(entries)
    stages = Counter(e.id.split("_")[1] for e in entries)
    return {
        "schema_version": "1.0",
        "split": "train",
        "description": f"run-local generated train manifest: families={list(families)}, root={root}",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "owner": "rl-training-engineer",
        "seeds": [],
        "stage_counts": dict(sorted(stages.items())),
        "maps": manifest_entries(entries, "train"),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--family", nargs="+", default=["a"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    root = Path(a.root) if Path(a.root).is_absolute() else REPO / a.root
    man = build(root, tuple(a.family))
    out = Path(a.out) if Path(a.out).is_absolute() else REPO / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(man, indent=1))
    print(json.dumps({"out": str(out), "maps": len(man["maps"]), "stage_counts": man["stage_counts"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
