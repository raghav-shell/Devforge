"""Load and validate map-split manifests (maps/splits/*.json). Schema: docs/EVALUATION_PROTOCOL.md §1.4.

    uv run --no-sync python -m training.eval.splits            # validate all manifests
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPLITS_DIR = ROOT / "maps" / "splits"
SCHEMA_VERSION = "1.0"
SPLITS = ("smoke", "train", "val", "test")
REQUIRED = ("id", "wad", "map", "wad_sha256", "source_group", "source", "license", "tier", "episode_ticks", "role")
ROLES = ("smoke", "train", "val", "test")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(split: str) -> dict:
    return json.loads((SPLITS_DIR / f"{split}.json").read_text())


def validate_all(check_hash: bool = True) -> list[str]:
    errs: list[str] = []
    groups: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    ids: set[str] = set()
    for split in SPLITS:
        p = SPLITS_DIR / f"{split}.json"
        if not p.is_file():
            errs.append(f"{split}: manifest missing"); continue
        man = json.loads(p.read_text())
        if man.get("schema_version") != SCHEMA_VERSION or man.get("split") != split:
            errs.append(f"{split}: bad schema_version/split field")
        for e in man.get("maps", []):
            miss = [k for k in REQUIRED if k not in e]
            if miss:
                errs.append(f"{split}:{e.get('id')}: missing {miss}"); continue
            if e["id"] in ids:
                errs.append(f"{split}:{e['id']}: duplicate id")
            ids.add(e["id"])
            if e["tier"] not in (None, 1, 2, 3, 4):
                errs.append(f"{split}:{e['id']}: tier must be 1-4 or null")
            if not 1 <= int(e["episode_ticks"]) <= 4200:
                errs.append(f"{split}:{e['id']}: episode_ticks out of range")
            wad = ROOT / e["wad"]
            if check_hash and not e["wad"].startswith("@vizdoom/"):
                if not wad.is_file():
                    errs.append(f"{split}:{e['id']}: {e['wad']} not found")
                elif sha256(wad) != e["wad_sha256"]:
                    errs.append(f"{split}:{e['id']}: sha256 mismatch")
            groups.setdefault(e["source_group"], set()).add(split)
            hashes.setdefault(f"{e['wad_sha256']}:{e['map']}", set()).add(split)
    # leakage: a source group or a (wad, map) may not span two of train/val/test (smoke may overlap train)
    for kind, table in (("source_group", groups), ("wad:map", hashes)):
        for k, ss in table.items():
            strict = ss & {"train", "val", "test"}
            if len(strict) > 1 or ({"val", "test"} & ss and "smoke" in ss):
                errs.append(f"LEAK: {kind} {k} appears in {sorted(ss)}")
    return errs


if __name__ == "__main__":
    problems = validate_all()
    for p in problems:
        print("ERROR", p)
    print("manifests OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
