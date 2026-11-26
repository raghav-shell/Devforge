"""Build maps/splits/{train,val,test}.json from the map registry (docs/EVALUATION_PROTOCOL.md §1).

Entries come from `training.envs.maps.all_entries()` + `manifest_entries(entries, role)`; nothing is hand-typed.
Assignment is per source_group (GROUP_SPLIT below), so near-duplicates cannot straddle splits. Maps whose
registry tier is null get the static rule tier (`training.eval.tiering`, tier_basis "static_rule_v1").
Manifest order = registry order (deterministic); the random baseline's per-map results depend on it (§5).

    uv run --no-sync python -m training.eval.build_splits --check   # exit 1 if committed manifests differ
    uv run --no-sync python -m training.eval.build_splits --write   # regenerate train/val/test (smoke untouched)
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import sys
from pathlib import Path

from . import splits
from .tiering import RULE_VERSION, describe, static_tier

# (source_group glob, split). First match wins; a group that matches nothing is an error (no silent default).
GROUP_SPLIT: list[tuple[str, str]] = [
    ("udmfgen:*-a-v*", "train"),   # generator family a, all stages
    ("udmfgen:*-b-v*", "val"),     # generator family b (held-out style, seeds 1000+)
    ("freedoom2:MAP25-32", "val"),
    ("freedoom2:MAP*", "train"),   # MAP01-08, 09-16, 17-24
    ("freedoom1:E3", "val"),
    ("freedoom1:E4", "test"),      # PROVISIONAL: same project as train (see TEST_STATUS)
    ("freedoom1:E*", "train"),     # E1, E2
    ("vizdoom:*", "train"),        # ViZDoom drill scenarios
]

CREATED = "2026-11-26"
OWNER = "evaluation-and-generalization-researcher"
SEEDS = {"train": [100, 101, 102], "val": [10000, 10001, 10002, 10003, 10004], "test": list(range(20000, 20010))}
DESCRIPTION = {
    "train": "Training maps: generator family a (all stages), freedoom2 MAP01-24, freedoom1 E1-E2, ViZDoom scenarios. "
             "train-eval seeds 100-102 (vanilla engine). Built by training.eval.build_splits.",
    "val": "Validation (model selection only): generator family b (seeds 1000+), freedoom2 MAP25-32, freedoom1 E3. "
           "Seeds 10000-10004. Built by training.eval.build_splits.",
    "test": "Held-out test, PROVISIONAL AND INCOMPLETE: freedoom1 E4 only. Access logged in "
            "reports/eval/test_access_log.jsonl. Built by training.eval.build_splits.",
}
TEST_STATUS = {
    "status": "INCOMPLETE",
    "incomplete": True,
    "incomplete_reason": (
        "Violates EVALUATION_PROTOCOL.md §1.1 'test must contain a source family with no group in train': freedoom1 E4 "
        "is the same Freedoom project (authors, textures) as freedoom1 E1-E2 and freedoom2 in train. A third-party "
        "family is pending an oss-scout licence check and main-session approval. Completing this manifest replaces "
        "it (still test v1, no test run has been made); after the first logged test run any change is test_v2. "
        "Tier coverage: no tier 1 or tier 3 maps."),
}


def assign(source_group: str) -> str:
    for pat, split in GROUP_SPLIT:
        if fnmatch.fnmatchcase(source_group, pat):
            return split
    raise ValueError(f"source_group {source_group!r} has no split assignment in GROUP_SPLIT")


def build() -> dict[str, dict]:
    from training.envs.maps import all_entries, manifest_entries

    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for e in all_entries():
        split = assign(e.source_group)
        d = manifest_entries([e], split)[0]
        if d["tier"] is None:
            tier, x = static_tier(e.path, e.map)
            d["tier"], d["tier_basis"] = tier, RULE_VERSION
            d["tier_inputs"] = x
            d["notes"] = (d["notes"] + " | " if d["notes"] else "") + describe(x)
        by_split[split].append(d)
    out = {}
    for split, maps in by_split.items():
        man = {"schema_version": splits.SCHEMA_VERSION, "split": split, "description": DESCRIPTION[split],
               "created_utc": CREATED, "owner": OWNER, "seeds": SEEDS[split],
               "generated_by": "uv run --no-sync python -m training.eval.build_splits --write"}
        man.update(TEST_STATUS if split == "test" else {"status": "complete", "incomplete": False})
        man["maps"] = maps
        out[split] = man
    return out


def tier_table(mans: dict[str, dict]) -> dict:
    """Per split, per tier: map count and distinct source groups (§1.3 minimums)."""
    tab = {}
    for split, man in mans.items():
        t: dict = {}
        for e in man["maps"]:
            k = str(e["tier"])
            t.setdefault(k, {"maps": 0, "groups": set()})
            t[k]["maps"] += 1
            t[k]["groups"].add(e["source_group"])
        tab[split] = {k: {"maps": v["maps"], "groups": len(v["groups"])} for k, v in sorted(t.items())}
    return tab


MINIMUM = {"train": (16, 3), "val": (4, 2), "test": (4, 2)}  # §1.3 (maps, groups) per tier


def shortfalls(mans: dict[str, dict]) -> list[str]:
    out = []
    for split, tiers in tier_table(mans).items():
        mn, mg = MINIMUM[split]
        for t in ("1", "2", "3", "4"):
            c = tiers.get(t, {"maps": 0, "groups": 0})
            if c["maps"] < mn or c["groups"] < mg:
                out.append(f"{split} tier {t}: {c['maps']} maps / {c['groups']} groups (minimum {mn} / {mg})")
    return out


def guard_check(mans: dict[str, dict], splits_dir: Path) -> list[str]:
    """check_trainable must accept every train entry and refuse every val/test entry (against `splits_dir`)."""
    from training.envs.maps import all_entries, check_trainable

    errs = []
    ents = {e.id: e for e in all_entries()}
    train = [ents[m["id"]] for m in mans["train"]["maps"]]
    try:
        check_trainable(train, splits_dir=splits_dir)
    except ValueError as ex:
        errs.append(f"check_trainable refused train entries: {ex}")
    for split in ("val", "test"):
        for m in mans[split]["maps"]:
            try:
                check_trainable([ents[m["id"]]], splits_dir=splits_dir)
                errs.append(f"check_trainable ACCEPTED held-out {split} map {m['id']}")
            except ValueError:
                pass
    return errs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m training.eval.build_splits")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--write", action="store_true")
    g.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    mans = build()
    rc = 0
    if a.write:
        for split, man in mans.items():
            (splits.SPLITS_DIR / f"{split}.json").write_text(json.dumps(man, indent=1) + "\n")
        print(f"wrote {', '.join(f'{s}={len(m['maps'])}' for s, m in mans.items())} maps to {splits.SPLITS_DIR}")
    else:
        for split, man in mans.items():
            cur = splits.load(split)
            for k in ("maps", "seeds", "status", "incomplete"):
                if cur.get(k) != man[k]:
                    print(f"DRIFT {split}.json field {k!r} differs from registry output (rerun --write)")
                    rc = 1
    errs = guard_check(mans, splits.SPLITS_DIR)
    for e in errs:
        print("GUARD", e)
    rc |= bool(errs)
    print(json.dumps(tier_table(mans)))
    for s in shortfalls(mans):
        print("SHORTFALL", s)
    print("build_splits OK" if rc == 0 else "build_splits FAILED")
    return rc


if __name__ == "__main__":
    sys.exit(main())
