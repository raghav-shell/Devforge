"""Load and validate map-split manifests (maps/splits/*.json). Schema: docs/EVALUATION_PROTOCOL.md §1.4.

    uv run --no-sync python -m training.eval.splits            # validate all manifests (exit 1 on any error)

Errors: schema, ids, tiers, ticks, sha256 (repo files AND @vizdoom/ bundled WADs), and leakage across
train/val/test by source_group, by (wad_sha256, map) and by map geometry (identical vertex/linedef or TEXTMAP
data under different hashes or groups). Warnings: test-family rule (§1.1) when test.json is flagged incomplete.
"""

from __future__ import annotations

import functools
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
TIER_BASIS = ("mechanic", "manual", "generator", "static_rule_v1", "none")
VIZDOOM_PREFIX = "@vizdoom/"


def resolve(wad: str) -> Path:
    if wad.startswith(VIZDOOM_PREFIX):
        import vizdoom

        return Path(vizdoom.__file__).resolve().parent / wad.removeprefix(VIZDOOM_PREFIX)
    p = Path(wad)
    return p if p.is_absolute() else ROOT / p


@functools.lru_cache(maxsize=256)
def _sha256(path: str, mtime_ns: int, size: int) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256(path: Path) -> str:
    st = path.stat()
    return _sha256(str(path), st.st_mtime_ns, st.st_size)


@functools.lru_cache(maxsize=32)
def _wad(path: str, mtime_ns: int):
    from doom_env.wad import read_wad

    return read_wad(path)


def geometry_key(path: Path, map_name: str) -> str:
    """Hash of the map's geometry: TEXTMAP for UDMF, VERTEXES for binary maps (catches retextured copies)."""
    from doom_env.wad import map_lumps

    lumps = map_lumps(_wad(str(path), path.stat().st_mtime_ns), map_name)
    data = lumps.get("TEXTMAP") or lumps.get("VERTEXES", b"")
    return hashlib.sha256(data).hexdigest()


def family(source_group: str) -> str:
    """Source family for the §1.1 unseen-family rule: project/author, or generator + style family."""
    head, _, tail = source_group.partition(":")
    if head == "udmfgen":  # udmfgen:<stage>-<family>-v<N>
        parts = tail.split("-")
        return f"udmfgen-{parts[1]}" if len(parts) >= 3 else source_group
    if head in ("freedoom1", "freedoom2"):
        return "freedoom"
    if head.startswith("template"):
        return "template"
    return head


def load(split: str) -> dict:
    return json.loads((SPLITS_DIR / f"{split}.json").read_text())


def _check(check_hash: bool) -> tuple[list[str], list[str]]:
    errs: list[str] = []
    warns: list[str] = []
    groups: dict[str, set[str]] = {}
    hashes: dict[str, set[str]] = {}
    geoms: dict[str, set[str]] = {}
    geom_ids: dict[str, list[str]] = {}
    ids: set[str] = set()
    mans: dict[str, dict] = {}
    for split in SPLITS:
        p = SPLITS_DIR / f"{split}.json"
        if not p.is_file():
            errs.append(f"{split}: manifest missing"); continue
        man = json.loads(p.read_text())
        mans[split] = man
        if man.get("schema_version") != SCHEMA_VERSION or man.get("split") != split:
            errs.append(f"{split}: bad schema_version/split field")
        for e in man.get("maps", []):
            miss = [k for k in REQUIRED if k not in e]
            if miss:
                errs.append(f"{split}:{e.get('id')}: missing {miss}"); continue
            if e["id"] in ids:
                errs.append(f"{split}:{e['id']}: duplicate id")
            ids.add(e["id"])
            if e["role"] != split:
                errs.append(f"{split}:{e['id']}: role {e['role']!r} != split")
            if e["tier"] not in (None, 1, 2, 3, 4):
                errs.append(f"{split}:{e['id']}: tier must be 1-4 or null")
            if e.get("tier_basis", "none") not in TIER_BASIS:
                errs.append(f"{split}:{e['id']}: tier_basis {e.get('tier_basis')!r} not in {TIER_BASIS}")
            if not 1 <= int(e["episode_ticks"]) <= 4200:
                errs.append(f"{split}:{e['id']}: episode_ticks out of range")
            wad = resolve(e["wad"])
            if check_hash:
                if not wad.is_file():
                    errs.append(f"{split}:{e['id']}: {e['wad']} not found"); continue
                if sha256(wad) != e["wad_sha256"]:
                    errs.append(f"{split}:{e['id']}: sha256 mismatch")
                try:
                    gk = geometry_key(wad, e["map"])
                    geoms.setdefault(gk, set()).add(split)
                    geom_ids.setdefault(gk, []).append(e["id"])
                except Exception as ex:  # noqa: BLE001 - reported as an error, never swallowed
                    errs.append(f"{split}:{e['id']}: cannot read map geometry: {type(ex).__name__}: {ex}")
            groups.setdefault(e["source_group"], set()).add(split)
            hashes.setdefault(f"{e['wad_sha256']}:{e['map']}", set()).add(split)
    # leakage: a source group, a (wad, map) or a geometry may not span two of train/val/test (smoke may overlap train)
    for kind, table in (("source_group", groups), ("wad:map", hashes), ("geometry", geoms)):
        for k, ss in table.items():
            strict = ss & {"train", "val", "test"}
            if len(strict) > 1 or ({"val", "test"} & ss and "smoke" in ss):
                extra = f" ({geom_ids[k][:6]})" if kind == "geometry" else ""
                errs.append(f"LEAK: {kind} {k[:24]} appears in {sorted(ss)}{extra}")
    # §1.1: test needs a source family with no group in train
    if "test" in mans and "train" in mans and mans["test"].get("maps"):
        train_fams = {family(e["source_group"]) for e in mans["train"]["maps"]}
        test_fams = {family(e["source_group"]) for e in mans["test"]["maps"]}
        if not test_fams - train_fams:
            msg = f"test families {sorted(test_fams)} all appear in train (§1.1 unseen-family rule)"
            if mans["test"].get("incomplete") is True:
                warns.append(msg + "; test.json is flagged INCOMPLETE")
            else:
                errs.append(msg)
    return errs, warns


def validate_all(check_hash: bool = True) -> list[str]:
    return _check(check_hash)[0]


def warnings_all(check_hash: bool = True) -> list[str]:
    return _check(check_hash)[1]


if __name__ == "__main__":
    problems, warns = _check(True)
    for p in problems:
        print("ERROR", p)
    for w in warns:
        print("WARNING", w)
    print("manifests OK" if not problems else f"{len(problems)} problem(s)")
    sys.exit(1 if problems else 0)
