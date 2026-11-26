"""MapSampler (docs/MODULE_CONTRACTS.md §5.1): train/smoke only, held-out refusal, tick randomisation, determinism."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from training.envs import maps as M
from training.envs.mapgen import generate

ROOT = M.ROOT


def _manifest(path, split, entries):
    path.write_text(json.dumps({"schema_version": "1.0", "split": split, "description": "t", "created_utc": "",
                                "owner": "t", "seeds": [], "maps": entries}))
    return path


def test_refuses_val_and_test_manifests(tmp_path):
    for split in ("val", "test"):
        with pytest.raises(ValueError, match="refusing"):
            M.MapSampler([ROOT / "maps" / "splits" / f"{split}.json"])
        smoke = json.loads((ROOT / "maps/splits/smoke.json").read_text())["maps"]
        with pytest.raises(ValueError, match="refusing"):
            M.MapSampler([_manifest(tmp_path / f"x_{split}.json", split, smoke)])


def test_refuses_heldout_entries_even_in_train_manifest(tmp_path):
    gm = generate("keys", 5)
    gm.write(tmp_path / "gen" / "keys")
    entry = M.generated_entries(tmp_path / "gen")[0]
    splits = tmp_path / "splits"
    splits.mkdir()
    for s in ("smoke", "train", "test"):
        _manifest(splits / f"{s}.json", s, [])
    _manifest(splits / "val.json", "val", [entry.manifest("val")])
    train = _manifest(tmp_path / "train.json", "train", [entry.manifest("train")])
    with pytest.raises(ValueError, match="held-out"):
        M.MapSampler([train], splits_dir=splits)
    with pytest.raises(ValueError, match="held-out"):
        M.MapSampler(entries=[entry], splits_dir=splits)
    # different group, not held out -> fine
    other = generate("doors", 5)
    other.write(tmp_path / "gen2" / "doors")
    ok = M.generated_entries(tmp_path / "gen2")
    assert M.MapSampler(entries=ok, splits_dir=splits).sample().map_id == ok[0].id


def test_tick_randomisation_uses_sidecar_range(tmp_path):
    generate("keys", 2).write(tmp_path / "keys")  # sidecar range [3000, 4200]
    s = M.MapSampler(entries=M.generated_entries(tmp_path), rng=np.random.default_rng(0))
    ticks = [s.sample().episode_ticks for _ in range(300)]
    assert min(ticks) >= 3000 and max(ticks) <= 4200 and len(set(ticks)) > 50
    smoke = M.MapSampler(["maps/splits/smoke.json"], rng=np.random.default_rng(0))
    t2 = [smoke.sample().episode_ticks for _ in range(300)]
    assert min(t2) >= 2100 and max(t2) <= 4200
    fixed = M.MapSampler(["maps/splits/smoke.json"], rng=np.random.default_rng(0), randomize_ticks=False)
    assert {fixed.sample().episode_ticks for _ in range(50)} == {2100}


def test_sampling_is_deterministic_and_weighted(tmp_path):
    a = M.MapSampler(["maps/splits/smoke.json"], rng=np.random.default_rng([7, 3, 1]))
    b = M.MapSampler(["maps/splits/smoke.json"], rng=np.random.default_rng([7, 3, 1]))
    assert [a.sample() for _ in range(20)] == [b.sample() for _ in range(20)]
    generate("locomotion", 1).write(tmp_path / "loco")
    gen = M.generated_entries(tmp_path)
    w = M.MapSampler(["maps/splits/smoke.json"], weights=[0.0, 1.0], entries=gen, rng=np.random.default_rng(0))
    assert {w.sample().map_id for _ in range(30)} == {gen[0].id}
    kept = w.sample(keep=w.sample())
    assert kept.map_id == gen[0].id
    with pytest.raises(ValueError):
        M.MapSampler(["maps/splits/smoke.json"], weights=[1.0, 1.0])


def test_write_manifest_refuses_splits_dir(tmp_path):
    es = M.freedoom2_entries()[:2]
    p = M.write_manifest(es, tmp_path / "m" / "train.json")
    assert M.MapSampler([p]).sample().map_id in {e.id for e in es}
    with pytest.raises(ValueError):
        M.write_manifest(es, ROOT / "maps" / "splits" / "train.json")
    with pytest.raises(ValueError):
        M.write_manifest(es, tmp_path / "v.json", split="val")


def test_map_registry_alias_is_backward_compatible():
    from training.envs import map_registry

    assert map_registry.MapEntry is M.MapEntry and map_registry.all_entries is M.all_entries
