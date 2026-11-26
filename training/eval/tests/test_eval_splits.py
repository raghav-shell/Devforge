"""Eval-owned checks: committed split manifests, the static tier rule, leak detection, the test-incomplete flag."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from training.eval import build_splits, splits, tiering

ROOT = Path(__file__).resolve().parents[3]


def test_committed_manifests_validate():
    assert splits.validate_all() == []


def test_committed_manifests_match_registry_and_guard():
    mans = build_splits.build()
    for split, man in mans.items():
        cur = splits.load(split)
        for k in ("maps", "seeds", "status", "incomplete"):
            assert cur[k] == man[k], (split, k)
    assert build_splits.guard_check(mans, splits.SPLITS_DIR) == []


def test_test_manifest_flagged_incomplete_and_never_in_train():
    test = splits.load("test")
    assert test["incomplete"] is True and test["status"] == "INCOMPLETE"
    assert any("unseen-family" in w for w in splits.warnings_all())
    train_groups = {e["source_group"] for e in splits.load("train")["maps"]}
    assert not train_groups & {e["source_group"] for e in test["maps"]}


def test_every_group_assigned_once():
    with pytest.raises(ValueError):
        build_splits.assign("thirdparty:unknown")
    assert build_splits.assign("udmfgen:keys-a-v1") == "train" and build_splits.assign("udmfgen:keys-b-v1") == "val"
    assert build_splits.assign("freedoom1:E4") == "test" and build_splits.assign("freedoom2:MAP25-32") == "val"


@pytest.mark.parametrize("x,tier", [
    (dict(lock_colours=["red", "blue"], static_monsters=0, doors=0, sectors=20), 4),
    (dict(lock_colours=["red"], static_monsters=6, doors=0, sectors=20), 4),
    (dict(lock_colours=["red"], static_monsters=5, doors=3, sectors=20), 3),
    (dict(lock_colours=[], static_monsters=40, doors=1, sectors=200), 2),
    (dict(lock_colours=[], static_monsters=0, doors=0, sectors=4), 2),
    (dict(lock_colours=[], static_monsters=8, doors=0, sectors=3), 1),
    (dict(lock_colours=[], static_monsters=0, doors=0, sectors=1), None),
])
def test_tier_rule(x, tier):
    assert tiering.tier_from_inputs(x) == tier


def test_lock_colour_normalisation():
    assert tiering.lock_colours([1, 129, 4]) == {"red"}
    assert tiering.lock_colours([2, 131]) == {"blue", "yellow"}
    assert tiering.lock_colours([101]) == {"red", "blue", "yellow"}
    assert tiering.lock_colours([0]) == set()


def test_rule_tiers_in_manifests_are_reproducible():
    for split in ("train", "val", "test"):
        for e in splits.load(split)["maps"]:
            if e["tier_basis"] == tiering.RULE_VERSION:
                t, x = tiering.static_tier(splits.resolve(e["wad"]), e["map"])
                assert (t, x) == (e["tier"], e["tier_inputs"]), e["id"]


def _write(d: Path, train, val, test, test_extra=None):
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "maps" / "splits" / "smoke.json", d / "smoke.json")
    for split, maps in (("train", train), ("val", val), ("test", test)):
        man = {"schema_version": "1.0", "split": split, "maps": [dict(m, role=split) for m in maps]}
        if split == "test" and test_extra:
            man.update(test_extra)
        (d / f"{split}.json").write_text(json.dumps(man))


def test_geometry_leak_detected(tmp_path, monkeypatch):
    """vizdoom basic and basic_audio: different sha256 and group, identical geometry."""
    from training.envs.maps import manifest_entries, vizdoom_scenario_entries

    vz = {e.id: e for e in vizdoom_scenario_entries()}
    a, b = manifest_entries([vz["vzd_basic_map01"], vz["vzd_basic_audio_map01"]], "train")
    assert a["wad_sha256"] != b["wad_sha256"] and a["source_group"] != b["source_group"]
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    _write(tmp_path, [a], [b], [])
    errs = splits.validate_all()
    assert any(p.startswith("LEAK: geometry") for p in errs), errs


def test_unseen_family_rule_is_error_unless_incomplete(tmp_path, monkeypatch):
    from training.envs.maps import freedoom1_entries, manifest_entries

    fd1 = {e.id: e for e in freedoom1_entries()}
    tr, te = manifest_entries([fd1["fd1_e1m1"], fd1["fd1_e4m1"]], "train")
    monkeypatch.setattr(splits, "SPLITS_DIR", tmp_path)
    _write(tmp_path, [tr], [], [te])
    assert any("unseen-family" in p for p in splits.validate_all())
    _write(tmp_path, [tr], [], [te], {"incomplete": True})
    assert splits.validate_all() == [] and splits.warnings_all()
