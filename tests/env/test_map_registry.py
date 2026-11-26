"""Map registry: entries, @vizdoom resolution, held-out guard, manifest validation via training.eval.splits."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from training.eval import splits
from training.envs import maps as R
from training.envs.mapgen import generate

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def gen_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("generated")
    for st, fam, s in (("locomotion", "a", 0), ("keys", "a", 1), ("combat", "a", 2), ("keys", "b", 100), ("full", "b", 101)):
        generate(st, s, family=fam).write(d / st)
    return d


def test_generated_entries(gen_dir):
    es = R.generated_entries(gen_dir)
    assert len(es) == 5
    for e in es:
        assert e.path.is_file() and R.sha256(e.path) == e.wad_sha256
        assert e.source_group.startswith("udmfgen:") and e.tier_basis == "generator" and e.tier in (1, 2, 3, 4)
        assert 1 <= e.episode_ticks <= 4200
    keys = next(e for e in es if e.id == "gen_keys_a_000001")
    assert {"key", "locked_door", "exit", "keys"} <= set(keys.tags)


def test_generated_entries_reject_tampered_wad(gen_dir, tmp_path):
    d = tmp_path / "g" / "locomotion"
    shutil.copytree(gen_dir / "locomotion", d)
    wad = next(d.glob("*.wad"))
    wad.write_bytes(wad.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="sha256"):
        R.generated_entries(tmp_path / "g")


def test_bundled_entries_load_in_place():
    fd2 = R.freedoom2_entries()
    assert [e.map for e in fd2] == [f"MAP{i:02d}" for i in range(1, 33)]
    assert all(e.wad == "@vizdoom/freedoom2.wad" for e in fd2)
    assert R.resolve_wad("@vizdoom/freedoom2.wad").is_file()
    assert {e.source_group for e in fd2} == {"freedoom2:MAP01-08", "freedoom2:MAP09-16", "freedoom2:MAP17-24", "freedoom2:MAP25-32"}
    fd1 = R.freedoom1_entries()
    assert len(fd1) == 36 and fd1[0].map == "E1M1" and fd1[0].source_group == "freedoom1:E1"
    vz = R.vizdoom_scenario_entries()
    ids = {e.id for e in vz}
    assert "vzd_my_way_home_map01" in ids and all(e.wad.startswith("@vizdoom/scenarios/") for e in vz)
    assert "vzd_cig_map01" not in ids and "vzd_multi_duel_map01" not in ids  # no player-1 start
    assert not any((ROOT / "maps").rglob("freedoom*.wad"))  # never copied into maps/


def _write_manifests(d: Path, train: list[dict], val: list[dict], test: list[dict]) -> None:
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(ROOT / "maps" / "splits" / "smoke.json", d / "smoke.json")
    for split, maps in (("train", train), ("val", val), ("test", test)):
        man = {"schema_version": splits.SCHEMA_VERSION, "split": split, "description": "test", "created_utc": "2026-11-26",
               "owner": "test", "seeds": [0], "maps": maps}
        (d / f"{split}.json").write_text(json.dumps(man))


def test_manifest_entries_validate(gen_dir, tmp_path, monkeypatch):
    gen = R.generated_entries(gen_dir)
    fam_a = [e for e in gen if "-a-" in e.source_group]
    fam_b = [e for e in gen if "-b-" in e.source_group]
    fd2 = R.freedoom2_entries()
    train = R.manifest_entries(fam_a + [e for e in fd2 if e.source_group != "freedoom2:MAP25-32"], "train")
    val = R.manifest_entries(fam_b + [e for e in fd2 if e.source_group == "freedoom2:MAP25-32"], "val")
    test = R.manifest_entries(R.vizdoom_scenario_entries()[:2], "test")
    d = tmp_path / "splits"
    _write_manifests(d, train, val, test)
    # generated wads live in tmp, so their "wad" is absolute; splits.ROOT / <absolute> resolves to it.
    monkeypatch.setattr(splits, "SPLITS_DIR", d)
    assert splits.validate_all() == []
    # leakage is caught: same group in train and val
    _write_manifests(d, train, val + [dict(train[0], id="dup")], test)
    assert any("LEAK" in p for p in splits.validate_all())


def test_check_trainable_refuses_heldout(gen_dir, tmp_path):
    gen = R.generated_entries(gen_dir)
    val = [e for e in gen if "-b-" in e.source_group]
    d = tmp_path / "splits"
    _write_manifests(d, [], R.manifest_entries(val, "val"), [])
    with pytest.raises(ValueError, match="held-out"):
        R.check_trainable(gen, splits_dir=d)
    ok = [e for e in gen if "-a-" in e.source_group]
    assert R.check_trainable(ok, splits_dir=d) == ok


def test_committed_samples_match_generator():
    """maps/generated/* must be exactly what the current generator produces (else sha256s in manifests drift)."""
    es = R.generated_entries()
    assert es, "no committed samples in maps/generated"
    for e in es:
        meta = json.loads((ROOT / e.meta_path).read_text())
        m = generate(meta["stage"], meta["seed"], family=meta["family"], map_name=meta["map"])
        assert m.sha256 == e.wad_sha256, e.id
