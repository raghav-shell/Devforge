"""Run-local generated manifest: family a accepted by MapSampler's held-out guard; family b refused."""

import json
import subprocess
import sys

import pytest
from conftest import ROOT

from training.configs.make_manifest import build


def _gen(out, stage, seed, family):
    from training.envs.mapgen import generate

    generate(stage, seed, family=family).write(out / stage)


def test_family_a_accepted_by_sampler(tmp_path):
    from training.envs.maps import MapSampler, load_trainable_manifest

    for st in ("locomotion", "doors"):
        _gen(tmp_path, st, 300, "a")
    man = build(tmp_path)
    assert man["split"] == "train" and len(man["maps"]) == 2
    p = tmp_path / "m.json"
    p.write_text(json.dumps(man))
    assert len(load_trainable_manifest(p)) == 2
    MapSampler([str(p)])  # constructs without raising


def test_family_b_refused(tmp_path):
    from training.envs.maps import check_trainable, generated_entries

    _gen(tmp_path, "doors", 1000, "b")  # a val-split map (maps/splits/val.json)
    with pytest.raises(ValueError, match="refusing"):
        build(tmp_path)  # family / seed filter
    with pytest.raises(ValueError, match="held-out"):
        check_trainable(generated_entries(tmp_path))  # A3 guard on its own also refuses it


def test_export_cli_without_pythonpath(tmp_path):
    r = subprocess.run([sys.executable, "-m", "training.ppo.export", "--help"], cwd=ROOT, capture_output=True,
                       text=True, timeout=60, env={"PATH": "/usr/bin:/bin"})
    assert r.returncode == 0, r.stderr
