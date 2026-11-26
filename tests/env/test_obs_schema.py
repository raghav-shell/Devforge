"""Tests for submission/doom_agent/obs/schema.py (module A1, docs/OBSERVATION_SCHEMA.md sections 2-3)."""

from __future__ import annotations

import ast
import dataclasses
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from doom_agent.obs import schema
from doom_agent.obs.schema import (
    FEATURE_SPECS,
    OBS_SCHEMA_VERSION,
    RESERVED_FEATURE_SPECS,
    FeatureSpec,
    dummy_features,
    obs_pipeline_hash,
)

REPO = Path(__file__).resolve().parents[2]
SUBMISSION = REPO / "submission"

# Exactly docs/OBSERVATION_SCHEMA.md section 2.
EXPECTED = {
    "ray_dist": ((64,), np.float32),
    "ray_near": ((64,), np.float32),
    "ray_ddist": ((64,), np.float32),
    "ray_hit": ((64,), np.int64),
    "ent_feat": ((32, 7), np.float32),
    "ent_type": ((32,), np.int64),
    "ent_mask": ((32,), np.bool_),
    "vitals_feat": ((27,), np.float32),
    "prev_action": ((9,), np.float32),
    "act_aux": ((6,), np.float32),
}

BATCHES = [(), 1, 7, (1,), (7,), (3, 4)]


def _batch(b: int | tuple[int, ...]) -> tuple[int, ...]:
    return (b,) if isinstance(b, int) else b


def test_version_constant() -> None:
    assert OBS_SCHEMA_VERSION == "1.0.0"


def test_feature_specs_match_schema_section_2() -> None:
    assert set(FEATURE_SPECS) == set(EXPECTED)
    for name, (shape, dtype) in EXPECTED.items():
        spec = FEATURE_SPECS[name]
        assert isinstance(spec, FeatureSpec)
        assert spec.shape == shape, name
        assert spec.np_dtype == np.dtype(dtype), name
        lo, hi = spec.low_array(), spec.high_array()
        assert lo.shape == shape and hi.shape == shape, name
        assert np.all(lo <= hi), name


def test_dtype_is_numpy_dtype() -> None:
    assert FEATURE_SPECS["ray_dist"].dtype == np.float32
    assert FEATURE_SPECS["ray_hit"].dtype == np.int64
    assert FEATURE_SPECS["ent_mask"].dtype == np.bool_
    assert isinstance(FEATURE_SPECS["act_aux"].dtype, np.dtype)


def test_float_width_is_458() -> None:
    floats = sum(s.size for s in FEATURE_SPECS.values() if s.np_dtype == np.float32)
    assert floats == 458
    assert sum(s.size for s in FEATURE_SPECS.values() if s.np_dtype == np.int64) == 64 + 32
    assert sum(s.size for s in FEATURE_SPECS.values() if s.np_dtype == np.bool_) == 32


def test_specific_ranges() -> None:
    assert FEATURE_SPECS["ray_near"].low == pytest.approx(1 / 17)
    assert FEATURE_SPECS["ray_hit"].high == 5
    assert FEATURE_SPECS["ent_type"].high == 11
    ent_hi = FEATURE_SPECS["ent_feat"].high_array()[0]
    np.testing.assert_array_equal(ent_hi, [1, 1, 1, 1, 3, 3, 1])
    vit_hi = FEATURE_SPECS["vitals_feat"].high_array()
    assert vit_hi[10] == 4.0 and vit_hi[0] == 2.0 and vit_hi[22] == 3.0 and vit_hi[26] == 1.0


def test_reserved_v2_slots_are_disabled() -> None:
    # Reserved keys are named and frozen, but never part of the v1 spec table.
    assert set(RESERVED_FEATURE_SPECS) == {"egomap", "depth_band"}
    assert RESERVED_FEATURE_SPECS["egomap"].shape == (7, 32, 32)
    assert RESERVED_FEATURE_SPECS["depth_band"].shape == (40,)
    assert not set(RESERVED_FEATURE_SPECS) & set(FEATURE_SPECS)
    assert not set(RESERVED_FEATURE_SPECS) & set(dummy_features(3))
    # act_aux slots 2, 4, 5 are pinned to exactly 0 in v1.
    aux = FEATURE_SPECS["act_aux"]
    for i in schema.ACT_AUX_RESERVED_V1:
        assert aux.low_array()[i] == 0.0 and aux.high_array()[i] == 0.0


@pytest.mark.parametrize("batch", BATCHES)
def test_dummy_features_match_specs(batch: int | tuple[int, ...]) -> None:
    b = _batch(batch)
    feats = dummy_features(batch, np.random.default_rng(123))
    assert set(feats) == set(FEATURE_SPECS)
    for name, spec in FEATURE_SPECS.items():
        x = feats[name]
        assert isinstance(x, np.ndarray), name
        assert x.shape == b + spec.shape, name
        assert x.dtype == spec.np_dtype, name
        if x.dtype != np.bool_:
            assert np.all(np.isfinite(x)), name
        assert np.all(x >= spec.low_array()), name
        assert np.all(x <= spec.high_array()), name


@pytest.mark.parametrize("batch", [(), 7, (3, 4)])
def test_dummy_features_structure(batch: int | tuple[int, ...]) -> None:
    f = dummy_features(batch, np.random.default_rng(7))
    mask = f["ent_mask"]
    assert np.all(f["ent_feat"][~mask] == 0.0)
    assert np.all(f["ent_type"][~mask] == 0)
    assert np.all(f["ent_type"][mask] >= 1)
    assert set(np.unique(f["ent_feat"][..., schema.ENT_FEAT_IN_RAY_RANGE])) <= {0.0, 1.0}
    assert set(np.unique(f["prev_action"])) <= {0.0, 1.0}
    vit = f["vitals_feat"]
    onehot = vit[..., schema.VITALS_WEAPON_ONEHOT]
    np.testing.assert_array_equal(onehot.sum(-1), 1.0)
    keys = vit[..., schema.VITALS_KEYS]
    assert set(np.unique(keys)) <= {0.0, 1.0}
    np.testing.assert_array_equal(vit[..., schema.VITALS_ANY_KEY], keys.max(-1))
    np.testing.assert_allclose(f["ray_near"], 1 / (1 + 16 * f["ray_dist"]), rtol=1e-6)
    for i in schema.ACT_AUX_RESERVED_V1:
        assert np.all(f["act_aux"][..., i] == 0.0)


def test_dummy_features_deterministic() -> None:
    a = dummy_features(5, np.random.default_rng(1))
    b = dummy_features(5, np.random.default_rng(1))
    c = dummy_features(5, np.random.default_rng(2))
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])
    assert any(not np.array_equal(a[k], c[k]) for k in a)
    d, e = dummy_features(5), dummy_features(5)  # default rng is seeded
    for k in d:
        np.testing.assert_array_equal(d[k], e[k])


def test_hash_deterministic_in_process() -> None:
    h = obs_pipeline_hash()
    assert len(h) == 64 and int(h, 16) >= 0
    assert obs_pipeline_hash() == h


def _fresh_python(code: str) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([str(SUBMISSION), str(REPO)])
    out = subprocess.run(
        [sys.executable, "-c", code], env=env, cwd="/", capture_output=True, text=True, timeout=60
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def test_hash_deterministic_across_processes() -> None:
    other = _fresh_python("from doom_agent.obs import obs_pipeline_hash; print(obs_pipeline_hash())")
    assert other == obs_pipeline_hash()


def _perturbed(name: str, **changes: object) -> dict[str, FeatureSpec]:
    specs = dict(FEATURE_SPECS)
    specs[name] = dataclasses.replace(specs[name], **changes)
    return specs


@pytest.mark.parametrize(
    "specs",
    [
        pytest.param(_perturbed("ray_dist", shape=(65,)), id="shape"),
        pytest.param(_perturbed("ray_hit", dtype="int32"), id="dtype"),
        pytest.param(_perturbed("ray_near", low=0.0), id="low"),
        pytest.param(_perturbed("ent_feat", high=(1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 2.0)), id="high"),
        pytest.param(
            {**FEATURE_SPECS, "extra": FeatureSpec((1,), "float32", 0.0, 1.0)}, id="add_key"
        ),
        pytest.param({k: v for k, v in FEATURE_SPECS.items() if k != "act_aux"}, id="drop_key"),
    ],
)
def test_hash_changes_when_spec_changes(
    monkeypatch: pytest.MonkeyPatch, specs: dict[str, FeatureSpec]
) -> None:
    base = obs_pipeline_hash()
    monkeypatch.setattr(schema, "FEATURE_SPECS", specs)
    assert obs_pipeline_hash() != base


def test_hash_changes_with_reserved_specs_and_version(monkeypatch: pytest.MonkeyPatch) -> None:
    base = obs_pipeline_hash()
    with monkeypatch.context() as m:
        m.setattr(
            schema,
            "RESERVED_FEATURE_SPECS",
            {**RESERVED_FEATURE_SPECS, "depth_band": FeatureSpec((41,), "float32", 0.0, 1.0)},
        )
        assert obs_pipeline_hash() != base
    with monkeypatch.context() as m:
        m.setattr(schema, "OBS_SCHEMA_VERSION", "1.0.1")
        assert obs_pipeline_hash() != base
    assert obs_pipeline_hash() == base


def test_hash_covers_source_files(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Point the hasher at a copy of the package and edit one source byte.
    pkg = Path(schema.__file__).resolve().parent
    copy = tmp_path / "obs"
    copy.mkdir()
    for p in pkg.glob("*.py"):
        (copy / p.name).write_bytes(p.read_bytes())
    monkeypatch.setattr(schema, "__file__", str(copy / "schema.py"))
    same = obs_pipeline_hash()
    monkeypatch.undo()
    assert same == obs_pipeline_hash()  # location-independent
    monkeypatch.setattr(schema, "__file__", str(copy / "schema.py"))
    (copy / "schema.py").write_bytes((pkg / "schema.py").read_bytes() + b"\n# edit\n")
    assert obs_pipeline_hash() != same


def test_import_is_light() -> None:
    code = (
        "import sys\n"
        "import doom_agent.obs\n"
        "from doom_agent.obs import FEATURE_SPECS, dummy_features, obs_pipeline_hash\n"
        "dummy_features(2); obs_pipeline_hash()\n"
        "bad = [m for m in sys.modules if m.split('.')[0] in "
        "('torch', 'training', 'doom_env', 'vizdoom', 'env', 'gymnasium')]\n"
        "print(','.join(bad))\n"
    )
    assert _fresh_python(code) == ""


def test_schema_source_has_no_forbidden_imports() -> None:
    forbidden = {"torch", "training", "env", "doom_env", "vizdoom"}
    pkg = Path(schema.__file__).resolve().parent
    for p in pkg.glob("*.py"):
        tree = ast.parse(p.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                roots = {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom):
                roots = {(node.module or "").split(".")[0]}
            elif isinstance(node, ast.Name | ast.Attribute):
                name = node.id if isinstance(node, ast.Name) else node.attr
                assert "privileged" not in name, f"{p.name}: {name}"
                continue
            else:
                continue
            assert not roots & forbidden, f"{p.name}: {roots & forbidden}"
