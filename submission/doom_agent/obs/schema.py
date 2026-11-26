"""v1 feature spec table: the code form of docs/OBSERVATION_SCHEMA.md section 2 (and section 3 reserved slots).

numpy only. Nothing here may import torch, `training/`, `env/` or `doom_env` (docs/OWNERSHIP.md rule 4).

Exports:
  OBS_SCHEMA_VERSION      schema version string, stored in checkpoints and exports.
  FeatureSpec             (shape, dtype, low, high) for one feature tensor, without batch dim.
  FEATURE_SPECS           the v1 tensors emitted by ObsPreprocessor.process, exactly the section 2 keys.
  RESERVED_FEATURE_SPECS  frozen names/shapes of v2 keys; NOT emitted in v1 (disabled).
  dummy_features          random, in-range, structurally valid v1 features for tests.
  obs_pipeline_hash       sha256 identifying the whole obs pipeline (source files + live spec table).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

OBS_SCHEMA_VERSION = "1.0.0"

Bound = float | tuple[float, ...]


@dataclass(frozen=True)
class FeatureSpec:
    """Spec of one feature tensor (no batch dim).

    `low` / `high` are inclusive bounds: a scalar, or a tuple broadcastable to `shape`
    (per-column for `ent_feat`, per-index for `vitals_feat` / `act_aux`).
    """

    shape: tuple[int, ...]
    dtype: np.dtype  # accepts anything np.dtype() accepts; normalised in __post_init__
    low: Bound
    high: Bound

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(s) for s in self.shape))
        object.__setattr__(self, "dtype", np.dtype(self.dtype))

    @property
    def np_dtype(self) -> np.dtype:
        return self.dtype

    def low_array(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.low, dtype=self.np_dtype), self.shape)

    def high_array(self) -> np.ndarray:
        return np.broadcast_to(np.asarray(self.high, dtype=self.np_dtype), self.shape)

    @property
    def size(self) -> int:
        return int(np.prod(self.shape, dtype=np.int64))


NUM_RAYS = 64
MAX_ENTITIES = 32
NUM_RAY_HIT_TYPES = 6  # env/doom_env/spaces.py RAY_HIT_TYPES
NUM_ENTITY_TYPES = 12  # env/doom_env/spaces.py ENTITY_TYPES (0 = pad)
NUM_BUTTONS = 9  # env/doom_env/spaces.py BUTTONS
ENT_FEAT_DIM = 7
VITALS_FEAT_DIM = 27
ACT_AUX_DIM = 6

# Index constants (OBSERVATION_SCHEMA.md 2.1-2.3) for consumers and tests.
ENT_FEAT_IN_RAY_RANGE = 6
VITALS_WEAPON_ONEHOT = slice(2, 10)
VITALS_KEYS = slice(15, 21)
VITALS_ANY_KEY = 21
ACT_AUX_STEPS_SINCE_USE_EDGE = 0
ACT_AUX_TURN_RUN = 1
ACT_AUX_BLOCKED_FLAG = 2  # reserved, 0.0 in v1
ACT_AUX_TIME_FRAC = 3
ACT_AUX_ODOM_FWD = 4  # reserved, 0.0 in v1
ACT_AUX_ODOM_LEFT = 5  # reserved, 0.0 in v1
ACT_AUX_RESERVED_V1 = (ACT_AUX_BLOCKED_FLAG, ACT_AUX_ODOM_FWD, ACT_AUX_ODOM_LEFT)

# section 2.1: far, near, sin, cos, vel_fwd, vel_right, in_ray_range
_ENT_LOW = (0.0, 0.0, -1.0, -1.0, -3.0, -3.0, 0.0)
_ENT_HIGH = (1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 1.0)

# section 2.2
_VIT_LOW = (
    (0.0, 0.0)  # health, armor
    + (0.0,) * 8  # weapon one-hot
    + (0.0,) * 5  # selected ammo, bullets, shells, rockets, cells
    + (0.0,) * 7  # 6 key flags + any key
    + (-3.0,) * 3  # vel fwd/right/up
    + (-1.0, -1.0)  # d health, d selected ammo
)
_VIT_HIGH = (
    (2.0, 2.0)
    + (1.0,) * 8
    + (4.0, 2.0, 2.0, 2.0, 2.0)
    + (1.0,) * 7
    + (3.0,) * 3
    + (1.0, 1.0)
)

# section 2.3: reserved v1 slots (2, 4, 5) are pinned to [0, 0].
_AUX_LOW = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
_AUX_HIGH = (1.0, 1.0, 0.0, 1.0, 0.0, 0.0)

FEATURE_SPECS: dict[str, FeatureSpec] = {
    "ray_dist": FeatureSpec((NUM_RAYS,), "float32", 0.0, 1.0),
    "ray_near": FeatureSpec((NUM_RAYS,), "float32", 1.0 / 17.0, 1.0),
    "ray_ddist": FeatureSpec((NUM_RAYS,), "float32", -1.0, 1.0),
    "ray_hit": FeatureSpec((NUM_RAYS,), "int64", 0, NUM_RAY_HIT_TYPES - 1),
    "ent_feat": FeatureSpec((MAX_ENTITIES, ENT_FEAT_DIM), "float32", _ENT_LOW, _ENT_HIGH),
    "ent_type": FeatureSpec((MAX_ENTITIES,), "int64", 0, NUM_ENTITY_TYPES - 1),
    "ent_mask": FeatureSpec((MAX_ENTITIES,), "bool", 0, 1),
    "vitals_feat": FeatureSpec((VITALS_FEAT_DIM,), "float32", _VIT_LOW, _VIT_HIGH),
    "prev_action": FeatureSpec((NUM_BUTTONS,), "float32", 0.0, 1.0),
    "act_aux": FeatureSpec((ACT_AUX_DIM,), "float32", _AUX_LOW, _AUX_HIGH),
}

# section 3: names and shapes frozen now; not emitted in v1 (enabling the gate raises in ObsPreprocessor).
RESERVED_FEATURE_SPECS: dict[str, FeatureSpec] = {
    "egomap": FeatureSpec((7, 32, 32), "float32", 0.0, 1.0),
    "depth_band": FeatureSpec((40,), "float32", 0.0, 1.0),
}
RESERVED_GATES: dict[str, str] = {"egomap": "enable_egomap", "depth_band": "enable_depth_band"}


def _batch_tuple(batch_shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if isinstance(batch_shape, (int, np.integer)):
        return (int(batch_shape),)
    return tuple(int(b) for b in batch_shape)


def _uniform(rng: np.random.Generator, spec: FeatureSpec, batch: tuple[int, ...]) -> np.ndarray:
    lo = spec.low_array().astype(np.float32)
    hi = spec.high_array().astype(np.float32)
    x = rng.uniform(lo.astype(np.float64), hi.astype(np.float64), size=batch + spec.shape)
    return np.clip(x.astype(np.float32), lo, hi)  # clip after the cast: float32 rounding can pass a bound


def dummy_features(
    batch_shape: int | tuple[int, ...] = (),
    rng: np.random.Generator | None = None,
) -> dict[str, np.ndarray]:
    """Random in-range v1 features with leading dims `batch_shape`, for tests only.

    Structural invariants of section 2 are honoured: masked entity rows/types are 0, binary
    features are exactly 0/1, the weapon slot is one-hot, `any key` = max(key flags),
    `ray_near` is consistent with `ray_dist`, and the v1 reserved `act_aux` slots are 0.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    b = _batch_tuple(batch_shape)
    S = FEATURE_SPECS
    out: dict[str, np.ndarray] = {}

    ray_dist = _uniform(rng, S["ray_dist"], b)
    out["ray_dist"] = ray_dist
    out["ray_near"] = np.clip(
        (1.0 / (1.0 + 16.0 * ray_dist)).astype(np.float32), np.float32(1.0 / 17.0), np.float32(1.0)
    )
    out["ray_ddist"] = _uniform(rng, S["ray_ddist"], b)
    out["ray_hit"] = rng.integers(0, NUM_RAY_HIT_TYPES, size=b + (NUM_RAYS,), dtype=np.int64)

    mask = rng.random(b + (MAX_ENTITIES,)) < 0.5
    m = mask[..., None].astype(np.float32)
    ent = _uniform(rng, S["ent_feat"], b)
    ent[..., ENT_FEAT_IN_RAY_RANGE] = (rng.random(b + (MAX_ENTITIES,)) < 0.5).astype(np.float32)
    out["ent_feat"] = (ent * m).astype(np.float32)
    ent_type = rng.integers(1, NUM_ENTITY_TYPES, size=b + (MAX_ENTITIES,), dtype=np.int64)
    out["ent_type"] = np.where(mask, ent_type, 0).astype(np.int64)
    out["ent_mask"] = mask.astype(np.bool_)

    vit = _uniform(rng, S["vitals_feat"], b)
    weapon = rng.integers(0, 8, size=b)
    vit[..., VITALS_WEAPON_ONEHOT] = np.eye(8, dtype=np.float32)[weapon]
    keys = (rng.random(b + (6,)) < 0.3).astype(np.float32)
    vit[..., VITALS_KEYS] = keys
    vit[..., VITALS_ANY_KEY] = keys.max(axis=-1)
    out["vitals_feat"] = vit

    out["prev_action"] = (rng.random(b + (NUM_BUTTONS,)) < 0.5).astype(np.float32)
    out["act_aux"] = _uniform(rng, S["act_aux"], b)  # reserved slots have low == high == 0

    return out


def _spec_manifest() -> str:
    """Canonical, process-independent JSON of the live spec tables and version."""

    def enc(bound: Bound) -> float | list[float]:
        return [float(v) for v in bound] if isinstance(bound, tuple) else float(bound)

    def table(specs: dict[str, FeatureSpec]) -> dict[str, dict]:
        return {
            name: {
                "shape": [int(s) for s in spec.shape],
                "dtype": spec.dtype.str,
                "low": enc(spec.low),
                "high": enc(spec.high),
            }
            for name, spec in specs.items()
        }

    return json.dumps(
        {
            "obs_schema_version": OBS_SCHEMA_VERSION,
            "feature_specs": table(FEATURE_SPECS),
            "reserved_feature_specs": table(RESERVED_FEATURE_SPECS),
        },
        sort_keys=True,
        separators=(",", ":"),
    )


def obs_pipeline_hash() -> str:
    """sha256 (hex) over the sorted `*.py` sources of this package plus the live spec manifest.

    Computed on every call (no caching). Files are keyed by their path relative to the package
    directory, so the hash does not depend on where the repo is checked out.
    """
    pkg_dir = Path(__file__).resolve().parent
    h = hashlib.sha256()
    for path in sorted(pkg_dir.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(pkg_dir).as_posix()
        data = path.read_bytes()
        h.update(f"file:{rel}:{len(data)}\n".encode())
        h.update(data)
    manifest = _spec_manifest().encode()
    h.update(f"specs:{len(manifest)}\n".encode())
    h.update(manifest)
    return h.hexdigest()
