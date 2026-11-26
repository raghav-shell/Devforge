"""Stateful v1 observation preprocessor: the code form of docs/OBSERVATION_SCHEMA.md sections 1-2.

numpy only. Nothing here may import torch, `training/`, `env/` or `doom_env` (docs/OWNERSHIP.md rule 4).
The same class runs in training workers and in `submission/agent.py`, so it is the single definition
of the policy's inputs. It reads only the official raw observation dict and our own committed actions.

Lifecycle (one instance per env / per agent):

    pre.reset()
    f0 = pre.process(obs0)      # prev_action = 0, deltas = 0, t = 0
    pre.commit_action(a0)       # a0 = the exact (9,) {0,1} vector sent to the env
    f1 = pre.process(obs1)      # prev_action = a0, t = 1
    ...
"""

from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from doom_agent.obs.schema import (
    ACT_AUX_DIM,
    ACT_AUX_STEPS_SINCE_USE_EDGE,
    ACT_AUX_TIME_FRAC,
    ACT_AUX_TURN_RUN,
    ENT_FEAT_DIM,
    FEATURE_SPECS,
    MAX_ENTITIES,
    NUM_BUTTONS,
    NUM_ENTITY_TYPES,
    NUM_RAY_HIT_TYPES,
    NUM_RAYS,
    VITALS_FEAT_DIM,
)

# Raw official obs layout (env/doom_env/spaces.py:73-95). Keys and shapes only; values are read, never trusted.
_RAW_SHAPES: dict[tuple[str, ...], tuple[int, ...]] = {
    ("rgb",): (120, 160, 3),
    ("depth",): (120, 160),
    ("vitals",): (17,),
    ("rays", "distance"): (NUM_RAYS,),
    ("rays", "hit_type"): (NUM_RAYS,),
    ("visible_entities", "features"): (MAX_ENTITIES, 5),
    ("visible_entities", "entity_type"): (MAX_ENTITIES,),
    ("visible_entities", "mask"): (MAX_ENTITIES,),
}
_TOP_KEYS = frozenset({"rgb", "depth", "vitals", "rays", "visible_entities"})
_NESTED_KEYS = {
    "rays": frozenset({"distance", "hit_type"}),
    "visible_entities": frozenset({"features", "entity_type", "mask"}),
}

# Button indices (docs/CONTRACT.md section 3, env/doom_env/spaces.py BUTTONS).
_TURN_LEFT = 4
_TURN_RIGHT = 5
_USE = 7

USE_EDGE_SATURATION = 16  # act_aux[0] = min(k, 16) / 16
TURN_RUN_SATURATION = 3  # act_aux[1] = min(r, 3) / 3


@dataclass(frozen=True)
class ObsConfig:
    time_feature: bool = True  # act_aux[3]; H12 ablation sets False -> slot is 0
    time_scale_steps: float = 1050.0  # scale constant only, NOT the true cap
    enable_depth_band: bool = False  # v2 slot, must stay False in v1
    enable_egomap: bool = False  # v2 slot, must stay False in v1
    enable_icp: bool = False  # v2 slot (blocked flag + odometry), must stay False in v1


_F32 = np.float32


def _bounds(name: str) -> tuple[np.ndarray, np.ndarray]:
    spec = FEATURE_SPECS[name]
    return spec.low_array().astype(_F32), spec.high_array().astype(_F32)


_BOUNDS = {name: _bounds(name) for name, spec in FEATURE_SPECS.items() if spec.np_dtype == np.float32}


class ObsPreprocessor:
    """Raw official obs dict -> v1 feature dict (no batch dim). See module docstring for the lifecycle.

    strict=True (default; training, tests, harness): unexpected keys raise KeyError and non-finite
    inputs raise ValueError. strict=False (DoomAgent only): unexpected keys are ignored and non-finite
    inputs are replaced by 0, each with one stderr warning. Missing keys and wrong shapes always raise.
    """

    def __init__(self, cfg: ObsConfig = ObsConfig(), *, strict: bool = True) -> None:
        for flag in ("enable_depth_band", "enable_egomap", "enable_icp"):
            if getattr(cfg, flag):
                raise NotImplementedError(f"ObsConfig.{flag} is a reserved v2 slot and must be False in v1")
        if not (np.isfinite(cfg.time_scale_steps) and cfg.time_scale_steps > 0):
            raise ValueError(f"time_scale_steps must be finite and > 0, got {cfg.time_scale_steps}")
        self.cfg = cfg
        self.strict = strict
        self._warned: set[str] = set()
        self.reset()

    # ------------------------------------------------------------------ lifecycle
    def reset(self) -> None:
        """Clear all per-episode history. Safe to call at any point (also mid-alternation)."""
        self._first = True
        self._prev_rays_dist = np.zeros(NUM_RAYS, _F32)
        self._prev_health = 0.0
        self._prev_sel_ammo = 0.0
        self._prev_sel_weapon = 0
        self._prev_action = np.zeros(NUM_BUTTONS, np.int8)
        self._steps_since_use_edge = USE_EDGE_SATURATION
        self._turn_run = 0
        self._t = 0
        self._awaiting_action = False

    def process(self, obs: Mapping) -> dict[str, np.ndarray]:
        """One raw obs -> fresh feature arrays matching FEATURE_SPECS. Must alternate with commit_action."""
        if self._awaiting_action:
            raise RuntimeError("process() called twice without commit_action() in between")
        raw = self._read(obs)
        d = raw["distance"]
        v = raw["vitals"]
        feats_ent = raw["features"]
        mask = raw["mask"] != 0
        m = mask.astype(_F32)

        out: dict[str, np.ndarray] = {}
        out["ray_dist"] = d.copy()
        out["ray_near"] = (1.0 / (1.0 + 16.0 * d)).astype(_F32)
        if self._first:
            out["ray_ddist"] = np.zeros(NUM_RAYS, _F32)
        else:
            out["ray_ddist"] = (4.0 * (d - self._prev_rays_dist)).astype(_F32)
        out["ray_hit"] = raw["hit_type"].copy()

        ed = np.maximum(feats_ent[:, 0], 0.0)
        ent = np.empty((MAX_ENTITIES, ENT_FEAT_DIM), _F32)
        ent[:, 0] = np.clip(ed / 4.0, 0.0, 1.0)
        ent[:, 1] = 1.0 / (1.0 + 8.0 * ed)
        ent[:, 2] = feats_ent[:, 1]
        ent[:, 3] = feats_ent[:, 2]
        ent[:, 4] = np.clip(feats_ent[:, 3] / 10.0, -3.0, 3.0)
        ent[:, 5] = np.clip(feats_ent[:, 4] / 10.0, -3.0, 3.0)
        ent[:, 6] = ed <= 1.0
        out["ent_feat"] = ent  # masked after the final clip (below), so padded rows are exactly 0
        out["ent_type"] = np.where(mask, raw["entity_type"], 0).astype(np.int64)
        out["ent_mask"] = mask.copy()

        health, sel_ammo = float(v[0]), float(v[3])
        sel_weapon = int(np.clip(np.rint(v[2]), 0, 7))
        vit = np.zeros(VITALS_FEAT_DIM, _F32)
        vit[0] = v[0] / 100.0
        vit[1] = v[1] / 100.0
        vit[2 + sel_weapon] = 1.0
        vit[10] = v[3] / 50.0
        vit[11] = v[4] / 200.0
        vit[12] = v[5] / 50.0
        vit[13] = v[6] / 50.0
        vit[14] = v[7] / 300.0
        keys = np.clip(v[8:14], 0.0, 1.0)
        vit[15:21] = keys
        vit[21] = keys.max()
        vit[22:25] = v[14:17] / 10.0
        if not self._first:
            vit[25] = (health - self._prev_health) / 20.0
            if sel_weapon == self._prev_sel_weapon:
                vit[26] = (sel_ammo - self._prev_sel_ammo) / 5.0
        out["vitals_feat"] = vit

        out["prev_action"] = self._prev_action.astype(_F32)

        aux = np.zeros(ACT_AUX_DIM, _F32)
        aux[ACT_AUX_STEPS_SINCE_USE_EDGE] = min(self._steps_since_use_edge, USE_EDGE_SATURATION) / USE_EDGE_SATURATION
        aux[ACT_AUX_TURN_RUN] = min(self._turn_run, TURN_RUN_SATURATION) / TURN_RUN_SATURATION
        if self.cfg.time_feature:
            aux[ACT_AUX_TIME_FRAC] = min(self._t / self.cfg.time_scale_steps, 1.0)
        out["act_aux"] = aux  # reserved slots 2, 4, 5 stay 0 (and are pinned to [0, 0] by the clip)

        for name, (lo, hi) in _BOUNDS.items():
            out[name] = np.clip(out[name], lo, hi).astype(_F32, copy=False)
        out["ent_feat"] *= m[:, None]
        out["ent_feat"] += 0.0  # normalise -0.0 from the mask product to +0.0

        # Commit state only after every feature was built, so a raise leaves the instance unchanged.
        self._prev_rays_dist = d.copy()
        self._prev_health = health
        self._prev_sel_ammo = sel_ammo
        self._prev_sel_weapon = sel_weapon
        self._first = False
        self._t += 1
        self._awaiting_action = True
        return out

    def commit_action(self, action: np.ndarray) -> None:
        """Record the (9,) MultiBinary action actually sent to the env (after the action codec)."""
        if not self._awaiting_action:
            raise RuntimeError("commit_action() called without a preceding process()")
        a = np.asarray(action)
        if a.shape != (NUM_BUTTONS,):
            raise ValueError(f"action shape {a.shape} != ({NUM_BUTTONS},)")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError(f"action values must be 0 or 1, got {a!r}")
        a = a.astype(np.int8)
        if a[_USE] == 1 and self._prev_action[_USE] == 0:
            self._steps_since_use_edge = 0
        else:
            self._steps_since_use_edge = min(self._steps_since_use_edge + 1, USE_EDGE_SATURATION)
        if (a[_TURN_LEFT] + a[_TURN_RIGHT]) == 1:
            self._turn_run = min(self._turn_run + 1, TURN_RUN_SATURATION)
        else:
            self._turn_run = 0
        self._prev_action = a
        self._awaiting_action = False

    def state_dict(self) -> dict:
        """Copy of the internal history, for debugging and determinism tests only."""
        return {
            "first": self._first,
            "prev_rays_dist": self._prev_rays_dist.copy(),
            "prev_health": self._prev_health,
            "prev_sel_ammo": self._prev_sel_ammo,
            "prev_sel_weapon": self._prev_sel_weapon,
            "prev_action": self._prev_action.copy(),
            "steps_since_use_edge": self._steps_since_use_edge,
            "turn_run": self._turn_run,
            "t": self._t,
            "awaiting_action": self._awaiting_action,
        }

    # ------------------------------------------------------------------ input validation
    def _problem(self, kind: str, msg: str, exc: type[Exception]) -> None:
        if self.strict:
            raise exc(msg)
        if kind not in self._warned:
            self._warned.add(kind)
            print(f"WARNING ObsPreprocessor: {msg} (strict=False: continuing)", file=sys.stderr)

    def _read(self, obs: Mapping) -> dict[str, np.ndarray]:
        if not isinstance(obs, Mapping):
            raise TypeError(f"obs must be a mapping, got {type(obs).__name__}")
        extra = set(obs) - _TOP_KEYS
        if extra:
            self._problem("extra_keys", f"unexpected obs keys {sorted(map(str, extra))}", KeyError)
        for top, allowed in _NESTED_KEYS.items():
            sub = obs[top] if top in obs else None
            if sub is None:
                raise KeyError(f"missing official obs key {top!r}")
            if not isinstance(sub, Mapping):
                raise TypeError(f"obs[{top!r}] must be a mapping, got {type(sub).__name__}")
            extra = set(sub) - allowed
            if extra:
                self._problem("extra_keys", f"unexpected obs[{top!r}] keys {sorted(map(str, extra))}", KeyError)

        arrays: dict[str, np.ndarray] = {}
        for path, shape in _RAW_SHAPES.items():
            node = obs
            for k in path:
                if k not in node:
                    raise KeyError(f"missing official obs key {'.'.join(path)!r}")
                node = node[k]
            a = np.asarray(node)
            if a.shape != shape:
                raise ValueError(f"obs {'.'.join(path)} shape {a.shape} != {shape}")
            arrays[path[-1]] = a

        out: dict[str, np.ndarray] = {}
        for name in ("distance", "vitals", "features"):
            x = arrays[name].astype(np.float64)
            bad = ~np.isfinite(x)
            if bad.any():
                self._problem("non_finite", f"non-finite values in obs field {name!r}", ValueError)
                x = np.where(bad, 0.0, x)
            out[name] = x
        out["distance"] = np.clip(out["distance"], 0.0, 1.0).astype(_F32)

        for name, n in (("hit_type", NUM_RAY_HIT_TYPES), ("entity_type", NUM_ENTITY_TYPES), ("mask", 2)):
            a = arrays[name]
            if not (np.issubdtype(a.dtype, np.integer) or np.issubdtype(a.dtype, np.bool_)):
                if not np.all(np.isfinite(a)) or not np.all(a == np.rint(a)):
                    self._problem("categorical", f"obs field {name!r} is not integer-valued", ValueError)
                a = np.nan_to_num(np.asarray(a, np.float64))
            a = a.astype(np.int64)
            if np.any((a < 0) | (a >= n)):
                self._problem("categorical", f"obs field {name!r} has values outside [0, {n - 1}]", ValueError)
                a = np.clip(a, 0, n - 1)
            out[name] = a
        return out


def stack_features(feats: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Stack per-env feature dicts along a new leading axis, in FEATURE_SPECS order, checking keys/dtypes.

    `stack_features([f])` equals `{k: v[None] for k, v in f.items()}` bit for bit, so training workers and
    DoomAgent build identical batches from identical features.
    """
    if not feats:
        raise ValueError("stack_features needs at least one feature dict")
    out: dict[str, np.ndarray] = {}
    for name, spec in FEATURE_SPECS.items():
        parts = []
        for i, f in enumerate(feats):
            if set(f) != set(FEATURE_SPECS):
                raise KeyError(f"feature dict {i} keys {sorted(f)} != FEATURE_SPECS keys")
            x = f[name]
            if x.shape != spec.shape or x.dtype != spec.np_dtype:
                raise ValueError(f"feature dict {i} {name}: {x.shape} {x.dtype} != {spec.shape} {spec.np_dtype}")
            parts.append(x)
        out[name] = np.stack(parts, axis=0)
    return out
