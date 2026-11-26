"""Training env factory (docs/MODULE_CONTRACTS.md §5.1). Training-only; never imported by `submission/`.

`TrainEnv` wraps the official `doom_env.make_env` and owns one `ObsPreprocessor` (the same class the agent
uses at eval), one `RewardComposer` (training/rewards/base.py) and the per-episode bookkeeping. It never
auto-resets; the vector env worker (training/envs/vector.py) does that.

    env = make_train_env(spec, seed=123, reward_cfg=[...], obs_cfg=ObsConfig())
    feats, info = env.reset()                      # or env.reset(seed=..., spec=...) for the next episode
    feats, reward, terminated, truncated, info = env.step(action_mb)

Privileged ground truth (`env.privileged_state()`) goes only to the reward terms (full dict, in-process) and,
opt-in, as a slim subset to `info["privileged"]`. The feature dict returned to the policy is built by
`ObsPreprocessor.process` from the official observation alone.

Engine crashes (ViZDoom exceptions from step/reset) raise `EngineCrash`; `recover()` reopens the engine. The
caller marks the transition (`info["crashed"]`), never hides it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import vizdoom as vd

from doom_agent.obs import FEATURE_SPECS, ObsConfig, ObsPreprocessor
from training.rewards.base import BaseRewardTerm, RewardComposer, StepContext, register_term

MAX_TRAIN_SEED = 10_000  # seeds >= 10000 are reserved for val/test (docs/EVALUATION_PROTOCOL.md §5)
PRIVILEGED_KEYS = ("x", "y", "angle", "health", "kills", "damage_taken", "keys", "dead", "exit_reached")
CELL = 64.0  # info["episode"]["cells"]: distinct 64x64 cells visited (same cell as the eval protocol)
BUTTON_USE, BUTTON_NEXT_WEAPON = 7, 8
SUPPORTED_MODS: tuple[str, ...] = ()  # training-time engine mods; none in v1 (vanilla engine)

# ViZDoom raises these when its engine process dies or stops answering.
ENGINE_ERRORS: tuple[type[BaseException], ...] = tuple(
    getattr(vd, n) for n in ("ViZDoomUnexpectedExitException", "ViZDoomIsNotRunningException", "ViZDoomErrorException",
                             "SharedMemoryException", "MessageQueueException") if hasattr(vd, n)
) + (BrokenPipeError, ConnectionResetError)


class EngineCrash(RuntimeError):
    """The ViZDoom engine died during reset/step. The env must be `recover()`ed before reuse."""


@dataclass(frozen=True)
class EnvSpec:
    map_id: str
    wad: str  # repo-relative, absolute, or @vizdoom/<file>
    map: str
    episode_ticks: int
    render_buffers: bool = False  # must be True if obs.enable_depth_band
    mods: tuple[str, ...] = ()  # training-time engine mods; () = vanilla

    def __post_init__(self) -> None:
        if not 1 <= int(self.episode_ticks) <= 100_000:
            raise ValueError(f"EnvSpec.episode_ticks out of range: {self.episode_ticks}")
        object.__setattr__(self, "mods", tuple(self.mods))


@register_term
class EnvDefaultReward(BaseRewardTerm):
    """Pass-through of the wrapper's default reward (+1 exit, -1 death, +0.1 per kill). Env plumbing for smoke
    runs and for logging `return_env`; the shaped v1 terms live in training/rewards/*.py."""

    name = "env_default"

    def value(self, ctx: StepContext) -> float:
        return float(ctx.env_reward)


def build_composer(reward_cfg: Any) -> RewardComposer:
    """reward_cfg: a term list, `{"terms": [...]}`, or `{"terms": [...], "modules": ["pkg.mod", ...]}`. The optional
    `modules` are imported first so their `@register_term` classes exist (needed in spawned workers)."""
    import importlib

    if isinstance(reward_cfg, dict):
        reward_cfg = dict(reward_cfg)
        for mod in reward_cfg.pop("modules", ()) or ():
            importlib.import_module(mod)
        return RewardComposer.from_config(reward_cfg)
    return RewardComposer.from_config(list(reward_cfg or []))


def _check_seed(seed: int) -> int:
    seed = int(seed)
    if not 0 <= seed < MAX_TRAIN_SEED:
        raise ValueError(f"training seed {seed} outside [0, {MAX_TRAIN_SEED}) (>= 10000 is reserved for val/test)")
    return seed


def _check_action(action) -> np.ndarray:
    a = np.asarray(action)
    if a.shape != (9,):
        raise ValueError(f"action shape {a.shape} != (9,)")
    if not np.all((a == 0) | (a == 1)):
        raise ValueError(f"action values must be 0 or 1, got {a!r}")
    return a.astype(np.int8)


def _check_feats(feats: dict) -> dict:
    """Fail loudly on anything the policy must never see: unknown keys, wrong shape/dtype, NaN/Inf."""
    if set(feats) != set(FEATURE_SPECS):
        raise KeyError(f"feature keys {sorted(feats)} != FEATURE_SPECS")
    for k, spec in FEATURE_SPECS.items():
        x = feats[k]
        if x.shape != spec.shape or x.dtype != spec.np_dtype:
            raise ValueError(f"feature {k}: {x.shape} {x.dtype} != {spec.shape} {spec.np_dtype}")
        if x.dtype.kind == "f" and not np.isfinite(x).all():
            raise ValueError(f"feature {k} has non-finite values")
    return feats


def slim_privileged(priv: dict) -> dict:
    """The opt-in `info["privileged"]` subset (no sector arrays). Training-only."""
    out = {k: priv[k] for k in PRIVILEGED_KEYS}
    out["keys"] = np.asarray(out["keys"], np.float32).copy()
    return out


class TrainEnv:
    def __init__(self, spec: EnvSpec, *, seed: int, reward_cfg: Any = (), obs_cfg: ObsConfig = ObsConfig(),
                 return_privileged: bool = False):
        self.spec = spec
        self.next_seed = _check_seed(seed)
        self.obs_cfg = obs_cfg
        if obs_cfg.enable_depth_band and not spec.render_buffers:
            raise ValueError("obs.enable_depth_band needs EnvSpec.render_buffers=True")
        self.pre = ObsPreprocessor(obs_cfg, strict=True)
        self.composer = build_composer(reward_cfg)
        self.return_privileged = bool(return_privileged)
        self.env = None
        self._key = None  # (wad, map, render_buffers) of the open engine
        self._inv_cache: dict[tuple, dict] = {}
        self._priv: dict | None = None
        self._ep: dict | None = None
        self.crash_count = 0
        self.engine_opens = 0

    # ------------------------------------------------------------------ engine lifecycle
    @property
    def vanilla(self) -> bool:
        """True when no training-time engine modification is active (eval-equivalent engine)."""
        return not self.spec.mods

    def _open(self, spec: EnvSpec) -> None:
        from doom_env import make_env

        from training.envs.maps import resolve_wad

        unknown = [m for m in spec.mods if m not in SUPPORTED_MODS]
        if unknown:
            raise NotImplementedError(f"engine mods {unknown} are not implemented (supported: {SUPPORTED_MODS})")
        self._close_engine()
        self.env = make_env(str(resolve_wad(spec.wad)), spec.map, episode_ticks=int(spec.episode_ticks),
                            render_buffers=bool(spec.render_buffers))
        self._key = (spec.wad, spec.map.upper(), bool(spec.render_buffers))
        self.engine_opens += 1

    def _ensure(self, spec: EnvSpec) -> None:
        key = (spec.wad, spec.map.upper(), bool(spec.render_buffers))
        if self.env is None or key != self._key:
            self._open(spec)
        elif int(spec.episode_ticks) != self.env.episode_ticks:
            # ADAPTER (training-time): change the tick cap on the running engine instead of a 0.3 s re-init.
            # vizdoom's set_episode_timeout applies from the next new_episode (MEASURED: 40 ticks -> truncated
            # at tic 41). `env.episode_ticks` keeps privileged_state() consistent.
            self.env._game.set_episode_timeout(int(spec.episode_ticks))
            self.env.episode_ticks = int(spec.episode_ticks)
        self.spec = spec

    def _close_engine(self) -> None:
        if self.env is not None:
            try:
                self.env.close()
            except Exception:  # noqa: BLE001 - a dead engine may fail to close; we are replacing it anyway
                pass
        self.env = None
        self._key = None

    def recover(self) -> None:
        """Drop a crashed engine; the next reset() opens a fresh one."""
        self._close_engine()
        self.pre.reset()
        self._priv = None
        self._ep = None

    def close(self) -> None:
        self._close_engine()

    def _inject_crash(self) -> None:
        """TEST HOOK: kill the engine so the next step raises like a real ViZDoom crash."""
        self.env._game.close()

    # ------------------------------------------------------------------ inventory (cached per map)
    def _inventory(self) -> dict:
        from training.envs.inventory import _count_live, map_inventory

        key = (self.spec.wad, self.spec.map.upper())
        inv = self._inv_cache.get(key)
        if inv is None:
            inv = map_inventory(self.env)
            self._inv_cache[key] = inv
            return inv
        monsters, keys = _count_live(self.env)
        return dict(inv, monsters=monsters, keys=keys)

    # ------------------------------------------------------------------ gym-like API (no auto-reset)
    def reset(self, *, seed: int | None = None, spec: EnvSpec | None = None) -> tuple[dict, dict]:
        seed = _check_seed(self.next_seed if seed is None else seed)
        spec = spec or self.spec
        try:
            self._ensure(spec)
            obs, winfo = self.env.reset(seed=seed)
            priv = self.env.privileged_state()
        except ENGINE_ERRORS as e:
            raise EngineCrash(f"reset {spec.map_id} seed={seed}: {type(e).__name__}: {e}") from e
        self.pre.reset()
        self.composer.reset(priv, obs, self._inventory(), spec)
        self._priv = priv
        door_lock = priv.get("door_lock")
        doors = np.flatnonzero(np.asarray(door_lock) >= 0) if door_lock is not None else np.zeros(0, np.int64)
        self._ep = {"map_id": spec.map_id, "seed": seed, "episode_ticks": int(spec.episode_ticks), "length": 0,
                    "return_shaped": 0.0, "return_env": 0.0, "cells": {self._cell(priv)}, "use_edges": 0,
                    "weapon_edges": 0, "door_reversals": 0, "prev_action": np.zeros(9, np.int8),
                    "doors": doors, "door_ceil": self._door_ceil(priv, doors), "door_vel": np.zeros(len(doors)),
                    "pending": None}
        feats = _check_feats(self.pre.process(obs))
        return feats, self._info(winfo, priv, {})

    def step(self, action) -> tuple[dict, float, bool, bool, dict]:
        if self._ep is None:
            raise RuntimeError("TrainEnv.step() before reset()")
        a = _check_action(action)
        self.pre.commit_action(a)
        try:
            obs, env_r, term, trunc, winfo = self.env.step(a)
            priv = self.env.privileged_state()
        except ENGINE_ERRORS as e:
            self.crash_count += 1
            raise EngineCrash(f"step {self.spec.map_id}: {type(e).__name__}: {e}") from e
        ep = self._ep
        ctx = StepContext(priv_prev=self._priv, priv=priv, obs=obs, action_mb=a, terminated=bool(term),
                          truncated=bool(trunc), env_reward=float(env_r), info=winfo, step=ep["length"])
        total, per_term = self.composer(ctx)
        if not math.isfinite(total):
            raise ValueError(f"non-finite shaped reward {total}")
        feats = _check_feats(self.pre.process(obs))
        self._priv = priv
        ep["length"] += 1
        ep["return_shaped"] += float(total)
        ep["return_env"] += float(env_r)
        ep["cells"].add(self._cell(priv))
        prev = ep["prev_action"]
        ep["use_edges"] += int(a[BUTTON_USE] == 1 and prev[BUTTON_USE] == 0)
        ep["weapon_edges"] += int(a[BUTTON_NEXT_WEAPON] == 1 and prev[BUTTON_NEXT_WEAPON] == 0)
        self._track_doors(priv, use_edge=bool(a[BUTTON_USE] == 1 and prev[BUTTON_USE] == 0))
        ep["prev_action"] = a
        info = self._info(winfo, priv, per_term)
        info["env_reward"] = float(env_r)
        if term or trunc:
            info["final_feats"] = feats
            info["episode"] = self._episode_summary(priv, bool(term), bool(trunc), bool(winfo.get("exit_reached")))
        return feats, float(total), bool(term), bool(trunc), info

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _door_ceil(priv: dict, doors: np.ndarray) -> np.ndarray:
        ceil = priv.get("sector_ceiling")
        return np.zeros(len(doors)) if ceil is None or not len(doors) else np.asarray(ceil, np.float64)[doors].copy()

    def _track_doors(self, priv: dict, use_edge: bool) -> None:
        """door_reversals (D11 logging): a door sector that was moving when a USE rising edge happened and moves
        the opposite way within that step or the next one. A Door_Raise closing after its delay is not counted
        (no USE edge, and the door pauses in between). Training-only diagnostics from privileged_state()."""
        ep = self._ep
        if not len(ep["doors"]):
            return
        ceil = self._door_ceil(priv, ep["doors"])
        vel = np.sign(ceil - ep["door_ceil"])
        pending = ep["pending"]
        if pending is not None:
            idx, sign, ttl = pending
            flipped = vel[idx] == -sign
            ep["door_reversals"] += int(flipped.sum())
            idx, sign = idx[~flipped], sign[~flipped]
            ep["pending"] = (idx, sign, ttl - 1) if ttl > 1 and len(idx) else None
        if use_edge:
            moving = np.flatnonzero(ep["door_vel"] != 0)  # moving on the step before the edge
            flipped = vel[moving] == -ep["door_vel"][moving]  # reversed within this step
            ep["door_reversals"] += int(flipped.sum())
            rest = moving[~flipped]
            if len(rest):
                ep["pending"] = (rest, ep["door_vel"][rest], 1)  # ...or on the next step
        ep["door_ceil"], ep["door_vel"] = ceil, vel

    @staticmethod
    def _cell(priv: dict) -> tuple[int, int]:
        return int(math.floor(priv["x"] / CELL)), int(math.floor(priv["y"] / CELL))

    def _info(self, winfo: dict, priv: dict, per_term: dict) -> dict:
        info = {"map_id": self.spec.map_id, "tic": int(winfo.get("tic", 0)), "exit_reached": bool(winfo.get("exit_reached")),
                "reward_terms": dict(per_term)}
        if self.return_privileged:
            info["privileged"] = slim_privileged(priv)
        return info

    def _episode_summary(self, priv: dict, term: bool, trunc: bool, exit_reached: bool) -> dict:
        ep = self._ep
        outcome = "exit" if exit_reached else ("death" if priv["dead"] else "timeout")
        stats = self.composer.episode_stats()
        return {"map_id": ep["map_id"], "seed": ep["seed"], "episode_ticks": ep["episode_ticks"], "length": ep["length"],
                "return_shaped": ep["return_shaped"], "return_env": ep["return_env"], "outcome": outcome,
                "cells": len(ep["cells"]), "keys": int(np.asarray(priv["keys"]).sum()), "kills": int(priv["kills"]),
                "use_edges": ep["use_edges"], "weapon_edges": ep["weapon_edges"], "door_reversals": ep["door_reversals"],
                "reward_terms": dict(stats.raw)}


def make_train_env(spec: EnvSpec, *, seed: int, reward_cfg: Any = (), obs_cfg: ObsConfig = ObsConfig(),
                   return_privileged: bool = False) -> TrainEnv:
    """docs/MODULE_CONTRACTS.md §5.1. The engine opens lazily on the first reset()."""
    return TrainEnv(spec, seed=seed, reward_cfg=reward_cfg, obs_cfg=obs_cfg, return_privileged=return_privileged)
