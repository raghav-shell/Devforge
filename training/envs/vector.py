"""Subprocess vector env (docs/MODULE_CONTRACTS.md §5.2). Training-only.

N = num_workers * envs_per_worker env slots; each worker process runs `envs_per_worker` TrainEnvs (one
ViZDoom engine each). Global slot i is fully determined by (seed_base, i):
    episode seeds  ~ default_rng([seed_base, i])     in [0, 9999]
    map sampling   ~ default_rng([seed_base, i, 1])
so results do not depend on how slots are split across workers.

Worker loop per slot and step:
    commit_action(a) -> env.step(a) -> reward terms -> feats = pre.process(obs)
    if done: final_feats = feats; env.reset(seed=next); pre.reset(); feats = pre.process(obs0)

Crashes are flagged, never hidden:
- engine crash in a slot (ViZDoom exception): engine reopened, slot reported truncated with
  info["crashed"] = True, reward 0, NO final_feats (trainer uses trunc_value 0), counted;
- a dead worker process (EOF on its pipe): the worker is restarted and every slot it ran is reported the same
  way, with info["worker_restarted"] = True. Restarted workers draw from fresh RNG streams
  (default_rng([seed_base, i, 2, restart_no])) so they do not replay old seeds.
Any other worker exception (bad action, NaN features, reward bug) is re-raised in the parent with the worker
traceback: fail loudly.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import signal
import sys
import time
import traceback
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from doom_agent.obs import FEATURE_SPECS, ObsConfig, stack_features

DEFAULT_START_METHOD = "forkserver"  # macOS: fork is unsafe with ViZDoom/threads; forkserver starts faster than spawn
# A new map costs an engine re-init (~0.3 s). MEASURED at 4x4 on generated maps (random agent, short episodes):
# episodes_per_map 1 -> 1180 SPS, 8 -> 2122 SPS. 4 keeps per-slot map diversity high (64 slots cover the pool).
DEFAULT_EPISODES_PER_MAP = 4


def default_num_workers() -> int:
    """MEASURED on the M4 (10 cores, docs/ENVIRONMENT.md A3 table): throughput saturates at about 8 workers."""
    return max(1, min(8, (os.cpu_count() or 2) - 2))


@dataclass
class StepResult:
    obs: dict[str, np.ndarray]  # (N, ...) features of the NEXT obs (post auto-reset where done)
    reward: np.ndarray  # (N,) f32 total shaped reward
    terminated: np.ndarray  # (N,) bool
    truncated: np.ndarray  # (N,) bool
    episode_start: np.ndarray  # (N,) bool == terminated | truncated
    infos: list[dict] = field(default_factory=list)  # per env; NEVER merged into obs


# sampler_cfg keys (env.* config, docs/MODULE_CONTRACTS.md §5.1): manifests, weights, entries (in-memory manifest
# entries), randomize_ticks, render_buffers, mods, return_privileged, episodes_per_map (map switch every k episodes).
# step_threads (default True): step a worker's slots concurrently from a thread pool.
# MEASURED: stepping several ViZDoom engines SEQUENTIALLY from one process intermittently collapses (bimodal; e.g.
# 2x4 on two_rooms: 356 vs 2629 decisions/s in back-to-back runs at the same load; engine-only make_action round
# trips show it too, so it is not wrapper code). Threaded stepping is stable (2329 / 2394) and bit-identical.
# HYPOTHESIS (not measured): engine wake-up latency after idling while sibling engines step.
SAMPLER_KEYS = frozenset({"manifests", "weights", "entries", "randomize_ticks", "render_buffers", "mods",
                          "return_privileged", "episodes_per_map", "step_threads"})


def _make_sampler(cfg: dict):
    from training.envs.maps import MapSampler

    return MapSampler(cfg.get("manifests", ()), cfg.get("weights"), None, randomize_ticks=cfg.get("randomize_ticks", True),
                      render_buffers=cfg.get("render_buffers", False), mods=tuple(cfg.get("mods", ())),
                      entries=cfg.get("entries"))


# ---------------------------------------------------------------------------------------------- worker
class _Slot:
    def __init__(self, i: int, base_sampler, seed_base: int, restart_no: int, reward_cfg, obs_cfg, return_priv: bool,
                 episodes_per_map: int):
        from training.envs.factory import make_train_env

        if restart_no == 0:
            self.seed_rng = np.random.default_rng([seed_base, i])
            map_rng = np.random.default_rng([seed_base, i, 1])
        else:
            self.seed_rng = np.random.default_rng([seed_base, i, 2, restart_no])
            map_rng = np.random.default_rng([seed_base, i, 3, restart_no])
        self.i = i
        self.sampler = base_sampler.with_rng(map_rng)
        self.episodes_per_map = max(1, int(episodes_per_map))
        self.episodes = 0
        spec = self.sampler.sample()
        self.env = make_train_env(spec, seed=0, reward_cfg=reward_cfg, obs_cfg=obs_cfg,
                                  return_privileged=return_priv)
        self._pending_spec = spec

    def _seed(self) -> int:
        return int(self.seed_rng.integers(0, 10_000))

    def _next_spec(self):
        if self._pending_spec is not None:
            spec, self._pending_spec = self._pending_spec, None
            return spec
        if self.episodes % self.episodes_per_map == 0:
            return self.sampler.sample()
        return self.sampler.sample(keep=self.env.spec)

    def reset(self) -> dict:
        from training.envs.factory import EngineCrash

        for attempt in range(3):
            spec, seed = self._next_spec(), self._seed()
            try:
                feats, _ = self.env.reset(seed=seed, spec=spec)
                self.episodes += 1
                return feats
            except EngineCrash:
                self.env.crash_count += 1
                self.env.recover()
        raise RuntimeError(f"slot {self.i}: engine failed to reset 3 times in a row")

    def step(self, a: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        from training.envs.factory import EngineCrash

        try:
            feats, r, term, trunc, info = self.env.step(a)
        except EngineCrash as e:
            map_id = self.env.spec.map_id
            self.env.recover()
            feats = self.reset()
            info = {"map_id": map_id, "tic": 0, "exit_reached": False, "reward_terms": {}, "crashed": True,
                    "crash_error": str(e), "crash_count": self.env.crash_count}
            return feats, 0.0, False, True, info
        if term or trunc:
            feats = self.reset()
        return feats, r, term, trunc, info


def _worker(conn, worker_idx: int, slot_ids: list[int], sampler_cfg: dict, seed_base: int, restart_no: int,
            reward_cfg, obs_cfg: ObsConfig) -> None:
    # Own process group: ViZDoom runs each engine as a CHILD process of this worker, which inherits the pipe and
    # survives a worker death (MEASURED). The parent kills the whole group on restart so no engine is orphaned.
    if hasattr(os, "setpgrp"):
        os.setpgrp()
    # Parent shutdown terminates daemon workers with SIGTERM: exit through `finally` so engines are closed.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        cfg = dict(sampler_cfg)
        base = _make_sampler(cfg)
        slots = [_Slot(i, base, seed_base, restart_no, reward_cfg, obs_cfg, bool(cfg.get("return_privileged", False)),
                       cfg.get("episodes_per_map", DEFAULT_EPISODES_PER_MAP)) for i in slot_ids]
        pool = None
        if len(slots) > 1 and cfg.get("step_threads", True):
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(len(slots), thread_name_prefix=f"doom-w{worker_idx}")
        conn.send(("ready", os.getpid()))
    except Exception:  # noqa: BLE001 - forwarded to the parent, which raises
        conn.send(("error", traceback.format_exc()))
        conn.close()
        return
    try:
        while True:
            cmd, arg = conn.recv()
            if cmd == "step":
                if pool is not None:  # results keep slot order; the first slot exception propagates
                    out = list(pool.map(lambda sa: sa[0].step(sa[1]), zip(slots, arg)))
                else:
                    out = [s.step(a) for s, a in zip(slots, arg)]
                feats = stack_features([o[0] for o in out])
                conn.send(("ok", (feats, np.array([o[1] for o in out], np.float32), np.array([o[2] for o in out], bool),
                                  np.array([o[3] for o in out], bool), [o[4] for o in out])))
            elif cmd == "reset":
                feats = list(pool.map(lambda s: s.reset(), slots)) if pool is not None else [s.reset() for s in slots]
                conn.send(("ok", stack_features(feats)))
            elif cmd == "inject_crash":
                slots[arg].env._inject_crash()
                conn.send(("ok", None))
            elif cmd == "stats":
                conn.send(("ok", {"crashes": sum(s.env.crash_count for s in slots),
                                  "engine_opens": sum(s.env.engine_opens for s in slots),
                                  "episodes": sum(s.episodes for s in slots)}))
            elif cmd == "close":
                break
            else:
                raise ValueError(f"unknown command {cmd!r}")
    except (EOFError, KeyboardInterrupt, SystemExit):
        pass
    except Exception:  # noqa: BLE001 - forwarded to the parent, which raises
        try:
            conn.send(("error", traceback.format_exc()))
        except (BrokenPipeError, OSError):
            pass
    finally:
        if pool is not None:
            pool.shutdown(wait=True)
        for s in slots:
            s.env.close()
        conn.close()


class WorkerError(RuntimeError):
    """A worker raised a non-crash exception (bad action, NaN features, reward bug...). Fail loudly."""


# ---------------------------------------------------------------------------------------------- parent
class SubprocVecEnv:
    def __init__(self, sampler_cfg: dict, *, num_workers: int | None = None, envs_per_worker: int = 8,
                 seed_base: int = 0, reward_cfg: Any = (), obs_cfg: ObsConfig = ObsConfig(),
                 start_method: str = DEFAULT_START_METHOD, timeout: float = 120.0):
        if not 0 <= int(seed_base) < 10_000:
            raise ValueError(f"seed_base {seed_base} outside [0, 10000)")
        self.num_workers = int(num_workers or default_num_workers())
        self.envs_per_worker = int(envs_per_worker)
        if self.num_workers < 1 or self.envs_per_worker < 1:
            raise ValueError("num_workers and envs_per_worker must be >= 1")
        self.num_envs = self.num_workers * self.envs_per_worker
        self.seed_base = int(seed_base)
        unknown = set(sampler_cfg) - SAMPLER_KEYS
        if unknown:
            raise ValueError(f"unknown sampler_cfg keys {sorted(unknown)}; allowed {sorted(SAMPLER_KEYS)}")
        _make_sampler(sampler_cfg)  # validate in the parent: val/test refusal and bad configs fail here, not in a worker
        self._args = (dict(sampler_cfg), self.seed_base, reward_cfg, obs_cfg)
        self._ctx = mp.get_context(start_method)
        self._timeout = float(timeout)
        self._conns: list = [None] * self.num_workers
        self._procs: list = [None] * self.num_workers
        self._restarts = [0] * self.num_workers
        self.crashed_steps = 0  # slot-steps reported as crashed (engine crash or worker restart)
        self.worker_restarts = 0
        self._closed = False
        try:
            for w in range(self.num_workers):
                self._start(w)
        except BaseException:
            self.close()
            raise

    # ------------------------------------------------------------------ process management
    def _slots(self, w: int) -> list[int]:
        return list(range(w * self.envs_per_worker, (w + 1) * self.envs_per_worker))

    def _start(self, w: int) -> None:
        parent, child = self._ctx.Pipe()
        sampler_cfg, seed_base, reward_cfg, obs_cfg = self._args
        p = self._ctx.Process(target=_worker, name=f"doom-vec-{w}", daemon=True,
                              args=(child, w, self._slots(w), sampler_cfg, seed_base, self._restarts[w], reward_cfg, obs_cfg))
        p.start()
        child.close()
        self._conns[w], self._procs[w] = parent, p
        tag, payload = self._recv(w)
        if tag != "ready":
            raise WorkerError(f"worker {w} failed to start:\n{payload}")

    def _recv(self, w: int):
        """Next message from worker w. A dead worker raises EOFError even though its orphaned engine child may
        still hold the pipe open (so EOF alone is not a reliable death signal)."""
        conn, proc = self._conns[w], self._procs[w]
        deadline = time.monotonic() + self._timeout
        while not conn.poll(0.05):
            if not proc.is_alive() and not conn.poll(0):
                raise EOFError(f"worker {w} (pid {proc.pid}) died, exit code {proc.exitcode}")
            if time.monotonic() > deadline:
                raise TimeoutError(f"worker {w} did not answer within {self._timeout}s")
        tag, payload = conn.recv()
        if tag == "error":
            raise WorkerError(f"worker {w} raised:\n{payload}")
        return tag, payload

    def _restart(self, w: int) -> dict:
        """Replace a dead worker; returns its fresh reset features."""
        try:
            self._conns[w].close()
        except OSError:
            pass
        p = self._procs[w]
        if p is not None:
            _kill_group(p)
            p.join(timeout=5)
        self._restarts[w] += 1
        self.worker_restarts += 1
        self._start(w)
        self._conns[w].send(("reset", None))
        return self._recv(w)[1]

    def _call(self, w: int, cmd: str, arg=None):
        self._conns[w].send((cmd, arg))
        return self._recv(w)[1]

    # ------------------------------------------------------------------ API
    def reset(self) -> dict[str, np.ndarray]:
        for w in range(self.num_workers):
            self._conns[w].send(("reset", None))
        parts = []
        for w in range(self.num_workers):
            try:
                parts.append(self._recv(w)[1])
            except (EOFError, BrokenPipeError, ConnectionResetError):
                parts.append(self._restart(w))
        return _concat(parts)

    def step(self, actions_mb: np.ndarray) -> StepResult:
        a = np.asarray(actions_mb)
        if a.shape != (self.num_envs, 9):
            raise ValueError(f"actions shape {a.shape} != ({self.num_envs}, 9)")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("actions must be 0/1")
        a = a.astype(np.int8)
        k = self.envs_per_worker
        sent = []
        for w in range(self.num_workers):
            try:
                self._conns[w].send(("step", a[w * k : (w + 1) * k]))
                sent.append(True)
            except (BrokenPipeError, ConnectionResetError, OSError):
                sent.append(False)
        feats, rew, term, trunc, infos = [], [], [], [], []
        for w in range(self.num_workers):
            try:
                if not sent[w]:
                    raise EOFError
                f, r, te, tr, inf = self._recv(w)[1]
            except (EOFError, BrokenPipeError, ConnectionResetError):
                f = self._restart(w)
                r, te, tr = np.zeros(k, np.float32), np.zeros(k, bool), np.ones(k, bool)
                inf = [{"map_id": None, "tic": 0, "exit_reached": False, "reward_terms": {}, "crashed": True,
                        "worker_restarted": True, "crash_error": f"worker {w} died"} for _ in range(k)]
            feats.append(f)
            rew.append(r)
            term.append(te)
            trunc.append(tr)
            infos.extend(inf)
        res = StepResult(_concat(feats), np.concatenate(rew).astype(np.float32), np.concatenate(term),
                         np.concatenate(trunc), np.zeros(self.num_envs, bool), infos)
        res.episode_start = res.terminated | res.truncated
        self.crashed_steps += sum(1 for i in infos if i.get("crashed"))
        return res

    def stats(self) -> dict:
        out = {"crashes": 0, "engine_opens": 0, "episodes": 0}
        for w in range(self.num_workers):
            for key, v in self._call(w, "stats").items():
                out[key] += v
        out.update(crashed_steps=self.crashed_steps, worker_restarts=self.worker_restarts)
        return out

    # ------------------------------------------------------------------ test hooks
    def _debug_inject_crash(self, env_index: int) -> None:
        """TEST HOOK: kill the ViZDoom engine of slot `env_index`; its next step is reported as crashed."""
        w, j = divmod(int(env_index), self.envs_per_worker)
        self._call(w, "inject_crash", j)

    def _debug_kill_worker(self, w: int) -> None:
        """TEST HOOK: SIGKILL worker process `w`; the next step restarts it and flags its slots."""
        self._procs[w].kill()
        self._procs[w].join(timeout=10)

    # ------------------------------------------------------------------ teardown
    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for c in self._conns:
            if c is not None:
                try:
                    c.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
        for p in self._procs:
            if p is not None:
                p.join(timeout=10)
                _kill_group(p)  # no-op after a clean exit; reaps a hung worker and any orphaned engine
                p.join(timeout=5)
        for c in self._conns:
            if c is not None:
                c.close()

    def __enter__(self) -> SubprocVecEnv:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:  # noqa: BLE001 - interpreter shutdown
            pass


def _kill_group(p) -> None:
    """SIGKILL worker p's process group (the worker and its ViZDoom engine children)."""
    if p.pid is None:
        return
    try:
        os.killpg(p.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        if p.is_alive():
            p.kill()


def _concat(parts: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.concatenate([p[k] for p in parts], axis=0) for k in FEATURE_SPECS}
