"""Synchronous recurrent PPO trainer (docs/MODULE_CONTRACTS.md section 4; DECISIONS D1).

Loop per iteration: collect T steps from N envs -> GAE -> `epochs` x `num_minibatches` updates on
whole env sequences via `policy.forward_sequence` (chunked if `bptt_chunk` < T).

Contract points implemented here (cited so reviewers can check them):
- Hidden state: the trainer never zeroes the carried state; the policy masks it with
  `episode_start` before consuming obs_t (section 3.4). `state0` is stored PRE-mask.
- Truncation (section 4.2): for each env with truncated & ~terminated & ~crashed at step t, the
  trainer runs `policy.step(final_feats, h_t_post, episode_start=False)` under no_grad and stores the
  value as trunc_value[t]. The returned state is discarded. A crash (info["crashed"]) gives
  trunc_value = 0 and is COUNTED. A non-crash truncation with no final_feats raises.
- `episode_start` from the env must equal terminated | truncated; a mismatch raises.
- Epoch-1 / minibatch-1 ratio check before any optimiser step (section 4.3), logged every iteration.
- Grad-norm clip (pre-clip norm logged); approx-KL early stop at `target_kl` (checked before the step).
- Entropy = sum over factor heads; coefficient linearly scheduled `ent_coef` -> `ent_coef_final`.
"""

from __future__ import annotations

import random
import sys
import time
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn

from training.configs import TrainConfig
from training.ppo.adapters import ActionAPI, features_to_torch
from training.ppo.loss import ReturnNormalizer, explained_variance, ppo_loss
from training.ppo.rollout import RolloutBuffer


class RatioCheckError(RuntimeError):
    pass


def _stack_feats(list_of_feats: list[dict]) -> dict:
    keys = list_of_feats[0].keys()
    return {k: np.stack([f[k] for f in list_of_feats], 0) for k in keys}


def linear_schedule(start: float, end: float, progress: float) -> float:
    p = min(max(progress, 0.0), 1.0)
    return start + (end - start) * p


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PPOTrainer:
    def __init__(self, policy: nn.Module, vec_env: Any, cfg: TrainConfig, action_api: ActionAPI, *,
                 obs_specs: dict, device: str | torch.device = "cpu", logger=None):
        self.cfg = cfg
        p = cfg.ppo
        if vec_env.num_envs != p.num_envs:
            raise ValueError(f"vec_env.num_envs={vec_env.num_envs} != ppo.num_envs={p.num_envs}")
        if cfg.aux.heads:
            raise NotImplementedError("aux heads are a v1 slot only (MODULE_CONTRACTS section 4.5)")
        self.device = torch.device(device)
        self.policy = policy.to(self.device)
        self.env = vec_env
        self.api = action_api
        self.logger = logger
        self.extra_value_heads = tuple(cfg.model.get("extra_value_heads", ()))
        if self.extra_value_heads:
            raise NotImplementedError("extra value heads (H7) are not wired into GAE yet")
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=p.lr, eps=p.adam_eps)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, self._lr_factor)
        self.gen = torch.Generator(device="cpu")
        self.gen.manual_seed(cfg.run.seed)
        self.return_norm = ReturnNormalizer(p.num_envs, p.gamma, clip=p.reward_clip, enabled=p.return_norm)
        self.buffer = RolloutBuffer(p.rollout_len, p.num_envs, policy.state_size, obs_specs,
                                    action_dim=action_api.action_dim, chunk=p.chunk)
        self.update_idx = 0
        self.global_step = 0
        self.counters: dict[str, int] = defaultdict(int)  # cumulative: crashed, episodes, ratio violations...
        self.obs: dict | None = None
        self.episode_start: np.ndarray | None = None
        self.state: torch.Tensor | None = None

    # ---------------------------------------------------------------- schedules
    def _lr_factor(self, update: int) -> float:
        p = self.cfg.ppo
        if p.lr_decay_frac <= 0:
            return 1.0
        start = p.total_updates * (1.0 - p.lr_decay_frac)
        if update <= start:
            return 1.0
        prog = (update - start) / max(1.0, p.total_updates - start)
        return linear_schedule(1.0, p.lr_final / p.lr, prog)

    def ent_coef(self) -> float:
        p = self.cfg.ppo
        return linear_schedule(p.ent_coef, p.ent_coef_final, self.update_idx / max(1, p.total_updates - 1))

    # ---------------------------------------------------------------- env carry
    def reset_env(self) -> None:
        self.obs = self.env.reset()
        self.episode_start = np.ones(self.env.num_envs, dtype=bool)
        self.state = self.policy.initial_state(self.env.num_envs, self.device)
        self.return_norm.ret[:] = 0.0

    # ---------------------------------------------------------------- collection
    @torch.no_grad()
    def collect(self) -> dict:
        if self.obs is None:
            self.reset_env()
        cfg, buf, N = self.cfg.ppo, self.buffer, self.env.num_envs
        buf.reset()
        self.policy.eval()
        t_env = t_inf = 0.0
        t0 = time.perf_counter()
        ep_infos: list[dict] = []
        term_sums: dict[str, float] = defaultdict(float)
        v_at_start: list[float] = []
        it = defaultdict(int)

        for _t in range(cfg.rollout_len):
            ti = time.perf_counter()
            buf.add_pre(self.obs, self.episode_start, self.state.cpu().numpy())
            x = features_to_torch(self.obs, self.device)
            es = torch.from_numpy(self.episode_start.copy()).to(self.device)
            out, new_state = self.policy.step(x, self.state, es)
            logits = out["logits"].float().cpu()
            dist = self.api.dist_cls(logits)
            idx = dist.sample(generator=self.gen)
            logp = dist.log_prob(idx)
            value = out["value"].float().cpu().numpy()
            if self.episode_start.any():
                v_at_start.extend(value[self.episode_start].tolist())
            idx_np = idx.numpy().astype(np.int64)
            action_mb = self.api.to_buttons(idx_np).astype(np.int8)
            t_inf += time.perf_counter() - ti

            te = time.perf_counter()
            res = self.env.step(action_mb)
            t_env += time.perf_counter() - te

            ti = time.perf_counter()
            term = np.asarray(res.terminated, dtype=bool)
            trunc = np.asarray(res.truncated, dtype=bool)
            done = term | trunc
            if not np.array_equal(np.asarray(res.episode_start, dtype=bool), done):
                raise RuntimeError("StepResult.episode_start != terminated | truncated (MODULE_CONTRACTS 5.2)")
            crashed = np.array([bool(info.get("crashed", False)) for info in res.infos], dtype=bool)
            if crashed.any():
                it["crashed"] += int(crashed.sum())
                if not np.all(trunc[crashed]):
                    raise RuntimeError("crashed env slot not reported as truncated (MODULE_CONTRACTS 5.2)")

            trunc_value = np.zeros(N, dtype=np.float32)
            boot = np.flatnonzero(trunc & ~term & ~crashed)
            if boot.size:
                finals = []
                for i in boot:
                    ff = res.infos[i].get("final_feats")
                    if ff is None:
                        raise RuntimeError(f"env {i}: truncated without info['final_feats'] and not crashed")
                    finals.append(ff)
                fx = features_to_torch(_stack_feats(finals), self.device)
                bi = torch.from_numpy(boot).to(self.device)
                out_f, _discard = self.policy.step(fx, new_state[bi],
                                                   torch.zeros(boot.size, dtype=torch.bool, device=self.device))
                trunc_value[boot] = out_f["value"].float().cpu().numpy()

            reward = np.asarray(res.reward, dtype=np.float32)
            reward_scaled = self.return_norm(reward, done)
            buf.add_post(action_idx=idx_np, action_mb=action_mb, logp=logp.numpy(), value=value,
                         reward=reward, reward_scaled=reward_scaled, terminated=term, truncated=trunc,
                         crashed=crashed, trunc_value=trunc_value)

            for i, info in enumerate(res.infos):
                for k, v in info.get("reward_terms", {}).items():
                    term_sums[k] += float(v)
                if done[i]:
                    if crashed[i]:
                        continue  # counted above as crashed; a crashed episode is a failed episode
                    if "episode" in info:
                        ep_infos.append(info["episode"])
                    else:
                        it["episodes_missing_info"] += 1

            self.obs = res.obs
            self.episode_start = np.asarray(res.episode_start, dtype=bool).copy()
            self.state = new_state
            t_inf += time.perf_counter() - ti

        ti = time.perf_counter()
        x = features_to_torch(self.obs, self.device)
        es = torch.from_numpy(self.episode_start.copy()).to(self.device)
        out, _ = self.policy.step(x, self.state, es)
        buf.set_next(self.obs, self.episode_start, self.state.cpu().numpy(), out["value"].float().cpu().numpy())
        buf.compute_advantages(cfg.gamma, cfg.gae_lambda)
        t_inf += time.perf_counter() - ti

        steps = cfg.rollout_len * N
        self.global_step += steps
        for k, v in it.items():
            self.counters[k] += v
        self.counters["episodes"] += len(ep_infos)
        return self._collect_stats(ep_infos, term_sums, v_at_start, it, t_env, t_inf,
                                   time.perf_counter() - t0, steps)

    def _collect_stats(self, ep_infos, term_sums, v_at_start, it, t_env, t_inf, t_total, steps) -> dict:
        buf = self.buffer
        s: dict[str, Any] = {
            "time/collect_s": t_total, "time/env_s": t_env, "time/policy_inference_s": t_inf,
            "sps/env_only": steps / max(t_env, 1e-9), "sps/collect": steps / max(t_total, 1e-9),
            "reward/raw_mean": float(buf.reward.mean()), "reward/scaled_mean": float(buf.reward_scaled.mean()),
            "reward/return_std": self.return_norm.std,
            "reward/extrinsic_mean": float(buf.reward.mean()),
            "reward/intrinsic_mean": 0.0,  # no intrinsic stream in v1 (DECISIONS D8: cell bonus is in r_ext)
            "env/crashed": it.get("crashed", 0), "env/crashed_total": self.counters["crashed"],
            "env/episodes_missing_info": it.get("episodes_missing_info", 0),
            "episodes/count": len(ep_infos), "episodes/total": self.counters["episodes"],
            "value/mean": float(buf.value.mean()),
            "adv/mean": float(buf.advantages.mean()),
        }
        for k, v in term_sums.items():
            s[f"reward_terms/{k}_per_step"] = v / steps
        if v_at_start:
            s["value/at_episode_start"] = float(np.mean(v_at_start))
        boot = buf.truncated & ~buf.terminated & ~buf.crashed
        mid = ~(buf.terminated | buf.truncated)
        if boot.any():
            s["adv/at_truncation"] = float(buf.advantages[boot].mean())
            s["value/trunc_value_mean"] = float(buf.trunc_value[boot].mean())
        if mid.any():
            s["adv/mid_episode"] = float(buf.advantages[mid].mean())
        if ep_infos:
            outcomes = [e.get("outcome") for e in ep_infos]
            n = len(ep_infos)
            s["episode/completion_rate"] = outcomes.count("exit") / n
            s["episode/death_rate"] = outcomes.count("death") / n
            s["episode/timeout_rate"] = outcomes.count("timeout") / n
            for key in ("return_shaped", "return_env", "length", "cells", "keys", "kills", "use_edges",
                        "weapon_edges", "door_reversals"):
                vals = [e[key] for e in ep_infos if key in e and e[key] is not None]
                if vals:
                    s[f"episode/{key}_mean"] = float(np.mean(vals))
            s["episode/reward_mean"] = s.get("episode/return_shaped_mean", float("nan"))
        return s

    # ---------------------------------------------------------------- update
    def _minibatch_tensors(self, env_idx: np.ndarray) -> dict:
        buf, C, L = self.buffer, self.buffer.num_chunks, self.buffer.chunk
        d = buf.env_minibatch(env_idx)
        M = len(env_idx)

        def seq(a: np.ndarray) -> torch.Tensor:  # (T, M, ...) -> (L, C*M, ...), chunk-major batch
            a = a.reshape((C, L, M) + a.shape[2:])
            a = np.swapaxes(a, 0, 1).reshape((L, C * M) + a.shape[3:])
            return torch.from_numpy(np.ascontiguousarray(a)).to(self.device)

        return {
            "obs": {k: seq(v) for k, v in d["obs"].items()},
            "episode_start": seq(d["episode_start"]),
            "state0": torch.from_numpy(np.ascontiguousarray(d["state0"].reshape(C * M, -1))).to(self.device),
            "action_idx": seq(d["action_idx"]),
            "logp": seq(d["logp"]),
            "value": seq(d["value"]),
            "advantages": seq(d["advantages"]),
            "returns": seq(d["returns"]),
        }

    def update(self) -> dict:
        cfg = self.cfg.ppo
        N = self.env.num_envs
        per_mb = N // cfg.num_minibatches
        ent_coef = self.ent_coef()
        self.policy.train()
        t0 = time.perf_counter()
        agg: dict[str, list] = defaultdict(list)
        ratio_check = float("nan")
        early_stop = False
        n_steps = 0
        for epoch in range(cfg.epochs):
            perm = torch.randperm(N, generator=self.gen).numpy()
            for mb in range(cfg.num_minibatches):
                e = np.sort(perm[mb * per_mb:(mb + 1) * per_mb])
                b = self._minibatch_tensors(e)
                out, _ = self.policy.forward_sequence(b["obs"], b["state0"], b["episode_start"])
                dist = self.api.dist_cls(out["logits"].float())
                new_logp = dist.log_prob(b["action_idx"])
                lo = ppo_loss(new_logp.reshape(-1), b["logp"].reshape(-1), b["advantages"].reshape(-1),
                              out["value"].float().reshape(-1), b["value"].reshape(-1), b["returns"].reshape(-1),
                              dist.entropy_per_head(), clip_eps=cfg.clip_eps, vf_coef=cfg.vf_coef,
                              ent_coef=ent_coef, norm_adv=cfg.norm_adv, value_clip=cfg.value_clip,
                              value_clip_eps=cfg.value_clip_eps)
                if epoch == 0 and mb == 0:
                    ratio_check = float(lo.ratio_max_dev)
                    if not ratio_check < cfg.ratio_check_tol:
                        self.counters["ratio_check_violations"] += 1
                        msg = (f"epoch-1 ratio check failed: max|ratio-1|={ratio_check:.3e} >= "
                               f"{cfg.ratio_check_tol:.0e} (sequence/mask/state bug, MODULE_CONTRACTS 4.3)")
                        if cfg.ratio_check_fail:
                            raise RatioCheckError(msg)
                        print("WARNING [PPOTrainer]: " + msg, file=sys.stderr)
                if cfg.target_kl > 0 and float(lo.approx_kl) > cfg.target_kl:
                    early_stop = True
                    agg["approx_kl_at_stop"].append(float(lo.approx_kl))
                    break
                self.optimizer.zero_grad(set_to_none=True)
                lo.loss.backward()
                gn = nn.utils.clip_grad_norm_(self.policy.parameters(), cfg.max_grad_norm)
                self.optimizer.step()
                n_steps += 1
                agg["loss/total"].append(float(lo.loss.detach()))
                agg["loss/policy"].append(float(lo.pg_loss.detach()))
                agg["loss/value"].append(float(lo.v_loss.detach()))
                agg["loss/entropy"].append(float(lo.entropy.detach()))
                agg["ppo/approx_kl"].append(float(lo.approx_kl))
                agg["ppo/clip_frac"].append(float(lo.clip_frac))
                agg["grad/norm_preclip"].append(float(gn))
                for h, name in enumerate(self.api.head_names):
                    agg[f"entropy_head/{name}"].append(float(lo.entropy_per_head[h].detach()))
            if early_stop:
                break
        self.scheduler.step()
        self.update_idx += 1
        if early_stop:
            self.counters["kl_early_stops"] += 1
        stats = {k: float(np.mean(v)) for k, v in agg.items()}
        stats.update({
            "ppo/ratio_check_max_dev": ratio_check,
            "ppo/ratio_check_violations_total": self.counters["ratio_check_violations"],
            "ppo/early_stop": float(early_stop),
            "ppo/optimizer_steps": n_steps,
            "ppo/explained_variance": explained_variance(self.buffer.value.reshape(-1),
                                                         self.buffer.returns.reshape(-1)),
            "sched/lr": self.optimizer.param_groups[0]["lr"],
            "sched/ent_coef": ent_coef,
            "time/learner_s": time.perf_counter() - t0,
        })
        return stats

    # ---------------------------------------------------------------- iteration
    def train_iteration(self) -> dict:
        c = self.collect()
        u = self.update()
        stats = {**c, **u}
        total = c["time/collect_s"] + u["time/learner_s"]
        steps = self.cfg.ppo.batch_size
        stats["sps/total"] = steps / max(total, 1e-9)
        stats["sps/learner_only"] = steps / max(u["time/learner_s"], 1e-9)
        stats["time/env_frac"] = c["time/env_s"] / max(total, 1e-9)
        stats["time/learner_frac"] = u["time/learner_s"] / max(total, 1e-9)
        stats["update"] = self.update_idx
        if self.logger is not None and self.update_idx % max(1, self.cfg.run.log_every) == 0:
            self.logger.log(self.global_step, stats)
        return stats

    # ---------------------------------------------------------------- state
    def state_dict(self, include_env_carry: bool = True) -> dict:
        d = {
            "policy_state": self.policy.state_dict(),
            "optimizer_state": self.optimizer.state_dict(),
            "scheduler_state": self.scheduler.state_dict(),
            "rng_states": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "trainer_generator": self.gen.get_state(),
            },
            "return_norm_state": self.return_norm.state_dict(),
            "update": self.update_idx,
            "global_step": self.global_step,
            "counters": dict(self.counters),
            "env_carry": None,
        }
        if include_env_carry and self.obs is not None:
            d["env_carry"] = {
                "obs": {k: np.array(v, copy=True) for k, v in self.obs.items()},
                "episode_start": self.episode_start.copy(),
                "state": self.state.detach().cpu().clone(),
                "env_state": self.env.get_state() if hasattr(self.env, "get_state") else None,
            }
        return d

    def load_state_dict(self, d: dict, *, restore_env: bool = True) -> None:
        self.policy.load_state_dict(d["policy_state"], strict=True)
        self.optimizer.load_state_dict(d["optimizer_state"])
        self.scheduler.load_state_dict(d["scheduler_state"])
        r = d["rng_states"]
        random.setstate(r["python"])
        np.random.set_state(r["numpy"])
        torch.set_rng_state(r["torch"])
        self.gen.set_state(r["trainer_generator"])
        self.return_norm.load_state_dict(d["return_norm_state"])
        self.update_idx = int(d["update"])
        self.global_step = int(d["global_step"])
        self.counters = defaultdict(int, d.get("counters", {}))
        carry = d.get("env_carry")
        if restore_env and carry is not None and carry.get("env_state") is not None and hasattr(self.env, "set_state"):
            self.env.set_state(carry["env_state"])
            self.obs = {k: np.array(v, copy=True) for k, v in carry["obs"].items()}
            self.episode_start = np.array(carry["episode_start"], dtype=bool)
            self.state = carry["state"].to(self.device)
        else:
            # Real vector envs cannot be snapshotted: episodes restart, return accumulators reset.
            self.obs = None


__all__ = ["PPOTrainer", "RatioCheckError", "features_to_torch", "linear_schedule", "seed_everything"]
