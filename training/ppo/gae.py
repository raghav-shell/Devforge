"""Generalised Advantage Estimation with separate terminated / truncated handling.

docs/MODULE_CONTRACTS.md section 4.2 (from research/RL_RESEARCH.md section 3.3):

    done_t  = terminated_t | truncated_t
    delta_t = r_t + gamma * [(1 - done_t) * V_{t+1} + truncated_t * (1 - terminated_t) * trunc_value_t] - V_t
    A_t     = delta_t + gamma * lambda * (1 - done_t) * A_{t+1}
    V_T     = last_value  (bootstrap V(next_obs, next_state))

- The recursion is cut at BOTH kinds of episode end.
- terminated (death / exit): no bootstrap at all.
- truncated (tick cap, or crash): bootstrap from trunc_value_t = V(final_obs_t, h_t_post). The vector
  env's obs_{t+1} is the NEXT episode's first observation, so V_{t+1} must never be used there.
  For crashes the trainer stores trunc_value = 0.
- If both flags are set, terminated wins (no bootstrap).
"""

from __future__ import annotations

import numpy as np


def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    terminated: np.ndarray,
    truncated: np.ndarray,
    trunc_value: np.ndarray,
    last_value: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """All inputs time-major (T, N) except last_value (N,). Returns (advantages, returns), float32 (T, N).

    returns = advantages + values (the value-function regression target, lambda-return).
    Computed in float64 internally.
    """
    r = np.asarray(rewards, dtype=np.float64)
    v = np.asarray(values, dtype=np.float64)
    term = np.asarray(terminated, dtype=bool)
    trunc = np.asarray(truncated, dtype=bool)
    tv = np.asarray(trunc_value, dtype=np.float64)
    lv = np.asarray(last_value, dtype=np.float64)
    if r.ndim != 2:
        raise ValueError(f"rewards must be (T, N), got {r.shape}")
    T, N = r.shape
    for name, a in (("values", v), ("terminated", term), ("truncated", trunc), ("trunc_value", tv)):
        if a.shape != (T, N):
            raise ValueError(f"{name} must be {(T, N)}, got {a.shape}")
    if lv.shape != (N,):
        raise ValueError(f"last_value must be ({N},), got {lv.shape}")

    done = term | trunc
    boot_trunc = trunc & ~term  # terminated wins
    adv = np.zeros((T, N), dtype=np.float64)
    next_adv = np.zeros(N, dtype=np.float64)
    next_value = lv
    for t in range(T - 1, -1, -1):
        not_done = 1.0 - done[t]
        boot = not_done * next_value + boot_trunc[t] * tv[t]
        delta = r[t] + gamma * boot - v[t]
        next_adv = delta + gamma * gae_lambda * not_done * next_adv
        adv[t] = next_adv
        next_value = v[t]
    ret = adv + v
    return adv.astype(np.float32), ret.astype(np.float32)


__all__ = ["compute_gae"]
