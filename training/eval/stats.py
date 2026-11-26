"""Cluster bootstrap CIs and paired effect sizes (docs/EVALUATION_PROTOCOL.md §6). Maps are the unit."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

N_BOOT = 10_000
BOOT_SEED = 0


def cluster_bootstrap(values: dict[str, list[float]], n_boot: int = N_BOOT, seed: int = BOOT_SEED,
                      weights: dict[str, float] | None = None) -> tuple[float, float, float]:
    """values: map_id -> per-episode values. Two-level resample: maps with replacement, then episodes
    within each drawn map. Statistic = weighted mean of per-map means. Returns (point, lo95, hi95)."""
    rng = np.random.default_rng(seed)
    keys = sorted(values)
    arrs = [np.asarray(values[k], float) for k in keys]
    w = np.array([1.0 if weights is None else weights[k] for k in keys])
    point = float(np.sum(w * [a.mean() for a in arrs]) / w.sum())
    if len(keys) < 2:  # maps are the unit: no CI from a single map
        return point, float("nan"), float("nan")
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, len(keys), len(keys))
        means = [arrs[i][rng.integers(0, len(arrs[i]), len(arrs[i]))].mean() for i in idx]
        stats[b] = np.sum(w[idx] * means) / w[idx].sum()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def paired_comparison(a: dict[tuple, float], b: dict[tuple, float], n_boot: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    """a, b: (map_id, seed, mode) -> score. Uses only keys present in both (must be identical sets)."""
    keys = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise ValueError("paired comparison needs identical (map, seed, mode) sets")
    d = np.array([b[k] - a[k] for k in keys])
    by_map = defaultdict(list)
    for k, v in zip(keys, d):
        by_map[k[0]].append(v)
    point, lo, hi = cluster_bootstrap(by_map, n_boot, seed)
    sd = d.std(ddof=1) if len(d) > 1 else float("nan")
    map_means = {m: float(np.mean(v)) for m, v in by_map.items()}
    return {
        "delta_mean": point, "ci95": [lo, hi], "d_z": float(d.mean() / sd) if sd and sd > 0 else float("nan"),
        "map_win_rate": float(np.mean([v > 0 for v in map_means.values()])), "n_pairs": len(d), "n_maps": len(by_map),
    }


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values for an ablation table."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m, out, running = len(items), {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out
