"""Factored action codec: 6 categorical heads <-> MultiBinary(9). numpy only (docs/ACTION_SPACE.md 2.1).

Head order, sizes and logit layout are frozen by ACTION_SPACE_VERSION. No torch import here: env
workers and logging use this module without torch.
"""

from __future__ import annotations

import numpy as np

ACTION_SPACE_VERSION = "1.0.0"

NUM_BUTTONS = 9
HEADS: tuple[tuple[str, int], ...] = (
    ("move", 3),
    ("strafe", 3),
    ("turn", 3),
    ("attack", 2),
    ("use", 2),
    ("next_weapon", 2),
)
HEAD_NAMES: tuple[str, ...] = tuple(name for name, _ in HEADS)
HEAD_SIZES: tuple[int, ...] = tuple(n for _, n in HEADS)
NUM_HEADS = len(HEADS)
NUM_LOGITS = sum(HEAD_SIZES)  # 15


def _slices(sizes: tuple[int, ...]) -> tuple[slice, ...]:
    out, start = [], 0
    for n in sizes:
        out.append(slice(start, start + n))
        start += n
    return tuple(out)


LOGIT_SLICES: tuple[slice, ...] = _slices(HEAD_SIZES)
NUM_JOINT_ACTIONS = int(np.prod(HEAD_SIZES))  # 216

# _TABLE[k][choice] -> button index set to 1, or -1 for "none".
_TABLE: tuple[tuple[int, ...], ...] = (
    (-1, 0, 1),  # move: none, forward, backward
    (-1, 2, 3),  # strafe: none, left, right
    (-1, 4, 5),  # turn: none, left, right
    (-1, 6),  # attack
    (-1, 7),  # use
    (-1, 8),  # next_weapon
)
assert tuple(len(r) for r in _TABLE) == HEAD_SIZES

# Button index -> head that drives it (for the Bernoulli arm's bias init and logging).
BUTTON_OF_BINARY_HEAD: dict[str, int] = {
    name: _TABLE[k][1] for k, (name, n) in enumerate(HEADS) if n == 2
}


def factors_to_buttons(idx: np.ndarray) -> np.ndarray:
    """(..., 6) integer factor indices -> (..., 9) int8 MultiBinary. Raises ValueError on bad input."""
    a = np.asarray(idx)
    if a.ndim < 1 or a.shape[-1] != NUM_HEADS:
        raise ValueError(f"expected (..., {NUM_HEADS}) factor indices, got shape {a.shape}")
    if not np.issubdtype(a.dtype, np.integer):
        if not (np.issubdtype(a.dtype, np.floating) and np.all(np.isfinite(a)) and np.all(a == np.round(a))):
            raise ValueError(f"factor indices must be integers, got dtype {a.dtype}")
        a = a.astype(np.int64)
    a = a.astype(np.int64, copy=False)
    out = np.zeros(a.shape[:-1] + (NUM_BUTTONS,), dtype=np.int8)
    for k, row in enumerate(_TABLE):
        col = a[..., k]
        if np.any(col < 0) or np.any(col >= len(row)):
            raise ValueError(f"head {HEAD_NAMES[k]!r}: index out of range [0, {len(row) - 1}]")
        for choice, button in enumerate(row):
            if button >= 0:
                out[..., button] = np.where(col == choice, np.int8(1), out[..., button])
    return out


def buttons_to_factors(buttons: np.ndarray) -> np.ndarray:
    """(..., 9) {0,1} -> (..., 6) int64. Both opposing buttons pressed -> 0 ("none", the engine's net
    effect). For tests and for logging Bernoulli-arm actions. Raises ValueError on non-binary input."""
    b = np.asarray(buttons)
    if b.ndim < 1 or b.shape[-1] != NUM_BUTTONS:
        raise ValueError(f"expected (..., {NUM_BUTTONS}) buttons, got shape {b.shape}")
    if not np.all((b == 0) | (b == 1)):
        raise ValueError("buttons must be 0/1")
    b = b.astype(np.int64)
    out = np.zeros(b.shape[:-1] + (NUM_HEADS,), dtype=np.int64)
    for k, row in enumerate(_TABLE):
        if len(row) == 3:
            p1, p2 = b[..., row[1]], b[..., row[2]]
            out[..., k] = np.where((p1 == 1) & (p2 == 0), 1, np.where((p2 == 1) & (p1 == 0), 2, 0))
        else:
            out[..., k] = b[..., row[1]]
    return out


def all_factor_tuples() -> np.ndarray:
    """(216, 6) int64, every joint action in C order. For tests."""
    grids = np.meshgrid(*[np.arange(n) for n in HEAD_SIZES], indexing="ij")
    return np.stack([g.reshape(-1) for g in grids], axis=-1).astype(np.int64)
