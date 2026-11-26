"""Shared helpers for the spatial probes (research only, read-only w.r.t. the repo).

Probes may use privileged_state() and env internals: they are diagnostics, never agent code.
Nothing in this folder may be imported by submission/.
"""
from __future__ import annotations

import glob
import math
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "training" / "experiments"))  # for DeadReckoner (read-only reuse)

import vizdoom  # noqa: E402

VZ = os.path.dirname(vizdoom.__file__)
FIXTURES = sorted(glob.glob(str(ROOT / "tests" / "fixtures" / "*.wad")))

MAPS = {
    "two_rooms": [(str(ROOT / "maps" / "two_rooms.wad"), "MAP01")],
    "fixtures": [(w, "MAP01") for w in FIXTURES],
    "freedoom2": [(os.path.join(VZ, "freedoom2.wad"), f"MAP{i:02d}") for i in (1, 2, 3, 4, 5, 6, 7, 8)],
}


def out_dir() -> Path:
    d = ROOT / "runs" / "spatial-intelligence-researcher"
    d.mkdir(parents=True, exist_ok=True)
    return d


class RandomSticky:
    """Same sticky random policy as training/experiments/dead_reckoning.py (re-sample every 6 steps)."""

    name = "random"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.t = 0
        self.cur = None

    def __call__(self, obs) -> np.ndarray:
        r = self.rng
        if self.t % 6 == 0:
            self.cur = (r.random() < 0.7, r.random() < 0.05, r.random() < 0.15, r.random() < 0.15, r.integers(0, 3))
        self.t += 1
        a = np.zeros(9, np.int8)
        a[:4] = self.cur[:4]
        a[4], a[5] = self.cur[4] == 1, self.cur[4] == 2
        return a


class ForwardBiased:
    """Forward 95 %, sticky turns (mean run 4 steps), rare strafes, USE pulse every ~12 steps, random ATTACK.
    A blind probe policy: it reads no observation (so it encodes no navigation logic)."""

    name = "forward"

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)
        self.turn = 0
        self.prev_use = 0

    def __call__(self, obs) -> np.ndarray:
        r = self.rng
        a = np.zeros(9, np.int8)
        a[0] = r.random() < 0.95
        if r.random() < 0.25:
            self.turn = int(r.choice([0, 0, 1, 2]))
        a[4], a[5] = self.turn == 1, self.turn == 2
        a[2], a[3] = r.random() < 0.05, r.random() < 0.05
        a[6] = r.random() < 0.2
        use = (not self.prev_use) and r.random() < 1 / 12
        a[7] = use
        self.prev_use = int(use)
        return a


POLICIES = {"random": RandomSticky, "forward": ForwardBiased}


def start_frame(p0):
    c0, s0 = math.cos(math.radians(p0["angle"])), math.sin(math.radians(p0["angle"]))

    def to_start(x, y):
        gx, gy = x - p0["x"], y - p0["y"]
        return gx * c0 + gy * s0, -gx * s0 + gy * c0

    return to_start


def dang(a, b):
    return (a - b + 540) % 360 - 180
