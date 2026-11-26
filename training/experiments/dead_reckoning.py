"""Feasibility check: track the agent's own pose using only legal observations.

Heading  : integrated from our own TURN_LEFT/TURN_RIGHT history (Doom slow-turn ramp).
Position : integrated from vitals vel_forward / vel_right (own frame, units per tic).
Truth    : env.privileged_state() x, y, angle (training only, used here just to measure error).

Run: .venv/bin/python training/experiments/dead_reckoning.py [wad] [map]
"""
import math
import sys

import numpy as np
from doom_env import make_env

# Measured on vizdoom 1.3.3 (see docs/RESEARCH_NOTES.md): degrees per 4-tic step by how many
# consecutive steps a turn key has been held (any direction counts toward the ramp).
TURN_RAMP = (7.031, 12.305, 14.063)
# Least-squares fit: displacement over one step ~= A*v_now + B*v_prev (own frame, units).
W_FWD = (2.33, 1.89)
W_SIDE = (2.09, 1.98)


class DeadReckoner:
    """Pose estimate in the frame of the episode's start pose. Pure observation preprocessing."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.x = self.y = self.h = 0.0  # h in degrees, CCW positive (Doom convention)
        self.turn_run = 0
        self.t = 0
        self.pvf = self.pvr = 0.0

    def update(self, action, vitals):
        left, right = bool(action[4]), bool(action[5])
        tdir = (1 if left else -1) if left != right else 0
        self.turn_run = self.turn_run + 1 if tdir else 0
        th = math.radians(self.h)  # displacement happens mostly before this step's turn lands
        self.t += 1
        if tdir:
            deg = TURN_RAMP[min(self.turn_run, 3) - 1]
            if self.t == 1:  # episode start: the first step's turn input is not applied
                deg = 0.0
            elif self.t == 2 and self.turn_run >= 2:  # ...and the second one only partly
                deg = 3.516
            self.h += tdir * deg
        vf, vr = float(vitals[14]), float(vitals[15])
        f = W_FWD[0] * vf + W_FWD[1] * self.pvf
        s = W_SIDE[0] * vr + W_SIDE[1] * self.pvr
        self.x += f * math.cos(th) + s * math.sin(th)
        self.y += f * math.sin(th) - s * math.cos(th)
        self.pvf, self.pvr = vf, vr


def dang(a, b):
    return (a - b + 540) % 360 - 180


def run(wad, mapname, seeds=4, steps=1000):
    env = make_env(wad, mapname, episode_ticks=4 * steps, render_buffers=False)
    rng = np.random.default_rng(0)
    for seed in range(seeds):
        env.reset(seed=seed)
        p0 = env.privileged_state()
        c0, s0 = math.cos(math.radians(p0["angle"])), math.sin(math.radians(p0["angle"]))
        dr = DeadReckoner()
        path, log = 0.0, []
        for t in range(steps):
            if t % 6 == 0:  # sticky random exploration policy
                cur = (rng.random() < 0.7, rng.random() < 0.05, rng.random() < 0.15,
                       rng.random() < 0.15, rng.integers(0, 3))
            a = np.zeros(9, np.int8)
            a[:4] = cur[:4]
            a[4], a[5] = cur[4] == 1, cur[4] == 2
            pre = env.privileged_state()
            obs, _, te, tr, _ = env.step(a)
            if te or tr:
                break
            ps = env.privileged_state()
            path += math.hypot(ps["x"] - pre["x"], ps["y"] - pre["y"])
            dr.update(a, obs["vitals"])
            # true pose expressed in the start frame
            gx, gy = ps["x"] - p0["x"], ps["y"] - p0["y"]
            tx, ty = gx * c0 + gy * s0, -gx * s0 + gy * c0
            log.append((t + 1, math.hypot(dr.x - tx, dr.y - ty), abs(dang(dr.h, ps["angle"] - p0["angle"])), path))
        L = np.array(log)
        marks = [k for k in (100, 250, 500, 1000) if k <= len(L)]
        print(f"seed {seed}: " + " | ".join(
            f"t={k}: pos err {L[k-1,1]:6.1f}u ({100*L[k-1,1]/max(L[k-1,3],1):4.1f}% of path), hdg err {L[k-1,2]:5.2f}deg"
            for k in marks))


if __name__ == "__main__":
    wad = sys.argv[1] if len(sys.argv) > 1 else "maps/two_rooms.wad"
    mapname = sys.argv[2] if len(sys.argv) > 2 else "MAP01"
    run(wad, mapname)
