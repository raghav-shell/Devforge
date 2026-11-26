"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6. This baseline acts randomly. Replace it with your trained agent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights here (e.g. from WEIGHTS_DIR) onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device
        self.rng = np.random.default_rng(0)

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory buffers and trackers here."""

    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int array of shape action_space.shape with values in {0, 1}."""
        return self.rng.integers(0, 2, size=self.action_space.shape, dtype=np.int8)
