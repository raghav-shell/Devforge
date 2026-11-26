"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6 (interface), docs/MODULE_CONTRACTS.md §9 (wiring), docs/DECISIONS.md.

Pipeline per decision: raw observation -> ObsPreprocessor (numpy) -> DoomPolicy.step (torch, CPU,
1 thread, recurrent GRU state) -> select_action -> MultiBinary(9) int8. Every action comes from the
trained network. There are no hand-written gameplay rules, and nothing here reads privileged game state.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:  # the scorer already does this; keeps direct imports working too
    sys.path.insert(0, str(SUBMISSION_DIR))

from doom_agent.models.config import PolicyConfig  # noqa: E402
from doom_agent.models.distributions import select_action  # noqa: E402
from doom_agent.models.io import features_to_torch  # noqa: E402
from doom_agent.models.policy import DoomPolicy  # noqa: E402
from doom_agent.obs import OBS_SCHEMA_VERSION, ObsConfig, ObsPreprocessor, obs_pipeline_hash  # noqa: E402

ZERO_ACTION = np.zeros(9, dtype=np.int8)
_MODES = ("sample", "greedy")


def _warn(msg: str) -> None:
    print(f"[DoomAgent] WARNING: {msg}", file=sys.stderr, flush=True)


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Loads the exported policy onto the CPU (docs/DECISIONS.md D9: CPU inference)."""
        torch.set_num_threads(1)
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = "cpu"  # the 0.4 ms CPU step makes CUDA pointless; `device` is accepted and ignored
        # Strict mode (tests / harness): re-raise instead of degrading to a no-op action (D12).
        self.strict = os.environ.get("DOOM_AGENT_STRICT", "") == "1"

        meta = json.loads((WEIGHTS_DIR / "policy.json").read_text())
        if meta.get("format_version") != "export-1":
            _warn(f"unexpected export format {meta.get('format_version')!r}")
        if meta.get("obs_schema_version") != OBS_SCHEMA_VERSION:
            _warn(f"obs schema {meta.get('obs_schema_version')!r} != code {OBS_SCHEMA_VERSION!r}")
        if meta.get("obs_pipeline_hash") != obs_pipeline_hash():
            _warn("obs_pipeline_hash differs from the one the weights were trained with")

        self.cfg = PolicyConfig.from_dict(meta["policy_config"])
        self.policy = DoomPolicy(self.cfg).eval()
        state = torch.load(WEIGHTS_DIR / "policy.pt", map_location="cpu", weights_only=True)
        self.policy.load_state_dict(state, strict=True)  # a bad export must fail loudly here
        self.pre = ObsPreprocessor(ObsConfig(**meta["obs_config"]), strict=False)

        mode = os.environ.get("DOOM_AGENT_INFERENCE") or meta.get("default_inference", "sample")
        if mode not in _MODES:
            _warn(f"invalid DOOM_AGENT_INFERENCE {mode!r}; using {meta.get('default_inference', 'sample')!r}")
            mode = meta.get("default_inference", "sample")
        self.mode = mode
        self.temperature = float(meta.get("inference_temperature", 1.0))
        try:
            self.seed = int(os.environ.get("DOOM_AGENT_SEED", "0"))
        except ValueError:
            _warn("invalid DOOM_AGENT_SEED; using 0")
            self.seed = 0
        self._reset_count = 0
        self.errors = 0  # caught act() exceptions (D12)

        self._warm_up()
        self.reset()
        self._reset_count = 0  # warm-up resets don't shift the per-map RNG stream

    def _warm_up(self) -> None:
        """One dummy forward pass so the first real act() pays no lazy-initialisation cost."""
        from doom_agent.obs import dummy_features

        feats = dummy_features(1)
        x = features_to_torch(feats, "cpu")
        with torch.inference_mode():
            self.policy.step(x, self.policy.initial_state(1), torch.tensor([True]))

    def reset(self) -> None:
        """Called before every map: clears the recurrent state and all observation history."""
        self.pre.reset()
        self.state = self.policy.initial_state(1)
        self.start = True
        self._reset_count += 1
        self.rng = np.random.default_rng([self.seed, self._reset_count])

    def act(self, observation: dict) -> np.ndarray:
        try:
            feats = self.pre.process(observation)
            x = features_to_torch({k: v[None] for k, v in feats.items()}, "cpu")
            with torch.inference_mode():
                out, self.state = self.policy.step(x, self.state, torch.tensor([self.start]))
            self.start = False
            a = select_action(out["logits"][0].numpy(), self.mode, self.temperature, self.rng, self.cfg.action_head)
            self.pre.commit_action(a)
            return a
        except Exception as exc:  # D12: one bad step must not freeze the player for the rest of the map
            if self.strict:
                raise
            self.errors += 1
            if self.errors <= 5:
                _warn(f"act() failed ({type(exc).__name__}: {exc}); returning a no-op action")
            if getattr(self.pre, "_awaiting_action", False):  # D12a: keep the preprocessor consistent
                try:
                    self.pre.commit_action(ZERO_ACTION)
                except Exception:
                    pass
            return ZERO_ACTION.copy()
