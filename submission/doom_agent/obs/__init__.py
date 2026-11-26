"""Eval-time observation preprocessing (docs/OBSERVATION_SCHEMA.md). numpy only.

Import-light: no torch, and nothing from `training/`, `env/` or `doom_env`. The same code runs in
training workers and in `submission/agent.py`.
"""

from doom_agent.obs.preprocess import ObsConfig, ObsPreprocessor, stack_features
from doom_agent.obs.schema import (
    FEATURE_SPECS,
    OBS_SCHEMA_VERSION,
    RESERVED_FEATURE_SPECS,
    FeatureSpec,
    dummy_features,
    obs_pipeline_hash,
)

__all__ = [
    "FEATURE_SPECS",
    "OBS_SCHEMA_VERSION",
    "RESERVED_FEATURE_SPECS",
    "FeatureSpec",
    "ObsConfig",
    "ObsPreprocessor",
    "dummy_features",
    "obs_pipeline_hash",
    "stack_features",
]
