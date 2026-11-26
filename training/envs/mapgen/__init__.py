"""Procedural UDMF training-map generator (training-only).

    from training.envs.mapgen import generate
    m = generate("keys", seed=123)          # same (stage, seed, family) -> byte-identical WAD
    m.write("maps/generated/keys")          # keys-a-000123.wad + keys-a-000123.json (metadata sidecar)

CLI: `python -m training.envs.mapgen --stage keys --seed 123 --out maps/generated/keys`.
See docs/ENVIRONMENT.md for stages and knobs. Generator-internal graph search is map construction only and
is never available to the agent (see generator.py / progression.py docstrings).
"""

from .config import FAMILIES, STAGES, StageConfig, stage_config
from .generator import GENERATOR, GENERATOR_VERSION, MAX_SECTORS, GeneratedMap, default_name, generate, source_group

__all__ = ["FAMILIES", "GENERATOR", "GENERATOR_VERSION", "MAX_SECTORS", "STAGES", "GeneratedMap", "StageConfig",
           "default_name", "generate", "source_group", "stage_config"]
