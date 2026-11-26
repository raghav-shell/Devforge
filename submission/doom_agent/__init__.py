"""DOOMSDAY agent package: the eval-time code that `submission/agent.py` loads.

The scorer and `validate.py` put `submission/` on `sys.path`, so this package is imported as the
top-level name `doom_agent` both in training and at evaluation (see docs/MODULE_CONTRACTS.md).

Keep this file import-light: no torch or numpy imports here. Subpackages:
  obs/     observation preprocessing   (environment-engineer)
  models/  policy network + action codec (model-architect)
  memory/  spatial memory, v2           (spatial-memory-engineer)

Nothing under this package may import `training/` or `env/`, or read privileged game state.
"""
