"""Policy network, action codec and action distributions (model-architect).

Contract: docs/MODULE_CONTRACTS.md section 3, docs/ACTION_SPACE.md.

This file is deliberately import-free: `doom_agent.models.action_codec` is numpy-only and is imported
by torch-free env workers, so importing this package must not pull in torch. Import the submodules
directly:

    from doom_agent.models.action_codec import factors_to_buttons          # numpy only
    from doom_agent.models.distributions import FactoredCategorical, select_action
    from doom_agent.models.config import PolicyConfig
    from doom_agent.models.policy import DoomPolicy
    from doom_agent.models.io import features_to_torch

Nothing under this package may import `training/`, `env/` or `doom_env` (tested by AST scan).
"""
