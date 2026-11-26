"""Training-only reward terms (docs/MODULE_CONTRACTS.md section 6). Never imported by `submission/`.

Framework: `base.py` (environment-engineer). Terms: `exploration.py` (exploration-engineer),
`progression.py` (main session), `combat.py` (combat-and-survival-engineer). Term modules register
themselves with `@register_term`; `RewardComposer` imports them on demand via `load_term_modules()`.
"""

from training.rewards.base import (
    REWARD_TERMS,
    TERM_MODULES,
    BaseRewardTerm,
    EpisodeRewardStats,
    FirstOccurrence,
    RewardComposer,
    RewardTerm,
    StepContext,
    load_term_modules,
    new_set_bits,
    register_term,
)

__all__ = [
    "REWARD_TERMS",
    "TERM_MODULES",
    "BaseRewardTerm",
    "EpisodeRewardStats",
    "FirstOccurrence",
    "RewardComposer",
    "RewardTerm",
    "StepContext",
    "load_term_modules",
    "new_set_bits",
    "register_term",
]
