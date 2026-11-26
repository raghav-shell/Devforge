"""TEST-ONLY reward terms, importable by spawned vector-env workers (reward_cfg {"modules": ["env_test_terms"]})."""

from training.rewards.base import BaseRewardTerm, register_term


@register_term
class FullPrivCheck(BaseRewardTerm):
    """Returns 1.0 per step, and raises unless the term sees the FULL privileged_state (incl. sector arrays)."""

    name = "test_full_priv"

    def value(self, ctx):
        for k in ("sector_floor", "sector_ceiling", "door_lock", "x", "y", "tic"):
            if k not in ctx.priv or k not in ctx.priv_prev:
                raise AssertionError(f"reward term did not get full privileged_state: missing {k}")
        return 1.0


@register_term
class StepIndex(BaseRewardTerm):
    """Value = the decision index in the episode (checks StepContext.step counting and auto-reset)."""

    name = "test_step_index"

    def value(self, ctx):
        return float(ctx.step)
