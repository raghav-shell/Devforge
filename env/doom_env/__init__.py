"""Official DevForge DOOM environment wrapper.

This package is the single source of truth for the observation/action contract.
The private scorer pins it by version. Do not fork it inside your submission.
"""

from .spaces import action_space, observation_space

CONTRACT_VERSION = "1.0.0-draft"

__all__ = ["CONTRACT_VERSION", "make_env", "observation_space", "action_space"]


def make_env(wad: str, map_name: str = "MAP01", **kwargs):
    """Create a DoomEnv for a map, e.g. make_env("maps/two_rooms.wad")."""
    from .wrapper import DoomEnv

    return DoomEnv(wad, map_name, **kwargs)
