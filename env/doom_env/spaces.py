"""Observation and action spaces: the code form of docs/CONTRACT.md.

Keep this file and docs/CONTRACT.md in sync, and bump CONTRACT_VERSION on any change.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

SCREEN_HEIGHT = 120
SCREEN_WIDTH = 160

NUM_RAYS = 64
RAY_RANGE = 1024.0  # map units
RAY_HIT_TYPES = ("none", "wall", "door", "locked_door", "enemy", "item")

MAX_ENTITIES = 32
ENTITY_FEATURES = ("distance", "sin_rel_angle", "cos_rel_angle", "rel_vel_forward", "rel_vel_right")
# Coarse entity classes. Index 0 is reserved for padding.
ENTITY_TYPES = (
    "pad",
    "enemy_melee",
    "enemy_hitscan",
    "enemy_projectile",
    "projectile",
    "health",
    "armor",
    "ammo",
    "weapon",
    "key",
    "barrel",
    "other",
)

VITALS_FIELDS = (
    "health",
    "armor",
    "selected_weapon",
    "selected_weapon_ammo",
    "ammo_bullets",
    "ammo_shells",
    "ammo_rockets",
    "ammo_cells",
    "key_blue_card",
    "key_yellow_card",
    "key_red_card",
    "key_blue_skull",
    "key_yellow_skull",
    "key_red_skull",
    "vel_forward",
    "vel_right",
    "vel_up",
)
VITALS_INDEX = {name: i for i, name in enumerate(VITALS_FIELDS)}

# Order matters: index i of the MultiBinary action presses BUTTONS[i].
BUTTONS = (
    "MOVE_FORWARD",
    "MOVE_BACKWARD",
    "MOVE_LEFT",
    "MOVE_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
    "ATTACK",
    "USE",
    "SELECT_NEXT_WEAPON",
)

FRAME_SKIP = 4


def observation_space() -> spaces.Dict:
    return spaces.Dict(
        {
            "rgb": spaces.Box(0, 255, (SCREEN_HEIGHT, SCREEN_WIDTH, 3), np.uint8),
            "depth": spaces.Box(0, 255, (SCREEN_HEIGHT, SCREEN_WIDTH), np.uint8),
            "vitals": spaces.Box(-np.inf, np.inf, (len(VITALS_FIELDS),), np.float32),
            "rays": spaces.Dict(
                {
                    "distance": spaces.Box(0.0, 1.0, (NUM_RAYS,), np.float32),
                    "hit_type": spaces.MultiDiscrete([len(RAY_HIT_TYPES)] * NUM_RAYS),
                }
            ),
            "visible_entities": spaces.Dict(
                {
                    "features": spaces.Box(
                        -np.inf, np.inf, (MAX_ENTITIES, len(ENTITY_FEATURES)), np.float32
                    ),
                    "entity_type": spaces.MultiDiscrete([len(ENTITY_TYPES)] * MAX_ENTITIES),
                    "mask": spaces.MultiBinary(MAX_ENTITIES),
                }
            ),
        }
    )


def action_space() -> spaces.MultiBinary:
    return spaces.MultiBinary(len(BUTTONS))
