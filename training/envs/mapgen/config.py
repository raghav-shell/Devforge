"""Curriculum-stage knobs for the map generator.

A stage config is a frozen dataclass; `stage_config(stage, family)` returns the preset. Every knob is a
range or a choice that the seeded RNG samples from, so a (stage, family, seed) triple fully determines a map.

Families are *style* variants of the same stage (texture palette, room shapes, sizes). They exist so the
evaluation split can hold out a whole generator family (docs/EVALUATION_PROTOCOL.md §1.1:
source_group = "udmfgen:<stage>-<family>-v<GENERATOR_VERSION>").
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

STAGES = ("locomotion", "doors", "keys", "combat", "full")
FAMILIES = ("a", "b")

# Textures and flats verified present in freedoom2.wad (tests/env/test_mapgen.py::test_textures_exist).
PALETTES = {
    "a": {
        "walls": ("STARTAN2", "STONE2", "BROWN1", "GRAY1", "TEKWALL4", "METAL", "BRICK7", "CEMENT9", "WOOD1", "BIGBRIK1"),
        "floors": ("FLOOR0_1", "FLOOR4_8", "FLAT5_4", "FLAT14", "FLOOR5_1", "FLAT1", "FLOOR7_1", "FLAT20", "FLAT2"),
        "ceilings": ("CEIL1_1", "CEIL3_5", "CEIL5_1", "FLAT18", "CEIL4_1", "TLITE6_4"),
    },
    "b": {
        "walls": ("ICKWALL3", "SUPPORT3", "PIPE2", "BRONZE1", "MARBLE1", "STUCCO", "ZIMMER2", "SILVER1", "TEKGREN2"),
        "floors": ("GRASS1", "RROCK09", "MFLR8_1", "DEM1_5", "FLAT5_1", "FLOOR1_1", "FLOOR3_3", "CRATOP2"),
        "ceilings": ("CEIL1_1", "FLAT5_1", "CEIL5_1", "FLOOR1_1", "CEIL3_5"),
    },
}
DOOR_TEXTURES = ("BIGDOOR2", "DOOR3")
LOCKED_DOOR_TEXTURES = {"red": "DOORRED2", "blue": "DOORBLU2", "yellow": "DOORYEL2"}
LOCKED_TRACK_TEXTURES = {"red": "DOORRED", "blue": "DOORBLU", "yellow": "DOORYEL"}
DOOR_TRACK = "DOORTRAK"
EXIT_SWITCH = "SW1EXIT"
EXIT_FLOOR = "GATE3"


@dataclass(frozen=True)
class StageConfig:
    stage: str
    family: str = "a"
    # --- layout -------------------------------------------------------------------------------
    n_rooms: tuple[int, int] = (3, 6)
    slot_size: tuple[int, int] = (14, 22)  # cells (1 cell = 32 map units) per macro-grid slot
    room_size: tuple[int, int] = (5, 14)  # room bbox side, cells
    shapes: tuple[tuple[str, float], ...] = (("rect", 3.0), ("l", 1.5), ("pillars", 1.0), ("cross", 1.0), ("octagon", 1.0))
    corridor_width: tuple[int, ...] = (2, 3)  # cells; 2 = 64 units, 3 = 96 units
    straight_prob: float = 0.5  # prefer an aligned (jog-free) corridor when one exists
    loop_prob: float = 0.15  # extra non-tree corridor between adjacent rooms
    arena_scale: float = 1.0  # >1 makes one room an arena (combat)
    # --- doors / keys -----------------------------------------------------------------------------
    door_prob: float = 0.0  # normal (USE) door on a non-locked corridor
    min_path_doors: int = 0  # at least this many normal doors on the start->exit path
    door_close_prob: float = 0.5  # Door_Raise (closes after a delay) instead of Door_Open
    n_locks: tuple[int, int] = (0, 0)
    skull_prob: float = 0.3  # skull key instead of keycard (same lock: "any key of colour")
    # --- things -------------------------------------------------------------------------------
    monsters: tuple[int, int] = (0, 0)
    roster_level: int = 0  # index into things.ROSTER (cumulative)
    ambush_prob: float = 0.2
    health_items: tuple[int, int] = (0, 2)
    armor_items: tuple[int, int] = (0, 1)
    ammo_items: tuple[int, int] = (0, 2)
    weapons: tuple[int, int] = (0, 0)
    # --- look and feel -----------------------------------------------------------------------------
    floor_choices: tuple[int, ...] = (0, 0, 0, 8, 16, 24)  # max pairwise difference 24 = max step-up
    room_height: tuple[int, int] = (128, 256)
    corridor_height: tuple[int, ...] = (96, 112, 128)
    light: tuple[int, int] = (144, 224)
    exit_switch_prob: float = 0.4  # exit switch on a wall instead of a walk-over exit pad
    # --- training metadata ------------------------------------------------------------------------
    episode_ticks: int = 2100  # manifest default
    episode_ticks_range: tuple[int, int] = (2100, 4200)  # suggested randomisation range for training
    tier: int = 2
    extra: dict = field(default_factory=dict)


_BASE = {
    "locomotion": StageConfig(
        stage="locomotion", n_rooms=(2, 5), loop_prob=0.1, health_items=(0, 1), armor_items=(0, 0),
        ammo_items=(0, 0), exit_switch_prob=0.3, episode_ticks=2100, tier=2,
    ),
    "doors": StageConfig(
        stage="doors", n_rooms=(3, 7), door_prob=0.6, min_path_doors=1, episode_ticks=2100, tier=2,
    ),
    "keys": StageConfig(
        stage="keys", n_rooms=(6, 12), loop_prob=0.05, door_prob=0.35, n_locks=(1, 3), episode_ticks=4200,
        episode_ticks_range=(3000, 4200), tier=3,
    ),
    "combat": StageConfig(
        stage="combat", n_rooms=(1, 3), slot_size=(20, 30), room_size=(10, 22), arena_scale=1.4, loop_prob=0.0,
        shapes=(("rect", 3.0), ("pillars", 2.0), ("octagon", 1.5), ("cross", 0.5)),
        monsters=(3, 10), roster_level=2, health_items=(2, 5), armor_items=(0, 2), ammo_items=(3, 7),
        weapons=(1, 2), door_prob=0.0, exit_switch_prob=0.3, episode_ticks=2100, tier=1,
    ),
    "full": StageConfig(
        stage="full", n_rooms=(7, 13), loop_prob=0.1, door_prob=0.4, n_locks=(1, 3), monsters=(4, 14),
        roster_level=3, health_items=(2, 6), armor_items=(1, 3), ammo_items=(3, 8), weapons=(1, 3),
        episode_ticks=4200, episode_ticks_range=(3000, 4200), tier=4,
    ),
}

# Family "b": a held-out style (disjoint wall palette, more irregular shapes incl. rings, wider corridors).
_FAMILY_B = {
    "shapes": (("rect", 1.0), ("l", 2.0), ("pillars", 1.5), ("cross", 1.5), ("octagon", 1.5), ("ring", 1.5)),
    "corridor_width": (2, 3, 3),
    "straight_prob": 0.3,
}


def stage_config(stage: str, family: str = "a", **overrides) -> StageConfig:
    if stage not in _BASE:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
    if family not in FAMILIES:
        raise ValueError(f"unknown family {family!r}; choose from {FAMILIES}")
    cfg = replace(_BASE[stage], family=family)
    if family == "b":
        upd = dict(_FAMILY_B)
        if stage == "combat":
            upd["shapes"] = (("rect", 1.0), ("pillars", 2.0), ("octagon", 2.0), ("ring", 1.5), ("cross", 1.0))
        cfg = replace(cfg, **upd)
    return replace(cfg, **overrides) if overrides else cfg
