"""Doom 2 / Freedoom2 thing editor numbers used by the generator.

Class names are the engine's (they are what ViZDoom reports in `state.objects[i].name`);
`tests/env/test_mapgen_load.py` checks them against the running engine.
"""

from __future__ import annotations

PLAYER1_START = 1

# (ednum, class name, radius). Radius matters for placement clearance.
MONSTERS: dict[str, tuple[int, str, int]] = {
    "zombieman": (3004, "Zombieman", 20),
    "shotgunguy": (9, "ShotgunGuy", 20),
    "imp": (3001, "DoomImp", 20),
    "demon": (3002, "Demon", 30),
    "spectre": (58, "Spectre", 30),
    "chaingunguy": (65, "ChaingunGuy", 20),
    "lostsoul": (3006, "LostSoul", 16),
    "cacodemon": (3005, "Cacodemon", 31),
    "hellknight": (69, "HellKnight", 24),
    "revenant": (66, "Revenant", 20),
    "baron": (3003, "BaronOfHell", 24),
    "painelemental": (71, "PainElemental", 31),
}
# Monster rosters by difficulty level (cumulative). Radius <= 31 everywhere, so a cell centre with a
# full 3x3 cell neighbourhood (48 units to the nearest wall) always fits.
ROSTER = (
    ("zombieman", "shotgunguy", "imp"),
    ("demon", "spectre", "chaingunguy", "lostsoul"),
    ("cacodemon", "hellknight", "revenant"),
    ("baron", "painelemental"),
)

HEALTH = {"stimpack": (2011, "Stimpack"), "medikit": (2012, "Medikit"), "healthbonus": (2014, "HealthBonus")}
ARMOR = {"armorbonus": (2015, "ArmorBonus"), "greenarmor": (2018, "GreenArmor"), "bluearmor": (2019, "BlueArmor")}
AMMO = {
    "clip": (2007, "Clip"), "clipbox": (2048, "ClipBox"), "shell": (2008, "Shell"), "shellbox": (2049, "ShellBox"),
    "rocketammo": (2010, "RocketAmmo"), "rocketbox": (2046, "RocketBox"), "cell": (2047, "Cell"),
}
WEAPONS = {
    "shotgun": (2001, "Shotgun"), "supershotgun": (82, "SuperShotgun"), "chaingun": (2002, "Chaingun"),
    "rocketlauncher": (2003, "RocketLauncher"), "chainsaw": (2005, "Chainsaw"), "plasmarifle": (2004, "PlasmaRifle"),
}
# Ammo that feeds each weapon (used to bias ammo drops towards weapons present).
WEAPON_AMMO = {
    "shotgun": ("shell", "shellbox"), "supershotgun": ("shell", "shellbox"), "chaingun": ("clip", "clipbox"),
    "rocketlauncher": ("rocketammo", "rocketbox"), "plasmarifle": ("cell",), "chainsaw": (),
}

# Key colour -> (keycard ednum, card class, skull ednum, skull class, ZDoom lock number "any key of colour").
# Lock numbers 129-131 accept the card or the skull of that colour (ZDoom LOCKDEFS); the template fixture
# tests/fixtures/e7ab54fb6834ffa3.wad uses 13 (RedCard) with lock 129.
KEYS: dict[str, tuple[int, str, int, str, int]] = {
    "red": (13, "RedCard", 38, "RedSkull", 129),
    "blue": (5, "BlueCard", 40, "BlueSkull", 130),
    "yellow": (6, "YellowCard", 39, "YellowSkull", 131),
}
KEY_COLORS = ("red", "blue", "yellow")
