"""DoomEnv: the official Gymnasium wrapper around a headless vizdoom.DoomGame.

Observations follow docs/CONTRACT.md and are ego-centric and partially observable.

`privileged_state()` returns ground truth (absolute position, kills, sector heights, ...). It is
what the scorer uses to compute metrics. During evaluation it lives in the scorer's process and is
never sent to the agent. You may use it freely for reward shaping during training.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import vizdoom as vd

from . import spaces as S
from .mapinfo import load_mapinfo
from .sensors import KeyTracker, LevelGeometry, Sensors, ego_rotate

IWAD = os.path.join(os.path.dirname(vd.__file__), "freedoom2.wad")
DEFAULT_EPISODE_TICKS = 2100
SKILL = 3  # "Hurt me plenty"

GV = vd.GameVariable
GAME_VARIABLES = [
    GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO,
    GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6,
    GV.POSITION_X, GV.POSITION_Y, GV.POSITION_Z, GV.ANGLE,
    GV.VELOCITY_X, GV.VELOCITY_Y, GV.VELOCITY_Z,
    GV.KILLCOUNT, GV.DAMAGE_TAKEN, GV.ITEMCOUNT,
]
_V = {v: i for i, v in enumerate(GAME_VARIABLES)}


class DoomEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 35 // S.FRAME_SKIP}

    def __init__(
        self,
        wad: str | os.PathLike,
        map_name: str = "MAP01",
        *,
        episode_ticks: int = DEFAULT_EPISODE_TICKS,
        render_buffers: bool = True,
        render_mode: str | None = None,
    ):
        """
        wad: path to a PWAD with the map (e.g. "maps/two_rooms.wad").
        map_name: map lump inside the WAD.
        episode_ticks: game tics before truncation (35 tics = 1 second of game time).
        render_buffers: False skips filling `rgb`/`depth` (zeros, same spaces) for faster training.
        """
        self.observation_space = S.observation_space()
        self.action_space = S.action_space()
        self.render_mode = render_mode
        self.wad = str(Path(wad).resolve())
        self.map_name = map_name.upper()
        self.episode_ticks = int(episode_ticks)
        self.render_buffers = render_buffers
        self.mapinfo = load_mapinfo(self.wad, self.map_name)

        g = vd.DoomGame()
        g.set_doom_game_path(IWAD)
        g.set_doom_scenario_path(self.wad)
        g.set_doom_map(self.map_name)
        g.set_doom_skill(SKILL)
        g.set_mode(vd.Mode.PLAYER)
        g.set_window_visible(False)
        g.set_sound_enabled(False)
        g.set_console_enabled(False)
        g.set_screen_resolution(vd.ScreenResolution.RES_160X120)
        g.set_screen_format(vd.ScreenFormat.RGB24)
        g.set_depth_buffer_enabled(True)
        g.set_render_hud(False)
        g.set_render_crosshair(False)
        g.set_render_weapon(True)
        g.set_render_decals(False)
        g.set_render_particles(False)
        g.set_render_messages(False)
        g.set_render_corpses(True)
        g.set_objects_info_enabled(True)
        g.set_sectors_info_enabled(True)
        g.set_available_buttons([getattr(vd.Button, b) for b in S.BUTTONS])
        g.set_available_game_variables(GAME_VARIABLES)
        g.set_episode_timeout(self.episode_ticks)
        g.init()
        self._game = g

        self._geometry: LevelGeometry | None = None
        self._sensors: Sensors | None = None
        self._keys = KeyTracker()
        self._last_obs: dict | None = None
        self._last_vars: np.ndarray | None = None
        self._last_sectors = None
        self._exit_reached = False
        self._closed = False

    # ------------------------------------------------------------------ gym API
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if self._closed:
            raise RuntimeError("environment is closed; create a new one with make_env()")
        super().reset(seed=seed)
        g = self._game
        if seed is not None:
            g.set_seed(int(seed) % (2**31))
        record = (options or {}).get("record_path")
        g.new_episode(str(record) if record else "")
        state = g.get_state()
        if self._geometry is None:
            self._geometry = LevelGeometry(state.sectors, self.mapinfo)
            self._sensors = Sensors(self._geometry)
        self._exit_reached = False
        obs = self._observe(state, first=True)
        return obs, self._info()

    def step(self, action):
        if self._closed:
            raise RuntimeError("environment is closed")
        g = self._game
        if self._last_obs is None:
            raise RuntimeError("call reset() before step()")
        if g.is_episode_finished():
            raise RuntimeError("episode finished; call reset()")
        buttons = self._to_buttons(action)
        kills_before = self._last_vars[_V[GV.KILLCOUNT]]
        g.make_action(buttons, S.FRAME_SKIP)

        terminated = truncated = False
        reward = 0.0
        if g.is_episode_finished():
            dead = g.is_player_dead()
            timeout = g.is_episode_timeout_reached()
            self._exit_reached = not dead and not timeout
            terminated = dead or self._exit_reached
            truncated = timeout and not terminated
            reward += 1.0 if self._exit_reached else 0.0
            reward -= 1.0 if dead else 0.0
            obs = self._last_obs  # engine has no state after the episode ends
            # ...but game variables are still readable: keep kills/damage/health from the final tics.
            self._last_vars = np.array([g.get_game_variable(v) for v in GAME_VARIABLES], np.float64)
        else:
            obs = self._observe(g.get_state())
        reward += 0.1 * float(self._last_vars[_V[GV.KILLCOUNT]] - kills_before)
        return obs, reward, terminated, truncated, self._info()

    def render(self):
        if self._last_obs is not None:
            return self._last_obs["rgb"]
        return None

    def close(self):
        if not self._closed:
            self._closed = True
            self._game.close()

    # ------------------------------------------------------------------ helpers
    def _to_buttons(self, action) -> list[int]:
        a = np.asarray(action)
        if a.shape != self.action_space.shape:
            raise ValueError(f"action shape {a.shape} != {self.action_space.shape}")
        if not np.all((a == 0) | (a == 1)):
            raise ValueError("action values must be 0 or 1")
        return [int(x) for x in a]

    def _observe(self, state, first: bool = False) -> dict:
        v = np.asarray(state.game_variables, np.float64)
        self._last_vars = v
        self._last_sectors = state.sectors
        self._geometry.update(state.sectors)

        heading = math.radians(v[_V[GV.ANGLE]])
        pvel = (v[_V[GV.VELOCITY_X]], v[_V[GV.VELOCITY_Y]])
        rays, entities = self._sensors.compute(v[_V[GV.POSITION_X]], v[_V[GV.POSITION_Y]], heading, pvel, state.objects)
        if first:
            self._keys.reset(self._sensors.present_keys)
        else:
            self._keys.update(self._sensors.present_keys)
        fwd, right = ego_rotate(pvel[0], pvel[1], heading)

        vitals = np.zeros(len(S.VITALS_FIELDS), np.float32)
        vitals[0:8] = [v[_V[x]] for x in (GV.HEALTH, GV.ARMOR, GV.SELECTED_WEAPON, GV.SELECTED_WEAPON_AMMO, GV.AMMO2, GV.AMMO3, GV.AMMO5, GV.AMMO6)]
        vitals[8:14] = self._keys.held
        vitals[14:17] = (fwd, right, v[_V[GV.VELOCITY_Z]])

        if self.render_buffers:
            rgb = np.ascontiguousarray(state.screen_buffer, dtype=np.uint8)
            depth = np.ascontiguousarray(state.depth_buffer, dtype=np.uint8)
        else:
            rgb = np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH, 3), np.uint8)
            depth = np.zeros((S.SCREEN_HEIGHT, S.SCREEN_WIDTH), np.uint8)
        obs = {"rgb": rgb, "depth": depth, "vitals": vitals, "rays": rays, "visible_entities": entities}
        self._last_obs = obs
        return obs

    def _info(self) -> dict:
        return {"tic": int(self._game.get_episode_time()), "exit_reached": self._exit_reached}

    # ------------------------------------------------------------------ scorer / training only
    def privileged_state(self) -> dict:
        """Ground truth for scoring and reward shaping. Never part of the agent's observation."""
        g = self._game
        v = self._last_vars
        finished = g.is_episode_finished()
        return {
            "tic": int(g.get_episode_time()),
            "episode_ticks": self.episode_ticks,
            "x": float(v[_V[GV.POSITION_X]]),
            "y": float(v[_V[GV.POSITION_Y]]),
            "z": float(v[_V[GV.POSITION_Z]]),
            "angle": float(v[_V[GV.ANGLE]]),
            "health": float(v[_V[GV.HEALTH]]),
            "kills": int(v[_V[GV.KILLCOUNT]]),
            "items": int(v[_V[GV.ITEMCOUNT]]),
            "damage_taken": float(v[_V[GV.DAMAGE_TAKEN]]),
            "keys": self._keys.held.copy(),
            "dead": bool(finished and g.is_player_dead()),
            "timeout": bool(finished and g.is_episode_timeout_reached()),
            "exit_reached": self._exit_reached,
            "finished": bool(finished),
            "sector_floor": None if self._geometry is None else self._geometry.floor.copy(),
            "sector_ceiling": None if self._geometry is None else self._geometry.ceil.copy(),
            "door_lock": None if self._geometry is None else self._geometry.door_lock.copy(),
        }

    def sector_polygons(self) -> list[tuple[int, np.ndarray]]:
        """(sector id, (K,4) line segments) for every sector; used by the scorer for coverage."""
        if self._last_sectors is None:
            raise RuntimeError("call reset() first")
        if self._geometry is not None and self._geometry.source == "wad":
            return sorted(self.mapinfo.sector_lines().items())
        return [
            (s.id, np.array([[ln.x1, ln.y1, ln.x2, ln.y2] for ln in s.lines], np.float64))
            for s in self._last_sectors
        ]
