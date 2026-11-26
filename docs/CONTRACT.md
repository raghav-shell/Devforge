# Environment & Agent Contract

**Contract version: `1.0.0-draft`** (frozen before the hackathon starts; check `doom_env.CONTRACT_VERSION`)

This document is the single specification shared by this template and the official scorer.
If your agent passes `validate.py` against this version, it runs identically during evaluation.

---

## 1. Engine isolation

- During evaluation your agent **never** gets a reference to `vizdoom.DoomGame`. The game runs in a
  separate process, and your agent receives only the observation dictionary below.
- Observations are **partially observable** and **ego-centric**. No global map coordinates,
  no god-view.
- During **training** you may use ViZDoom however you like (custom maps, ACS, game variables,
  engine tweaks). We strongly recommend training on `doom_env` observations so your agent sees
  the same inputs at evaluation.

## 2. Observation space

`gymnasium.spaces.Dict` with these keys (definitions in `env/doom_env/spaces.py`):

| Key | Space | Description |
| --- | --- | --- |
| `rgb` | `Box(0, 255, (120, 160, 3), uint8)` | Low-res RGB screen buffer (HUD hidden) |
| `depth` | `Box(0, 255, (120, 160), uint8)` | Depth buffer aligned with `rgb` |
| `vitals` | `Box(-inf, inf, (17,), float32)` | Player state; see §2.1 |
| `rays` | `Dict` | Ego-centric raycasts; see §2.2 |
| `visible_entities` | `Dict` | Entities in field of view and line of sight; see §2.3 |

### 2.1 `vitals` layout (`doom_env.spaces.VITALS_INDEX`)

| Index | Field | Notes |
| --- | --- | --- |
| 0 | `health` | 0–200 |
| 1 | `armor` | 0–200 |
| 2 | `selected_weapon` | Weapon slot 0–7 |
| 3 | `selected_weapon_ammo` | |
| 4–7 | `ammo_bullets`, `ammo_shells`, `ammo_rockets`, `ammo_cells` | |
| 8–13 | `key_blue_card`, `key_yellow_card`, `key_red_card`, `key_blue_skull`, `key_yellow_skull`, `key_red_skull` | 0/1 flags |
| 14–16 | `vel_forward`, `vel_right`, `vel_up` | Velocity in the player's own frame, map units per tic |

Key flags turn to `1` when the key item is picked up and stay `1` for the rest of the episode.

### 2.2 `rays`

`NUM_RAYS = 64` rays spread evenly over 360°, starting straight ahead (index 0) and going
counter-clockwise (index 16 = 90° to the left, index 32 = behind), with max range `RAY_RANGE = 1024` map units.

A ray stops at the first thing that blocks **movement**: a solid wall, an impassable line, a
ledge or step higher than 24 units, an opening lower than the player (56 units), an alive
monster (`enemy`) or a pickup (`item`). A closed door gives `door` (or `locked_door` if it needs
a key). Once the door is open, rays pass through it. Corpses, decorations and projectiles don't
stop rays.

| Sub-key | Space | Description |
| --- | --- | --- |
| `distance` | `Box(0, 1, (64,), float32)` | Hit distance / `RAY_RANGE`; `1.0` = nothing hit |
| `hit_type` | `MultiDiscrete([6] * 64)` | `0` none, `1` wall, `2` door, `3` locked door, `4` enemy, `5` item |

### 2.3 `visible_entities`

Entities that are in the player's 90° horizontal field of view **and** in line of sight
(not hidden behind walls or closed doors), up to 4,096 units away. At most `MAX_ENTITIES = 32`,
sorted nearest first, zero-padded. Corpses, decorations, blood and bullet puffs are never included.

| Sub-key | Space | Description |
| --- | --- | --- |
| `features` | `Box(-inf, inf, (32, 5), float32)` | `[distance / 1024, sin(rel_angle), cos(rel_angle), rel_vel_forward, rel_vel_right]`. `rel_angle` is 0 straight ahead and positive to the left; velocities are the entity's velocity minus yours, in your frame (units/tic) |
| `entity_type` | `MultiDiscrete([NUM_ENTITY_TYPES] * 32)` | Class id; see `doom_env.spaces.ENTITY_TYPES` |
| `mask` | `MultiBinary(32)` | `1` = slot holds a real entity |

## 3. Action space

`gymnasium.spaces.MultiBinary(9)`. Each element presses one button for the step:

| Index | Button |
| --- | --- |
| 0 | `MOVE_FORWARD` |
| 1 | `MOVE_BACKWARD` |
| 2 | `MOVE_LEFT` (strafe) |
| 3 | `MOVE_RIGHT` (strafe) |
| 4 | `TURN_LEFT` |
| 5 | `TURN_RIGHT` |
| 6 | `ATTACK` |
| 7 | `USE` (doors, switches) |
| 8 | `SELECT_NEXT_WEAPON` |

`act()` must return something convertible to `np.ndarray` of shape `(9,)` with values in `{0, 1}`.
Anything else counts as an invalid action, and the episode ends with the score earned so far.

Buttons are held for all 4 tics of a step (standard ViZDoom behaviour). Two vanilla Doom rules matter here:

- **`USE` fires when the button goes down.** Holding it across steps presses it only once. To use
  again, output `USE = 0` for at least one step first.
- **Pressing `USE` on a door that is already moving reverses it.** Spamming `USE` in a doorway keeps
  the door bouncing. Press once, then wait about 1–2 seconds for it to open.

`SELECT_NEXT_WEAPON` also fires on the button-down edge.

## 4. Game settings

- IWAD: `freedoom2.wad` (bundled with ViZDoom). Skill 3 ("Hurt me plenty"). Single player.
- Engine: ViZDoom `1.3.3`, unmodified, vanilla rules.

## 5. Timing

- **Frame skip:** 4 game ticks per `act()` call.
- **Episode cap:** 2,100–4,200 ticks depending on the map (≈525–1,050 decisions).
- Wall-clock limits are listed in [EVALUATION.md](EVALUATION.md#timeouts).

## 6. Agent interface

File `submission/agent.py`, class `DoomAgent`:

```python
class DoomAgent:
    def __init__(self, observation_space, action_space, device: str): ...
    def reset(self) -> None: ...
    def act(self, observation: dict) -> np.ndarray: ...
```

- `__init__` is called **once** per evaluation run. Load weights from paths relative to the
  `submission/` directory (`Path(__file__).parent`). `device` is `"cuda"` or `"cpu"`.
- `reset` is called **before every map**. Clear recurrent state, memory buffers and trackers.
- `act` is called once per decision step and must return an action matching §3.
- The repository is mounted **read-only**. Write scratch files to `/tmp` only.
- There is **no network access** at evaluation time.
- Anything printed to stdout/stderr ends up in your logs, which organizers can read.

## 7. Using the wrapper for training

```python
from doom_env import make_env
env = make_env("maps/two_rooms.wad", "MAP01", episode_ticks=2100)
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(action)
```

- `terminated`: the player reached the exit or died. `truncated`: tick cap reached.
- `reward` is a basic default (+1 exit, −1 death, +0.1 per kill) for convenience. It is **not**
  the evaluation score. Shape your own.
- `render_buffers=False` skips filling `rgb`/`depth` (zeros) when you don't need pixels.
- `env.privileged_state()` returns ground truth (position, kills, damage, …) for reward shaping.
  It is **not available to your agent during evaluation**.
- ViZDoom reports at most **2,048 sectors** per map. On bigger maps, rays and line-of-sight miss walls
  and doors beyond that limit (the wrapper warns you). Evaluation maps stay under the limit.
- `make_env` accepts any ViZDoom-compatible WAD (`make_env("my.wad", "MAP07")`), so you can train
  on whatever maps you find or make. Door and locked-door ray types are detected from the map's linedef
  specials (Doom, Boom, Hexen and UDMF formats).

## 8. Versioning

Any change to this file bumps `CONTRACT_VERSION` in `env/doom_env/__init__.py`. During the event only
backwards-compatible bug fixes (patch versions) will be released, and they will be announced.
