# Observation Schema (v1)

Owner: project-architect. Status: **v1 contract, pending main-session approval**. `OBS_SCHEMA_VERSION = "1.0.0"`.
Implementer: environment-engineer, in `submission/doom_agent/obs/`. Consumers: model-architect, rl-training-engineer, `submission/agent.py`.

Sources:
- `docs/SPATIAL_OBSERVATION_DESIGN.md` §1 (proposal) and `research/SPATIAL_INTELLIGENCE.md` §1–2 (evidence, cited as R§n).
- `research/RL_RESEARCH.md` §3.1 (action feedback).
- Main-session rulings for this pass.

Where this file differs from those documents, this file wins and the difference is marked `DECISION:`.

---

## 0. Raw observation (fixed contract, never changed)

The agent receives the dict from `env/doom_env/spaces.py:73-95`. At eval it comes over the `doom_env.remote` socket with the same key paths, dtypes and shapes (`env/doom_env/remote.py:58-90,152`).

| Raw path | Shape | Dtype (as delivered) | Range / meaning | Source |
| --- | --- | --- | --- | --- |
| `rgb` | (120, 160, 3) | uint8 | 0–255, HUD hidden | `spaces.py:76`, `wrapper.py:189-194` |
| `depth` | (120, 160) | uint8 | 0–255, about 8 u per level, saturates near 2,050 u (R§0.4) | `spaces.py:77` |
| `vitals` | (17,) | float32 | layout `spaces.py:36-55`, `docs/CONTRACT.md` §2.1 | `wrapper.py:184-187` |
| `rays.distance` | (64,) | float32 | [0, 1]; dist / 1024. 1.0 = no hit. Never below about 0.0156 (player radius 16 u, R§2.1) | `sensors.py:267-271` |
| `rays.hit_type` | (64,) | int64 | 0 none, 1 wall, 2 door, 3 locked_door, 4 enemy, 5 item | `sensors.py:25,270` |
| `visible_entities.features` | (32, 5) | float32 | `[d/1024, sin, cos, rel_vf, rel_vr]`, zero-padded | `sensors.py:289-298` |
| `visible_entities.entity_type` | (32,) | int64 | 0 pad … 11 other (`spaces.py:21-34`) | `sensors.py:290` |
| `visible_entities.mask` | (32,) | int8 | 1 = real slot | `sensors.py:291` |

Ray index 0 points straight ahead, and indices go counter-clockwise (16 = left, 32 = behind) (`docs/CONTRACT.md` §2.2, `sensors.py:190`).

**Wrapper facts the preprocessor must respect**
- **The terminal observation is stale.** When an episode ends, `step()` returns the *previous* obs (`env/doom_env/wrapper.py:142`).
- **Pixels are zero without render buffers.** With `render_buffers=False`, `rgb` and `depth` are zeros (`wrapper.py:192-194`).
- **The first steps after reset mishandle turns.** The first step after `reset()` ignores turn input, and the second applies only 1–2 tics (R§9). This matters only for v2 odometry.

---

## 1. Preprocessor API (stateful)

File: `submission/doom_agent/obs/preprocess.py`. **numpy only.** It must not import torch, `training/` or `env/` (OWNERSHIP rule 4), so env workers stay torch-free.

```python
OBS_SCHEMA_VERSION = "1.0.0"

@dataclass(frozen=True)
class ObsConfig:
    time_feature: bool = True          # act_aux[3]; H12 ablation sets False -> slot is 0
    time_scale_steps: float = 1050.0   # scale constant only, NOT the true cap
    enable_depth_band: bool = False    # v2 slot, must stay False in v1
    enable_egomap: bool = False        # v2 slot, must stay False in v1
    enable_icp: bool = False           # v2 slot (blocked flag + odometry), must stay False in v1

class ObsPreprocessor:
    def __init__(self, cfg: ObsConfig = ObsConfig(), *, strict: bool = True): ...
    def reset(self) -> None: ...
    def process(self, obs: dict) -> dict[str, np.ndarray]: ...
    def commit_action(self, action: np.ndarray) -> None: ...
    def state_dict(self) -> dict: ...            # for debugging / determinism tests only
```

**Per-episode lifecycle**

```
pre.reset()
f0 = pre.process(obs0)          # uses prev_action = 0, deltas = 0, t = 0
a0 = policy(f0) ; pre.commit_action(a0)      # a0 = the exact (9,) int8 vector sent to the env
obs1 = env.step(a0) ; f1 = pre.process(obs1)  # prev_action = a0, t = 1
...
```

**State it keeps** (all reset by `reset()`):
- `prev_rays_dist` (64,) f32;
- `prev_health` f32 and `prev_sel_ammo` f32;
- `prev_sel_weapon` int;
- `prev_action` (9,) int8 (the action committed last);
- `steps_since_use_edge` int, initialised to 16 (saturated, meaning "long ago");
- `turn_run` int;
- `t` int (count of `process` calls since reset);
- a `_awaiting_action` flag.

**Ordering rules (asserted, not silently tolerated)**
- `process` → `commit_action` → `process` … must strictly alternate. Calling `process` twice without a `commit_action` in between raises `RuntimeError`. Calling `commit_action` before any `process` also raises.
- `commit_action` must receive the action **after** the action codec: the 9-bit MultiBinary vector actually sent to the env. It must not receive factor indices. This keeps the pipeline independent of the action head (the H2 Bernoulli arm uses it unchanged).
- On the first `process` after `reset()`, all deltas are 0 and `prev_action` is zeros.

**Terminal observation.** On the step that ends an episode, the raw obs is the stale previous obs (`wrapper.py:142`). `process` treats it normally. So `ray_ddist` is 0 and the Δ-vitals are 0, and `prev_action` and `t` advance. This "final features" tensor is used **only** to bootstrap the value on truncation (`docs/MODULE_CONTRACTS.md` §4). The vector env calls `pre.reset()` immediately afterwards.

**Batching.** One `ObsPreprocessor` instance per environment (training) or per agent (eval). Each worker stacks the outputs along a new leading axis. There is no batched variant in v1, because the cost is negligible next to the env step (ESTIMATE: well under 0.1 ms of numpy per step).

**Spec table.** `submission/doom_agent/obs/schema.py` exports the following, and model-architect builds against it:
- `FEATURE_SPECS: dict[str, FeatureSpec(shape, dtype, low, high)]`, matching §2 exactly;
- `dummy_features(batch_shape: tuple[int, ...], rng) -> dict[str, np.ndarray]`, which produces random in-range features for tests.

**Pipeline hash.** `obs_pipeline_hash()` in `schema.py` is the sha256 over the sorted `*.py` files in `submission/doom_agent/obs/`. It is stored in every checkpoint and export (`docs/EVALUATION_PROTOCOL.md` §8.2, §9.4).

---

## 2. v1 feature tensors

Shapes have no batch dimension. Ranges are guaranteed after clipping. Every listed tensor is always present in the dict returned by `process`.

| Name | Shape | Dtype | Range | Normalisation (exact) | Source field(s) |
| --- | --- | --- | --- | --- | --- |
| `ray_dist` | (64,) | float32 | [0, 1] | raw `rays.distance` | `rays.distance` |
| `ray_near` | (64,) | float32 | [1/17, 1] | `1 / (1 + 16 * d)`, where d = `rays.distance` (that is 1/(1 + u/64), u in map units) | `rays.distance` |
| `ray_ddist` | (64,) | float32 | [-1, 1] | `clip(4 * (d_t - d_{t-1}), -1, 1)`; 0 on the first step after reset | `rays.distance`, previous step |
| `ray_hit` | (64,) | int64 | {0..5} | raw categorical; the model embeds it (6 → 4) | `rays.hit_type` |
| `ent_feat` | (32, 7) | float32 | see §2.1 | §2.1; **rows with mask 0 are exactly 0** | `visible_entities.features` |
| `ent_type` | (32,) | int64 | {0..11} | raw categorical, 0 where mask is 0; the model embeds it (12 → 8) | `visible_entities.entity_type` |
| `ent_mask` | (32,) | bool | {0, 1} | `mask.astype(bool)` | `visible_entities.mask` |
| `vitals_feat` | (27,) | float32 | see §2.2 | §2.2 | `vitals`, previous step |
| `prev_action` | (9,) | float32 | {0, 1} | the last committed MultiBinary action; zeros after reset | own action |
| `act_aux` | (6,) | float32 | see §2.3 | §2.3 | own action history, step counter |

Total flat width, excluding categoricals: 64·3 + 32·7 + 27 + 9 + 6 = 458 floats, plus 64 + 32 int64 and 32 bool.

**DECISION: use fixed normalisation constants only. No running mean/std in v1.**
- Rationale:
  - Every constant is grounded in measured ranges (R§2.1–2.3).
  - `obs_pipeline_hash` stays a complete description of the pipeline, and no statistics need to be exported with the weights.
  - The training and eval paths can't drift apart through a stats file.
- Alternative considered: running mean/std with ±5 clipping (`research/RL_RESEARCH.md` §3, "Observation normalisation"). Rejected for v1. Revisit only if per-feature activation statistics on a trained checkpoint show badly scaled inputs.

**DECISION: deltas are differences between consecutive `process` calls (frame stacking of depth 1).** Frame stacking is explicitly allowed (`docs/RULES.md` §1). There is no deeper stack in v1, because the GRU carries history.

### 2.1 `ent_feat` (32, 7)

Notation: d = `features[:,0]` (distance / 1024), m = mask.

| Col | Content | Formula | Range |
| --- | --- | --- | --- |
| 0 | far distance | `clip(d / 4, 0, 1) * m` (entities reach 4,096 u = d 4.0, `sensors.py:22`) | [0, 1] |
| 1 | near distance | `1 / (1 + 8 d) * m` (= 1/(1 + u/128)) | [0, 1] |
| 2 | sin(rel_angle) | `features[:,1] * m` | [-1, 1] |
| 3 | cos(rel_angle) | `features[:,2] * m` | [-1, 1] |
| 4 | rel_vel_forward | `clip(features[:,3] / 10, -3, 3) * m` | [-3, 3] |
| 5 | rel_vel_right | `clip(features[:,4] / 10, -3, 3) * m` | [-3, 3] |
| 6 | in ray range | `(d <= 1) * m` | {0, 1} |

The mask multiplication is mandatory, because `1/(1+8·0) = 1` would otherwise leak a fake "very near" entity into padded slots. Slot order (nearest-first) is passed through, but **the model must be permutation-invariant over slots** (R§5, `docs/SPATIAL_OBSERVATION_DESIGN.md` §1.2).

### 2.2 `vitals_feat` (27)

v = raw `vitals` (`spaces.py:36-54`). Every value is clipped to the stated range.

| Idx | Content | Formula | Range |
| --- | --- | --- | --- |
| 0 | health | `clip(v[0] / 100, 0, 2)` | [0, 2] |
| 1 | armor | `clip(v[1] / 100, 0, 2)` | [0, 2] |
| 2–9 | selected_weapon one-hot | `one_hot(clip(round(v[2]), 0, 7), 8)` (CONTRACT §2.1: slot 0–7; observed 1–3, R§2.3) | {0, 1} |
| 10 | selected_weapon_ammo | `clip(v[3] / 50, 0, 4)` | [0, 4] |
| 11 | bullets | `clip(v[4] / 200, 0, 2)` | [0, 2] |
| 12 | shells | `clip(v[5] / 50, 0, 2)` | [0, 2] |
| 13 | rockets | `clip(v[6] / 50, 0, 2)` | [0, 2] |
| 14 | cells | `clip(v[7] / 300, 0, 2)` | [0, 2] |
| 15–20 | key flags (blue card, yellow card, red card, blue skull, yellow skull, red skull) | raw `v[8:14]` | {0, 1} |
| 21 | any key held | `max(v[8:14])` | {0, 1} |
| 22 | vel_forward | `clip(v[14] / 10, -3, 3)` | [-3, 3] |
| 23 | vel_right | `clip(v[15] / 10, -3, 3)` | [-3, 3] |
| 24 | vel_up | `clip(v[16] / 10, -3, 3)` | [-3, 3] |
| 25 | Δhealth | `clip((v[0] - prev_health) / 20, -1, 1)`; 0 on the first step | [-1, 1] |
| 26 | Δselected ammo | `clip((v[3] - prev_sel_ammo) / 5, -1, 1)`; **0 on the first step and on any step where `selected_weapon` changed** | [-1, 1] |

**DECISION: zero Δselected-ammo on a weapon change.** Without it, a weapon switch would look like a large ammo gain or loss, which is a false "my shot fired" cue. This re-encodes observed data and chooses no action.

### 2.3 `act_aux` (6)

| Idx | Name | v1 status | Formula |
| --- | --- | --- | --- |
| 0 | `steps_since_use_edge` | **on** | `min(k, 16) / 16`. k = steps since the last committed USE rising edge (USE 0→1, or USE = 1 on the first committed action after reset). Starts at 16 after reset. Makes the rising-edge rule learnable (CONTRACT §3, `research/RL_RESEARCH.md` §3.1) |
| 1 | `turn_run` | **on** | `min(r, 3) / 3`. r = number of consecutive committed actions, ending with the last one, that pressed exactly one of TURN_LEFT/TURN_RIGHT, **in either direction**. A direction switch does not reset r. r = 0 if the last action pressed neither or both. This is the same counter as `HeadingTracker.run` (`research/spatial_probes/probe_dr_variants.py:43-45`), so v2 heading integration can reuse it. It exposes the Doom turn ramp (R§4.1) |
| 2 | `blocked_flag` | **RESERVED, always 0.0** | v2: ICP bump cue (`docs/SPATIAL_OBSERVATION_DESIGN.md` §1) |
| 3 | `time_frac` | **on** if `ObsConfig.time_feature`, else 0.0 | `min(t / 1050, 1)`. t = number of `process` calls since reset, minus 1. 1050 is a scale constant; the true tick cap is not observable |
| 4 | `odom_fwd` | **RESERVED, always 0.0** | v2: ICP odometry step, forward |
| 5 | `odom_left` | **RESERVED, always 0.0** | v2: ICP odometry step, left |

**DECISION: keep `act_aux` at width 6, with slots 2, 4 and 5 hard-zeroed in v1.**
- Rationale:
  - Main-session ruling: ICP is deferred to v2.
  - `blocked_flag` and `odom_*` depend on ICP (`docs/SPATIAL_OBSERVATION_DESIGN.md:41-50`).
  - The cheaper speed-based blocked cue is useless (F1 0.17–0.26, R§4.3), so there is no non-ICP substitute worth adding.
  - A fixed width keeps the vitals-branch input at 27 + 9 + 6 = 42 for v1 and v2.
- Alternative considered: dropping the slots. That would change the model's input width at v2 and break checkpoint compatibility.
- Test: in v1, slots 2, 4 and 5 are exactly 0 on every step.

**DECISION: `time_frac` is on in v1. H12 runs it as an ablation (set to 0).**
- Rationale: it costs nothing, and training randomises `episode_ticks` over 2,100–4,200 (`research/RL_RESEARCH.md` §3.3).
- Kill criterion (H12): no gain at 15M decisions → set `time_feature=False` for the export, because a feature that is off is harder to misuse.

---

## 3. Reserved v2 slots (named, disabled in v1)

These keys are **not emitted** in v1. Enabling a flag in v1 raises `NotImplementedError` in `ObsPreprocessor.__init__`. The model raises if the config asks for an encoder of a key that is missing. Names and shapes are frozen now, so v2 work doesn't rename anything.

| Name | Shape | Dtype | Range | Gate | Intended source | Status |
| --- | --- | --- | --- | --- | --- | --- |
| `egomap` | (7, 32, 32) | float32 | [0, 1] | `enable_egomap` | heading-up crop of a 64 u allocentric grid built from rays plus legal pose; channels: visited, free, wall, door, locked_door, enemy_seen, item_seen (`docs/SPATIAL_OBSERVATION_DESIGN.md` §3) | **DEFERRED.** Gated on the oracle-pose ego-map experiment (Proposal B) showing value first |
| `depth_band` | (40,) | float32 | [0, 1] | `enable_depth_band` | `depth` rows 56–63, mean over the 8 rows and over 4-column groups → 40 bins, /255 | **DEFERRED** (v1.5/v2 candidate, R§3). Needs `render_buffers=True` in training |
| `act_aux[2]` `blocked_flag` | scalar slot | float32 | {0, 1} | `enable_icp` | move pressed ∧ V1 step > 2 u ∧ ICP step < 0.5 × V1 step | **DEFERRED** with ICP |
| `act_aux[4:6]` `odom_fwd`, `odom_left` | 2 slots | float32 | [-1, 1] | `enable_icp` | ICP-refined step / 32, clipped | **DEFERRED** with ICP |

**DECISION: v1 excludes `depth`, `egomap` and the ICP flag (main-session ruling). Consequently v1 training may use `render_buffers=False` for speed** (`wrapper.py:189-194`).
- A run that enables `depth_band` must switch back to `render_buffers=True`.
- The env factory must refuse `render_buffers=False` when `enable_depth_band` is set.

**DECISION: the RGB key-colour hint is DEFERRED (main-session ruling).** If it is ever needed, it is a *learned* RGB encoder (for example a horizon strip into a Conv1d) under a new reserved key, `rgb_strip`. It is never a hand-designed hue rule.

---

## 4. Legality audit

Rule: every policy input is a function of the official observation dict plus our own past actions only (`docs/CONTRACT.md` §1, §6; `docs/RULES.md` §1; `docs/OWNERSHIP.md` rule 3). Audit source: R§1.

| Feature | Available at eval? | Derived only from legal data? | Notes |
| --- | --- | --- | --- |
| `ray_dist`, `ray_near`, `ray_hit` | yes (`rays`) | yes | pure re-encoding |
| `ray_ddist` | yes | yes (obs history) | frame difference, allowed (RULES §1) |
| `ent_feat`, `ent_type`, `ent_mask` | yes (`visible_entities`) | yes | FOV and LOS limited by the wrapper |
| `vitals_feat[0:25]` | yes (`vitals`) | yes | key flags are legal obs (`wrapper.py:186`) |
| `vitals_feat[25:27]` | yes | yes (obs history) | frame differences |
| `prev_action` | yes | yes (own output) | |
| `act_aux[0,1]` | yes | yes (own action history) | |
| `act_aux[3]` `time_frac` | yes | yes (own call counter) | does **not** reveal the map's tick cap |
| `act_aux[2,4,5]` (v1 zeros) | yes | yes | constants |
| reserved `egomap` | yes | yes, **if** the pose comes only from vitals, rays and own actions | it is an input image, never read by a planner. It never contains x, y or start-relative heading as numbers |
| reserved `depth_band` | yes (`depth`) | yes | |

**Training-only, never a policy input** (allowed only in `training/rewards/`, `training/eval/`, aux-loss labels and diagnostics):
- `env.privileged_state()` (`wrapper.py:203-227`): x, y, z, angle, kills, items, damage_taken, keys, dead/timeout/exit flags, sector_floor/ceiling, door_lock;
- `env.sector_polygons()`;
- the engine object list, WAD linedef specials and geometry;
- the `info` dict (`tic`, `exit_reached`) and the reward. The agent never receives these at eval (`docs/CONTRACT.md` §6).

**Enforcement (tests the testing-and-debugging-engineer should own):**
- `grep` / import check: no file under `submission/` imports `training`, `env`, `doom_env`, `vizdoom`, or references `privileged`.
- `ObsPreprocessor.process` reads **only** the official keys. Behaviour on unexpected extra keys:
  - With `strict=True` (the default, used in training, tests and the harness) it raises `KeyError`. This makes it impossible to smuggle `info` or privileged data in through the obs path.
  - With `strict=False`, used only by `DoomAgent`, it warns once to stderr and ignores them, so that a scored map is never lost to an exception.
  - Missing official keys always raise.
- Identity test: the same raw obs sequence and committed actions produce bit-identical features in the training worker and in `DoomAgent`, because both import the same `ObsPreprocessor`.

---

## 5. Tests the implementer must ship (`tests/env/`)

1. **Shapes, dtypes and ranges** match `FEATURE_SPECS` on 500 random-policy steps of `maps/two_rooms.wad` and one fixture.
2. **Padded entity rows** are exactly 0 in `ent_feat`, and `ent_type` is 0 there.
3. **Reset semantics.** After `reset()`:
   - `ray_ddist`, Δhealth, Δammo and `prev_action` are 0;
   - `act_aux[0] == 1.0`, `act_aux[1] == 0` and `act_aux[3] == 0`.
4. **USE edge counter.** For the committed sequence USE = 1, 1, 0, 1, k reads 0, 1, 2, 0 on the following `process` calls. Holding USE does not reset k.
5. **`turn_run`** for L, L, L, L, R, none reads 1, 2, 3, 3, 3, 0 → /3 (clipped at 3; the switch to R does not reset). Pressing L and R together gives 0.
6. **Ordering assertions** raise as specified in §1.
7. **The stale terminal obs** gives zero deltas.
8. **v1 reserved slots** are 0, and enabling a v2 flag raises.
9. **Microbenchmark.** `process` p50 and p99 on 1 thread, reported in `docs/ENVIRONMENT.md`.
