# Spatial Intelligence Research (Phase 1)

Owner: spatial-intelligence-researcher. Design proposal: [`docs/SPATIAL_OBSERVATION_DESIGN.md`](../docs/SPATIAL_OBSERVATION_DESIGN.md).
Probes: `research/spatial_probes/` (diagnostic only; they may read `privileged_state()` and env
internals, so **nothing there may be imported by `submission/`**). Raw JSON outputs:
`runs/spatial-intelligence-researcher/*.json` (gitignored scratch).

Environment: vizdoom 1.3.3, doom_env contract `1.0.0-draft`, numpy 2.5.5, Apple M4. **torch is not
installed in `.venv`.** All latencies are numpy single-thread references (§6).

Probe maps:
* `maps/two_rooms.wad`
* all 13 `tests/fixtures/*.wad` (tiny corridor layouts; 4 of 13 have an exit line, counted with
  `probe_aux_labels.exit_points`)
* Freedoom 2 MAP01–MAP08, bundled with vizdoom. These are real Doom levels with doors, locked doors,
  keys and monsters.

Probe policies (`_common.py`) are blind: neither reads the observation.
* `random`: sticky random actions, re-sampled every 6 steps. Same as
  `training/experiments/dead_reckoning.py`.
* `forward`: 95 % forward, sticky turns, 5 % strafes, a USE pulse about every 12 steps, 20 % ATTACK.

Survivorship caveat: blind policies die often on maps with monsters (freedoom2/forward: 25 of 32
episodes ended in death). So every 1,050-step drift number is also reported from a `-nomonsters`
diagnostic run, which gives full-length episodes, and every table gives n at each time mark.

---

## 0. Key findings (all measured)

1. **Rays are dominated by walls.** 84–96 % of rays hit a wall. Doors are 3–10 %, locked doors 0–5 %,
   enemies 0–2.4 %, items 0–5 %, and "none" (beyond 1,024 u) only 0–0.4 %. The median ray is 34–115 u
   long, and 39–78 % of rays are shorter than 64 u. A linear distance channel wastes resolution where
   it matters, so add a near-field transform.
2. **Entities are sparse.** 68–97 % of steps have *zero* visible entities. The mean is 0.03–1.35 per
   step, p99 is ≤ 16, and the max is 27 (32 slots, so never saturated). On freedoom2, pickups (ammo,
   health, armor, weapon, key, other) make up 69–76 % of entity occurrences and enemies 20–21 %. This favours a cheap masked set
   encoder over a transformer.
3. **Rays sense enemies that `visible_entities` does not.** On 5–19 % of steps an enemy is hit by a ray
   (360°) but no enemy is in `visible_entities` (90° FOV + LOS). Fuse ray hit types with entities.
   Don't treat entities as the only enemy channel.
4. **Depth is not redundant on real maps.** A linear map from rays to 15 × 20 pooled depth, held out
   by episode, gets R² = 0.87–0.92 on two_rooms but only 0.40–0.66 on freedoom2. Depth is linear in
   distance (≈ 8 u per level, saturates at ≈ 2,050 u, r = 0.99 on walls in simple maps). Candidate
   v1.5 input: a cheap eye-level depth band.
5. **Heading from own actions is exact (≤ 0.05° over 1,050 steps) in 316 of 320 episodes, once a
   start-of-episode bug in `DeadReckoner` is fixed** (§4.1).
   * The bug gives a constant 5.27° offset whenever a turn first starts at step 2. It hit 21 of the 81
     forward-policy episodes in the with-monsters run.
   * The 4 remaining episodes are all freedoom2/forward:
     * one teleport (152.6°, MAP05);
     * three unexplained engine-side heading events of 3–15°.
   * So heading needs a legal **jump / rotation guard** (§4.1).
6. **Velocity-only position drift comes mostly from blocked steps.** Residual is about 10 u/step when
   the player pushes into a wall, versus 1.2–2.0 u/step otherwise. Blocked steps are 9–38 % of steps but
   36–83 % of the error.
7. **Ray scan matching (point-to-line ICP) is a legal preprocessing step that fixes this.**
   * Blocked-step residual drops to 0.9–1.7 u/step.
   * 1,050-step drift falls from 236–1,011 u (V0) to 65–249 u (V2) median, i.e. 0.5–2.2 % of path
     instead of 1.7–9.9 %.
   * On full-length freedoom2 episodes, V2 keeps the agent within one 64 u cell of truth on 36–48 % of
     steps (128 u: 86–94 %). Relative drift over 200 steps is 55–75 u median.
   * Cost: 0.46–0.62 ms/step in numpy for odometry plus map write (§4.5).
8. **Every candidate encoder is far under budget.** DeepSets 13.5 k params / 23.5 µs, attention pooling
   17.8 k / 34.4 µs, 2-layer transformer 80 k / 353 µs (numpy, 1 thread, uncontended run). All three pass the
   permutation, garbage-padding and n = 0 tests. Latency is not the deciding factor; sample efficiency is.

---

## 1. Legality audit (every candidate policy input)

"Legal" means it is available to `DoomAgent.act()` at evaluation and computed only from the
observation dict plus our own past actions (CONTRACT §1, §2, §6; RULES §1; OWNERSHIP rule 3).

| Candidate input | Source | Eval-legal? | Notes / use |
| --- | --- | --- | --- |
| `rgb` 120×160×3 | obs | **Legal** | Only legal source of key / locked-door *colour* (textures, sprites). Not used in v1 |
| `depth` 120×160 | obs | **Legal** | Linear ≈ 8 u/level (§3). v1.5 eye-level band, Proposal A |
| `vitals` (17) | obs | **Legal** | Incl. own-frame velocity 14–16 and key flags 8–13 (`env/doom_env/wrapper.py:184-187`) |
| `rays.distance`, `rays.hit_type` | obs | **Legal** | 360°, movement-blocking geometry (`env/doom_env/sensors.py:207-219`) |
| `visible_entities.*` | obs | **Legal** | FOV + LOS (`sensors.py:273-298`). Order is nearest-first, but encoders must be order-invariant |
| Previous action(s) | own output | **Legal** | Needed for USE edge timing (CONTRACT §3) |
| Steps since last USE, turn-run counter | own action history | **Legal** | Pure functions of own actions |
| `act()` call counter `t` | own history | **Legal** | The episode cap itself is **not** given (2,100–4,200 ticks), so "time remaining" is unknown |
| Δvitals, Δrays (frame differences), frame stacks | obs history | **Legal** | "frame stacking" is explicitly allowed (RULES §1) |
| "Blocked" flag (move pressed, ICP step < 0.5 × velocity step) | obs + own action | **Legal** | Re-encoding, not a decision. The speed-based variant doesn't work (§4.6) |
| Dead-reckoned heading (action-integrated) | own actions | **Legal** as internal state | Used to write the ego-map. Don't feed it as a number (start-relative frame, generalisation risk) |
| Dead-reckoned / ICP-matched position (V0/V1/V2) | vitals + rays + own actions | **Legal** as internal state | Same as above. The map is the input, not x, y |
| Ego-centric visit / occupancy grid from rays + V2 pose | derived | **Legal** input | Only if no planner reads it. The CNN reads it, the policy decides (RESEARCH_NOTES §1, correction 4) |
| Episodic novelty (kNN over own embeddings) | derived | **Legal** | NGU-style, an input feature or intrinsic reward |
| Entity "in ray range" flag (d ≤ 1024) | obs | **Legal** | Trivial re-encoding |
| `info` dict (`tic`, `exit_reached`) | env return | **Not delivered at eval** | The agent only gets `observation` (CONTRACT §6) |
| Reward | env return | **Not delivered at eval** | Training only |
| `privileged_state()` (x, y, z, angle, kills, items, damage, keys, sector floor/ceiling, door_lock) | env | **Training-only** label source | Aux targets, reward shaping, diagnostics (`wrapper.py:203-227`) |
| Sector geometry / polygons (`sector_polygons`, `env._geometry`) | env / WAD | **Training-only** label source | Free-space, door-near, exit-distance labels |
| Object list (positions, names, hidden monsters) | ViZDoom state | **Training-only** label source | Hidden-enemy / key-distance labels |
| WAD linedef specials (exit lines, lock numbers) | WAD | **Training-only** label source | Exit-distance label. The lock colour of a door is **not** in the obs |
| Absolute heading / global coordinates | engine | **Illegal** | CONTRACT §1 |
| Hand-computed "turn toward enemy" / "go to frontier" | derived | **Illegal** | RULES §1 (aimbot, planning) |

Two edge cases for main-session review:
* **ICP scan matching** aligns two of our own ray scans to estimate motion. It is observation
  preprocessing and chooses no action. Judges might still read it as "classical robotics", so document
  it as a *sensor-fusion preprocessor* and keep it in `submission/doom_agent/obs/`.
* **RGB colour hints** (e.g. mean hue of pixels in columns where a ray reports `locked_door`) are a
  legal *re-encoding* of the observation. They encode a feature, not a decision, but they are
  hand-designed. Q3 in §10.

---

## 2. What the observation actually contains

Probe: `probe_obs_stats.py --steps 1050 --seeds 4` (two_rooms 4 episodes per policy, fixtures 52,
freedoom2 32).

### 2.1 Rays

| Group / policy | none | wall | door | locked | enemy | item | median dist (u) | p10 / p90 (u) | < 64 u | < 128 u |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_rooms / random | 0.000 | 0.920 | 0.037 | 0.000 | 0.000 | 0.043 | 87 | 17 / 338 | 0.44 | 0.58 |
| two_rooms / forward | 0.000 | 0.931 | 0.030 | 0.000 | 0.007 | 0.033 | 66 | 17 / 348 | 0.50 | 0.61 |
| fixtures / random | 0.000 | 0.893 | 0.041 | 0.043 | 0.022 | 0.001 | 36 | 17 / 112 | 0.78 | 0.93 |
| fixtures / forward | 0.000 | 0.935 | 0.033 | 0.024 | 0.006 | 0.002 | 34 | 17 / 126 | 0.74 | 0.90 |
| freedoom2 / random | 0.004 | 0.841 | 0.099 | 0.011 | 0.013 | 0.033 | 115 | 18 / 402 | 0.39 | 0.53 |
| freedoom2 / forward | 0.002 | 0.866 | 0.047 | 0.012 | 0.024 | 0.049 | 88 | 18 / 408 | 0.45 | 0.58 |

(Fractions over all 64 rays. FOV-only fractions are within ±0.03 and are in the JSON.)

Consequences:
* **Ray distance never goes below about 16 u**, because rays start at the player centre and the player
  radius is 16. **"none" is almost never seen.** The useful dynamic range is 16–400 u, so
  `near = 1/(1 + d/64)` spreads it far better than d/1024.
* **Hit-type classes are very imbalanced.** Use a learned embedding (6 → 4). Any aux loss that predicts
  hit type needs class weights.
* **Caveats from reading the code.**
  * Rays stop only at movement-blocking *lines* and at alive monsters and pickups
    (`sensors.py:185-189, 251-265`). Barrels (`_RAY_CODE_BY_TYPE` maps `barrel` to `RAY_NONE`) and
    solid decorations (TorchTree, Column, ...; category `Vegetation`/`LightSource`, not classified at
    `sensors.py:40-54`) **do not stop rays** (code fact). That they block player movement is Doom
    knowledge and was not measured here.
  * A door counts as "closed" (`door` hit) until its opening is ≥ 56 u (`sensors.py:139`), so a
    half-open door still reads as `door`.
  * Ledges taller than 24 u read as `wall` (`sensors.py:134-135`).

### 2.2 Visible entities

| Group / policy | steps with ≥1 entity | ≥1 enemy entity | mean count | p99 | max | ≥ 2 | ≥ 5 | ≥ 17 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_rooms / random | 0.114 | 0.000 | 0.11 | 1 | 1 | 0 | 0 | 0 |
| two_rooms / forward | 0.057 | 0.008 | 0.06 | 1 | 2 | 0.001 | 0 | 0 |
| fixtures / random | 0.068 | 0.066 | 0.07 | 1 | 3 | 0.003 | 0 | 0 |
| fixtures / forward | 0.030 | 0.025 | 0.03 | 1 | 2 | 0.001 | 0 | 0 |
| freedoom2 / random | 0.247 | 0.099 | 0.95 | 12 | 24 | 0.176 | 0.064 | 0.0026 |
| freedoom2 / forward | 0.320 | 0.153 | 1.35 | 16 | 27 | 0.232 | 0.100 | 0.0066 |

Entity-type mix on freedoom2 (share of entity occurrences; random / forward):
* pickups: ammo .26/.30, health .27/.20, armor .18/.12, weapon .03/.02, key .019/.027, other .005/.015
* enemies: enemy_hitscan .13/.13, enemy_projectile .05/.06, enemy_melee .024/.029
* other: barrel .03/.08, projectile .005/.011

The fixtures are dominated by hitscan enemies (0.71–0.87).

**Enemy sensed by rays but not by entities** (an enemy-type ray hit, no enemy entity) happens on
0.144 (two_rooms/forward), 0.05–0.18 (fixtures) and 0.12–0.19 (freedoom2) of steps. The cause is
that rays are 360° (`sensors.py:190`, `_RAY_ANGLES` over 2π) while entities are FOV + LOS only
(`sensors.py:274`).

Feature ranges (`probe_vitals_range.py`, 12,229 entity rows):
* `distance` 0.013–1.55 (p99 1.27). Entities extend to 4,096 u (`sensors.py:22`), beyond the 1,024 u
  ray range.
* `sin` reaches ±0.99. The FOV test is widened by the entity radius (`sensors.py:274`).
* Relative velocities: p1–p99 about −7.9..+6.0 u/tic, extremes −25.8..+20.0 (projectiles). They are
  *relative to the player*: static pickups show −(own velocity), median −2.76 u/tic forward
  (`sensors.py:293`).

Hence `/10` with a ±3 clip for velocities, and `d/4` plus a near transform for distance.

### 2.3 Vitals ranges (probe policies, 8,244 steps)

| Vital | p1 / median / p99 |
| --- | --- |
| health | 9 / 100 / 108 |
| armor | 0 / 0 / 100 |
| selected_weapon | 1 / 2 / 3 |
| selected_weapon_ammo | 0 / 42 / 65 |
| bullets, shells | 0–70, 0–20 |
| vel_fwd | −2.2 / 1.3 / 7.5 |
| vel_right | −5.9 / 0 / 5.6 |
| vel_up | −4 / 0 / 0 |
| planar speed | max 10.4 u/tic |

Hence /100 for health and armor and /10 for velocity.

---

## 3. Depth: worth its compute?

Probe: `probe_obs_stats.py`.
* **Fit.** For each in-FOV ray (17 rays, ±45°), mean depth over rows 56–63 at the screen column that
  looks along that ray, against the ray's perpendicular distance `d·cos θ`.
* **Redundancy test.** Ridge regression from ray features (64 distances + 64 × 6 one-hot hit types) to
  15 × 20 average-pooled depth. **Held out by episode**: the last 25 % of episodes are the test set, so
  near-duplicate frames never straddle the split.

| Group / policy | units per depth level | saturation (u) | wall fit MAE (levels) / r | depth ≫ ray (>1.25×+16 u) | depth ≪ ray (<0.8×−16 u) | R² depth ← rays | R² rows 0–39 / 40–79 / 80–119 | R² FOV rays ← horizon row |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_rooms / random | 7.39 | 1,885 | 0.69 / 1.000 | 0.019 | 0.000 | 0.92 | 0.92 / 0.94 / 0.82 | 0.93 |
| two_rooms / forward | 7.49 | 1,911 | 0.78 / 0.997 | 0.012 | 0.001 | 0.87 | 0.88 / 0.88 / 0.72 | 0.86 |
| fixtures / random | 7.91 | 2,016 | 0.64 / 0.996 | 0.013 | 0.000 | 0.67 | 0.67 / 0.66 / 0.68 | 0.95 |
| fixtures / forward | 7.93 | 2,023 | 0.71 / 0.992 | 0.007 | 0.000 | 0.62 | 0.64 / 0.61 / 0.60 | 0.95 |
| freedoom2 / random | 8.12 | 2,070 | 2.38 / 0.924 | 0.049 | 0.016 | 0.66 | 0.61 / 0.73 / 0.46 | 0.66 |
| freedoom2 / forward | 8.24 | 2,100 | 3.01 / 0.842 | 0.078 | 0.020 | 0.40 | 0.34 / 0.49 / 0.25 | 0.27 |

Interpretation:
* **Depth is linear in distance**, about 8 u per uint8 level, saturating near 2,050 u (2× the ray
  range). On simple maps the eye-level band reproduces the in-FOV rays (R² 0.86–0.95).
* **On freedoom2, depth carries information rays lack.**
  * Sight past movement blockers: ledges, windows, low walls. Depth is much farther than the ray on
    4.9–7.8 % of FOV rays.
  * Depth is much nearer than the ray on 1.6–2.0 %. *Inferred* cause, not attributed per pixel:
    sprites (decorations, barrels, corpses) and other things rays ignore.
  * Vertical structure: the floor band (rows 80–119) is the least explained by rays, R² 0.25–0.46.
* **This is a *linear* lower bound on redundancy.** A nonlinear model would explain more.
* The fixture R² (0.62–0.67) sits below two_rooms. The *inferred* reason, not tested, is low depth
  variance in tiny corridors. Either way, the fixtures are not representative of depth content.
* **Cost** (§6): the eye-level band is 2.6 k params, 3.8 k MACs, about 7 µs. A full-frame depth CNN is
  215 k params, 3.7 M MACs, about 0.12 ms numpy. Both are affordable. The real cost is sample
  efficiency and the overfitting risk of a pixel encoder.
* **Recommendation.** v1: no depth. v1.5: the eye-level band, the cheapest piece with distinct
  content. Full depth CNN only if Proposal A shows a gain on held-out maps.
* `render_buffers=False` only substitutes zeros (`wrapper.py:189-194`). No simulation time is saved,
  so the cost of depth is entirely on the model side. `probe_obs_stats.py --timing-only`, run alone:
  0.174 vs 0.176 ms/step on two_rooms and 0.99 vs 1.00 ms/step on freedoom2 MAP01 (render on vs off).

---

## 4. Dead reckoning and odometry

Probes: `probe_dead_reckoning.py --steps 1050 --seeds 4` (with monsters, and again with
`--nomonsters`) and `probe_dr_variants.py` (estimator definitions and coefficient refit). Truth is
`privileged_state()`, expressed in the episode-start frame.

### 4.1 Heading: one bug found in `training/experiments/dead_reckoning.py`

* `DeadReckoner.update` (`training/experiments/dead_reckoning.py:41-47`) applies the full 7.031° when
  a turn key is *first* pressed at step 2. The engine applies only 1.758° (one tic).
* Measured on two_rooms, forward policy, seed 2: step 2 `L` → true Δ 1.758°, error 5.273°.
* This leaves a constant 5.27° offset for the rest of the episode.
* First (pre-fix) run: 23 of the 81 forward-policy episodes exceeded 0.5°. 21 of them were the
  step-2 bug (all seed 2, whose RNG presses TURN at step 2); the other 2 are explained below.
* Fix, as used in the probes (`probe_dr_variants.py`, `HeadingTracker`):
  `if t == 2: deg = 3.516 if run >= 2 else 1.758`.
* After the fix, max heading error is ≤ 0.051° at p90 in every group, in both runs.
* **4 of 320 episodes still exceed 0.5°**, all freedoom2/forward. A diagnostic pass printed the step
  where the error jumps:
  * **MAP05 s1 (`-nomonsters`), step 957.** A **teleport**: true displacement 1,156 u, z −120 → 128,
    velocity 0, heading change 152.6°. This is the only jump > 64 u in any episode.
  * **MAP01 s3 (`-nomonsters`), step 507.** A held TURN_LEFT was not applied and the heading moved
    −0.40° instead. Net error 14.5°. Cause not identified: it happens one step after an ATTACK press
    with forward held, health unchanged.
  * **MAP07 s3 (6.97°, step 438) and MAP06 s1 (3.13°, step 763)**, with monsters (health 60 and 52).
    Cause not identified.
* So action-integrated heading is exact except at rare engine events. Teleports have a legal
  signature: velocity → 0, a ray scan that no longer matches (ICP inliers collapse), a large z change
  visible as a ray-pattern jump.
* **Recommendation.** Add a rotation term to the ICP (solve 3-DoF when the scan match residual is
  high) as a legal heading correction, and the jump guard of design §3.2. Neither is implemented or
  measured yet.
* `training/experiments/` is not mine to edit: see the request in the report.

### 4.2 Position drift (full length, medians; n = episodes still alive at that mark)

Estimators (all legal):
* **V0**: `DeadReckoner`, with the heading fix.
* **V1**: per-term frames. `v_now` is measured *after* this step's turn, so it is rotated by the new
  heading, and `v_prev` by the old one. In V0 both are rotated by the old heading.
* **V2**: V1 plus point-to-line ICP of consecutive wall-hit ray scans. Rotation is fixed to the known
  heading change. Translation is initialised at, and regularised toward (λ = 0.05), the V1 step. 4
  iterations, 24 u gate.

`-nomonsters` (no survivorship bias, the primary numbers):

| Group / policy | est | err@100 | err@250 | err@525 | err@1050 (n) | % path @1050 | rel. err over 50 / 200 steps | in 64 u cell ±1 | in 128 u cell ±1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| two_rooms / random | V0 | 77 | 102 | 213 | 236 (4) | 1.7 | 65 / 128 | 0.35 | 0.67 |
| | V1 | 85 | 87 | 213 | 281 (4) | — | 58 / 124 | 0.36 | 0.62 |
| | **V2** | 12 | 24 | 46 | **69** (4) | 0.5 | 9 / 19 | **1.00** | 1.00 |
| two_rooms / forward | V0 | 178 | 402 | 452 | 1,011 (2) | 8.9 | 114 / 275 | 0.07 | 0.13 |
| | **V2** | 14 | 36 | 40 | **86** (2) | 0.7 | 10 / 26 | **1.00** | 1.00 |
| fixtures / random | V0 | 81 | 174 | 224 | 403 (48) | 5.4 | 95 / 189 | 0.14 | 0.50 |
| | **V2** | 11 | 29 | 41 | **65** (48) | 0.8 | 13 / 26 | **1.00** | 1.00 |
| fixtures / forward | V0 | 235 | 351 | 449 | 653 (40) | 9.9 | 134 / 345 | 0.10 | 0.20 |
| | **V2** | 29 | 35 | 40 | **69** (40) | 1.1 | 16 / 40 | **1.00** | 1.00 |
| freedoom2 / random | V0 | 60 | 98 | 209 | 340 (32) | 2.7 | 74 / 160 | 0.29 | 0.55 |
| | V1 | 50 | 111 | 225 | 311 (32) | — | 67 / 149 | 0.34 | 0.58 |
| | **V2** | 30 | 53 | 106 | **173** (32) | 1.4 | 20 / 55 | **0.48** | 0.94 |
| freedoom2 / forward | V0 | 126 | 367 | 527 | 572 (30) | 5.5 | 106 / 270 | 0.09 | 0.17 |
| | V1 | 125 | 352 | 513 | 627 (30) | — | 101 / 277 | 0.10 | 0.15 |
| | **V2** | 40 | 93 | 139 | **249** (30) | 2.2 | 28 / 75 | **0.36** | 0.86 |

With monsters (deaths shorten episodes; n shrinks):
* freedoom2/random: V2 err@1050 is 169 u (n = 18 of 32), cell64 ±1 = 0.76.
* freedoom2/forward: 176 u (n = 7 of 32), 0.97.

The higher cell agreement there is survivorship (shorter episodes), not better odometry.

Tails:
* freedoom2 V2 err@1050 has p10 23–64 u and p90 451–473 u.
* The worst episodes were MAP05/MAP06/MAP07. The worst of all (MAP05 s1 forward, 1,193 u) is the
  teleport episode in §4.1, and its 5.4 u/step free-step residual is the teleport step averaged in.
* Other tail causes are *inferred*, not measured: large open areas with few wall hits, and moving
  floors.

### 4.3 Where the drift comes from

A step is **blocked** when the true step is under 50 % of the V1-predicted step and the predicted step
is over 2 u. This avoids the degenerate "a ray is near" criterion: the ray p25 is 19–30 u, so nearly
every step would count.

| Group / policy (nomonsters) | blocked steps | V0 residual free / blocked (u/step) | share of V0 residual on blocked steps | V2 residual free / blocked |
| --- | --- | --- | --- | --- |
| two_rooms / random | 0.09 | 1.90 / 10.98 | 0.38 | 0.41 / 1.08 |
| fixtures / forward | 0.38 | 1.60 / 9.62 | 0.79 | 0.35 / 0.97 |
| freedoom2 / random | 0.11 | 1.98 / 10.98 | 0.42 | 1.01 / 1.66 |
| freedoom2 / forward | 0.21 | 1.99 / 10.00 | 0.58 | 1.04 / 1.56 |

* When the player pushes into a wall, the reported own-frame velocity keeps a large component the
  wall prevents. The velocity model then over-predicts motion by about 10 u/step.
* The V1 frame fix lowers the *free*-step residual from about 2.0 to about 1.2–1.4 u/step. It cannot
  fix blocked steps.
* ICP removes about 85–90 % of the blocked-step error.
* Refit of the velocity coefficients by least squares (`probe_dr_variants.py --steps 1050 --seeds 2`,
  52,413 steps, all groups and both policies): fwd (2.20, 1.48), side (2.09, 1.03), against the
  current (2.33, 1.89) and (2.09, 1.98). RMSE moves only from 6.26 to 5.83 u/step (p99 residual
  18.5 u), so the coefficients are not the problem; wall contact is.

### 4.4 Implications for an ego-map

* **Within about 200 steps, V2 is accurate to about one 64 u cell** (relative error 19–75 u median).
  Over a full episode on real maps it is not (36–48 % within ±1 cell).
* So the map must be **local and recency-aware**:
  * a heading-up ±1,024 u window;
  * an `age` channel or confidence decay for old writes;
  * no assumption that a cell written 800 steps ago lines up.
* Long-range memory ("locked door is behind-left, far away") should live in the GRU or the episodic
  memory, with the map as a fading aid. This agrees with the plan in RESEARCH_NOTES §3.
* Map consistency is measured directly in §4.5.

### 4.5 Ego-map consistency (estimated pose vs true pose)

Probe: `probe_egomap.py --steps 1050 --seeds 2` (and `--nomonsters`).
* Two grids (64 u cells, 5 channels) are written every step from the *same* rays, one with the V2
  pose and one with the true pose. A V1-pose grid is added for comparison.
* The heading-up 32 × 32 crops are compared at t = 250/525/1,050 by exact-cell IoU, and by
  **F1 with 1-cell tolerance** (a cell matches if the other grid has the class within ±1 cell). The
  CNN's first 3 × 3 conv provides this tolerance.

F1 with 1-cell tolerance (wall / free / visited), median over episodes, `-nomonsters` run:

| Group / policy | t | n | **V2 pose** F1±1 w / f / v | V1 pose F1±1 w / f / v | V2 exact-cell IoU w / f / v |
| --- | --- | --- | --- | --- | --- |
| two_rooms / random | 525 | 2 | **0.99 / 0.99 / 1.00** | 0.78 / 0.76 / 0.81 | 0.44 / 0.65 / 0.75 |
| two_rooms / random | 1050 | 2 | **0.98 / 0.98 / 0.98** | 0.65 / 0.62 / 0.57 | 0.37 / 0.59 / 0.62 |
| fixtures / random | 1050 | 24 | **1.00 / 0.90 / 0.95** | 0.42 / 0.34 / 0.31 | 0.43 / 0.23 / 0.31 |
| fixtures / forward | 1050 | 20 | **1.00 / 0.93 / 0.96** | 0.39 / 0.28 / 0.27 | 0.43 / 0.29 / 0.40 |
| freedoom2 / random | 250 | 16 | **0.98 / 1.00 / 1.00** | 0.98 / 0.98 / 0.99 | 0.44 / 0.74 / 0.51 |
| freedoom2 / random | 525 | 16 | **0.99 / 0.99 / 0.99** | 0.92 / 0.92 / 0.96 | 0.43 / 0.70 / 0.63 |
| freedoom2 / random | 1050 | 16 | **0.94 / 0.97 / 0.93** | 0.85 / 0.86 / 0.71 | 0.41 / 0.63 / 0.47 |
| freedoom2 / forward | 525 | 16 | **0.94 / 0.97 / 0.97** | 0.76 / 0.81 / 0.64 | 0.35 / 0.71 / 0.44 |
| freedoom2 / forward | 1050 | 15 | **0.83 / 0.90 / 0.87** | 0.75 / 0.80 / 0.70 | 0.28 / 0.61 / 0.32 |

(With-monsters run: same picture. freedoom2/random t = 1050 V2 F1±1 is 0.87 / 0.88 / 0.88 with n = 8.)

Scope: these numbers are for the probe's **5-channel hard-write grid** (visited / free / wall / door
/ locked). The proposed 7-channel grid with bilinear splatting and an age channel (design §3.2) has
**not** been measured.

Reading:
* **The local crop stays self-consistent** (F1±1 ≥ 0.83 at 1,050 steps on real maps, ≥ 0.94 up to
  525 steps), even though the *global* pose error reaches 170–250 u. Global error does not matter for
  a heading-up local window. What matters is whether recent and older writes in the window agree.
* **Exact-cell IoU is only 0.3–0.7 even early.** Sub-cell offsets flip thin wall cells between
  neighbours. So the grid should be written with **bilinear splatting**, or read by a CNN with a 3×3
  first layer, and evaluated with tolerance.
* **V2 vs V1:** on corridor maps the velocity-only map falls apart (F1±1 of 0.27–0.42 at 1,050 steps);
  the ICP map does not.
* **Cost** (measured with other probes running, so an upper bound): V2 pose update plus map write is
  0.46–0.62 ms per step, and the heading-up crop is 0.04–0.05 ms.

### 4.6 A legal "blocked" (bump) cue

Probe: `probe_blocked_flag.py --steps 1050 --seeds 2`, run with `-nomonsters`. The privileged label is
the one defined in §4.3.

| Group / policy | label rate | speed < 1 u/tic: P / R / F1 | ICP ratio (V2 step < 0.5 × V1 step): P / R / F1 | front/back rays not shrinking while pushing (d < 64 u): P / R / F1 |
| --- | --- | --- | --- | --- |
| two_rooms / random | 0.09 | 0.20 / 0.25 / 0.22 | 0.99 / 0.97 / **0.98** | 0.23 / 0.67 / 0.35 |
| two_rooms / forward | 0.21 | 0.21 / 0.18 / 0.20 | 1.00 / 0.99 / **0.99** | 0.26 / 0.79 / 0.39 |
| fixtures / random | 0.19 | 0.23 / 0.22 / 0.22 | 1.00 / 0.92 / **0.96** | 0.33 / 0.62 / 0.43 |
| fixtures / forward | 0.39 | 0.33 / 0.21 / 0.26 | 1.00 / 0.97 / **0.98** | 0.43 / 0.81 / 0.56 |
| freedoom2 / random | 0.11 | 0.18 / 0.22 / 0.20 | 0.98 / 0.94 / **0.96** | 0.31 / 0.63 / 0.41 |
| freedoom2 / forward | 0.22 | 0.19 / 0.16 / 0.17 | 0.98 / 0.93 / **0.96** | 0.36 / 0.82 / 0.50 |

* A low speed reading is **not** a bump signal: the reported velocity stays high while pushing into a
  wall (§4.3).
* The ICP-ratio flag agrees with the label almost perfectly. This is partly by construction, because
  both are defined against the same V1 step. The point is that ICP's estimate of the true step is
  accurate.
* The raw ray-difference heuristic is weak (F1 0.35–0.56). In the schema `ray_ddist` is a raw input
  channel, not a hand-thresholded flag.

---

## 5. Entity encoder: comparison and permutation invariance

Probe: `probe_encoders.py`. numpy reference implementations, batch 1, `*_NUM_THREADS=1`.

Per-slot input is 14 dims: 5 raw features + near transform + 8-dim type embedding.

| Encoder | Params | MACs, n = 32 padded | MACs, n = 2 sliced | Latency padded 32 slots (any n), median / p99 | Latency sliced n = 2 / n = 16 | Ops per forward |
| --- | --- | --- | --- | --- | --- | --- |
| **Masked DeepSets** (φ 14-64-64, mean ‖ max ‖ count, ρ 129-64) | **13,536** | 168 k | 18 k | **23.5 / 28.6 µs** | 18.9 / 20.2 µs | 16 |
| Attention pooling (φ 14-64-64, 4-head PMA, 1 query + null token) | 17,824 | 438 k | 39 k | 34.4 / 38.4 µs | 33.3 / 30.5 µs | 24 |
| Tiny transformer (2 × [4-head SA + FFN 128], pre-LN) + attention pooling | 80,096 | 2.67 M | 163 k | 353 / 377 µs | 122 / 189 µs | 74 |

Invariance tests (all three encoders, n ∈ {0, 1, 2, 5, 16, 32}, 5 random permutations each):
* Joint permutation of (features, type, mask) over the real slots, and over all 32 slots so padded
  slots are interleaved with real ones. Max |Δ| ≤ 2e-7 (float32 rounding).
* Padded slots filled with N(0, 100) garbage features and random types: |Δ| = 0.0.
* Slicing to the n real entities gives the same output as the 32-slot padded pass: |Δ| ≤ 2e-7. At
  eval, with batch 1, the encoder can run on the n real slots only.
* n = 0: finite output for all three encoders.
  * DeepSets gets it from a count channel and a max-pool floor of 0 (h ≥ 0 after ReLU).
  * Attention pooling and the transformer get it from a **learned null token** that is always present
    in the softmax.
  * In torch, `masked_fill(-inf)` followed by `softmax` over an all-masked row returns **NaN**. Since
    68–97 % of steps have n = 0, this is the default case, not an edge case.
  * The transformer's self-attention block must also be skipped, or kept finite, at n = 0. The probe
    crashed on it before the guard was added.

Recommendation:
* **v1: masked DeepSets** (mean ‖ max ‖ count), optionally with grouped pooling (enemies /
  projectiles / pickups).
* **Ablation: attention pooling** (PMA, 4 heads, 1 query + null token).
* **The tiny transformer is not justified by the data**: mean ≤ 1.35 entities, ≥ 5 entities on only
  6–10 % of freedoom2 steps.
* Revisit if combat maps (Tier 1/4) show many simultaneous enemies.

References:
* Zaheer et al. 2017, *Deep Sets*, arXiv 1703.06114.
* Lee et al. 2019, *Set Transformer* (PMA pooling), arXiv 1810.00825.

Both are cited from prior knowledge; the PDFs were not re-read this session.

---

## 6. Compute (1 CPU thread)

**torch is not installed in `.venv`, so these are not torch measurements.** The latencies are numpy
single-thread references (`OPENBLAS/OMP/MKL/VECLIB_NUM_THREADS=1`, batch 1, median of 300–2,000
calls, measured with no other probe running).

| Block | Params | MACs | numpy median / p99 | Ops |
| --- | --- | --- | --- | --- |
| Ray encoder (circular Conv1d 6-32 k5 s1, 32-32 k5 s2, 32-64 k3 s2, FC 1024-128) | 143,576 | 455 k | 39.3 / 40.4 µs | 21 |
| Entity: masked DeepSets | 13,536 | 168 k | 23.5 / 28.6 µs | 16 |
| Trunk Linear 384-256 (v2 width, incl. map) | 98,560 | 98 k | 2.6 / 2.8 µs | 3 |
| GRU cell 256-256 | 394,752 | 393 k | 12.8 / 13.4 µs | 12 |
| Heads 256 → 26 (9 logits + V + 16 aux) | 6,682 | 7 k | 1.5 / 1.7 µs | 2 |
| *v1 total: rays + DeepSets + vitals MLP 40-64 (2.6 k) + trunk 256-256 (65.8 k) + GRU + heads (sum of measured blocks)* | *≈ 627 k* | *≈ 1.09 M* | *≈ 82 µs* | *≈ 58* |
| v2 ego-map CNN (7×32×32 → 3 × conv3×3 s2 → 512 → 128); measured with C = 6 | 80,432 | 729 k | 74.3 / 84.8 µs | 18 |
| v1.5 depth eye-level band (rows 56–63 → 40 bins → 64) | 2,624 | 3.8 k | 5.5 / 5.8 µs | 6 |
| Full depth CNN (120×160 → 8×8 s4 16 → 4×4 s2 32 → 3×3 s2 32 → 1,536 → 128) | 215,248 | 3.71 M | 92.5 / 103 µs | 18 |
| Preprocessing: V2 odometry + map write | — | — | 0.46–0.62 ms (contended) | — |
| Preprocessing: heading-up crop 32×32 | — | — | 0.04–0.05 ms | — |
| Env step, two_rooms (render on / off; run alone) | — | — | 0.174 / 0.176 ms | — |
| Env step, freedoom2 MAP01 (render on / off; run alone) | — | — | 0.99–1.00 / 1.00–1.01 ms | — |

Ray-encoder check: circularly rolling the 64 rays by 3 rolls the stride-1 first-layer feature map by exactly 3 (max error 0.0). Circular padding is implemented correctly in the reference.

numpy here links **Apple Accelerate**, which may use the M4 matrix units. The x86 eval CPU will
likely be slower per MAC, but since the whole model is about 1–2 M MACs, that does not change the
conclusion.

The two env-step rows confirm the earlier note that `render_buffers=False` saves nothing (§3).

ESTIMATE:
* torch eager on 1 thread adds roughly 5–10 µs of dispatch per op. The full v2 model is about 90 ops,
  so about 0.5–1 ms of overhead on top of the MACs.
* The full v2 forward pass is about 1.6–2.3 M MACs (without the depth CNN), roughly 0.5 ms of arithmetic.
* Expected total is 1–3 ms per `act()` including preprocessing, against a 40 ms budget.
* To verify once torch 2.15.0 is pinned: `training/benchmarks/model_*.py` (model-architect).

---

## 7. Auxiliary targets (training-only labels)

Probe: `probe_aux_labels.py --steps 1050 --seeds 2`. Targets that are a function of the *current*
legal input are excluded, because predicting them adds no information. That rules out "enemy in view"
(= `ent_mask` + `ent_type`) and "door in FOV" (= ray hit types).

| Target (label source) | two_rooms r / f | fixtures r / f | freedoom2 r / f | Verdict |
| --- | --- | --- | --- | --- |
| hidden enemy ≤ 512 u, not visible (objects + privileged x, y) | 0.51 / 0.73 | 0.15 / 0.04 | 0.95 / 0.83 | Use as **count buckets** {0, 1, 2+}. Binary saturates on real maps |
| door line ≤ 256 u but no door ray (WAD lines + door_lock) | 0.05 / 0.26 | 0.03 / 0.07 | 0.07 / 0.12 | Keep (BCE, pos-weight). Door ≤ 256 u at all: 0.47–0.71 |
| locked door seen earlier, not in rays now (legal history) | 0 / 0 | 0 / 0.002 | 0.14 / 0.18 | Memory probe, self-supervised: tests whether the GRU remembers |
| key ≤ 1,024 u, not visible (objects) | 0 / 0 | 0 / 0 | 0.68 / 0.64 | Keep as distance buckets. Only meaningful on key maps |
| exit distance (WAD exit specials) | median 503 / 432 u | 304 / 255 u (exit on 48 % / 16 % of steps) | 1,719 / 1,235 u | Keep, log-bucketed. Strong long-horizon shaping signal for the trunk |
| novel 64 u cell (privileged x, y) | 0.03 / 0.06 | 0.01 / 0.01 | 0.07 / 0.12 | Keep. It is also the exploration reward, so check it doesn't just duplicate V_int |
| rays at t+4 (future legal obs) | persistence R² 0.06 / 0.19 | −0.48 / −0.20 | −0.22 / −0.01 | Keep. **Not** predictable by copying the present, so it needs action-conditioning (UNREAL-style) |

**Local free space (occupancy completion).** Probe: `probe_free_space.py --steps 1050 --seeds 2`,
every 5th step.
* Label: a heading-up 7 × 7 grid of 64 u cells around the player, excluding the player's own cell. A
  cell is *walkable* if its centre lies inside a sector polygon (`env.sector_polygons()`, even-odd
  test) whose current ceiling − floor is ≥ 56 u. Training-only.
* Legal counterpart: *ray-observed free*, meaning the centre is closer than the ray in its direction.

| Group / policy | samples | walkable | ray-observed free | walkable but not observed | share of walkable hidden |
| --- | --- | --- | --- | --- | --- |
| two_rooms / random | 420 | 0.41 | 0.35 | 0.06 | 0.14 |
| two_rooms / forward | 246 | 0.41 | 0.33 | 0.09 | 0.19 |
| fixtures / random | 2,644 | 0.07 | 0.05 | 0.02 | 0.22 |
| fixtures / forward | 3,347 | 0.09 | 0.07 | 0.01 | 0.12 |
| freedoom2 / random | 2,372 | 0.60 | 0.39 | 0.21 | 0.35 |
| freedoom2 / forward | 1,530 | 0.64 | 0.37 | 0.27 | 0.39 |

* On real maps, 35–39 % of the walkable space around the agent is hidden from the current rays:
  behind corners, past ledges, through windows. Predicting it is a genuine inference task, and the
  label is cheap.
* Caveat: "walkable" means inside a tall-enough sector, not *reachable*. A cell beyond a wall in
  another room counts as walkable.
* Proposed head: 49 × BCE, Proposal C.

References: Lample & Chaplot 2017 (arXiv 1609.05521), game-feature aux heads; Jaderberg et al. 2016,
UNREAL (arXiv 1611.05397); Dosovitskiy & Koltun 2017, DFP (arXiv 1611.01779). All from prior knowledge.

Caveat: the probe counts every ViZDoom object with category `Monster` as an alive enemy. This matches
how `sensors.py:42-47` classifies them, but dead-monster handling was not separately verified.

---

Fusion options (concat / FiLM / cross-attention) are compared, with parameter and MAC estimates, in
design §5. The recommendation is concat for v1, FiLM as an ablation, and cross-attention for v3.

## 8. Benchmarkable architecture proposals

The shared trunk for all three is v1 (design doc §2) with a GRU of 256. Default protocol:
* train on fixtures + freedoom2 train split;
* evaluate on held-out maps from `maps/splits/` (owned by the evaluation researcher);
* 3 seeds;
* fixed env-step budget (e.g. 20 M).

### A. Ray encoder and depth ablation

* **Hypothesis.** Circular Conv1d over rays (shift-equivariant, near-field channel) generalises to
  unseen layouts better than an MLP on flat rays. The eye-level depth band adds a measurable gain on
  freedoom-like maps (R² 0.40–0.66 says depth carries extra content there), and a full depth CNN does
  not add more.
* **Metric.**
  * Held-out exploration: unique 64 u cells per episode.
  * Collision rate: fraction of blocked steps (§4.3 definition).
  * Exit rate.
  * Area under the learning curve.
* **Arms.**
  1. MLP(rays)
  2. CircConv(rays) without the near channel
  3. CircConv + near (v1)
  4. v1 + depth band
  5. v1 + depth CNN
* **Ablation control.** v1 with the depth band input replaced by its per-dataset mean, which keeps the
  parameter count fixed.

### B. Ego-map value and pose quality

* **Hypothesis.** A heading-up ego-map (64 u, ±1,024 u, 7 channels, age-aware) built from V2 odometry
  reduces revisits and increases unique cells on multi-room held-out maps versus GRU-only memory. The
  benefit scales with pose quality.
* **Metric.**
  * Unique cells at 525 / 1,050 steps.
  * Revisit ratio: steps in already-visited cells.
  * "Stuck" rate: no new cell in 100 steps.
  * Tier-2-like exit rate.
* **Arms.**
  1. No map
  2. Map from V1 pose
  3. Map from V2 pose (v2)
  4. Map from true pose. This is a **diagnostic oracle, training-only**, never shipped. It bounds what
     better odometry could give.
  5. V2 map with only the `visited` channel
  6. 128 u cells
* **Ablation control.** V2 map with the crop randomly rotated each step, which destroys ego-alignment
  while keeping channel statistics.

### C. Spatial auxiliary heads

* **Hypothesis.** Aux heads on hidden-state labels speed up learning of navigation and memory:
  * rays at t+4 (action-conditioned);
  * local free space (7 × 7 occupancy completion);
  * hidden-enemy buckets;
  * door-near-hidden;
  * exit-distance buckets;
  * locked-door-seen-earlier.

  This is the Lample & Chaplot result transferred to navigation. Expected gain: ≥ 1.3× fewer env steps
  to reach the baseline's final held-out exploration score.
* **Metric.** Env steps to reach X % of the baseline's final held-out score, final held-out score, and
  aux-head accuracy as a diagnostic.
* **Arms.**
  1. No aux
  2. All aux
  3. All aux minus each head (leave-one-out)
  4. All aux with **labels shuffled across the batch**, a control for the regularisation-only effect
* **Loss weight sweep.** {0.05, 0.1, 0.3}, aux-loss total ≤ 0.3× the PPO loss.

---

## 9. Wrapper facts that matter for design (from code reading)

* **Rays vs line of sight.** Rays use `move_block` lines, while entity LOS uses `sight_block` lines
  (opening ≤ 0) (`sensors.py:134-137, 279`). So an enemy behind a ledge or window can be visible as an
  entity while the ray in that direction reports `wall`.
* **Key flags** are set when a key *object disappears* from the level (`sensors.py:302-320`). Picking
  up a key gives no colour hint before pickup; the colour shows up only in vitals 8–13 afterwards.
* **No pitch.** The view is level, so the horizon is screen row 60. Rows 56–63 are the eye-level band.
* **Depth matches `rgb` geometry.** 90° hfov over 160 columns, so column c looks along
  `atan((80 − c − 0.5)/80)`. Used in `probe_obs_stats.py`.
* **Step 1 after `reset()` ignores turn input**, and step 2 applies 1–2 tics (§4.1).

## 10. Open questions

* **Q1. Teleporters, lifts and conveyors.** The jump detector is untested: 0 jumps on the probe maps.
  The MAP05 outlier suggests moving floors break the velocity model. Probe freedoom2 MAP09–32 next.
* **Q2. ICP in open areas.** Rays beyond 1,024 u give "none" (0.2–0.4 % on freedoom2). In big
  outdoor areas the inlier count falls and V2 reverts to V1. Not yet measured per area type.
* **Q3. Key / locked-door colour from RGB.** The only legal colour source. Should we add a learned
  RGB column-strip encoder (e.g. 8 rows × 160 cols around the horizon → Conv1d), or a hand-designed
  "hue at locked-door columns" feature? Main-session ruling needed on whether the hand-designed
  variant counts as acceptable preprocessing.
* **Q4. ICP heavy tail.** p90 1,050-step drift is about 450 u on freedoom2. Robust weighting or a
  scan-to-local-map match (rather than scan-to-previous-scan) would likely cut it. Not done.
* **Q5. Are blind-policy statistics representative?** Entity counts and novelty rates will shift
  under a trained policy. Rerun `probe_obs_stats.py` with the first trained checkpoint.

## 11. Reproduce

```bash
cd research/spatial_probes
../../.venv/bin/python probe_obs_stats.py --steps 1050 --seeds 4
../../.venv/bin/python probe_obs_stats.py --timing-only        # run alone
../../.venv/bin/python probe_dead_reckoning.py --steps 1050 --seeds 4
../../.venv/bin/python probe_dead_reckoning.py --steps 1050 --seeds 4 --nomonsters
../../.venv/bin/python probe_dr_variants.py --steps 1050 --seeds 2  # coefficient refit
../../.venv/bin/python probe_egomap.py --steps 1050 --seeds 2 [--nomonsters]
../../.venv/bin/python probe_aux_labels.py --steps 1050 --seeds 2
../../.venv/bin/python probe_vitals_range.py
../../.venv/bin/python probe_blocked_flag.py --steps 1050 --seeds 2
../../.venv/bin/python probe_free_space.py --steps 1050 --seeds 2
../../.venv/bin/python probe_encoders.py                       # run alone
../../.venv/bin/python summarize.py                            # DR tables
```
