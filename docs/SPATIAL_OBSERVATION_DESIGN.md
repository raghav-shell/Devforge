# Spatial Observation Design (v1 / v2)

Owner: spatial-intelligence-researcher. Status: **proposal**. This document feeds
`docs/OBSERVATION_SCHEMA.md` (owned by project-architect). Nothing here is an agreed interface until
the architect adopts it; field names below are marked `PROPOSED`.

Evidence for every number lives in [`research/SPATIAL_INTELLIGENCE.md`](../research/SPATIAL_INTELLIGENCE.md)
(section numbers given as R§n) and the probe scripts in `research/spatial_probes/`.

## 0. Principles

1. **Inputs = observation dict + own past actions.** `rgb`, `depth`, `vitals`, `rays`,
   `visible_entities`, and the actions our own policy emitted. Nothing else (R§1).
2. **Preprocessing is a pure function of that history.** Module boundary: `submission/doom_agent/obs/`
   produces tensors; `submission/doom_agent/models/` decides. Preprocessing never outputs an action, a
   direction to go, a target angle, or a "best cell". It only re-encodes what was observed.
3. **No planner.** The v2 ego-map is an *input image* to a CNN. No BFS / A* / frontier selection reads it.
4. **No coordinates as inputs.** The dead-reckoned pose is internal state used to *write* the map. The
   network never receives x, y or start-relative heading as numbers (generalisation risk, and it is the
   closest thing to a "map-coordinate shortcut").
5. **Privileged data only as labels.** `privileged_state()`, sectors, objects, WAD geometry: aux-loss
   targets and diagnostics in `training/`, never tensors in `submission/`.

## 1. Schema v1 (recurrent-PPO baseline)

All tensors are per step, batch dimension omitted. `float32` unless stated. "clip" = clamp after scaling.

| PROPOSED name | Shape | Dtype | Built from | Normalisation | Evidence |
| --- | --- | --- | --- | --- | --- |
| `ray_dist` | (64,) | f32 | `rays.distance` | raw (already /1024, in [0.016, 1]) | R§2.1 |
| `ray_near` | (64,) | f32 | `rays.distance` | `1 / (1 + d*1024/64)`: resolves the 16–128 u band where 53–93 % of rays fall | R§2.1 |
| `ray_ddist` | (64,) | f32 | `rays.distance` now − previous step | ×4, clip ±1; zero at reset | Raw motion channel. As a hand-thresholded bump detector it is weak (F1 0.35–0.56 vs the privileged blocked label, R§4.6), so it is given raw and the network combines it |
| `ray_hit` | (64,) | int64 | `rays.hit_type` | categorical 0..5 → embedding(6, 4) in the model | R§2.1 |
| `ent_feat` | (32, 7) | f32 | `visible_entities.features` | `[d/4 (clip 0..1), near=1/(1+8d), sin, cos, vf/10, vr/10 (clip ±3)]` + `in_ray_range = d<=1` | R§2.2 |
| `ent_type` | (32,) | int64 | `visible_entities.entity_type` | categorical 0..11 → embedding(12, 8) | R§2.2 |
| `ent_mask` | (32,) | bool | `visible_entities.mask` | — | R§2.2 |
| `vitals_feat` | (27,) | f32 | `vitals` | see §1.1 | R§2.3 |
| `prev_action` | (9,) | f32 | own last action | {0,1}; zeros at reset | R§1 |
| `act_aux` | (6,) | f32 | own actions + odometry | `[steps_since_USE/16 clip 1, turn_run/3 clip 1, blocked_flag, t/1050 clip 1, odom_fwd/32, odom_left/32 (clip ±1)]` | R§1, R§4.6 |

* `blocked_flag` = 1 if a move button was pressed, the V1 velocity-model step is over 2 u, and the
  ICP-refined (V2) step is under 50 % of it.
  * Against the privileged blocked label, this gives precision 0.98–1.00 and recall 0.92–0.99
    (F1 0.96–0.99, R§4.6). Part of that agreement is by construction, because the label and the flag
    share the V1 step.
  * The obvious alternative, "move pressed and speed < 1 u/tic", is useless: F1 0.17–0.26. The
    reported velocity stays high while pushing into a wall (R§4.3).
  * The flag needs the V2 odometry preprocessor in v1 too (0.5 ms).
* `odom_fwd`, `odom_left` = the V2 step of the last action, in the *current* ego frame. This is our own
  motion estimate, not a coordinate.
`t` = number of `act()` calls since `reset()` (the true tick cap is **not** observable; 1050 is only a
scale constant).

### 1.1 `vitals_feat` (27)

| Slice | Content | Scale |
| --- | --- | --- |
| 0 | health | /100 (clip 0..2) |
| 1 | armor | /100 (clip 0..2) |
| 2–9 | selected_weapon one-hot (slots 0–7) | — |
| 10 | selected_weapon_ammo | /50 (clip 0..4) |
| 11–14 | bullets, shells, rockets, cells | /200, /50, /50, /300 |
| 15–20 | 6 key flags | raw 0/1 |
| 21 | any key held | max of 15–20 |
| 22–24 | vel_fwd, vel_right, vel_up | /10 (measured p1–p99: −2.2..7.5, −5.9..5.6, −4..0 u/tic) |
| 25 | Δhealth since last step | /20, clip ±1 (damage cue) |
| 26 | Δselected ammo since last step | /5, clip ±1 (did my shot fire) |

### 1.2 Not in v1 (and why)

| Input | Decision | Evidence |
| --- | --- | --- |
| `depth` full frame | Excluded from v1; ablation in Proposal A | Linear R² of pooled depth from rays is 0.87–0.92 on two_rooms but only 0.40–0.66 on freedoom2 (held out by episode). So depth adds information on real maps, but at 3.7 M MACs per frame for the CNN. Decide by ablation, not up front (R§3) |
| `depth` eye-level band (rows 56–63 → 40 bins) | **v1.5 candidate** (cheap: 2.6 k params, 3.8 k MACs) | Detects solid decorations and barrels that rays pass through (depth nearer than ray on 1.6–2.0 % of FOV rays in freedoom2), and sight past ledges and windows (depth farther than ray on 4.9–7.8 %) (R§3) |
| `rgb` | Excluded from v1 | No measured need for v1. It is the *only* legal source of key and locked-door colour (R§1, open question Q3) |
| entity order / slot index | Never used as a feature | Permutation invariance is required (R§5) |

## 2. Encoders (v1)

```
ray_*  (64 x 7: dist, near, ddist, emb4(hit))  -> circular Conv1d 7-32 k5 s1 -> 32-32 k5 s2 -> 32-64 k3 s2
                                              -> flatten 1024 -> Linear 128              z_ray (128)
ent_*  (32 x 7 + emb8(type), mask)            -> phi: 15-64-64 ReLU (shared per slot)
# (benchmarked with 6 ray channels / 14 entity inputs; +1 input changes params by <0.1 %)
                                              -> masked mean ‖ masked max ‖ count/32 -> rho 129-64   z_ent (64)
vitals_feat ‖ prev_action ‖ act_aux (42)      -> MLP 42-64                               z_vit (64)
concat (256) -> Linear 256 -> GRU 256 -> heads (9 Bernoulli logits, V, aux)
```

* **Rays: circular padding**, because index 63 neighbours index 0. A stride-1 circular conv is
  exactly shift-equivariant (tested in `probe_encoders.py`, error 0.0, R§6). The flatten afterwards keeps the ego direction.
  This is intended, because "ahead" matters.
* **Entities: masked DeepSets by default.** Attention pooling is the first ablation and the tiny
  transformer the second (R§5 has measured costs). Mean visible entities per step is ≤ 1.35, and
  68–97 % of steps have **zero** entities. Entity–entity interaction is rare, so self-attention is
  unlikely to pay for its 6× parameters and ~10× latency.
  * Mandatory in every variant: a learned **null token** or a count channel, so n = 0 produces a
    finite, learnable "nothing visible" code.
  * Torch-specific trap: a softmax over a row whose keys are all masked to `-inf` gives NaN.
* **Optional grouped pooling** (cheap, still permutation-invariant): pool separately over enemies
  {1, 2, 3}, projectiles {4} and pickups {5..9, 11}, then concatenate. This keeps "an enemy is near"
  from being washed out by ammo clutter (pickups are 69–76 % of entity occurrences on freedoom2, R§2.2).

## 3. Schema v2 additions: ego-centric map (input representation only)

| PROPOSED name | Shape | Dtype | Notes |
| --- | --- | --- | --- |
| `egomap` | (7, 32, 32) | f32 | Heading-up crop of an allocentric grid, 64 u cells, ±1,024 u (= ray range) |

### 3.1 Pose source: "V2" legal odometry (preprocessing)

* **Heading.** Integrated from own TURN actions with the Doom turn ramp, plus the step-2 start quirk
  fix (R§4.1).
  * Error is ≤ 0.05° over 1,050 steps in 316 of 320 probe episodes.
  * The exceptions were one teleport (152.6°) and three unexplained 3–15° engine events, all on
    freedoom2. So the preprocessor also needs the jump guard (§3.2) and, ideally, a 3-DoF ICP heading
    correction (not yet built).
* **Translation.**
  1. Predict with the velocity model, each term rotated in its own frame (V1).
  2. Refine with **point-to-line ICP** of this step's wall-hit ray points against the previous step's.
     Rotation is fixed to the known heading change, and the solve is regularised toward the velocity
     prediction (2×2 linear solve, 4 iterations). Odometry + map write measured at 0.46–0.62 ms/step
     in numpy (R§4.5).
* **Why.** Velocity-only odometry is wrong mainly on steps where the player is blocked by a wall: about
  10 u/step residual versus about 1–2 u/step on free steps. ICP cuts the blocked-step residual about 7×
  (R§4.2).
* **ICP is not planning.** It aligns two observations and outputs a pose. No action depends on it
  except through the learned network.

### 3.2 Grid and update rule

* **Storage.** An allocentric uint8/f16 array of 512 × 512 cells at 64 u (32,768 u span, the Doom map
  limit). Origin is at the episode start pose, offset by half a cell. Size is 1.8 MB for 7 channels. A
  dict of touched cells also works.
* **Per step, after the pose update:**
  1. `visited[cell(p)] = min(visited + 1, 255)`. Network input is `log1p(visited)/log(256)`.
  2. For each of the 64 rays, walk the cells from p to the hit point with a vectorised DDA (at most
     16 cells at 64 u). Interior cells get `free = 1`. The end cell gets the hit-type channel. When a
     ray now passes through a cell that was a door, the door is cleared: rays pass open doors.
  3. `enemy_seen` and `item_seen` are written at the entity position (from ent_feat distance and
     angle) with value 1. These decay by ×0.97 per step (half-life about 23 steps), because entities move.
  4. `age` channel = steps since the cell was last written, `/256` clipped to 1. The network uses it
     to discount stale, drifted cells.
* **Channels (7):** visited, free, wall, door, locked_door, enemy_seen, item_seen. (`age` is folded
  in by multiplying free/wall by `exp(-age/512)`, or kept as an 8th channel.)
* **Read-out.** Rotate a 32 × 32 window about p so heading points up (nearest-neighbour sampling of
  1,024 points). Measured at 0.04–0.05 ms.
* **CNN.** 3 × conv3×3 s2 (7 → 16 → 32 → 32) → 512 → 128: 80 k params, 0.73 M MACs, 74 µs numpy
  (measured with 6 input channels).
* **Write with bilinear splatting**, spreading each write over the 4 nearest cells. This is not yet
  measured: the probe used hard writes (R§4.5).
* **Teleport / lift guard.** If the ICP inlier count is below 6 for 3 consecutive steps, or the
  velocity-predicted step and the ICP step differ by more than 64 u, shift the grid to a fresh origin
  and keep the old one read-only.
  * Measured need: 1 teleport in 320 probe episodes (freedoom2 MAP05). Displacement was 1,156 u, the
    velocity read 0, and heading changed 152.6° (R§4.1).
  * The guard itself is not yet implemented or tested.

### 3.3 Why 64 u cells and a ±1,024 u window

* **Global pose drifts; local pose doesn't (much).** Full-length `-nomonsters` runs:
  * V2 keeps the agent within ±1 cell of truth on 100 % of steps (two_rooms, fixtures), but only on
    36–48 % (64 u) and 86–94 % (128 u) on freedoom2.
  * Relative error over 200 steps is 19–75 u median, about one 64 u cell (R§4.2).
* **What matters for a local heading-up window is self-consistency, and that holds** for the probe's
  5-channel hard-write grid. Comparing V2-pose and true-pose crops with 1-cell tolerance gives F1
  0.94–1.00 up to 525 steps and 0.83–0.97 at 1,050 steps on freedoom2 (R§4.5). Exact-cell IoU is only 0.3–0.7, so **write with bilinear splatting**
  (spread each write over the 4 nearest cells) and rely on the CNN's 3×3 first layer for tolerance.
* 32 u cells would not be supported by the odometry. 128 u cells blur doorways, which are 64–128 u
  wide in Doom.
* ±1,024 u matches the ray range, so nothing in the window is out of sensor reach when it is written.

### 3.4 Legal alternatives (benchmarked against the grid, not instead of it)

* **Episodic novelty scalar.** kNN distance of the current GRU/ray embedding to an episodic buffer
  (NGU-style), fed as one input feature: 1,050 × 64 floats and 67 k MACs per step. No pose needed, so
  it is robust to drift. Weaker at "which direction is unexplored".
* **Learned place embedding.** An aux head regresses V2-odometry displacement between two steps
  (self-supervised). It trains the trunk to encode place without a hand-built grid.
* **Topological memory.** Deferred: it needs place recognition (RESEARCH_NOTES §1, correction 3).

## 4. Auxiliary targets (training-only labels, heads ignored at eval)

Only targets that are **not** a function of the current input are listed. "Enemy in view" (= mask +
type) and "door in FOV" (= ray hit types) are excluded because they add nothing.

| Target | Label source (training only) | Base rate on probe maps | Head / loss |
| --- | --- | --- | --- |
| `aux_rays_future` rays_{t+4} distance (64) | future *legal* observation | persistence R² ≈ −0.5..0.19 (not trivially predictable) | MSE on `ray_near`, conditioned on actions t..t+3 |
| `aux_hidden_enemy` count of alive monsters within 512 u not visible, bucketed {0, 1, 2+} | objects + privileged x, y | ≥1 on 4–15 % (fixtures), 83–95 % (freedoom2) | 3-way CE. Binary ≥1 is too imbalanced on real maps |
| `aux_door_near_hidden` door line within 256 u, no door ray | WAD lines + door_lock | 3–26 % | BCE |
| `aux_key_dist` nearest uncollected key, 4 log buckets + "none" | objects | key within 1,024 u but not visible on 64–68 % of freedoom2 steps | 5-way CE |
| `aux_exit_dist` Euclidean distance to the nearest exit line, log buckets | WAD specials 11/51/52/124/197/198, 243/244 | median 300–1,700 u | 6-way CE |
| `aux_novel` true 64 u cell first visited | privileged x, y | 1–12 % with probe policies | BCE (pos-weighted) |
| `aux_free_space` 7×7 ego grid (64 u) of walkable cells: centre inside a sector with ceiling − floor ≥ 56 | `sector_polygons()` + privileged heights | walkable 7–64 % of cells. **Not visible to current rays: 12–22 % of walkable cells (two_rooms, fixtures), 35–39 % (freedoom2)** (R§7) | 49 × BCE, occupancy completion (Active-Neural-SLAM-style) |
| `aux_odom` V2 displacement t−k → t, k = 8 | legal odometry | — | MSE, self-supervised |

## 5. Fusion (rays, entities, vitals, memory)

Sized for 1 thread, batch 1. MACs are analytic; numbers for blocks that were benchmarked come from R§6.

| Option | Extra params | Extra MACs / step | Pros | Cons | Verdict |
| --- | --- | --- | --- | --- | --- |
| **Concat → Linear 256 → GRU** | 65.8 k (trunk 256-256) | 66 k | Simplest; measured 2.6 µs for the 384-wide variant | Vitals can only gate rays after the GRU | **v1 default** |
| **FiLM**: z_vit → (γ, β) for the ray-conv channels (64) and z_ent (64) | 64 → 256: 16.6 k | 16 k + 2 k elementwise | Lets state (health, ammo, key held) re-weight spatial features *before* memory, e.g. "seek health when low". About 3 extra ops | Small expected gain on navigation | Cheap ablation in Proposal A |
| **Cross-attention**: query = h_{t−1} (GRU, 256 → 64); keys/values = 16 ray tokens (pre-flatten conv, 64-d) + n entity tokens (64-d) + 16 map tokens (v2 CNN 4×4×32 → 64), 4 heads | Q 16.4 k + K, V 2 × 4.2 k + out 4.2 k ≈ 29 k | ≈ (32 + n) × 2 × 64 × 64 + 16 k ≈ 0.28 M | One mechanism attends over all spatial evidence. Memory-driven focus ("where was the door") | Needs the n = 0 null-token guard. About 25 more ops (~0.2 ms in torch eager, estimate). Harder to train early | **v3** candidate, after v2 is stable |

All three fit easily in 40 ms. The choice is about sample efficiency and stability, and that can only
be settled by the ablations below.

## 6. Benchmarkable proposals (details in R§8)

* **A. Ray-encoder and depth ablation.** CircConv vs MLP rays; +depth band; +depth CNN.
* **B. Ego-map value and pose quality.** No map / map from V1 pose / map from V2 pose / map from true
  pose (diagnostic oracle, never shipped).
* **C. Aux spatial heads.** Each head ablated, plus a label-shuffled control.

## 7. Compute budget (numpy 1-thread reference; torch not installed, see R§6)

Measured in numpy, 1 thread, Apple Accelerate (R§6):

| Configuration | Params | MACs | Latency |
| --- | --- | --- | --- |
| v1 | ≈ 627 k | ≈ 1.09 M | ≈ 82 µs (sum of blocks) |
| v2 (v1 + 80 k map CNN, 0.73 M MACs, 74 µs) | ≈ 740 k | ≈ 1.85 M | ≈ 0.16 ms |

* Preprocessing (V2 odometry + map write) is 0.46–0.62 ms, plus 0.05 ms for the crop.
* ESTIMATE: torch eager adds about 5–10 µs dispatch per op over about 80–90 ops, so the total is about
  1–2 ms per `act()` against the 40 ms budget.
* Verify on torch 2.15.0 once it is pinned.
