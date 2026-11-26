# Exploration: reward term, metrics, and the v2+ plan

Owner: exploration-engineer. Status: **v1** (D8, D10 in `docs/DECISIONS.md`). Everything here is **training- or
eval-side only**. Nothing under `submission/` imports it, and no intrinsic or shaping signal runs at evaluation.

Files:
- `training/rewards/exploration.py`: the `cell_visit` reward term.
- `training/eval/metrics/exploration.py`: the `expl.` metric plug-in.
- `tests/exploration/test_cell_visit.py`, `tests/exploration/test_expl_metrics.py`.

## 1. Principle

The scorer rewards "unique areas of the map visited (spinning in place earns nothing)" (`docs/EVALUATION.md:46`).
The v1 bonus mirrors that category exactly, and it counts **true positions**, not observations:
- It is drift-free.
- It is immune to the noisy-TV problem. Monsters, projectiles and moving doors cannot inflate it.

Every later exploration method has to beat this baseline on **completion**, not just on coverage
(`research/RL_RESEARCH.md` §4.1). **If coverage rises while completion falls, report it as a failure.**

## 2. `cell_visit` reward term (v1)

Config (`docs/MODULE_CONTRACTS.md` §6.3). The weight is a config value, not contract (D10):
```
reward:
  terms:
    - {name: cell_visit, weight: 0.05, params: {cell: 64}}
```

**Definition.** It is identical to the eval coverage metric (`training/eval/metrics/core.py`,
`docs/EVALUATION_PROTOCOL.md` §4.3).
- **Cell.** c(p) = (⌊x/64⌋, ⌊y/64⌋), with true x, y from `privileged_state()`. `math.floor` is used, not
  `int`, so the four cells around the origin stay distinct.
- **Sampling.** One sample per decision, at the **end-of-step** position, with no interpolation of the
  segment walked. This is exactly what the eval trace records (one row per decision).
- **Spawn.** The spawn cell is pre-recorded at `reset`, so it never pays.
- **Payment.** +1 (unweighted) the first time each other cell is entered in the episode, and never again (D10,
  via `FirstOccurrence`). The ledger is cleared on every `reset`.
- **Terminal step.** The term pays normally on the terminal step. MEASURED: on generated `locomotion`/`keys`/`doors`
  maps the post-exit row moves only 0 to 7 units from the step before, so it is a real position. The eval
  trace counts that row too.
- **Non-finite x or y** pays 0.

**Invariant (tested on a real engine walk).** Σ_episode `cell_visit` = |distinct non-spawn cells| =
`core.cells_new` = `expl.coverage_cells`.

**What pays 0 by construction:**
- spinning in place and standing still;
- re-entering a visited cell;
- oscillating across a cell border (only the first crossing pays);
- returning to the spawn cell.

An episode's maximum is bounded by the number of reachable cells, so the term cannot be farmed.

**Budget against the exit reward.** The most the term can pay in an episode is w · (cells_total − 1). That bound
counts unreachable cells too, so the real maximum is lower. MEASURED at w = 0.05 on the 4 walker maps:

| Map | cells_total − 1 (about) | Maximum payout |
| --- | --- | --- |
| `locomotion/0` | 122 | 6.1 |
| `doors/1` | 140 | 7.0 |
| `keys/3` | 242 | 12.1 |
| `keys/11` | 329 | 16.5 |

The exit pays 10. On large maps, therefore, full coverage can be worth more than the exit. Exiting early forfeits
up to that much, but staying costs nothing except time. Watch `core.completed` against `expl.coverage_frac` in H1.
If completion lags, lower w or scale it by 1 / cells_total. That is a config change, not a code change.

**Safeguards and known limits:**
- **Door bouncing.** `cell_visit` cannot pay for door motion, because it reads only the player position. Door
  rewards (`door_open`, `training/rewards/progression.py`) latch once per sector at opening ≥ 56 (D10), and D11
  rules out a USE-debounce mask. Bounces are therefore only *measured* (`expl.door_bounces`, §3.3). Per D11, log
  them in H1.
- **Re-entry farming across episodes** is impossible, because the ledger is per episode. There is no lifelong
  count in v1.
- **Speed.** MEASURED: the largest per-decision displacement was 34.8 units, over the test walker's
  forward-only walks on 3 generated maps. That walker never strafes or runs, so a strafe-run step may be longer.
  Either way it is less than one 64-unit cell. A diagonal corner can still be skipped. That gap is the same in the eval metric, so the two
  stay consistent.
- **No hand-written exploration policy, and no action override.** The term is a scalar reward. Nothing reads it
  to choose actions.
- **The bonus depends on the episode's history.** A memoryless critic cannot predict it, so the GRU (and later
  the ego-map) has to carry visitation (`research/RL_RESEARCH.md` §4.2).

## 3. `expl.` metrics plug-in

`METRICS = [coverage_metrics, repetition_metrics, objective_metrics]`. These are pure functions of
`(EpisodeTrace, MapInventory)` (`docs/MODULE_CONTRACTS.md` §7).
- Values are Python `int` or `float`.
- An undefined value is `float("nan")`, as in core's `kills_per_100_ammo`.
- The plug-in imports core's `CELL`, `_stuck_mask` (W = 32, R = 48) and `PLAYER_HEIGHT`, so the shared
  definitions cannot drift.
- n = number of rows (row 0 is reset), and steps = n − 1.

### 3.1 Coverage

| Key | Definition |
| --- | --- |
| `expl.coverage_cells` | distinct non-spawn cells; equals `core.cells_new` and the episode sum of `cell_visit` |
| `expl.coverage_frac` | min(1, coverage_cells / (cells_total − 1)); equals `core.cell_coverage` |
| `expl.coverage_auc` | mean over t = 1..n−1 of coverage_frac(t). It rewards *early* discovery: two agents with the same final coverage differ if one found the cells sooner |
| `expl.cells_per_100_steps` | 100 · coverage_cells / steps |
| `expl.cell_entries` | the number of steps t whose cell differs from the cell at t−1 |
| `expl.revisit_ratio` | the share of cell entries that go into a cell already seen this episode (NaN if there were no entries) |
| `expl.long_revisit_ratio` | the share of cell entries that go into a cell last occupied **more than 32 steps** earlier (32 = core's stuck window, about 3.7 s). This captures coming back to explored ground (loops, re-sweeping) rather than local wobble between neighbouring cells |

**Reference values.** MEASURED: the scripted shortest-path walker (`tests/env/env_walker.py`) on generated
`locomotion/0`, `keys/3`, `keys/11` and `doors/1`, all of which it completed:

| Metric | Walker range |
| --- | --- |
| `revisit_ratio` | 0.43–0.52 |
| `long_revisit_ratio` | 0.02–0.11 |
| `repeated_state_frac` | 0.22–0.36 |

- **`revisit_ratio` is high even for a near-optimal path.** The revisit gaps were bucketed by steps since the
  cell was last occupied:
  - ≤ 2 steps: 0–3 per walk;
  - 3–8 steps: 3–14;
  - 9–32 steps: 16–67;
  - > 32 steps: 1–21.

  So these are not 1-step border flickers. They are slower: drifting back and forth along a cell boundary,
  plus backtracking out of dead-end key rooms. The gap cannot separate the two. Compare `revisit_ratio` only
  between agents on the same maps. `long_revisit_ratio` is the more discriminating loop signal.

### 3.2 Repetition and stuck (diagnostics only, never an action override)

| Key | Definition |
| --- | --- |
| `expl.repeated_state_frac` | state = (cell, 45° heading octant). Runs of identical consecutive states are collapsed first (dwell is measured by still/stuck, not here). The metric is the share of state **changes** that land in a state already occupied earlier this episode (NaN if the state never changes). Spinning in place cycles through 8 states and then only repeats them: 33/40 = 0.825 in the unit test, against 0.22–0.36 for the walker |
| `expl.still_steps_frac` | the share of steps with displacement < 1 unit |
| `expl.spin_steps_frac` | the share of steps that are still **and** turn ≥ 5° (spinning in place) |
| `expl.stuck_steps` | the count of stuck steps, using core's rule: step t ≥ 32 is stuck iff every position in [t−32, t] is within 48 units of p_t. The same mask as `core.stuck_steps_frac`, reported as a count |
| `expl.longest_stuck_run` | the longest run of consecutive stuck steps |

### 3.3 Objective discovery (reached) and door bouncing

| Key | Definition |
| --- | --- |
| `expl.keys_reached_frac` | the number of key bits newly set after spawn, divided by `keys_present` (NaN if 0) |
| `expl.first_key_step` | the first row at which a new key bit is set (NaN if none) |
| `expl.doors_reached_frac` | doors (normal and locked) that start closed (< 56) and reach an opening ≥ 56, divided by the doors that start closed. This is the same "opened" rule as core |
| `expl.first_door_step` | the first row at which any such door is ≥ 56 (NaN if none) |
| `expl.exit_reached`, `expl.exit_step` | outcome == exit; the episode's step count (NaN if the exit was not reached) |
| `expl.objectives_reached` / `_total` / `_frac` | (new keys + doors opened + exit) / (keys_present + doors closed at spawn + 1) |
| `expl.door_reversals` | across all door sectors, the number of times a door's opening goes from rising to falling (zero-change steps are ignored). A normal open-then-close cycle counts 1 |
| `expl.door_bounces` | reversals whose rising phase peaked **below 56**, i.e. the door was reversed before it ever became passable. This is the USE-spam signature (`docs/EVALUATION_PROTOCOL.md` §4.4 measured 0↔48 bouncing) |

### 3.4 Not yet computable

"Seen" (key, door or exit *observed* before it is reached) needs per-step observation fields that
`EpisodeTrace` doesn't record. This is requested from the evaluation researcher in the v1 report.
"Seen" is deliberately **not** approximated from positions.

## 4. Testing

```
uv run --no-sync python -m pytest tests/exploration -q --basetemp=/tmp/ptq_explore
```

**Unit tests:**
- pays once per cell;
- the spawn cell is excluded;
- border jitter pays once;
- negative coordinates;
- spinning and standing pay 0;
- resets per episode;
- params are validated;
- the contract config loads through `RewardComposer.from_config`;
- `CELL` equals both `training.envs.inventory.CELL` and the eval `CELL`.

**Real-engine test.** The test-only walker runs on generated `locomotion/0` and `keys/3`. The test asserts
reward sum = distinct non-spawn cells = `core.cells_new` = `expl.coverage_cells`, and
`expl.coverage_frac` = `core.cell_coverage`.

**Metric tests.** Synthetic traces cover:
- a straight walk;
- back and forth;
- spinning;
- standing still;
- a slow purposeful walk (dwell is not repetition);
- long against short revisits;
- a door bounce plus a full open;
- keys and exit;
- spawn-held keys and spawn-open doors;
- 0-, 1- and 2-row traces;
- registration in `run_all` with JSON-scalar types.

## 5. v2+ plan (not built; gated per `research/RL_RESEARCH.md` §4 and `research/EXPERIMENT_HYPOTHESES.md` H7)

Order: (1) this v1 cell bonus. (2) Make it learnable: the GRU, then the legal ego-map visitation channel; an
optional privileged critic (H5). (3) Only then, **NGU-lite lifelong novelty** (H7).

**H7 design:**
- **Intrinsic stream.** The stream is r_int = w_cell · 1[new cell] · (clip(α_RND, 1, 5) − 1). It is
  *gated by the new-cell indicator*, so it can only amplify real exploration and never pays when the agent is
  standing, spinning or revisiting. The base bonus stays in r_ext, so nothing is counted twice.
- **α_RND.** This is the RND predictor error, normalised as in NGU. Its input is a running-mean/std-normalised
  **compact legal embedding**: ray distances and hit types, entity-type counts, and key flags. It never sees raw
  pixels.
  - Entity velocities and health are **excluded**. They are the noisy-TV channels: animated or moving
    monsters, and damage flicker.
- **Value heads.** A separate V_int head with γ_I = 0.99 and γ_E = 0.995. The combined advantage is
  A = 2·A_E + 1·A_I, then per-minibatch normalisation of the combined A. r_int is normalised by a running std of
  intrinsic returns.
- **Predictor training.** The predictor trains on 25 % of each batch. The intrinsic stream is episodic.
- **Interface.** PROPOSED CONTRACT, not yet in `docs/MODULE_CONTRACTS.md`: §4.5 currently lists only aux
  heads and a privileged critic. The proposal is an `IntrinsicReward` module in `training/exploration/` with:
  - `compute(batch) -> r_int` of shape (T, N);
  - `update(batch) -> dict` of stats;
  - `state_dict()` / `load_state_dict()`, for checkpointing the predictor and the running statistics.

  It needs the project-architect to add an §4.5 slot and the rl-training-engineer to add a PPO hook (V_int and the
  advantage mix) before anything is built. It never runs at evaluation.
- **Decay.** β starts at 1.0 and decays linearly to 0.1 over the last 30 % of each curriculum stage. It is never 0
  in exploration stages. The cell bonus itself stays constant, because it mirrors a scored category.
- **Kill criteria (H7):**
  - intrinsic return rises while `expl.coverage_cells` falls (noisy-TV-like); check the embedding for velocity
    leakage;
  - no gain in keys or doors at 20M steps;
  - **any drop in completion (`core.completed`)**.

**ICM is not prioritised (H11, spare budget only).** Its forward-model error peaks where the agent itself
causes the randomness: re-pressing USE reverses a door, and firing provokes monsters. The inverse-model features
keep both, so ICM would likely reward door bouncing. If H11 runs, `expl.door_bounces` and
`expl.spin_steps_frac` are its hacking alarms.

**Ablation protocol (once v2 exists):**
- Arms: none / cell count / cell + RND (NGU-lite) / cell + ICM.
- Same budget and the same seeds, on held-out val maps through `training/eval/harness.py`.
- Report `core.completed` first, then coverage, `expl.objectives_frac`, `expl.revisit_ratio`,
  `expl.door_bounces` and `expl.spin_steps_frac`.
