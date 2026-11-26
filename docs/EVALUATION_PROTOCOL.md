# Evaluation Protocol

Owner: evaluation-and-generalization-researcher. Status: **phase 1 (design + skeleton)**, 2026-11-26.
Primary metric: the **held-out-map proxy score** (§3) on the `test` split, reported with a 95 % CI.
Model selection uses only the `val` split.

Ground truth this protocol follows: `docs/EVALUATION.md` (tiers, non-binary scoring, timeouts),
`docs/CONTRACT.md` (observation, actions, game settings), `validate.py` (agent loop) and
`env/doom_env/wrapper.py` (`privileged_state()`, `sector_polygons()`). Where this document
approximates something hidden (the official weights), it says so.

Code: `training/eval/` (`harness.py`, `splits.py`, `inventory.py`, `metrics/core.py`, `proxy_score.py`,
`stats.py`). Manifests: `maps/splits/{smoke,train,val,test}.json`. Test access log:
`reports/eval/test_access_log.jsonl`. Run outputs: `runs/evaluation-and-generalization-researcher/<run-id>/`.

---

## 1. Map split policy

### 1.1 Unit of splitting: the source group

A map belongs to exactly one **`source_group`**. Splits are assigned per group, never per map, so
near-duplicates can't straddle splits. Hashes alone don't catch this. For example, fixtures
`509aca7cabb8cd7f.wad` and `c70e4f3a64c9f8f9.wad` are the same layout translated by a few cells.
Their sha256 values differ, but they share the group `template_fixtures_layout_dz` (MEASURED: same grid
in `tests/test_env.py:124-125`).

Group rules:

| Source | `source_group` | Rationale |
| --- | --- | --- |
| Third-party multi-level WAD of independently built levels (e.g. `freedoom1.wad`, `freedoom2.wad`) | `<wad>:<block>`, where a block is a contiguous range of maps (proposal: Freedoom2 MAP01–08, 09–16, 17–24, 25–32; Freedoom1 per episode) | Levels in a block often share an author, theme and texture set |
| A map we edit, crop, re-thing or monster-randomise | **inherits the parent's group** | Same geometry means the same layout |
| Procedural generator (Oblige/ObAddon, our UDMF generator) | `<generator>:<config-family>`, e.g. `udmfgen:corridor-v1` | Same generator config gives the same style. Seeds within a family are near-duplicates |
| ViZDoom bundled scenarios | `vizdoom:<scenario>` | One map each |
| Template maps (`maps/two_rooms.wad`, `tests/fixtures/*.wad`) | `template_*` | Smoke only (§1.5) |

`DECISION (needs main-session approval)`: how to sub-group `freedoom2.wad`, since it is a single WAD.
The proposal is blocks of 8 maps, with at least one block held out entirely for `val`, and `test` drawn from
source families that never appear in training. Reasoning: the hidden maps are **unseen** and use the
freedoom2 IWAD textures. If train, val and test are all Freedoom, a high score could reflect Freedoom-specific
style rather than generalisation.

Hard rules (enforced by `training/eval/splits.py`, which exits non-zero on violation):
- A `source_group` or a `(wad_sha256, map)` pair may appear in at most one of `train`, `val` and `test`.
- `smoke` maps may also appear in `train`, never in `val` or `test`.
- Every manifest entry's `wad_sha256` must match the file on disk.
- `test` must contain at least one source family (generator or author) with **no** group in `train`
  (checked manually at split-freeze time and recorded in `test.json` `description`).

Splits are frozen once `val`/`test` are populated. Adding maps to `test` later starts a new test
version (`test_v2.json`), and scores from different test versions are never compared.

### 1.2 Tier tagging

`tier` follows `docs/EVALUATION.md`: 1 = combat and survival, 2 = multi-room exploration (doors, dead ends),
3 = long-horizon progression (key, then locked door, then exit), 4 = full gauntlet (all of the above with ambushes,
scarce resources and multiple locks). A map's tier is the **highest** tier whose defining mechanic it needs
in order to exit:
- at least one locked door on the exit path: tier ≥ 3;
- at least two locks, or ambushes plus scarce resources: tier 4;
- doors or more than one room and no locks: tier 2;
- an arena or a single room whose challenge is the monsters: tier 1.

`tier_basis` records how the tier was assigned:
- `mechanic`: from the map contents, for tiny or synthetic maps;
- `manual`: judged by someone playing it;
- `generator`: from the generator config;
- `none`: no tier.

`tier: null` means the map is not representative of any tier (pure environment sanity).

### 1.3 Minimum maps per tier per split

| Split | Minimum per tier | Target per tier | Minimum distinct source groups per tier | Episode ticks |
| --- | --- | --- | --- | --- |
| train | 16 | 50+ (procedural, unbounded) | 3 | as authored, 2,100–4,200 |
| val | 4 | 6 | 2 | 2,100 (T1–T2), 4,200 (T3–T4) unless the map defines otherwise |
| test | 4 | 6 | 2 (at least one family unseen in train) | same rule as val |

`ESTIMATE:` the size of differences we can detect. Assume a between-map SD of about 20 proxy points (§3 scale:
about −15 to 100) and paired (map, seed) comparisons that cancel most map variance. With about 0.5 of the
unpaired variance left, 4 maps per tier gives a per-tier CI half-width of about 2·20·√0.5/√4 ≈ 14 points.
The full 16–24 map suite gives about 5–7 points. In other words, per-tier claims are coarse, and
**decisions are made on the whole-suite paired delta**. Revisit the SD once real val numbers exist.

### 1.4 Manifest schema (`maps/splits/<split>.json`, `schema_version` "1.0")

```json
{
  "schema_version": "1.0",
  "split": "smoke | train | val | test",
  "description": "free text: purpose, freeze notes",
  "created_utc": "YYYY-MM-DD",
  "owner": "evaluation-and-generalization-researcher",
  "seeds": [0, 1, 2],
  "maps": [{
    "id": "unique across all manifests",
    "wad": "repo-relative path, or @vizdoom/<file> for WADs bundled in the vizdoom package",
    "map": "MAP01",
    "wad_sha256": "hex digest of the WAD file",
    "source_group": "see 1.1",
    "source": "where it came from (URL / generator + version + seed)",
    "license": "license id or 'template'",
    "tier": 1, "tier_basis": "mechanic | manual | generator | none",
    "episode_ticks": 2100,
    "role": "same as split",
    "tags": ["door", "key", "locked_door", "exit", "monster", "negative_control"],
    "notes": ""
  }]
}
```

`training/eval/splits.py` validates required keys, tier ∈ {1, 2, 3, 4, null}, `1 ≤ episode_ticks ≤ 4200`, unique
ids, sha256 and leakage (§1.1).

### 1.5 Maps that exist locally (phase 1)

`train.json`, `val.json` and `test.json` are **empty**. There are no licensed training maps yet, and the
environment-engineer owns `maps/<source>/`. `smoke.json` lists all 14 local maps. They are **usable for smoke
tests only**: the fixtures are one-row corridors, or two rows for the door/zombie pair, with 2–14 64-unit cells and
1–4 sectors. `two_rooms` has 60 cells and 4 sectors. They exercise every metric code path, but say nothing about generalisation.

Fixture-to-layout mapping: MEASURED by hashing every grid string in `tests/test_env.py` and `tests/test_remote.py`
with `tests/fixtures.py:fixture_wad`. All 13 matched. Outcomes are from the random baseline
(`submission/agent.py`), 3 seeds, run `20261126T114755Z-smoke-61983206-default` (proxy_v1):

| id | file | grid / content | tier | Random-agent outcomes (seeds 0, 1, 2) | Smoke use |
| --- | --- | --- | --- | --- | --- |
| two_rooms | maps/two_rooms.wad | 1 door, 1 monster, exit; 4 sectors | 2 | timeout ×3, 29/60 cells, door not opened | end-to-end, validate.py parity |
| fx_tiny_room | 57a739… | `#P.#` | null | timeout ×3 | sanity |
| fx_corridor_empty | 2f617a… | `#P..#` | null | timeout ×3 | stuck/coverage sanity |
| fx_zombie_adjacent | 68d6a9… | `#P.z#` | 1 | timeout ×3, kill 1/1 ×3 | kill metric |
| fx_stim_zombie | 78a888… | `#P..h..z#` | 1 | death ×3 (168, 182, 430 steps) | death metric |
| fx_imp_corridor | 99c841… | `#P......i#` | 1 | death ×3, early | early death |
| fx_lethal_c | 279106… | `#P......c#` | 1 | death ×3 (52–74 steps) | early death |
| fx_door_zombie | 509aca… | 2-row room, door | 2 | death ×3, door opened ×3 | door metric |
| fx_door_zombie_shifted | c70e4f… | same layout, translated | 2 | death ×3 | near-duplicate grouping example |
| fx_door_between_zombies | 5c688a… | `#z.P..D..z#` | 2 | death ×3 | door metric |
| fx_exit_only | 6a392a… | `#P...E#` | 2 | **exit ×3** | completion metric |
| fx_door_exit | 9a5147… | `#P..D...E#` | 2 | timeout ×3, door opened ×3 | door-then-exit |
| fx_key_locked_exit | e7ab54… | `#P.r..R...E#`, lock code 129 | 3 | exit ×1, timeout ×2; key 3/3; locked opened 2/3 | key/locked-door metric |
| fx_locked_no_key | 7965e3… | `#P..B.E#`, lock code 130, no key | 3 | timeout ×3, locked 0/1 | **negative control**: exit unreachable |

---

## 2. `privileged_state()` fields used (eval-side only)

From `env/doom_env/wrapper.py:203-227`:

| Field | Used for |
| --- | --- |
| `x`, `y` | cell coverage, sector coverage, stuck detection, trajectories |
| `angle` | trajectories only |
| `health` | health remaining, exit efficiency |
| `kills` | combat term, ammo efficiency |
| `damage_taken` | damage penalty (all sources: **not** self-damage only, see §3) |
| `keys` | keys picked up (latched; same bits as `vitals[8:14]`) |
| `dead`, `timeout`, `exit_reached`, `finished` | cross-check of the outcome taken from `terminated`/`truncated`/`info["exit_reached"]` (`core.outcome_consistent`; MEASURED 42/42 consistent) |
| `sector_floor`, `sector_ceiling` | door-opened detection |
| `door_lock` | door inventory: `-1` = not a door, `0` = ordinary door, `> 0` = locked |
| `episode_ticks` | normalising time |

Also used: `env.sector_polygons()` (`wrapper.py:229`), whose docstring says the scorer uses it for
coverage. We use it for map area and reachable-cell denominators. `tic` is **not** used for terminal
time (see the MEASURED note in §4.1). Unused: `z`, `items`.

Not in `privileged_state()` but needed as denominators: monsters at reset, and keys at reset.
`training/eval/inventory.py` currently reads them from `env._game.get_state().objects` with
`doom_env.sensors.classify`. That is an adapter on wrapper internals until the env contract exists (§9.1).
Ammo comes from the observation `vitals[4:8]`, which the harness receives anyway.

---

## 3. Proxy scorer `proxy_v1` (APPROXIMATION of the hidden official score)

The official weights are hidden. `proxy_v1` mirrors the four official categories and their stated
ordering. Weights are **frozen per version** (`training/eval/proxy_score.py:PROXY_VERSION`). Raw
components are stored per episode, so a later version can be recomputed from saved `episodes.jsonl` without
replaying. Numbers from different proxy versions are never compared. `proxy_v0` was the pre-freeze label for two drafts
(area-only, then the blend without spawn exclusion) and is retired.

Per-map score (all notation from §4):

| Component | Formula | Max | Why this weight |
| --- | --- | --- | --- |
| Exit | 50 · `completed` | 50 | EVALUATION.md: exit is the **largest** category. 50 is more than any other single category and close to the sum of the rest, so exiting always dominates partial progress |
| Exit efficiency | if exit: 5 · min(1, H_end/100) + 5 · (1 − T_end/T_cap) | 10 | "small efficiency bonus for remaining health and time" |
| Progression | 15 · U_got / U_avail, with U = 1·keys + 1·doors + 2·locked doors (0 if U_avail = 0) | 15 | keys, doors and locked doors are listed. Locked doors are worth double because each needs a key plus backtracking |
| Exploration | 15 · (0.5 · area_coverage + 0.5 · cell_coverage), both excluding the spawn sector/cell (§4.3) | 15 | "unique areas visited". The scorer uses `sector_polygons()`, but it is unknown whether it counts sectors or cells within them. MEASURED: on the single-sector corridors, area coverage is 1.0 at spawn while cell coverage is 2/8, so area alone is degenerate. The 50/50 blend hedges between the two readings. Excluding spawn means a motionless agent earns 0 |
| Combat | 10 · min(1, kills / monsters_at_reset) (0 if no monsters) | 10 | "enemies killed". Smaller than exit and progression because the four categories are listed with exit largest |
| Damage penalty | −min(5, 0.02 · damage_taken) | −5 | "small penalties for self-damage". We can't separate self-damage, so all damage counts and the penalty is kept small |
| Early-death penalty | if death: −10 · (1 − T_end/T_cap) | −10 | "dying early" is penalised, and earlier is worse |

The score is **not** clipped at 0 (MEASURED: random agent on `fx_lethal_c` scores −9.7 to −10.0).
Suite score = tier-weighted mean of per-map means, with `TIER_WEIGHT = {1: 1.0, 2: 1.5, 3: 2.0, 4: 3.0}`
("harder tiers carry more weight"; the values are provisional). We use a weighted mean rather than a weighted
sum, so suites of different sizes stay on the same scale. Untiered smoke maps get weight 1.

Proxy validity check (to run as soon as there are 2+ trained checkpoints): the proxy must rank checkpoints
the same way under ±50 % perturbations of each weight. A ranking that flips under a reasonable perturbation is
reported as "not distinguishable by proxy", not as an improvement.

---

## 4. Metric definitions (per episode; `training/eval/metrics/core.py`)

Notation:
- N: number of decision steps.
- p_t = (x_t, y_t): position after step t (t = 0 is right after reset).
- T_cap: `episode_ticks`.
- F = 4: frame skip.

### 4.1 Outcome, completion, survival, length
- `outcome` ∈ {exit, death, timeout}: from `info["exit_reached"]` and `terminated`/`truncated`, as in
  `validate.py:418`.
- `completed` = 1[outcome = exit]. **Completion rate** = mean of `completed` over episodes.
- `survived` = 1[outcome ≠ death]. **Survival rate** = mean of `survived`.
- `steps` = N. `T_end` = min(N · F, T_cap). MEASURED: after an exit, `privileged_state()["tic"]` reads **0**
  (level change), so the engine `tic` cannot be used for terminal time. On death the two agree (65 steps,
  tic 260).
- `time_frac` = T_end / T_cap.
- `health_end` = 0 if dead. On exit it is max(0, health_{N−1}) from the last pre-terminal row, because the terminal row is read after
  the level change. Otherwise it is max(0, health_N).
- `kills` and `damage_taken` are the max over the trace: both are monotone, which is robust to a post-exit reset of the terminal row.
  OPEN: whether `kills`, `damage_taken` and `health` stay valid on the post-exit row is untested, because no measured
  exit episode had kills or damage.

### 4.2 Reward (reported separately)
`env_return` = Σ of the wrapper's default reward (+1 exit, −1 death, +0.1 per kill). Shaped training
returns are logged by rl-training-engineer, never by this harness. **Neither is the task score.** The
summary reports `env_return_mean` next to, never merged with, `proxy_suite`. See §8.3 for the divergence check.

### 4.3 Exploration coverage (eval-side only, from true x, y)
- Cell c(p) = (⌊x/64⌋, ⌊y/64⌋). `cells_visited` = |{c(p_t) : t = 0..N}|, including spawn (reported raw).
  `cells_new` = cells_visited − 1, which excludes the spawn cell c(p_0).
- `cells_total` = the number of 64-unit cells containing at least one 16-unit raster sample inside some sector
  polygon (`inventory.py`, even-odd test on `sector_polygons()`). It includes cells in unreachable sectors,
  so coverage is a lower bound on the "reachable" fraction.
- `cell_coverage` = min(1, cells_new / (cells_total − 1)), or 0 if the denominator is 0.
- `area_coverage` = Σ_{s ∈ visited, s ≠ s_0} area(s) / Σ_{s ≠ s_0} area(s), where s_0 is the spawn sector, area comes from a
  16-unit raster, and the visited sectors are those containing some p_t (smallest-area match if nested). It is 0 if the
  denominator is 0, as on single-sector maps.
- `max_no_new_cell_steps` = the longest run of steps with no new cell.
- Spinning or standing in place therefore scores 0 coverage: the spawn sector and spawn cell are excluded from both the
  numerator and the denominator.
- The exploration-engineer's `metrics/exploration.py` may add more (frontier, revisit ratio) under the `expl.` prefix.

### 4.4 Objectives
- `keys_picked` = Σ keys_N (6 latched bits). `keys_present` = key items in the level at reset.
- Door d is any sector with `door_lock[d] ≥ 0`. Its opening at step t is o_d,t = ceil_d,t − floor_d,t.
  Door d is **opened** iff o_d,0 < 56 and max_t o_d,t ≥ 56, where 56 is the player height and the
  passability threshold in `sensors.py:135`. It is latched once per sector, because doors close again.
  MEASURED on `fx_door_exit`: spamming USE bounces the door between 0 and 48, which is never passable and
  correctly not counted. On `fx_key_locked_exit` the door crosses 56 at step 25 and the player passes through.
- `doors_opened` / `doors_total` cover `door_lock = 0`. `locked_opened` / `locked_total` cover `door_lock > 0`.
  Locked means `> 0`, never a list of codes: the fixtures use codes 129 and 130, which fall outside the 1–3/100/101
  comment in `mapinfo.py:22-24`.
- Caveat: monsters can open ordinary doors, and remote doors open from switches. The metric counts the
  door state, not who caused it. Locked doors can only be opened by a key holder.

### 4.5 Stuck frequency
- Step t ≥ W is **stuck** iff max over k in [t−W, t] of ‖p_k − p_t‖ < R, with W = 32 steps (about 3.7 s) and
  R = 48 units.
- `stuck_steps_frac` = (number of stuck steps) / N.
- `stuck_events_per_1k` = 1000 · (number of maximal stuck runs) / N.
- A **stuck episode** is one with `stuck_steps_frac ≥ 0.5`. The RESEARCH_NOTES §8 variant ("no new cell
  for 100 steps") is also reported, as `max_no_new_cell_steps ≥ 100`.
- Caveat: standing still on purpose (for example waiting for a door) also counts. That is acceptable for a
  frequency diagnostic.

### 4.6 Combat, health and ammo
- `kills` = kills_N. `monsters_present` = monsters at reset (skill-3 spawns).
- `damage_taken` = damage_taken_N.
- `ammo_spent` = Σ_t Σ_{i ∈ bullets, shells, rockets, cells} max(0, a_i,t−1 − a_i,t), from `vitals[4:8]`.
  Pickups only increase ammo, so only decreases count.
- `kills_per_100_ammo` = 100 · kills / ammo_spent (NaN if nothing was spent).
- `attack_rate` and `use_rate` = the fraction of steps with ATTACK or USE pressed.
- The combat engineer's `metrics/combat.py` adds the `combat.` prefix (accuracy, threat exposure).

### 4.7 Generalisation gap
For splits A, B and proxy suite score S (or any metric):
- `gap(A→B)` = S_A − S_B.
- Relative gap = (S_A − S_B) / max(|S_A|, 1).

We report train→val and val→test. The CI comes from independent cluster bootstraps of each split (the maps
differ, so the comparison is unpaired), with the 2.5th and 97.5th percentiles of the bootstrap difference.
The gap is reported per tier and overall, always with the same checkpoint, mode and seeds protocol.
**Status: formula defined, computation PLANNED** (needs a populated train and val; `stats.cluster_bootstrap` is the
building block). "Train"
means **held-in training maps evaluated with the vanilla engine and eval seeds**, not training-run statistics.

---

## 5. Harness and agent loop (`training/eval/harness.py`)

- **Process boundary identical to `validate.py:360-433`.** The agent is served by
  `python -m doom_env.remote serve`, with `limits.THREAD_ENV` (1 thread) and the same `INIT/RESET/FIRST_ACT/ACT`
  timeouts and per-map budget `agent_budget_s(ticks)`. It uses `env.reset(seed)` followed by `conn.reset`,
  and CPU pinning where `sched_setaffinity` exists. The env is always `doom_env.make_env(wad, map, episode_ticks)`:
  the vanilla engine, never the `training/envs` factory.
- **Agent failure → stand still.** `validate.py` aborts on `AgentFailure`, but the scorer
  (EVALUATION.md "When your agent stops") keeps the game running while the player stands still, then
  restarts the agent for the next map. The harness does the latter: it sends NOOP until the cap, death or
  exit, records `agent_failure = "<kind>@step<k>"`, and restarts the process. MEASURED (run
  `failtest-agent-raises`): an agent that raises at act #50 gets `timeout`, `agent_failure:error`, and
  the next map exits normally with a fresh agent.
- **Lifecycle.** One agent init per pass over the suite, with maps in manifest order (like the scorer). Each seed is a
  new pass with a fresh process. MEASURED: the random baseline seeds its RNG in `__init__`, so
  its actions on map k depend on the episode lengths of maps 1..k−1. Per-map results are therefore
  reproducible only for a fixed manifest order. Reordering a manifest counts as a protocol change.
- **Determinism.** MEASURED: the same env seed and the same action sequence give bit-identical traces
  (two_rooms, 525 steps). Different seeds change outcomes once monsters engage (`fx_imp_corridor`: final
  damage 108, 117 and 111 for seeds 0, 1 and 2 under NOOP). On `two_rooms` the random agent never meets the monster, so
  seeds 0–2 are identical there.
- **Seeds.** smoke {0, 1, 2}; train-eval {100, 101, 102}; val {10000..10004}; test {20000..20009}. Training
  code must never use seeds ≥ 10000 (request to rl-training-engineer).
- **Episodes per map.** val: 5 seeds × each inference mode. test: 10 seeds × each mode. Release candidates:
  test plus `validate.py --docker`.
- **Inference modes.** `default` (whatever the submission does), `sample` and `greedy` are passed with
  `DOOM_AGENT_INFERENCE` and `DOOM_AGENT_SEED` (§9.2). They are **always reported separately**. The submitted mode
  is chosen on **val**, then frozen.
- **Timing.** macOS has no `sched_setaffinity`, so local runs set THREAD_ENV but are not pinned, and
  latency numbers are **not authoritative**. Authoritative timing comes from `validate.py --docker` on Linux.
- **Outputs per run** (`runs/evaluation-and-generalization-researcher/<run-id>/`):
  - `run.json`: argv, git commit, `submission_dirty`, agent code hash, weights hash, doom_env hash,
    contract and vizdoom versions, and proxy version;
  - `episodes.jsonl`: **every** episode, with all metrics and raw proxy components;
  - `<map>_s<seed>.npz`: trajectory (x, y, angle, health, kills, damage, keys, actions);
  - optional `.lmp` ViZDoom replays (`--record`);
  - `summary.json`.
  No episode is ever dropped or filtered. Agent failures count in every aggregate.

**Failure taxonomy** (`core.classify_failure`, evaluated in order):
1. `success` (or `success_after_agent_failure`);
2. `agent_failure:<kind>`;
3. `death_early` (time_frac < 0.25) or `death_late`;
4. `timeout_stuck` (stuck_steps_frac ≥ 0.5);
5. `timeout_missing_key` (an unopened locked door and fewer keys than present);
6. `timeout_looping` (no new cell for ≥ 50 % of the episode);
7. otherwise `timeout_exploring`.

Planned refinements, once the combat and exploration plug-ins exist: deaths with no enemy in view (hazard), and
locked doors seen in rays without the key.

---

## 6. Statistics and comparisons (`training/eval/stats.py`)

- **Unit = map.** The CI uses a two-level cluster bootstrap. Each replicate resamples maps with replacement, then
  episodes within each map, and takes the tier-weighted mean of per-map means. 10,000 replicates, rng seed 0,
  percentile 95 % CI.
- **Checkpoint comparison (paired).** Both checkpoints run on identical (map, seed, mode) sets.
  d = score_B − score_A per key. We report the mean Δ with a cluster-bootstrap CI over maps, Cohen's
  d_z = mean(d)/sd(d), the map win rate (fraction of maps where the per-map mean Δ > 0), and n_pairs/n_maps.
  - **Promote** B only if the lower CI bound of Δ(val proxy) > 0.
  - Report "no detectable difference" if the CI contains 0.
  - Non-inferiority (for speed or size wins): the lower bound of Δ is > −2 proxy points.
- **Ablations** (the RESEARCH_NOTES §8 table A–G; design by the architect and researchers): each arm uses ≥ 3
  training seeds where compute allows. Each training seed is evaluated on val with the standard seeds. The arm
  score is the mean over training seeds, and the CI bootstraps over training seeds × maps. Every arm is compared
  with its parent (B vs A, C vs B, …), paired on (map, eval seed, mode). Multiple arms get Holm-adjusted
  p-values (bootstrap p = 2·min(P(Δ ≤ 0), P(Δ ≥ 0))). The ablation table lists all arms, including
  failed ones. **Status:** `holm` and `paired_comparison` are implemented, and the bootstrap p-value is PLANNED
  (it needs `cluster_bootstrap` to return its replicate distribution).
- **Checkpoint selection** uses val only. "Best of N checkpoints on val" is biased upward. The final
  held-out number is the test score of the single selected checkpoint, measured once.

---

## 7. Test-set access policy

- The harness refuses `--split test` without `--test-purpose "<why>"`. It appends a JSON line to
  `reports/eval/test_access_log.jsonl` **before** playing: UTC time, run id, git commit, agent and weights
  hashes, seeds, mode, maps, purpose and argv.
- Budget: **at most 1 test run per milestone**, decided by the main session. Expected total for the event is
  ≤ 5. Each logged run is reported in full, whatever the result.
- No decision (hyperparameters, reward weights, inference mode, checkpoint choice, curriculum) may cite a
  test result. Test replays and trajectories are not watched for debugging. Debugging uses val.
- Test WADs never enter `train.json` or `val.json`, or any training manifest under `training/` (enforced for the
  manifests by `splits.py`. Request: environment-engineer's map registry should call
  `training.eval.splits.validate_all()` and reject any WAD whose sha256 is in `test.json`).
- Changing anything in §§3–5 after a test run starts a new protocol version, and earlier test numbers
  are not compared with later ones.

---

## 8. Training/evaluation mismatch detection

1. **Engine.** The harness always uses vanilla `doom_env.make_env`. `run.json` stores the vizdoom version, the
   contract version and a hash of `env/doom_env/*.py`. Every checkpoint trained with engine mods (auto-doors,
   infinite ammo) must also be evaluated vanilla. The **mod-on minus mod-off gap** on val is reported, and
   a gap > 20 % of the score is a blocker.
2. **Observation pipeline drift.** PLANNED (not built). Checkpoints must record a hash of
   `submission/doom_agent/obs/` at training time (§9.4). At eval, the harness will compare it with the current hash and fail on
   mismatch. Today `run.json` only records `agent_code_hash`, which covers all of `submission/**/*.py`. Also planned: per-feature
   distribution checks of obs seen in training vs eval (vitals mean/std, ray-distance histogram, hit-type
   frequencies, entity counts). The check flags a population stability index > 0.2.
3. **Reward vs score divergence.** PLANNED. On every val run, compute the Spearman correlation between each
   episode's training-shaped return (if logged by the env) and `proxy.score`, and also the trend across checkpoints. A
   training return that rises while the val proxy is flat or falling for 3 consecutive evals flags reward hacking.
4. **Train-map overfitting.** The train→val gap (§4.7) is tracked per checkpoint. If it grows while val is
   flat, the agent is overfitting the train maps.

---

## 9. Proposed contracts and needs from other owners

### 9.1 environment-engineer
`PROPOSED CONTRACT: training/envs/inventory.py: map_inventory(env) -> dict` called right after
`env.reset()`:
`{"monsters": int, "keys": int, "door_sectors": {sector: lock}, "sector_area": float[S], "cells_total": int}`.
This replaces the private-attribute adapter in `training/eval/inventory.py`. Also requested:
- a map registry that emits manifest entries in the §1.4 schema, and refuses test WADs;
- a vanilla-mode flag on the training env factory, so eval can confirm "no mods";
- the training env logging both the shaped return and the default env return per episode.

### 9.2 Model/agent owners (main session owns `submission/agent.py`)
`PROPOSED CONTRACT:` `DoomAgent.__init__` reads two optional environment variables:
- `DOOM_AGENT_INFERENCE` ∈ {`sample`, `greedy`}; if unset, the submission's frozen default;
- `DOOM_AGENT_SEED` (int); seeds the sampling RNG. `reset()` re-derives the RNG from (seed, reset count).

Environment variables are the only channel that crosses the `doom_env.remote` boundary without editing `env/`
(the protocol only has init, reset and act). The scorer never sets them, so the default path is exactly what gets scored.
`greedy` means a per-button threshold at 0.5, or the argmax for categorical heads. Either way it must not change
anything except action selection.

### 9.3 Metric plug-in interface (exploration-engineer, combat-and-survival-engineer)
`PROPOSED CONTRACT:` `MetricFn = Callable[[EpisodeTrace, MapInventory], dict[str, float|int|str]]`
(`training/eval/metrics/core.py`):
- it is a pure function and never touches the env;
- its keys are prefixed `expl.` or `combat.`;
- values are JSON scalars.

If they need extra per-step fields (for example visible-entity counts), they request them, and the harness adds them
to `EpisodeTrace.record`.

### 9.4 rl-training-engineer
Checkpoints should store:
- `obs_pipeline_hash`;
- `action_space_version`;
- the training map manifest ids;
- the training seeds, which must be < 10000.

Training should log the shaped return per episode, and val evals should be called through
`training.eval.harness` rather than a separate loop.

---

## 10. Reproduce

```bash
uv run --no-sync python -m training.eval.splits                                   # validate manifests
uv run --no-sync python -m training.eval.harness --split smoke                    # all smoke maps, seeds 0 1 2
uv run --no-sync python -m training.eval.harness --split smoke --only two_rooms --seeds 0
uv run --no-sync python -m training.eval.harness --split val --mode greedy        # once val is populated
uv run --no-sync python -m training.eval.harness --split test --test-purpose "M3 final"   # logged
```

Phase-1 reference run: `20261126T114755Z-smoke-61983206-default` (proxy_v1). Random baseline, agent code hash
`61983206…`, no weights, 42 episodes:
- proxy suite 16.9 (95 % CI 6.0–30.3);
- completion 0.095, survival 0.571;
- env return −0.33;
- 0 agent failures.

Every other run on disk is labelled `proxy_v0` (pre-freeze) and is **not comparable**:
- `20261126T113644Z…` (two_rooms ×2) and `20261126T113655Z…` (smoke ×42): area-only exploration;
- `20261126T113838Z…` (smoke ×42, with `.lmp`): blend without spawn exclusion;
- `failtest-agent-raises` (3 episodes): a scratch agent that raises at act #50, used to test the failure path.

## 11. Known limitations
- The proxy weights are guesses (§3). Only the ordering of the categories comes from the docs.
- `damage_taken` includes enemy damage, so the "self-damage" penalty is overstated.
- `cells_total` and the sector area include unreachable sectors.
- The door-opened metric can't attribute who opened a door.
- **Zero-point rules are not mirrored.** EVALUATION.md gives 0 points for a map when init (60 s) or `reset()` (10 s) times out.
  The harness gives stand-still credit after a reset failure, and aborts the pass on an init failure. TODO: score those
  maps as 0 and record them.
- The tier-weighted suite score includes untiered smoke maps at weight 1. It is only meaningful once val and test exist.
- The dashboard (TensorBoard or static HTML in `reports/eval/`) is not built yet. It is phase 2, once there are 2+ checkpoints.
