# Evaluation Protocol

Owner: evaluation-and-generalization-researcher. Status: **phase 2 (real splits; test INCOMPLETE)**, 2026-11-26.
Primary metric: the **held-out-map proxy score** (§3) on the `test` split, reported with a 95 % CI.
Model selection uses only the `val` split.

Ground truth this protocol follows: `docs/EVALUATION.md` (tiers, non-binary scoring, timeouts),
`docs/CONTRACT.md` (observation, actions, game settings), `validate.py` (agent loop) and
`env/doom_env/wrapper.py` (`privileged_state()`, `sector_polygons()`). Where this document
approximates something hidden (the official weights), it says so.

Code: `training/eval/` (`harness.py`, `splits.py`, `build_splits.py`, `tiering.py`, `inventory.py`,
`metrics/__init__.py`, `metrics/core.py`, `proxy_score.py`, `stats.py`; checks in `training/eval/tests/`). Manifests: `maps/splits/{smoke,train,val,test}.json`. Test access log:
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

Freedoom sub-grouping as implemented by the registry (`training/envs/maps.py`): freedoom2 in blocks of 8 maps
(`freedoom2:MAP01-08` … `MAP25-32`), freedoom1 per episode (`freedoom1:E1` … `E4`). Reasoning for keeping the
test family separate: the hidden maps are **unseen** and use the freedoom2 IWAD textures. If train, val and test
are all Freedoom, a high score could reflect Freedoom-specific style rather than generalisation.

Hard rules (enforced by `training/eval/splits.py`, which exits non-zero on violation):
- A `source_group`, a `(wad_sha256, map)` pair or a **map geometry** may appear in at most one of `train`, `val`
  and `test`. Geometry = sha256 of `TEXTMAP` (UDMF) or `VERTEXES` (binary maps), which catches retextured or
  re-thinged copies under a new hash or group. MEASURED: over all 155 registry maps the only geometry duplicates
  are ViZDoom scenarios (basic = basic_audio = basic_notifications; cig MAP02 = cig_with_unknown MAP02;
  rocket_basic = simpler_basic), all in train. No freedoom1/freedoom2 map shares geometry with another.
- `smoke` maps may also appear in `train`, never in `val` or `test`.
- Every manifest entry's `wad_sha256` must match the file on disk, **including** `@vizdoom/` bundled WADs.
- `tier_basis` ∈ {mechanic, manual, generator, static_rule_v1, none}; `role` equals the split.
- `test` must contain at least one source family with **no** group in `train`. Family = the project or generator
  style (`freedoom` for freedoom1 and freedoom2, `udmfgen-a` / `udmfgen-b`, `vizdoom`, `template`;
  `splits.family`). This is an **error**, downgraded to a printed WARNING only while `test.json` carries
  `"incomplete": true`.

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
- `mechanic`: from the map contents, for tiny or synthetic maps (smoke fixtures);
- `manual`: judged by someone playing it (not used; never from replays of val/test maps);
- `generator`: from the generator (`training/envs/mapgen/generator.py:317-319`);
- `static_rule_v1`: the static rule below, for every map whose registry tier is null (Freedoom, ViZDoom);
- `none`: no tier.

**Rule `static_rule_v1`** (`training/eval/tiering.py`). Inputs are static WAD data only: `doom_env.mapinfo`
door/lock sectors and `training.envs.inventory.static_things` (skill 3, single player). No engine run, no
outcomes, no replays. The inputs are stored per entry in `tier_inputs` and in `notes`.

| Inputs | Definition |
| --- | --- |
| L | distinct key colours that lock a door. Lock codes are normalised: 1/4/129 red, 2/5/130 blue, 3/6/131 yellow, 100 = one colour ("any"), 101 = all three |
| M | static monsters; D = ordinary doors; S = sectors |

| Tier | Condition (first match) |
| --- | --- |
| 4 | L ≥ 2, or L ≥ 1 and M ≥ 6 (multiple locks, or a lock plus a monster gauntlet) |
| 3 | L = 1 |
| 2 | L = 0 and (D ≥ 1, or M = 0 and S ≥ 2) |
| 1 | L = 0, D = 0, M ≥ 1 |
| null | otherwise (one sector, nothing to do) |

The thresholds are the generator's own (stage tiers in `config.py`, and the tier-4 rule "≥ 2 locks, or a full
stage with ≥ 6 monsters"), so a single rule covers generated and authored maps. MEASURED: the rule reproduces the
generator tier on **70/70** generated maps (`uv run --no-sync python -m training.eval.tiering`).
Monster density was considered for tier 1 vs 2 and **rejected**: EVALUATION.md lists doors as a tier-2 skill, and
the density of generator combat-a maps (MEASURED 0.88–4.0 monsters per 100 cells) overlaps door-heavy Freedoom
levels without locks (0.16–6.1), so a density threshold would label multi-room door maps as combat maps.
Limits: the rule cannot tell whether a lock is on the exit path (authored maps can have optional locks), and
ACS-spawned monsters (most ViZDoom scenarios) are invisible to it, so `basic`, `defend_the_*`, `take_cover` and
friends come out `null`. Result: Freedoom levels are almost all tier 4 (they have 2–3 key colours and dozens of
monsters). The maps with no locks are tier 2, and no Freedoom map is tier 1 or 3. Changing the rule is a new
`static_rule_v2` label; it is never adjusted to move particular val/test maps.

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

Episode ticks: manifests keep the registry value. Generated maps use their stage cap (2,100 for locomotion,
doors and combat; 4,200 for keys and full), ViZDoom scenarios 2,100, and **Freedoom levels 4,200 whatever their
tier**. Freedoom levels are full-size (441–7,687 cells), so the 2,100-tick rule for T1–T2 above counts as "the map
defines otherwise". This is recorded per entry in `episode_ticks`.

`build_splits` prints the per-tier counts against these minimums (`SHORTFALL` lines); shortfalls are reported, never
hidden by moving maps between splits (§1.5).

### 1.4 Manifest schema (`maps/splits/<split>.json`, `schema_version` "1.0")

```json
{
  "schema_version": "1.0",
  "split": "smoke | train | val | test",
  "description": "free text: purpose, freeze notes",
  "created_utc": "YYYY-MM-DD",
  "owner": "evaluation-and-generalization-researcher",
  "seeds": [0, 1, 2],
  "status": "complete | INCOMPLETE", "incomplete": false,
  "incomplete_reason": "only when incomplete",
  "generated_by": "uv run --no-sync python -m training.eval.build_splits --write",
  "maps": [{
    "id": "unique across all manifests",
    "wad": "repo-relative path, or @vizdoom/<file> for WADs bundled in the vizdoom package",
    "map": "MAP01",
    "wad_sha256": "hex digest of the WAD file",
    "source_group": "see 1.1",
    "source": "where it came from (URL / generator + version + seed)",
    "license": "license id or 'template'",
    "tier": 1, "tier_basis": "mechanic | manual | generator | static_rule_v1 | none",
    "tier_inputs": {"sectors": 0, "doors": 0, "locked_sectors": 0, "lock_colours": [], "static_monsters": 0, "static_keys": 0},
    "episode_ticks": 2100,
    "role": "same as split",
    "tags": ["door", "key", "locked_door", "exit", "monster", "negative_control"],
    "notes": ""
  }]
}
```

`tier_inputs` is present only for `static_rule_v1` entries. `status` / `incomplete` / `incomplete_reason` /
`generated_by` are manifest-level, and the smoke manifest predates them. `training/eval/splits.py` validates:
- required keys, `role` = split, tier ∈ {1, 2, 3, 4, null}, `tier_basis`, and `1 ≤ episode_ticks ≤ 4200`;
- unique ids and sha256, including `@vizdoom/` WADs;
- leakage by group, hash and geometry, and the unseen-family rule (§1.1).

### 1.5 Split contents (frozen 2026-11-26, test INCOMPLETE)

Built from the registry, never hand-typed: `uv run --no-sync python -m training.eval.build_splits --write`
(`--check` fails if the committed manifests drift from registry output). Entries come from
`training.envs.maps.all_entries()` through `manifest_entries(entries, role)`. Splits are assigned per
`source_group` by the glob table `build_splits.GROUP_SPLIT`, where the first match wins and an unmatched group is
an error. Manifest order is registry order. The registry guard `check_trainable` is asserted both ways: it accepts
every train entry and refuses every val and test entry. As a result, `check_trainable(all_entries())` now
**raises by design**. Trainers must use the train manifest or `MapSampler`.

| Split | Maps | Groups (tiers) | Seeds |
| --- | --- | --- | --- |
| train | 109 | generator family a: combat-a (10 × T1), locomotion-a (10 × T2), doors-a (10 × T2), keys-a (4 × T3, 6 × T4), full-a (10 × T4); freedoom2 MAP01-08 (1 × T2, 7 × T4), MAP09-16 (8 × T4), MAP17-24 (8 × T4); freedoom1 E1 (2 × T2, 7 × T4), E2 (2 × T2, 7 × T4); 17 ViZDoom scenarios, one group each (deadly_corridor T1; cig, cig_with_unknown, deathmatch, multi_deathmatch, my_way_home, predict_position T2; 10 null) | 100–102 |
| val | 37 | generator family b, seeds 1000–1003: combat-b (4 × T1), locomotion-b (4 × T2), doors-b (4 × T2), keys-b (3 × T3, 1 × T4), full-b (4 × T4); freedoom2 MAP25-32 (MAP30 T2, 7 × T4); freedoom1 E3 (E3M1, E3M8 T2; 7 × T4) | 10000–10004 |
| test | 9 | freedoom1 E4 (E4M8 T2, 8 × T4). **INCOMPLETE** | 20000–20009 |

Per-tier counts (maps / groups) against §1.3:

| Split | T1 | T2 | T3 | T4 | null |
| --- | --- | --- | --- | --- | --- |
| train | 11 / 2 (short: 16 / 3) | 31 / 11 | 4 / 1 (short: 16 / 3) | 53 / 7 | 10 / 10 |
| val | 4 / 1 (short: 2 groups) | 11 / 4 | 3 / 1 (short: 4 / 2) | 19 / 4 | 0 |
| test | 0 (short) | 1 / 1 (short) | 0 (short) | 8 / 1 (short: 2 groups) | 0 |

The train shortfalls are cheap to fix, because the generator is unbounded: the generated samples are only a sample,
and training generates more family-a maps. The val shortfalls are T1 (only generator combat-b) and T3 (only 3
keys-b maps with one lock). Both need more **family-b** maps from the environment-engineer: more keys-b seeds with
one lock, and a second tier-1 source. Until then, val per-tier numbers for T1 and T3 are single-group and are not
used for decisions.

**What "frozen" means for val.** Adding val maps (for example the requested keys-b seeds) creates a new val version.
It changes the suite and the manifest order, so the random baseline's per-map results change too (§5). Val numbers
from different versions are never compared. The checkpoints being compared and the random floor are rerun on the new
version. Val is never reshuffled with train.

**Stratified reporting.** Freedoom maps are 17 of the 37 val maps and are mostly tier 4 (weight 3), so they carry more
than half of the tier-weighted val score. Val results are therefore reported overall **and** per stratum (generated
family b vs Freedoom). The ViZDoom drills have no val counterpart, so the train→val gap (§4.7) excludes them.

**The test gap.** `test.json` carries `"status": "INCOMPLETE"`, `"incomplete": true` and an `incomplete_reason`.
- It holds freedoom1 E4 only. That is a held-out **episode**, but not an unseen **family**: it comes from the same
  Freedoom project (authors, textures) as freedoom1 E1–E2 and freedoom2 in train. It therefore violates the §1.1
  unseen-family rule (a WARNING while flagged, an ERROR once the flag is removed).
- It has no tier 1 or tier 3 maps.
- A third-party family needs an `oss-scout` licence check and main-session approval before download.
- Until then, the test split must not be called "unseen-map" performance. The harness stamps
  `manifest_status` and `manifest_incomplete` into `run.json` and into the access-log line of any test run.
- Completing the manifest before the first logged test run is still test v1. After a logged test run, any change
  is `test_v2`.

Leak cross-checks run on every `splits` validation: source group, (sha256, map) and geometry. None fire on the
committed manifests. The ViZDoom geometry duplicates are all in train. Request to the environment-engineer: give
basic/basic_audio/basic_notifications, cig/cig_with_unknown and rocket_basic/simpler_basic one `source_group` each,
so the grouping states what the geometry check enforces.

#### Smoke split (`smoke.json`, unchanged since phase 1)

`smoke.json` lists the 14 template maps. They are **usable for smoke tests only**: the fixtures are one-row
corridors, or two rows for the door/zombie pair, with 2–14 64-unit cells and 1–4 sectors. `two_rooms` has 60 cells
and 4 sectors. They exercise every metric code path, but say nothing about generalisation.

Fixture-to-layout mapping: MEASURED by hashing every grid string in `tests/test_env.py` and `tests/test_remote.py`
with `tests/fixtures.py:fixture_wad`. All 13 matched. Outcomes are from the random baseline
(`submission/agent.py`), 3 seeds, run `20261126T114755Z-smoke-61983206-default` (proxy_v1), and were reproduced
episode for episode by `20261126T160850Z-smoke-3081520d-default` after the D13 inventory swap:

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

Not in `privileged_state()` but needed as denominators: monsters at reset, and keys at reset. Since D13 they come
from the single implementation `training.envs.inventory.map_inventory(env)` (environment-engineer; its one
engine-object read is isolated there in `_live_objects`). `training/eval/inventory.py` is now a shim that wraps
that dict in the `MapInventory` dataclass of the metric contract and adds `sector_at` from the public
`env.sector_polygons()`. The mapping is `monsters_present ← monsters` (live at reset, **not** `static_monsters`;
they differ on fd2 MAP16 and MAP30 and on every ACS-spawned scenario) and `keys_present ← keys`. MEASURED: the smoke
run after the swap reproduces all 42 episodes of the reference run, every non-timing field identical.
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

- **Metrics.** Each episode runs `training.eval.metrics.run_all`: every `MetricFn` in `ALL_METRICS` (§9.3), core
  first, then the `expl.` and `combat.` plug-ins once their modules exist. The proxy is computed from the
  `core.` keys only.
- **Subsets.** `--only <ids>` and `--groups <source_group globs>` select maps, keeping manifest order. `run.json`
  records `subset: true`, plus `manifest_status` and `manifest_incomplete`. A subset run is never reported as a split score.
- **Summary.** `summary.json` adds `means` (episode means of the main metrics) and `per_tier` (maps, episodes, proxy mean
  with a per-tier cluster-bootstrap CI, completion, survival, coverage), next to the tier-weighted `proxy_suite`.

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
  hashes, seeds, mode, maps, purpose, argv and `manifest_status` / `manifest_incomplete`. While `test.json` is
  INCOMPLETE (§1.5), a test result is labelled "held-out Freedoom episode", never "unseen-map".
- Test log status 2026-11-26: **no test run has been made** (`reports/eval/test_access_log.jsonl` is empty).
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
`training/envs/inventory.py: map_inventory(env) -> dict` is **landed and adopted** (D13). The eval side imports it
(§2). The registry (`training/envs/maps.py`: `manifest_entries`, `check_trainable`, `MapSampler`) is **landed**,
and `maps/splits` is built from it (§1.5). Still requested:
- more family-b maps for val: keys-b seeds with one lock (T3 has 3 maps, needs ≥ 4) and a second tier-1 source group
  (§1.5);
- merge the geometry-duplicate ViZDoom scenarios into shared source groups (§1.5);
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
`MetricFn = Callable[[EpisodeTrace, MapInventory], dict[str, float|int|str]]`
(`training/eval/metrics/core.py`). **Registry implemented** (MODULE_CONTRACTS §7): each module exports
`METRICS: list[MetricFn]`, and `training/eval/metrics/__init__.py` collects them into `ALL_METRICS` (core first). A
plug-in module that does not exist yet is skipped; any other import error propagates. `run_all` raises on a key
without the module's prefix, on a duplicate key and on a non-scalar value. Rules:
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
uv run --no-sync python -m training.eval.splits                                   # validate manifests + leak checks
uv run --no-sync python -m training.eval.build_splits --check                     # manifests == registry output
uv run --no-sync python -m training.eval.tiering                                  # static_rule_v1 vs generator (70/70)
uv run --no-sync python -m pytest training/eval/tests -q --basetemp=/tmp/ptq_eval # eval-owned checks
uv run --no-sync python -m training.eval.harness --split smoke                    # all smoke maps, seeds 0 1 2
uv run --no-sync python -m training.eval.harness --split smoke --only two_rooms --seeds 0
uv run --no-sync python -m training.eval.harness --split val --groups 'udmfgen:*-b-v*' --seeds 10000   # family-b floor
uv run --no-sync python -m training.eval.harness --split val --mode greedy        # full val, 5 seeds
uv run --no-sync python -m training.eval.harness --split test --test-purpose "M3 final"   # logged
```

Phase-1 reference run: `20261126T114755Z-smoke-61983206-default` (proxy_v1). Random baseline, agent code hash
`61983206…`, no weights, 42 episodes:
- proxy suite 16.9 (95 % CI 6.0–30.3);
- completion 0.095, survival 0.571;
- env return −0.33;
- 0 agent failures.

Post-D13 runs (2026-11-26, git `269d031`, `submission_dirty` true because other agents added files under
`submission/doom_agent/`; `submission/agent.py` is the unchanged random baseline; agent code hash `3081520d…`,
weights hash `efc09868…`, doom_env hash `d1829786…`, vizdoom 1.3.3, proxy_v1, mode `default`):
- `20261126T160850Z-smoke-3081520d-default`: smoke, seeds 0–2, 42 episodes, with `.lmp` (42 files). It is
  **identical, episode for episode, to the phase-1 reference** (all non-timing fields): proxy suite 16.93 (CI 6.04–30.32),
  completion 0.095, survival 0.571, env return −0.326, 0 agent failures.
- `20261126T160929Z-val-3081520d-default`: **random-agent floor on the val family-b subset** (20 maps × seed 10000,
  20 episodes; `subset: true`, so this is not the val suite score). Proxy suite −2.20 (95 % CI −5.09 to 0.67);
  completion 0.00; survival 0.65; env return −0.335; mean cell coverage 0.115; 0 agent failures. Failure classes:
  death_early 5, death_late 2, timeout_exploring 8, timeout_missing_key 5. Per tier (proxy mean, CI):
  - T1: 4 maps, −8.92 (−9.54 to −8.29), survival 0.00;
  - T2: 8 maps, 2.10 (1.39 to 3.14), survival 1.00;
  - T3: 3 maps, 1.10 (0.52 to 1.92), survival 1.00;
  - T4: 5 maps, −5.18 (−10.69 to 0.34), survival 0.40.

  No key was picked up, no locked door was opened, and 1 of 35 ordinary doors was opened (0 of 12 keys, 0 of 12 locked doors).
  The Freedoom part of val has not been run yet.
- `harnesscheck-freedoom-train`: a **harness check, not a score**. It runs train maps fd2_map01 and fd1_e1m1 at seed 100
  (binary Freedoom format, 1,050 steps each). Both ran to timeout with no error, p99 act latency 1.0 and 7.2 ms, 0 agent failures.

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
- The dashboard (TensorBoard or static HTML in `reports/eval/`) is not built yet. It waits for 2+ checkpoints.
- **Test is INCOMPLETE** (§1.5): one Freedoom episode, no unseen family, no T1/T3 maps.
- The val T1 and T3 tiers are single-group, and T3 has 3 maps (§1.5).
- `static_rule_v1` cannot see whether a lock is on the exit path, or see ACS-spawned monsters (§1.2).
- Eval-owned tests live in `training/eval/tests/`, because `tests/eval/` is unassigned in OWNERSHIP.md.
