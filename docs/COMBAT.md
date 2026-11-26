# Combat and survival (v1)

Owner: combat-and-survival-engineer. Code: `training/rewards/combat.py` (reward terms, training-only),
`training/eval/metrics/combat.py` (metrics plug-in, prefix `combat.`), tests in `tests/combat/`.

All combat behaviour comes from the learned policy. This module shapes rewards from privileged ground
truth and measures outcomes. It never chooses actions, and nothing under `submission/` imports it.

## 1. Sources

- `env/doom_env/wrapper.py:203-227`, `privileged_state()`: `kills` (KILLCOUNT), `damage_taken`
  (DAMAGE_TAKEN), `health`, `dead`. Armor is not in privileged state; it is only in obs `vitals[1]`.
- `wrapper.py:140-146`: on the terminal step the game variables are re-read, but `obs` is the previous
  (stale) one.
- `docs/EVALUATION_PROTOCOL.md` §4.1: after an exit the terminal row is read after the level change
  (`tic` reads 0). Whether `kills`, `damage_taken` and `health` survive on that row is still OPEN.
- `docs/EVALUATION.md`: kills count toward the score; damage taken and dying early cost points.

## 2. Reward terms (`training/rewards/combat.py`)

All terms return UNWEIGHTED values. Weights are config values (D10). H1 start weights are in
`training/configs/h1.json`.

| Term | Value per step | H1 weight | Notes |
| --- | --- | --- | --- |
| `kill` | new kills this step (count) | +0.5 | KILLCOUNT above its episode high-water mark; episode total capped at monsters present at reset (`cap_at_monsters`, default on) |
| `damage_taken` | HP of damage taken this step (≥ 0) | −0.01 | DAMAGE_TAKEN above its episode high-water mark |
| `ammo_spent` | ammo units spent this step | off (not in any config) | optional, see §3 |
| `death` | 1 on the death step | −3 | **defined in `training/rewards/progression.py`**, not here |

### 2.1 High-water mark (both `kill` and `damage_taken`)
Each term stores the episode maximum of its variable, seeded from `priv0` at reset, and pays
`max(0, current − max)`. So:
- spawn values (non-zero kills at reset, for instance) are the baseline and never pay;
- a row that drops, such as a post-exit row reading 0, pays 0, not a negative amount;
- the episode total equals `max_t(var_t) − var_0` exactly, the same formula the metrics use;
- nothing pays twice.

Anti-farming cap on `kill`: the episode total is capped at `inventory["monsters"]`, the live count at reset,
which `training/envs/factory.py:198-221` passes in. Pain Elementals (roster level 3,
`training/envs/mapgen/things.py:28-33`, used by the `full` stage, tier 4) spawn Lost Souls mid-episode.
Without the cap, those could pay beyond what the evaluator credits, `min(1, kills / monsters_at_reset)`.
The `combat` stage (roster level 2) has no spawners. With no `monsters` key the cap is off, and
`params: {cap_at_monsters: false}` disables it.

### 2.2 What `damage_taken` measures (MEASURED)
Measured on generated combat maps (per-step deltas on seeds 0–3, episode totals on seeds 0–9, random
attack-heavy driver):
- the per-step increase equals the **health lost** exactly. For example, with green armor: health −4,
  armor −2, damage_taken +4;
- **armor-absorbed points are not counted.** Picking up armor therefore lowers future `damage_taken`
  per hit, which is the right direction for survival;
- **overkill is counted:** final health on death goes below 0 (0 to −20 over seeds 0–9, so up to 20 HP of overkill).
  Episode totals of 100–138 also include damage healed by pickups;
- health pickups did not reduce it; it was monotone in every measured episode;
- all sources count: enemies, the player's own rocket splash and hazards. This is per
  EVALUATION_PROTOCOL §3 and not measured here (no self-damage occurred). The evaluator can't separate
  self-damage either, so training matches evaluation.

The real-engine test (`tests/combat/test_combat_real_engine.py`) asserts damage value == HP lost on
every step where health did not also rise.

### 2.3 Known limitations (v1, accepted)
- UNVERIFIED: whether KILLCOUNT includes monsters killed by infighting or barrels. If it does, a policy
  could learn to provoke infighting. That is legitimate play and counts in evaluation too.
- `damage_taken` includes overkill. MEASURED: at most 20 HP over 10 deaths, which is ≤ 0.2 reward at −0.01
  and small next to `death` −3.
- Damage dealt is **not derivable**: privileged state has no monster health or hit events.

## 3. `ammo_spent` (implemented, off by default) and a future `ammo_waste`
`ammo_spent` sums the per-type decreases of obs `vitals[4:8]` (bullets, shells, rockets, cells),
optionally weighted per type (`params: {type_weights: [1, 1, 1, 1]}`). Pickups are ignored. It returns 0
on the terminal step because the obs is stale.

It penalises **all** shooting, not only misses, so it must never be used without `kill`. A weight such
as −0.005 per unit is cheaper than one kill (+0.5) at up to about 100 shots per kill. It is intended for
the scarce-ammo curriculum stage only.

PROPOSED (not implemented): `ammo_waste` = ammo spent on steps that are not followed by a kill within
K steps, or a true miss signal. A true miss signal needs hit or damage-dealt ground truth. That would be
a training-time engine change (for example an ACS or ZScript damage counter in `training/scenarios/combat/`),
which needs the environment-engineer and a check that vanilla evaluation is unaffected.

No aim-related shaping (angle to enemy, enemy in crosshair) is used in v1. "Enemy in crosshair" and
"projectile incoming" are candidates for **auxiliary prediction targets** (labels, not reward), later.

## 4. Metrics (`training/eval/metrics/combat.py`, `METRICS = [combat_metrics]`)
This is a pure function of `EpisodeTrace` and `MapInventory`. Quirks are handled as in `core.py`:
kills and damage are the max over the trace (minus row 0), on exit health ignores the terminal row,
and the stale terminal obs means ammo spent on the last step is not counted. Undefined ratios are NaN.

| Key | Definition |
| --- | --- |
| `combat.kills` | max kills − kills at row 0 |
| `combat.monsters_present` | live monsters at reset (inventory) |
| `combat.kill_frac` | min(1, kills / monsters_present); NaN if there are no monsters (elimination rate) |
| `combat.cleared` | 1 if monsters_present > 0 and every one was killed (combat completion) |
| `combat.kills_per_min` | kills per minute of game time (T_end = min(4N, episode_ticks), 35 tics/s) |
| `combat.first_kill_step`, `combat.first_damage_step` | first row with a rise; −1 if none |
| `combat.damage_taken`, `combat.damage_per_min` | max damage − row 0; per game minute |
| `combat.health_end` | 0 if dead; otherwise the last valid row, clipped at 0 |
| `combat.health_min` | minimum valid health (0 if dead) |
| `combat.health_gained` | sum of health increases (health pickups) |
| `combat.died`, `combat.survived` | outcome == death; its complement (survival rate = mean) |
| `combat.death_time_frac` | T_end / episode_ticks if dead, otherwise NaN ("dying early") |
| `combat.ammo_spent`, `combat.ammo_spent_{bullets,shells,rockets,cells}` | sum of per-type decreases |
| `combat.kills_per_100_ammo` | ammo efficiency; NaN if nothing was spent |
| `combat.attack_steps`, `combat.attack_rate` | steps with ATTACK pressed; fraction of steps |
| `combat.kills_per_100_attacks` | kills per 100 ATTACK steps; NaN if none |
| `combat.no_ammo_steps_frac` | fraction of rows with all four ammo counters at 0 |
| `combat.weapon_switch_presses` | rising edges of SELECT_NEXT_WEAPON (it is edge-triggered) |

The trace has no armor or selected-weapon columns, so armor pickups, dry fire (ATTACK with
`selected_weapon_ammo == 0`) and weapon usage are not measurable yet. See the request in §6.

### 4.1 Random-driver reference (MEASURED; a seeded random button driver, not a trained policy)
Random buttons (each p = 0.3) with ATTACK at p = 0.7, generated `combat` stage family a, map seeds 0–9,
policy seed = map seed, episode_ticks 2100 (`tests/combat/test_combat_real_engine.py::_episode`):
- 10/10 died, mean `death_time_frac` 0.22;
- mean kills 0.6 of 5.6 monsters present, mean damage 111;
- mean ammo spent 3.5, despite an attack rate of 0.70;
- the same driver with SELECT_NEXT_WEAPON never pressed: mean ammo spent **29.5** and kills **1.1**,
  still 10/10 deaths with mean `death_time_frac` 0.22. So constant weapon switching suppresses firing,
  and indiscriminate switching is a concrete failure mode to track with `weapon_switch_presses`
  (D14's next_weapon bias of −2 points the same way);
- mean H1 combat return (kill + damage_taken + death) about −3.8 per episode.

Tier-1 combat maps kill this random driver within about 450 tics on average. Survival, not kills, is
the first thing the curriculum below needs to teach.

## 5. Planned combat curriculum (outline only, not built)
These will be scenarios in `training/scenarios/combat/` and are built on the environment-engineer's
mapgen `combat` stage plus overrides. Each stage is promoted on val metrics from §4.
1. **C0, single stationary target:** one weak monster (zombieman), small rect arena, plenty of ammo.
   Promote when `kill_frac` ≥ 0.8.
2. **C1, mobile single enemies:** randomised class among melee, hitscan and projectile; randomised
   spawn and facing. This teaches turning toward and engaging enemies through the learned policy only.
3. **C2, mixed groups:** 3–6 monsters, pillars for cover, health and armor pickups. Measure
   `health_gained`, `damage_per_min` and projectile dodging (damage from projectile enemies).
4. **C3, ambushes with scarce ammo:** ambush flags, few ammo items, optional `ammo_spent` penalty and
   weapon pickups (`weapon_switch_presses`, `no_ammo_steps_frac`).
5. **C4, vanilla finish:** stock mapgen `combat` and `full` stages and the train split with **no**
   training-time modifications (no infinite ammo, default monster counts). This is mandatory before any
   evaluation.

Training-time modifications such as infinite ammo or extra monsters are allowed only in C0–C3.

## 6. Open items
- The death term ownership mismatch: MODULE_CONTRACTS §6.2 lists `death` under combat.py, but it
  lives in progression.py. The registry forbids duplicates, so it stays there.
- Request (evaluation researcher): record `vitals[1]` (armor), `vitals[2]` (selected weapon) and
  `vitals[3]` (selected-weapon ammo) in `EpisodeTrace`, for armor and dry-fire metrics.
- Whether the post-exit terminal row keeps kills and damage is still unmeasured. Both the terms and the
  metrics are already robust to either answer.
