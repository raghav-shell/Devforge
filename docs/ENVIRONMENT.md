# Environment: training env factory, vector env, maps, generator and registry

Owner: environment-engineer. Contents:
- the training env factory, vector env and `MapSampler` (A3, docs/MODULE_CONTRACTS.md §5);
- the procedural map generator, the map registry and the smoke-load tool;
- the `map_inventory` contract.

The observation preprocessor (A2) is documented in `docs/OBSERVATION_SCHEMA.md`.

Everything here is **training-only**. Nothing is imported from `submission/`, and `env/` is never edited.
Maps are loaded through the official `doom_env.make_env(wad, map, episode_ticks=…, render_buffers=…)`.

## Environment facts (measured)

| Item | Value |
| --- | --- |
| `doom_env.CONTRACT_VERSION` | `1.0.0-draft` |
| vizdoom | 1.3.3 (`pyproject.toml`), IWAD `freedoom2.wad` from the vizdoom package, skill 3 |
| Exit detection | `wrapper.py`: the episode finished, not dead and not timed out, gives `exit_reached`. Any exit special works. We use `Exit_Normal` (243), either a walk-over (`playercross`) or a USE switch (`playeruse`). |
| Door detection | `mapinfo.py` with UDMF `zdoom` namespace uses Hexen specials 11/12/13/14/202. `arg0 = 0` targets the **back** sector. The lock argument is `arg3` for 13. `lock > 0` means `locked_door`. |
| Ray door types | `sensors.py`: a closed (opening < 56) two-sided line next to a door sector reads `door`, or `locked_door` if locked |
| Sector limit | ViZDoom reports ≤ 2,048 sectors. The generator caps at 1,024 (`MAX_SECTORS`); real maps use 1–49. |
| Deathmatch-only maps | `make_env` raises `No player 1 start`: vizdoom `cig` MAP01, `cig_with_unknown` MAP01/03/04/05, `multi_duel`. The registry filters them statically. |
| Scenario ACS | ViZDoom scenario scripts still run under `doom_env` (e.g. `basic` ends the map when the monster dies, which reads as `exit_reached`). Their `.cfg` rewards and buttons are **not** applied. |

## Training env factory and vector env (A3)

Code: `training/envs/factory.py` (`EnvSpec`, `TrainEnv`, `make_train_env`), `training/envs/vector.py`
(`SubprocVecEnv`, `StepResult`), `training/envs/maps.py` (`MapSampler`). Contract: docs/MODULE_CONTRACTS.md §5.

```python
from training.envs.vector import SubprocVecEnv
vec = SubprocVecEnv({"manifests": ["maps/splits/train.json"], "randomize_ticks": True},
                    num_workers=8, envs_per_worker=8, seed_base=0, reward_cfg=cfg.reward.terms, obs_cfg=ObsConfig())
obs = vec.reset()                       # dict of (N, ...) features, every slot starts an episode
res = vec.step(actions_mb)              # StepResult(obs, reward, terminated, truncated, episode_start, infos)
vec.stats(); vec.close()
```

### Worker loop, per slot (the eval feature path is the same code)
1. `pre.commit_action(a)`.
2. `env.step(a)` on the official `doom_env`.
3. The `RewardComposer` gets a `StepContext` with the **full** `privileged_state()` before and after the step. It runs in the worker.
4. `feats = pre.process(obs)`. `ObsPreprocessor(strict=True)` is the same class `submission/agent.py` uses, so NaN, Inf, wrong shapes or unknown keys raise.
5. If done: `info["final_feats"] = feats`, then `env.reset(seed=next)`, `pre.reset()`, and `feats = pre.process(obs0)`.

Each worker returns its slots' features stacked with `stack_features`. `tests/env/test_factory.py::test_features_match_eval_path` checks that TrainEnv features equal
`ObsPreprocessor` applied to the raw official env, bit for bit.

### `sampler_cfg` keys (`env.*` config)
| Key | Default | Meaning |
| --- | --- | --- |
| `manifests` | `()` | train/smoke manifest paths. A val or test manifest, or any entry held out in `maps/splits/val.json` / `test.json`, raises `ValueError` in the parent |
| `entries` | none | in-memory manifest entries, e.g. `[e.manifest("train") for e in maps.generated_entries()]`; same held-out check |
| `weights` | uniform | one weight per manifest (plus one for `entries`) |
| `randomize_ticks` | true | per episode, `episode_ticks` is drawn uniformly from the generated map's sidecar `episode_ticks_range`, else from [2100, 4200]. The tick cap is changed on the running engine (no re-init) |
| `episodes_per_map` | 4 | sample a new map every k episodes. An engine re-init costs about 0.3 s, while the same map reuses the engine and only re-draws `episode_ticks` |
| `render_buffers` | false | must be true if `obs.enable_depth_band` |
| `mods` | `()` | training-time engine mods. None exist in v1, so any value raises `NotImplementedError`. `TrainEnv.vanilla` reports whether the engine is unmodified |
| `return_privileged` | false | opt-in slim `info["privileged"]` |
| `step_threads` | true | step a worker's slots concurrently (see the throughput section) |

`reward_cfg`: a term list, `{"terms": [...]}`, or `{"terms": [...], "modules": ["pkg.mod"]}`. The extra modules are
imported in every worker so their `@register_term` classes exist. `env_default` (registered in `factory.py`) passes the
wrapper's default reward through. It is for smoke runs only; the v1 shaped terms live in `training/rewards/`.

### Seeding (reproducible from `seed_base` < 10000)
Global slot i draws episode seeds from `default_rng([seed_base, i])` in [0, 9999] and maps from
`default_rng([seed_base, i, 1])`. Results do not depend on how slots are split across workers, or on
`step_threads` (both are tested bit-exact). A restarted worker uses fresh streams (`[seed_base, i, 2|3, restart_no]`), so it does not
replay seeds. `TrainEnv` refuses any seed ≥ 10000.

### Info keys (training-only; never merged into `obs`)
| Key | When | Content |
| --- | --- | --- |
| `map_id`, `tic`, `exit_reached`, `reward_terms` (unweighted, per term), `env_reward` | every step | |
| `privileged` | every step, opt-in | `x, y, angle, health, kills, damage_taken, keys, dead, exit_reached` (no sector arrays) |
| `final_feats` | done steps that are **not** crashes | features of the terminal (stale) obs, for truncation bootstrapping |
| `episode` | done steps | `map_id, seed, episode_ticks, length, return_shaped, return_env, outcome ∈ {exit, death, timeout}, cells (distinct 64-unit cells), keys, kills, use_edges, weapon_edges, door_reversals, reward_terms` (per-term episode sums) |
| `crashed`, `crash_error`, `crash_count`, `worker_restarted` | crash steps | see below |

`door_reversals` is D11 logging. It counts a door sector that was moving when a USE rising edge happened and then moves the opposite way
within that step or the next one. A `Door_Raise` that closes after its delay does not count.
MEASURED on `fx_door_exit`: one press gives 0; USE, NOOP, USE gives 1; ten USE/NOOP pairs give 9.

### Crash handling (flagged, counted, never hidden)
- **Engine crash** (a ViZDoom exception in step or reset): `EngineCrash` is raised, the engine is reopened, and the slot
  is auto-reset. The step reports `truncated=True`, `terminated=False`, `reward=0`, `info["crashed"]=True` and **no**
  `final_feats` (the trainer uses `trunc_value` 0). It is counted in `vec.stats()["crashes"]` and `crashed_steps`.
- **Dead worker process**: the parent notices through `Process.is_alive()`. EOF alone is not reliable, because ViZDoom runs each engine
  as a child of the worker, and that child inherits the pipe and survives the worker (MEASURED). The worker is
  restarted, and every slot it ran is flagged the same way with `info["worker_restarted"]=True`. Each worker leads its own
  process group, and the parent SIGKILLs the whole group on restart or close, so no engine is orphaned (tested). SIGTERM
  makes a worker exit through `finally`, which closes its engines.
- **Anything else** (a bad action, NaN features, a reward-term exception, an unknown term) raises `WorkerError` in the parent with the
  worker traceback. Bad configs and val/test maps fail in the parent before any worker starts.
- Workers start with `forkserver` (fork is unsafe on macOS). Test hooks: `_debug_inject_crash(i)` and `_debug_kill_worker(w)`.

### Throughput (MEASURED 2026-11-26/27, M4 10 cores = 4 performance + 6 efficiency, shared machine)

Commands:
```bash
uv run --no-sync python -m training.benchmarks.env_vector_throughput --maps smoke,two_rooms,generated,fd2_map01 \
    [--workers 1,2,4,8] [--epw 1,4,8] [--no-threads] [--episodes-per-map K] --out runs/environment-engineer/bench/x.json
uv run --no-sync python -m training.benchmarks.env_step_profile --terms all
```
SPS is env decisions per second summed over all envs, with random actions, no policy, render off, the real
preprocessor, and `env_default` (vector grid) or every registered term (profile). Other agents' jobs shared the machine. Every
result in `runs/environment-engineer/bench/*.json` records its load average. Run-to-run noise at a fixed config is about ±35 %
(1×1 generated: 617 vs 837 SPS in two runs).

**The slow H1 smoke (docs/TRAINING.md §6.1: 310–620 env SPS at 2 workers × 4 envs).**
- Stepping several engines **sequentially** in one process collapses intermittently, so the rate is bimodal. The engine-only `DoomGame.make_action`
  round trip shows it too, so it is not wrapper or A3 code. Idle engines use 0 % CPU, so it is not a busy-wait.
- HYPOTHESIS, not measured: wake-up latency of engines left idle while their siblings step.
- Fix: workers step their slots from a thread pool (`step_threads`, default on). This is bit-identical to sequential stepping, which is tested,
  including with the real H1 reward terms.

Threads on vs off at the same load (2 × 4, load average 2–5, each config run twice):

| maps | threads on (run 1 / run 2) | threads off (run 1 / run 2) |
| --- | --- | --- |
| smoke manifest | 1,459 / 1,446 | 327 / 636 |
| two_rooms | 2,329 / 2,394 | 356 / 2,629 |
| generated | 1,330 / 1,332 | 686 / 338 |

When sequential stepping happens to land in its good mode, threading costs about 10 % (two_rooms 2,629 vs 2,394). Threading removes the collapse
and the variance.

**Trainer end to end after the fix.** The docs/TRAINING.md §1 `real-two-rooms-smoke` command (2 × 4, real DoomPolicy, 6 updates, load average 1.8–2.4, run dir
`runs/rl-training-engineer/a3-recheck-two-rooms-smoke`) gives **env-only 2,960–3,030 SPS** and total 1,150–1,200 with 0 crashes. The original report was 310–620.

Grid (`vector_before.json` sequential at load 6–8; `vector_after.json` threaded at load 8–36), generated maps, `episodes_per_map` 1:

| workers × envs/worker | before | after |
| --- | --- | --- |
| 1 × 8 | 188 | 1,215 |
| 2 × 4 | 449 | 1,438 |
| 4 × 4 | 528 | 1,938 |
| 4 × 8 | 566 | 1,741 |
| 8 × 8 | 973 | 1,139 (load about 30) |

freedoom2 MAP01 after the fix: about 700–1,000 SPS at 4–10 workers.

**Map switching.** At 4 × 4 on generated maps, `episodes_per_map` 1 gives 1,180 SPS and 8 gives 2,122. A new map is an engine re-init of about 0.3 s. The default is now 4.

**Per-decision cost**, single process, load average 4–7 (`step_profile_lowload.json`):

| map | official `DoomEnv.step` µs | `privileged_state` µs | all registered reward terms µs | `ObsPreprocessor` µs | `TrainEnv.step` µs |
| --- | --- | --- | --- | --- | --- |
| two_rooms | 346 | 7 | 23 | 92 | 489 |
| generated keys | 506 | 7 | 27 | 92 | 655 |
| freedoom2 MAP01 | 1,850–1,937 | 8 | 24 | 95 | 2,059–2,073 |

- The official wrapper step (engine plus doom_env's Python ray and entity sensors, which can't be changed) is 70–95 % of a decision.
- A3 adds about 125–150 µs per decision: 6 % on Freedoom and 25–30 % on small maps.
- The A2 preprocessor latency on a quiet machine is **p50 0.041 ms, p99 0.052 ms** (`training/benchmarks/env_obs_latency.py`); it is about 92 µs inside this loop.

**The remaining ceiling** (generated keys map, load average 5–8): independent single-env processes (no vec env) sum to 1,489 SPS for 1 process, 3,625 for 4 (about 905 each) and **5,549 for 8** (about 690 each).
The synchronous vec env at k × 1 gives 1,184, 2,607 and **2,899**.

Two separate effects:
1. Per-process speed falls as processes are added even with no coordination. That is machine contention: 8 engines plus 8 Python processes on 10 cores, with other agents' load on top.
2. The lockstep vec env loses about half of the independent total at 8 workers, because each step waits for the slowest worker. About 20 % at 1 × 1 is the pipe round trip. HYPOTHESIS, not measured: the mix of performance and efficiency cores widens the slowest-worker gap.

The `~13k/s at 8 processes` in RL_RESEARCH §1 was not reproduced on this machine: 8 independent processes reached 5.5k here.
Options if more is needed: more envs per worker with threads (it amortises the lockstep), or an asynchronous collector in the trainer.

**Startup**: engines open in parallel within a worker. 1 × 8 starts in 0.97 s (2.27 s before); 10 × 8 takes about 9 s.

**Recommended config**: `num_workers` 8 (`default_num_workers()` = min(8, cores − 2)), `envs_per_worker` 4–8, threads on,
`episodes_per_map` 4. Re-measure on a quiet machine before sizing H1.

## Procedural generator: `training/envs/mapgen/`

```bash
uv run --no-sync python -m training.envs.mapgen --stage keys --seed 123 --out maps/generated/keys
uv run --no-sync python -m training.envs.mapgen --stage all --seed 0 --count 10 --out maps/generated
uv run --no-sync python -m training.envs.mapgen --stage full --seed 7 --family b --out /tmp/x.wad
```

```python
from training.envs.mapgen import generate
m = generate("keys", seed=123, family="a")    # GeneratedMap: .wad bytes, .meta dict, .layout, .geometry
m.write("maps/generated/keys")                 # keys-a-000123.wad + keys-a-000123.json
m = generate("combat", 5, monsters=(12, 12))   # any StageConfig field can be overridden
```

**Determinism.** The same `(stage, family, seed[, overrides])` gives a byte-identical WAD. This is tested in-process and
across subprocesses with different `PYTHONHASHSEED`. All randomness comes from `random.Random` seeded with a
sha256 of `"doomsday-udmfgen:v<VERSION>:<stage>:<family>:<seed>:<attempt>:<override-hash>"`. An attempt that
violates a constraint is retried with `attempt + 1`, up to 64 tries. The attempt number is stored in the sidecar.

**Format.** A UDMF `namespace = "zdoom"` PWAD with lumps `MAP01, TEXTMAP, ENDMAP` and no nodes, the same layout
as `maps/two_rooms.wad`. GZDoom builds the nodes on load.

**Geometry.** The map is a grid of 32-unit cells, and each cell belongs to one sector or is void. A linedef goes on every cell edge
whose two sides differ, and collinear edges merge while both sides stay the same. That gives closed sectors with
no T-junctions. Rooms sit in the slots of a jittered macro grid. Corridors are 2 or 3 cells wide (64 or 96 units) and connect
rooms in neighbouring slots, either straight or with one jog. A corridor keeps a one-cell clearance from every other region, so
regions never merge by accident. Every walkable sector has at least 96 units of headroom. The player is 56 tall and steps up at most 24, and
room floors are in {0, 8, 16, 24}.

**Line orientation.** One-sided lines face their sector. Door lines have front = corridor and back = door sector, on both faces.
Exit-pad lines have front = room and back = pad.

### Stages (`config.py`, `stage_config(stage, family)`)

| Stage | Rooms | Doors | Locks | Monsters | Items | Tier | `episode_ticks` (manifest / suggested range) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `locomotion` | 2–5 | none | 0 | 0 | 0–1 health | 2 | 2100 / 2100–4200 |
| `doors` | 3–7 | 60 % of corridors, ≥ 1 on the exit path | 0 | 0 | few | 2 | 2100 / 2100–4200 |
| `keys` | 6–12 | 35 % | 1–3 key → locked-door chains | 0 | few | 3 (1 lock) / 4 (≥ 2) | 4200 / 3000–4200 |
| `combat` | 1–3, one arena (×1.4 slot) | none | 0 | 3–10, roster level 2 | 2–5 health, 3–7 ammo, 1–2 weapons | 1 | 2100 / 2100–4200 |
| `full` | 7–13 | 40 % | 1–3 | 4–14, roster level 3 | health, armour, ammo, 1–3 weapons | 3 or 4 | 4200 / 3000–4200 |

Knobs, all `StageConfig` fields and overridable:
- layout: `n_rooms`, `slot_size`, `room_size`, `shapes` (rect, L, pillars, cross, octagon, ring), `corridor_width`, `straight_prob`, `loop_prob`, `arena_scale`;
- doors: `door_prob`, `min_path_doors`, `door_close_prob` (`Door_Raise` closes after 150 tics; `Door_Open` stays open);
- locks: `n_locks`, `skull_prob`;
- things: `monsters`, `roster_level` (0 = zombieman, shotgunguy, imp; 1 adds demon, spectre, chaingunguy, lost soul; 2 adds cacodemon, hell knight, revenant; 3 adds baron, pain elemental), `ambush_prob`, `health_items`, `armor_items`, `ammo_items`, `weapons`;
- look: `floor_choices`, `room_height`, `corridor_height`, `light`, `exit_switch_prob`.

**Overrides.** A map built with overrides gets the override hash appended to its file name, registry id and `source_group`, so it never overwrites the plain seed's map.

**Families.** `a` is the default style. `b` is a held-out style with a disjoint wall palette, more irregular shapes (including rings) and wider corridors.
Families exist so a whole generator family can be held out of training.

**Keys and locks.** Locks use `Door_LockedRaise` (13) with ZDoom lock 129/130/131 ("any red/blue/yellow key", the same
pattern as the template fixture). A key is a card or a skull (`skull_prob`). Locks go on **bridges** of the
start→exit room path. Each key goes in the region that lock opens, and for the first 8 attempts it must sit in a
**side branch** off the main path, so backtracking is needed. MEASURED: 100 % of keys were in a branch over 100 seeds per family.
Lock counts over 100 seeds: family a keys 1/2/3 = 55/34/11, full 57/37/6.

**Randomised per map:** player start (room, position with ±8 jitter, angle), exit kind (pad or switch), key, monster and item
placement, textures, light, heights, and door open/close behaviour.
**Not yet randomised per episode:** the spawn on a fixed map. Doom allows exactly one player-1 start, and extras
become voodoo dolls. Per-episode spawn and `episode_ticks` randomisation belong in the env factory, which comes later.

### Solvability guarantee

The layout is solvable by construction: a connected room graph, locks on path bridges, keys before their lock.
`progression.check_solvable` then re-checks the **emitted cell grid** independently of the room graph. It
flood-fills the player's centre over grid vertices (a 32×32 box over 4 cells, step ≤ 24, opening ≥ 56,
doors open, locked doors only with the key) to a key fixpoint. It requires:
1. the exit is reachable;
2. **every** lock gates the exit: with that key withheld, the exit is unreachable, so there is no bypass.

Maps failing either check are rejected. `solvable` and the details go into the sidecar.

MEASURED on the final code (2026-11-26):
- **Generation:** 0 failures across 100 seeds × 5 stages × 2 families. Locomotion and doors take about 2–3 ms per map, keys and full about 11–14 ms, combat about 15–18 ms.
- **Engine walk:** the test-only walker reaches the exit on 120/120 maps (20 seeds × locomotion/doors/keys × families a and b). That covers every key and every lock, both pad and switch exits, and every key colour, card and skull.
- **Monster maps:** the walker dies on combat and full maps because it does not fight. That is expected.
- **Rendering:** walks with `render_buffers=True` produce real frames. PNGs are in `runs/environment-engineer/render/` (normal door BIGDOOR2/DOOR3, coloured locked-door faces, blue GATE3 exit pad, SW1EXIT switch) with no hall-of-mirrors. The only near-constant frames (`rgb` std < 1) were close-up wall views while the player was pressing USE.

> These graph searches (room-graph BFS, vertex flood fill) are **generator-internal, training-only map
> construction**. The agent never sees them or the room graph, and nothing in `submission/` imports them.
> In the tests, a scripted walker (`tests/env/env_walker.py`, test-only, using `privileged_state`) also
> completes maps in the real engine.

### Sidecar (`<name>.json`, training/eval metadata only, never a policy input)

It contains:
- identity: `generator`, `generator_version`, `stage`, `family`, `seed`, `attempt`, `config` (all knobs), `source_group`;
- scoring: `tier`, `episode_ticks`, `episode_ticks_range`;
- counts: `num_sectors`, `num_linedefs`, `num_things`, `room_count`, `num_doors`, `num_locked_doors`, `num_keys`, `num_monsters`;
- layout: `rooms` (bbox, shape, floor, ceiling), `corridors` (endpoints, sectors, door kind, on-path), `path_rooms`;
- objects: `start`, `exit` (kind, position), `doors`, `locked_doors`, `keys` (colour, class, room, in_branch), `monsters`, `items`;
- result: `solvable`, `solvability`, `wad_sha256`.

### Sample set (`maps/generated/<stage>/`)

There are 10 maps per stage in family a (seeds 0–9) and 4 per stage in family b (seeds 1000–1003): 70 WADs plus sidecars, **2.4 MB** in total.
`tests/env/test_map_registry.py::test_committed_samples_match_generator` fails if the generator output drifts.
After an intentional generator change, bump `GENERATOR_VERSION` (this changes `source_group`), regenerate,
and tell the eval researcher that the manifest sha256 values changed.

```bash
rm -rf maps/generated && for st in locomotion doors keys combat full; do
  uv run --no-sync python -m training.envs.mapgen --stage $st --seed 0 --count 10 --family a --out maps/generated/$st
  uv run --no-sync python -m training.envs.mapgen --stage $st --seed 1000 --count 4 --family b --out maps/generated/$st
done
```

For training, generate as many maps as you need into `runs/…` or `maps/generated/`. A generated map costs 2–18 ms (MEASURED).

## Map registry: `training/envs/maps.py` (alias `training/envs/map_registry.py`)

```bash
uv run --no-sync python -m training.envs.maps                              # list all 155 entries
uv run --no-sync python -m training.envs.maps --sources generated --role train --json
```

| Source | Entries | `wad` | `source_group` | License |
| --- | --- | --- | --- | --- |
| generated | 70 (samples) | `maps/generated/<stage>/<name>.wad` | `udmfgen:<stage>-<family>-v1` | project |
| freedoom2 | 32 (MAP01–32) | `@vizdoom/freedoom2.wad` | `freedoom2:MAP01-08` … `MAP25-32` | BSD-3-Clause (Freedoom) |
| freedoom1 | 36 (E1M1–E4M9, over the freedoom2 IWAD) | `@vizdoom/freedoom1.wad` | `freedoom1:E1` … `E4` | BSD-3-Clause (Freedoom) |
| vizdoom | 17 scenario maps with a player start | `@vizdoom/scenarios/<f>.wad` | `vizdoom:<scenario>` | MIT (ViZDoom) |

The licence strings are from the packages' documentation and still need an `oss-scout` confirmation. Bundled WADs load in place and are never
copied. `MapEntry.manifest(role)` and `manifest_entries(entries, role)` give dicts in the `maps/splits` schema
(docs/EVALUATION_PROTOCOL.md §1.4). The registry **never writes `maps/splits/*.json`**, which the evaluation researcher owns.
`check_trainable(entries)` raises if any entry's id, `source_group` or (sha256, map) appears in `val.json` or `test.json`.
`resolve_wad("@vizdoom/…")` resolves the same way as `training/eval/harness.py`.

### Adding maps
- **More generated maps:** run the CLI with new seeds. They show up in `generated_entries()` automatically.
- **A new generator style:** add a family in `config.py`. It gets its own `source_group`.
- **Third-party WADs:** they need an `oss-scout` licence check and main-session approval *before* download. Put them in
  `maps/<source>/` and add a loader function to `_LOADERS` that emits a `source_group` per author or block. Run
  `python -m training.envs.smoke_load --only <id>` and the eval researcher's `python -m training.eval.splits`.

## Smoke-load tool: `training/envs/smoke_load.py`

```bash
uv run --no-sync python -m training.envs.smoke_load --jobs 6 --out runs/environment-engineer/smoke/smoke_all.json
```

For each map it runs `make_env` and `reset`, then 200 seeded random steps, resetting on done. It records:
- whether the map loads;
- WAD and engine sector counts and the geometry source;
- mapinfo door and locked counts, and the number of lines classified `door` / `locked_door` at reset;
- the ray hit types actually observed;
- for generated maps, a sidecar check: doors, spawn, thing classes and sector count against the engine.

Each map runs in a spawned worker with a per-map timeout, so an engine crash is recorded rather than hidden.

MEASURED, 2026-11-26, all 155 entries load in 23.3 s with 6 workers:

| group | maps | loads | sectors | geometry = wad | with doors | with locks | door ray seen (random, 200 steps) | locked ray seen | sidecar ok | median steps/s per process (6 parallel, render off) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gen locomotion | 14 | 14 | 4–10 | 14 | 0 | 0 | 0 | 0 | 14/14 | 2252 |
| gen doors | 14 | 14 | 8–20 | 14 | 14 | 0 | 11 | 0 | 14/14 | 1823 |
| gen keys | 14 | 14 | 21–32 | 14 | 13 | 14 | 4 | 8 | 14/14 | 1796 |
| gen combat | 14 | 14 | 1–6 | 14 | 0 | 0 | 0 | 0 | 14/14 | 1692 |
| gen full | 14 | 14 | 17–49 | 14 | 14 | 14 | 7 | 6 | 14/14 | 1326 |
| freedoom2 | 32 | 32 | 27–863 | 32 | 32 | 30 | 21 | 6 | n/a | 292 |
| freedoom1 | 36 | 36 | 33–983 | 36 | 36 | 29 | 17 | 5 | n/a | 223 |
| vizdoom | 17 | 17 | 1–179 | 17 | 0 | 0 | 0 | 0 | n/a | 1782 |

Random play only sometimes points a ray at a door. The deterministic checks are in the tests: at reset, every generated map has
exactly 2 `door` lines per door and 2 `locked_door` lines per lock, and the walker sees `locked_door` before unlocking.
The steps/s figures are short-run rough numbers. Proper throughput benchmarks (`training/benchmarks/env_*.py`) come with the env factory.

## `training/envs/inventory.py: map_inventory(env) -> dict` (D13: the single implementation)

Call it once after `env.reset()`. It returns `num_sectors`, `door_sectors {sector: lock}`, `doors_total`, `locked_total`,
`monsters` and `keys` (live engine objects at reset), `static_monsters` and `static_keys` (from WAD THINGS at skill 3, single player),
`sector_area` (float64 (S,), map units²) and `cells_total` (64-unit cells). It uses only the public `env.mapinfo` and
`env.sector_polygons()`, except for one isolated adapter read of the engine object list (`_live_objects`).
MEASURED: on generated full-a-000000, combat-a-000003, freedoom2 MAP01, freedoom1 E1M1 and two_rooms, it matches
`training/eval/inventory.py` exactly (counts, `cells_total`, total area).

## Tests

```bash
uv run --no-sync python -m pytest tests/env -q --basetemp=/tmp/ptq_env_a3
```

The suite covers:
- determinism, in-process and across hash seeds;
- the CLI;
- sector cap, headroom and a single player start;
- mapinfo door and lock detection equal to the generator's;
- stage semantics, solvability and lock necessity;
- textures that exist in freedoom2;
- load through `make_env`, with sidecar checks against the engine;
- every thing class spawning;
- walker completion on locomotion, doors and keys maps (pad and switch exits);
- a locked door blocking without its key;
- registry entries and a tamper check, the held-out guard, and manifest validation through `training.eval.splits.validate_all`;
- sample drift;
- the smoke tool;
- `map_inventory`.

A3 tests (`tests/env/test_factory.py`, `test_factory_sampler.py`, `test_vector.py`) cover:
- the `StepResult` shapes and dtypes;
- the info-key layout per step and on done steps;
- auto-reset ordering: `obs` is the fresh episode, and `final_feats` is the terminal one;
- `episode_start == terminated | truncated`;
- `final_feats` present exactly on non-crash done steps;
- `StepContext.step` restarting at 0 after an auto-reset;
- reward terms receiving the full privileged dict;
- no privileged key in `obs`, and `info["privileged"]` opt-in and slim;
- determinism from `seed_base`, across worker splits and threads vs sequential;
- seeds < 10000;
- an injected engine crash, and a SIGKILLed worker restarted with its orphaned engines reaped;
- val/test manifest and held-out-entry refusal;
- unknown config keys, bad actions and worker exceptions failing loudly;
- NaN observations raising;
- `episode_ticks` changed in place without a re-init, and tick ranges from the sidecar;
- `episodes_per_map` reusing the engine;
- `door_reversals` and `use_edges`;
- train/eval feature parity.
