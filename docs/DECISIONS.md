# Main-session decision log

Owned by the main session. These rulings were reviewed with Fable 5.2. Where they add to or override an
architect doc (`docs/MODULE_CONTRACTS.md`, `docs/OBSERVATION_SCHEMA.md`, `docs/ACTION_SPACE.md`),
the ruling here wins until the architect folds it in.

## 2026-11-26: Phase 1 rulings

| # | Ruling | Status |
| --- | --- | --- |
| D1 | Core algorithm: custom synchronous recurrent PPO, GRU-256 (`research/RL_RESEARCH.md`) | approved |
| D2 | Action head: factored categorical heads (move / strafe / turn / attack / use / next_weapon), 15 logits mapped one-to-one onto MultiBinary(9). The mapping in `ACTION_SPACE.md` §2.1 was checked against `CONTRACT.md` §3 | approved |
| D3 | Privileged-input critic: allowed as training-only scaffolding. Never imported by `submission/`, stripped from exports. Optional slot, not part of v1 | approved |
| D4 | Geodesic or path-distance reward shaping: deferred | deferred |
| D5 | ICP scan matching and the ego-map: deferred to v2, gated on an oracle-pose ego-map experiment | deferred |
| D6 | RGB key-colour hint: deferred. If ever needed, a learned RGB encoder, never hand rules | deferred |
| D7 | v1 excludes depth, ego-map and the ICP flag. Named reserved slots only | approved |
| D8 | v1 exploration reward: episodic true-cell count bonus in r_ext (training-only reward term) | approved |
| D9 | The architect's provisional decisions are approved as provisional: CPU inference, fixed normalisation constants, opaque (B, S) hidden state, the `episode_start` done-mask, the `trunc_value` buffer, top-level `doom_agent` import, the sampler refusing val/test, codec under `models/` | provisional |
| D10 | Reward start weights are **config values, not contract**. Cells, keys, doors and locked doors pay **only on first occurrence per episode**, so a door that re-closes and reopens can't pay twice | approved |
| D11 | **No USE-debounce mask.** It keys off the agent's own action history and reads as a hand door rule. Rely on the USE bias of −2, the `steps_since_use_edge` input and the doors curriculum stage. Log door reversals in H1 | approved |
| D12 | `DoomAgent.act()` catches exceptions at the top level, logs to stderr, returns the all-zero action and keeps going. A strict mode (env var or constructor flag) re-raises for tests and the harness. Caught exceptions are counted | approved |
| D12a | Implementation note for D12, from A2: if `act()` catches an exception after `pre.process` but before `pre.commit_action`, call `pre.commit_action(zero_action)` (only if an action is still pending) before returning the zero action. Otherwise every later `process` raises until `reset()` | approved |
| D13 | One `map_inventory` implementation, in `training/envs/inventory.py`. `training/eval/inventory.py` becomes an import of it once it lands | approved |
| D14 | `next_weapon` bias −2 for now. A bias of 0 is a cheap ablation | provisional |
| D16 | Learner device: CPU with `torch_threads=2`. Measured: MPS is no faster for the learner and 2.5× slower for rollout inference | approved |
| D17 | H1 runs with `ppo.ratio_check_fail=true`: a ratio ≠ 1 at epoch 1 means a sequence, mask or state bug, so the run must stop | approved |
| D18 | Splits adopted as proposed. Train: generator family a, freedoom2 MAP01–24, freedoom1 E1–E2, ViZDoom drills. Val: generator family b, freedoom2 MAP25–32, freedoom1 E3. Test: freedoom1 E4, **INCOMPLETE** until a licence-cleared third-party family is added. Every test number is labelled "held-out Freedoom episode", never "unseen-map" | approved |
| D19 | Tier rule `static_rule_v1` (static WAD data only, inputs stored per entry). No monster-density split between tier 1 and tier 2. Freedoom maps run at 4,200 ticks. Generator families a and b count as separate families, but val results on family b are reported as **optimistic**, since both families come from the same generator. Val is reported per stratum (generated vs Freedoom) | approved |
| D20 | E1 shipped: `submission/agent.py` wires obs → `DoomPolicy` → `select_action`. CPU, 1 thread. `DOOM_AGENT_STRICT=1` re-raises (D12), and D12a handles a pending action. The bundled weights are a 6-update smoke policy, kept only so the baseline stays working | approved |
| D21 | Combat terms use high-water-mark deltas. `kill` stops paying once kills reach `inventory["monsters"]` (`cap_at_monsters`, default on): Pain Elementals spawn Lost Souls, and the scorer credits only `min(1, kills / monsters_at_reset)`, so uncapped kills could be farmed. `ammo_spent` is off in v1. `death` stays in `progression.py` | approved |
| D22 | Watch-item for H1: `cell_visit` (0.05/cell) can exceed the exit reward (10) on large maps, and `death` −3 dominates early returns on combat maps. Judge H1 on completion rate and the proxy score, not on reward. If coverage rises while completion lags, change the weights in config | open |
| D23 | H1 grad-norm kill criterion: pre-clip grad norm above 10× the **running median over the last 20 updates**, not the whole-run median. The pilot hit a transient 21× spike against the whole-run median at updates 28–31, while KL and explained variance stayed healthy. Applied by whoever monitors the run | approved |
| D24 | **H1 launched 2026-11-27 03:04 IST**, run `h1-gru-s0`. Config: `training/configs/h1.json`, 64 envs (8 workers × 8), T=128, 3 epochs × 4 minibatches, CPU with 2 threads, 20M decisions, a checkpoint every 100 updates. Maps: generated family a only, 200 seeds × 5 stages (`runs/maps/gen-a-v1`); no Freedoom, no ViZDoom scenarios. Source snapshot: `runs/main/h1-gru-s0-src/` (HEAD + tarball, since this round isn't committed yet). Pilot: 1,526 decisions/s, 4.6 GB peak RSS. Expected about 3.6–4 h | running |
| D25 | **Deadline 2026-11-27 08:00 IST.** No new training runs. H1 runs to completion (about 06:30). Interim submission: H1 `ckpt_000800` (6.55M decisions) exported to `submission/weights/`, `validate.py` passed (exit reached on two_rooms, p99 4.2 ms). After H1 ends, ship whichever of ckpt 800 and the final checkpoint scores higher on val (proxy score, seed 0); the repo must be pushed by about 07:30 | approved |
| D15 | Eval inference default `sample` at T = 1.0, provisional until H13 on val | provisional |

## Open
- H13: the frozen inference mode and temperature, decided on val only.
- The deadline, which is needed to size the experiment list.
