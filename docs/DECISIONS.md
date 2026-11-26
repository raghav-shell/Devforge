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
| D13 | One `map_inventory` implementation, in `training/envs/inventory.py`. `training/eval/inventory.py` becomes an import of it once it lands | approved |
| D14 | `next_weapon` bias −2 for now. A bias of 0 is a cheap ablation | provisional |
| D15 | Eval inference default `sample` at T = 1.0, provisional until H13 on val | provisional |

## Open
- H13: the frozen inference mode and temperature, decided on val only.
- The deadline, which is needed to size the experiment list.
