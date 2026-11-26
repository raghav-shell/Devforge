# Research Notes: from the DOOM RL guide to our actual build

Working notes for Track 1. They map the team's "DOOM RL technical learning guide" onto the real
contract in [CONTRACT.md](CONTRACT.md), [RULES.md](RULES.md) and [EVALUATION.md](EVALUATION.md),
correct the parts that don't fit, and record what we measured.

> **Source honesty.** The paper summaries in §5 are written from prior knowledge of these papers.
> The PDFs were **not** downloaded and re-read in this session. Treat numbers marked *(verify)* as
> "check against the paper before quoting in the brief". Everything in §2 and §3 was **measured**
> in this repo (vizdoom 1.3.3, `doom_env` contract 1.0.0-draft).

> **Errata (2026-11-26, Phase 1). `research/*.md` supersedes this file where they disagree.**
> - **Throughput:** about 2,300 steps/s per core holds only on `two_rooms`. Freedoom maps are 5–10× slower: about 3.3k decisions/s in total on MAP01 and 1.4k on MAP10 with 8 processes.
> - **RND discounts are reversed in §5.** The paper uses γ_E = 0.999 and γ_I = 0.99. The PPO Atari settings use 3 epochs.
> - **CleanRL's LSTM PPO does not bootstrap on truncation.** We do: see `research/RL_RESEARCH.md` §3.3.
> - **The §3 dead-reckoning script has a step-2 heading bug.** With the fix, heading is exact. Ray scan matching cuts the position drift. See `research/SPATIAL_INTELLIGENCE.md`.

---

## 1. The guide vs our real contract

| Guide concept | What it actually is for us | Consequence |
| --- | --- | --- |
| Observation dict (`health`, `spatial_rays`, …) | Fixed: `rays` (64 × distance + hit_type 0–5), `visible_entities` (32 × 5 features + type 0–11 + mask), `vitals` (17), `rgb` 120×160×3, `depth` 120×160 | No observation design freedom, only **encoders and preprocessing** |
| Raycasting (guide §23, "cast 32 rays") | Already done by the wrapper: 64 rays over 360°, index 0 = ahead, CCW, range 1024, stops at walls, doors, enemies, items. Line of sight is already applied | Don't build raycasting. Treat the rays as a 1-D circular "image" → **circular 1-D conv** (wrap padding) |
| Entities, permutation invariance (§27–29) | 32 slots, nearest first, zero-padded, `mask`. Types: melee / hitscan / projectile enemy, projectile, health, armor, ammo, weapon, key, barrel, other | **Masked set encoder**: type embedding + features → small self-attention, then attention pooling |
| Keycards | Entity type `key` has **no colour**. Colour only appears in vitals 8–13 once picked up. A ray hit of `locked_door` also has no colour | The memory must tie "saw a locked door" to "now hold some key". Colour matching can't be learned from observations. Fine: most maps need any held key → go back to the locked door |
| Coordinate transforms (§32) | No global x, y or heading. Only own-frame velocity (vitals 14–16) | Our own pose must be integrated (dead reckoning, §3). Measured: feasible |
| Actions: "SHOOT, INTERACT…" | `MultiBinary(9)`: fwd, back, strafe L, strafe R, turn L, turn R, attack, use, next weapon. One decision = 4 tics | **9 independent Bernoulli heads** (factorised policy), not one softmax over 512 combos |
| USE | Fires on the button's down edge. Pressing it on a moving door **reverses the door** | Feed **previous action** into the network (it must learn to release then press). Possibly a learned "USE refractory" input feature (steps since last USE) |
| Terminated vs truncated (§85) | `terminated` = exit or death, `truncated` = tick cap (2,100–4,200 ticks ≈ 525–1,050 steps) | In GAE, **bootstrap V(s′) on truncation**, zero only on termination |
| Compute (§86, "investigate MPS") | Eval: **1 CPU thread** (or a 2 GB slice of a 3090), 40 ms average per step | Model must run in about 5–10 ms on 1 CPU thread → under ~1M params, no CNN on RGB by default |
| Reward (§94) | The wrapper's reward (+1 exit, −1 death, +0.1 per kill) is not the score. `env.privileged_state()` gives the true x, y, angle, kills, damage_taken, keys, door_lock, exit_reached (**training only**) | Our own shaped reward, mirroring the scorer's 4 categories (§6) |

### Corrections to the guide

1. **Apple MPS is probably slower than CPU for us.** Our nets are tiny (MLP + GRU, batch of a few
   thousand). MPS kernel-launch overhead dominates at that size. The bottleneck is simulation
   (measured 2,300 steps/s per core). Plan: rollouts on N CPU worker processes, PPO update on CPU;
   benchmark MPS once and only switch if the update phase dominates.
2. **Sample Factory** is fine as a *throughput* reference, but the rules say the core architecture
   and training pipeline must be ours and "off-the-shelf … policies" score poorly. Decision: write
   our own recurrent PPO with a multiprocessing vector env. Sample Factory is not in the eval image
   anyway (eval only needs `submission/agent.py` + torch).
3. **GNN / topological map (guide §41, "strongest recommendation").** Deferred. Without global
   coordinates, building room nodes needs reliable place recognition. That's a research project of
   its own and adds inference cost. Our first spatial memory is an **ego-centric visit/occupancy
   grid** built from dead-reckoned pose + rays (§3), read by a small CNN. A topological layer can
   come later if time allows.
4. **Occupancy grid from "observations it legitimately received".** Rules explicitly allow
   "preprocessing and encoding observations … frame stacking, learned memory modules". A map
   *written* from our own rays and integrated velocity is preprocessing. **Banned:** planning over
   it (BFS / A* / frontier search that picks actions). The policy network alone reads the map and
   decides.
5. **No aimbot shortcuts.** Entity features already include sin/cos of the relative angle. The
   network may use them; we must not compute "turn toward enemy" in code.
6. **ICM on pixels (guide §46) → do it on our features.** The noisy-TV problem is worse on RGB
   (animated textures, muzzle flashes). Compute curiosity on ray/entity/map embeddings, or use a
   count bonus on dead-reckoned cells.

---

## 2. Measured environment facts

| Fact | Value |
| --- | --- |
| Env throughput, 1 core, random policy | ~2,300 steps/s ≈ 9,000 tics/s (rgb on or off makes little difference on the tiny test map) |
| Turn per 4-tic step, by consecutive steps held | 7.031°, 12.305°, then 14.063° (Doom slow-turn ramp; held direction switch L↔R keeps the fast rate) |
| Both turn keys pressed | 0° (cancel) |
| First step after `reset()` | turn input ignored (0°). Second step 3.516° if the key was already held |
| Own-frame displacement per step | ≈ 2.3·v_now + 1.9·v_prev (forward), ≈ 2.1·v_now + 2.0·v_prev (strafe), v in units/tic from vitals |
| Validator, random agent | passes. Init 0.9 s, p99 step 0.6 ms |

## 3. Dead-reckoning experiment (feasibility of a legal ego-map)

Script: [`training/experiments/dead_reckoning.py`](../training/experiments/dead_reckoning.py).
Policy: sticky random actions (re-sampled every 6 steps), 4 seeds, 500 steps. Heading comes from
our own turn actions only. Position comes from integrating vitals velocity. Truth from
`privileged_state()`.

| Map | Heading error after 500 steps | Position error after 500 steps (% of path walked) |
| --- | --- | --- |
| `maps/two_rooms.wad` | ≤ 0.04° | 66–237 units (1.0–3.7 %) |
| `tests/fixtures/2f617ad2…wad` | ≤ 0.04° | 146–474 units (4.4–18.6 %) |
| `tests/fixtures/509aca7c…wad` | ≤ 0.02° | 69–591 units (3–15 %) |

**Conclusion.** Heading is exact and comes for free. Position drifts at roughly 1–19 % of distance
travelled, probably from wall collisions and sliding, where the reported velocity and the actual
movement differ *(to investigate)*. So:

- A **local, coarse** ego-map (cells of ~64 units, window of ~±1,024 units) is reliable over tens to
  a few hundred steps. That's enough for "have I been here", "dead end", "door I just saw".
- For long-horizon memory (keycard → back to the locked door), add drift correction:
  1. **Scan matching** of the 64 rays against the map (cheap 1-D/2-D correlation; it's
     preprocessing, not planning), and/or
  2. let the GRU carry abstract memory ("a locked door exists, roughly behind-left") and treat the
     map as a fading aid. Map confidence decays with distance travelled.
- Engine quirks to handle: teleporters and lifts break dead reckoning. Detect a jump (rays change
  completely) and reset the map's confidence.

---

## 4. Architecture plan (v1 → v3)

```
rays (64×[dist, onehot6])  ──► circular Conv1d ×3 ──► z_ray (128)
entities (32×[5 + emb(type)], mask) ──► 2-layer masked self-attn (d=64, 4 heads) ──► attn-pool ──► z_ent (64)
vitals (17, normalised) + prev_action (9) + steps_since_use ──► MLP ──► z_vit (64)
ego-map (C×32×32, rotated so heading points up) ──► small CNN ──► z_map (128)      [v2]
                    concat ──► MLP 256 ──► GRU 256 ──► heads:
                        π: 9 × Bernoulli logits
                        V_ext, V_int (two value heads, RND-style)                   [v2]
                        aux: game-features (enemy visible / door ahead / key visible),
                             future-vitals Δ (DFP-style), next-ray prediction       [v2]
```

Ego-map channels: visited count, ray-observed free space, wall, door, locked door, item/key seen,
enemy last seen (decaying). All written only from observations we received.

v3 options (pick by ablation): high-level option policy (explore / fight / go-to-objective) emitting a
latent goal every k = 8 steps to the low-level policy (FuN/options style), or a learned topological
memory.

Budget check: the above is roughly 0.5–1.0 M params. One forward pass on 1 CPU thread should be
about 2–5 ms *(measure)*, well within 40 ms.

---

## 5. Paper takeaways (what we actually use)

**Foundations from the guide's list**

- **DQN** (Mnih et al. 2013, arXiv 1312.5602). Replay + target network + ε-greedy. *Use:* only as
  background. On-policy PPO suits recurrent memory better, and our action space is combinatorial.
- **PPO** (Schulman et al. 2017, arXiv 1707.06347). Clipped ratio objective, several epochs over
  minibatches of each fresh rollout. Typical settings: clip 0.1–0.2, γ = 0.99, λ = 0.95, entropy
  0.01, Adam lr ≈ 2.5e-4–3e-4, grad-norm clip 0.5 *(verify)*. *Use:* the core learner, with
  recurrent minibatches made of whole sequence chunks (e.g. 64 steps) carrying the stored initial
  hidden state. Advantage normalisation per minibatch. Value clipping is optional.
- **GAE** (Schulman et al. 2015, arXiv 1506.02438). δ_t = r_t + γV(s_{t+1}) − V(s_t),
  Â = Σ(γλ)^l δ. *Use:* λ = 0.95. Bootstrap on truncation (see §1).
- **ICM** (Pathak et al. 2017, arXiv 1705.05363). Feature encoder φ trained by an **inverse model**
  (predict a_t from φ(s_t), φ(s_{t+1})). Intrinsic reward = forward-model error in φ-space. Because φ
  only keeps what the agent's actions can affect, it ignores uncontrollable noise (noisy-TV
  mitigation). The paper's own setting was ViZDoom "My Way Home" with sparse reward. Very relevant.
  *Use:* candidate intrinsic signal on our fused features. The inverse model also suits our
  factorised action space (9 BCE losses).
- **RND** (Burda et al. 2018, arXiv 1810.12894). Fixed random target net f, trained predictor f̂,
  r_int = ‖f̂(o) − f(o)‖². Key engineering: **normalise observations** (running mean/std, clip) for the
  target, **normalise intrinsic reward** by a running std of intrinsic *returns*, **two value heads**
  (extrinsic episodic, intrinsic non-episodic) with separate γ (0.99 / 0.999 *(verify)*), and train
  the predictor on a subset of each batch. *Use:* simpler and steadier than ICM. First curiosity
  candidate.
- **Recurrent Models of Visual Attention** (Mnih et al. 2014, arXiv 1406.6247). A recurrent net that
  picks glimpses, trained with REINFORCE. *Use:* mostly conceptual (recurrent state + learned focus).
  Our entity attention is the practical equivalent. No glimpse sensor needed because rays and
  entities are already compact.

**High-value extras the guide didn't list (strongly relevant to DOOM)**

- **Lample & Chaplot 2017, "Playing FPS Games with Deep RL"** (arXiv 1609.05521). DRQN plus an
  auxiliary **game-feature head** (is an enemy visible?), trained with supervision from engine
  info. It hugely sped up learning. *Use:* aux heads supervised by `privileged_state` and wrapper
  info at training time only (enemy in view, door ahead, key visible, distance to the nearest
  unexplored cell). At eval the heads are simply ignored, so this is legal.
- **Dosovitskiy & Koltun 2017, "Learning to Act by Predicting the Future" (DFP)** (arXiv
  1611.01779). Won the ViZDoom 2016 full-deathmatch track. Predicts future changes in
  *measurements* (health, ammo, frags) at several horizons, and acts by goal-weighting them. *Use:*
  an auxiliary head that predicts Δvitals and Δkills at +4/+16/+32 steps. Cheap, dense, directly tied
  to the scoring categories.
- **UNREAL** (Jaderberg et al. 2016, arXiv 1611.05397). Auxiliary tasks (pixel control, reward
  prediction, value replay) give large gains in 3-D mazes. *Use:* ray-depth prediction (next-step
  rays) and reward prediction as aux losses.
- **NGU / episodic curiosity** (Badia et al. 2020, arXiv 2002.06038. Savinov et al. 2018, arXiv
  1810.02274). Per-episode novelty from a memory of embeddings (kNN distance) × lifelong novelty
  (RND). *Use:* our ego-map visit count is a cheap, explicit episodic novelty. Episodic + RND
  lifelong = NGU-lite.
- **Active Neural SLAM** (Chaplot et al. 2020, arXiv 2004.05155). Learned mapper + pose estimator +
  global/local policies. *Use:* validates the ego-map + pose-correction design. **Their global
  planner is analytical, which our rules ban.** We keep only the mapper/pose ideas.
- **Sample Factory** (Petrenko et al. 2020, arXiv 2006.11751). APPO hitting ~100k+ fps on ViZDoom
  with async rollouts. *Use:* throughput engineering ideas (double-buffered rollouts, shared memory
  tensors). Not a dependency.

---

## 6. Reward plan (training only, from `privileged_state()`)

Mirror the scorer's categories so that maximising our reward ≈ maximising the score:

| Term | Signal | Starting weight | Anti-hacking guard |
| --- | --- | --- | --- |
| Exit | `exit_reached` | +10 | terminal, once |
| Time / health bonus at exit | remaining ticks, health | +0–2 | only on exit |
| Keycard | new bit in `keys` | +3 | once per key |
| Door opened / unlocked | `door_lock` and sector ceiling change near the player | +1 / +3 | once per door (dedupe by sector id) |
| Exploration | new 64×64-unit cell of true (x, y) visited | +0.05 per cell | once per cell per episode. Spinning in place gives 0 |
| Kill | Δ`kills` | +0.5 | once per kill. No per-hit reward (avoids farming damage) |
| Damage taken | Δ`damage_taken` | −0.01 per HP | |
| Death | `dead` | −3, larger if early | |
| Intrinsic | RND / ICM / ego-map count | β, decayed 0.5 → 0.05 | normalised; separate value head |

Optional potential-based shaping (Ng et al. 1999): Φ = −geodesic distance to the exit or the next
objective, computed **offline in training** from the WAD geometry. This keeps the optimal policy
unchanged. Allowed because it never runs at evaluation.

## 7. Curriculum and maps

Curriculum (allowed to modify the engine in training, evaluated vanilla):

1. **Locomotion:** open rooms, reward = new cells. Learn to avoid walls.
2. **Doors:** auto-open doors off → USE timing.
3. **Combat:** ViZDoom scenarios `basic`, `defend_the_center`, `deadly_corridor`, `health_gathering`
   (in vizdoom's scenarios folder), then our arenas at skill 3.
4. **Exploration:** `my_way_home`-style mazes, generated multi-room maps.
5. **Keycard chains:** key → locked door → exit, with the key placed behind branches.
6. **Full levels:** Freedoom Phase 1/2 maps, generated gauntlets. Mix earlier stages in
   (replay ~20 %) to avoid catastrophic forgetting.

Map sources:
- **Freedoom 2** `freedoom2.wad` (MAP01–MAP32) and **Freedoom Phase 1** `freedoom1.wad`
  (E1M1–E4M9). Both are bundled inside the vizdoom package, no download needed. Note: the eval IWAD is freedoom2, so the hidden maps may *be* or resemble
  Freedoom maps. **Hold out a set of Freedoom maps purely for validation.**
- **ViZDoom scenarios** (`vizdoom/scenarios/*.wad`, bundled): basic, simpler_basic, rocket_basic,
  defend_the_center, defend_the_line, deadly_corridor, health_gathering(_supreme), my_way_home,
  take_cover, predict_position, deathmatch, cig. Several rely on their `.cfg` and ACS rewards, so
  check each one loads sensibly through `make_env`.
- **Procedural:** Oblige / ObAddon (generates full Doom 2 levels with keys and locked doors), and our
  own Python WAD generator (UDMF text maps are easy to emit). Domain randomisation: monster mix, item
  placement, start position, key placement.
- `tests/fixtures/*.wad` (13 small maps that ship with the template) as smoke and validation maps.

Always keep three splits: **train / validation (held-out maps) / sanity (two_rooms)**. Report the
generalisation gap.

## 8. Ablation table (for the brief and the judges)

| Run | Model | Measures |
| --- | --- | --- |
| A | PPO + MLP on rays/entities/vitals, no memory | baseline |
| B | A + GRU | memory value |
| C | B + set-attention entity encoder | variable-count enemies |
| D | C + ego-map (dead-reckoned) | spatial memory |
| E | D + RND / ego-count curiosity | exploration |
| F | E + aux heads (game features, DFP Δvitals) | sample efficiency |
| G | F + hierarchical option policy | long horizon |

Metrics on held-out maps: exit rate, keys/doors per map, unique cells, kills, damage taken, death
rate, stuck rate (no new cell for 100 steps), and our own estimate of the scorer's score. ≥ 3
seeds where compute allows.

## 9. Pitfalls checklist (from guide Part 12, applied)

- [ ] Never train on one map. Hold out validation maps from day 1.
- [ ] No `privileged_state` anywhere in `submission/`. Grep for it before every commit.
- [ ] Every observation-derived feature is computed in `agent.py` from `observation` only.
- [ ] Validate on vanilla settings (no auto-doors or infinite ammo) every evaluation cycle.
- [ ] `validate.py` (and `--docker` when possible) passes before every push.
- [ ] Checkpoints include optimizer, observation-normaliser stats, RNG states and config.
- [ ] Choose the inference mode (sample vs threshold at 0.5) by validation score. Sampling usually
      helps break loops.
- [ ] Keep `act()` under ~10 ms on 1 thread (`torch.set_num_threads(1)`).

## 10. Environment note (macOS)

The project sits on an iCloud-synced Desktop, which keeps setting the `UF_HIDDEN` flag on files
inside `.venv`. Python 3.13 skips hidden `.pth` files, so `import doom_env` breaks. Fix used: the
real venv lives at `~/.venvs/doomsday` and `.venv` is a symlink to it. To recreate:

```bash
UV_PROJECT_ENVIRONMENT=$HOME/.venvs/doomsday uv sync && rm -rf .venv && ln -s $HOME/.venvs/doomsday .venv
```
