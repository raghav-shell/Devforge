# Experiment hypotheses (ranked)

Owner: rl-researcher. Every item is a **HYPOTHESIS** until it has been run.

## Conventions

**Budgets** are in decisions. The planning rate is about 8M decisions/h on a mixed curriculum (ESTIMATE, RL_RESEARCH.md §2). Freedoom-scale maps alone run at about 3–6M/h.

**Seeds.** Use 2 seeds at the smoke stage and 3 seeds for any claim that goes into the brief.

**Metrics** are computed on **held-out validation maps**. The splits are owned by the evaluation-and-generalization-researcher (`maps/splits/`):

| Metric | Definition |
| --- | --- |
| `S_est` | Our estimate of the scorer's score (`training/eval/`) |
| `exit%` | Fraction of episodes that reach the exit |
| `cells` | Unique 64×64 true cells per episode |
| `keys` | Keys picked up per episode |
| `doors` | Doors opened per episode |
| `kills` | Kills per episode |
| `dmg` | Damage taken per episode |
| `death%` | Fraction of episodes that end in death |
| `stuck%` | Fraction of 100-step windows with no new cell |
| `gap` | S_est on train maps minus S_est on validation maps |

**Kill criteria** stop a run early so the compute can be reused.

**Run records.** Every run records its config, seed, commit and metrics (`CLAUDE.md` execution rule 10).

---

## H1: Plain recurrent-PPO baseline (must run first)

**Hypothesis.** A custom synchronous PPO with an MLP encoder and a GRU-256 learns to explore and open doors on small multi-room maps. It beats a random policy and a feed-forward PPO run on `cells`, using the RL_RESEARCH.md §3 hyperparameters.

**Setup**
- Reward is RESEARCH_NOTES §6 minus the intrinsic term: cell bonus, key, door, exit, kill, damage and death.
- No ego-map and no auxiliary heads.
- Maps: fixtures plus `two_rooms` plus any generated small maps; validation on held-out fixtures.
- Control arm: the same configuration with the GRU replaced by an MLP of equal width (H1b).

**Metrics.** `cells`, `doors`, `exit%` and `S_est` on validation maps, plus throughput (decisions/s) and wall-time fractions.

**Baseline.** A random policy with p = 0.3 per button on the same validation maps.

**Expected effect**
- `cells` ≥ 2× random by 10M decisions.
- GRU ≥ MLP on `doors` by 15M decisions.
- This is the go/no-go check for the whole pipeline.

**Budget.** 20M decisions per arm (about 2–3 h), × 2 seeds.

**Kill criteria**
- Stop at 5M decisions if any of these hold:
  - `cells` ≤ 1.2× random;
  - explained variance < 0;
  - the epoch-1 ratio check fails, i.e. π_new/π_old ≠ 1 before the first gradient step.
- Stop at any time on NaN or a pre-clip gradient norm 10× its median.
- A failure of H1 means **debugging, not tuning**: run the leak, truncation and ratio checks in RL_RESEARCH.md §7.

## H2: Categorical factor heads beat 9 Bernoulli heads

**Hypothesis.** The categorical heads (move 3, strafe 3, turn 3, attack 2, use 2, weapon 2) remove the cancelling button combinations and USE spam at initialisation. That should speed up early learning.

**Metric**
- Decisions needed to reach the H1 arm's `cells` at 10M decisions.
- USE rising edges per episode.
- Door-reversal count (privileged).

**Baseline.** H1 with Bernoulli heads.

**Expected effect.** 1.3× or better sample efficiency to the same `cells`, and about 50 % fewer door reversals.

**Budget.** 10M decisions × 2 seeds.

**Kill criterion.** If there is no difference beyond seed noise at 10M, keep whichever head is simpler for `submission/`.

## H3: Discount γ for 1,000-decision horizons

**Hypothesis.** γ = 0.995 beats 0.99 on maps where the exit is more than 200 decisions away. γ = 0.998 adds variance without adding gain.

**Arms.** γ ∈ {0.99, 0.995, 0.998}, with λ = 0.95.

**Metric.** `exit%` and `keys` on keycard validation maps, and explained variance.

**Baseline.** H1 (γ = 0.995) is the middle arm.

**Expected effect.** 0.99 gives at least 20 % lower `exit%` on long maps.

**Budget.** 20M × 3 arms × 2 seeds. Run on the tier-3-style subset only.

**Kill criterion.** Drop an arm whose explained variance stays below 0.2 after 10M decisions while the other arms are above 0.5.

## H4: The ego-map input makes exploration learnable

The ego-map belongs to the spatial team; this hypothesis is mine.

**Hypothesis.** The cell bonus depends on episode history. Giving the policy and critic a visitation channel raises `cells` and lowers `stuck%`.

**Metric.** `cells`, `stuck%` and `S_est` on validation maps. Also measure inference ms per step on 1 thread.

**Baseline.** H1 GRU (the winner of H2).

**Expected effect.** `cells` +25 % and `stuck%` −30 % at an equal budget.

**Budget.** 30M decisions × 2 seeds.

**Kill criterion.** No gain at 15M, *or* p99 inference above 15 ms.

## H5: Privileged (asymmetric) critic

This needs main-session approval.

**Hypothesis.** A training-only value branch gets privileged extras: true local visit counts, geodesic distance to the current objective, and true pose relative to the start. It takes them *after* the shared torso. This lowers value error and speeds learning. The actor stays legal at evaluation.

**Metric**
- Explained variance of V_ext.
- `exit%` and `keys`.
- A grep-verified check that `submission/` never loads the critic branch.

**Baseline.** The winner of H4.

**Expected effect.** Explained variance +0.15, and `exit%` +20 % relative.

**Budget.** 30M × 2 seeds.

**Kill criteria**
- The actor's validation score falls below the baseline.
- The training–validation gap widens by more than 2×, a sign the actor relies on critic-shaped features that are missing at evaluation.

## H6: Auxiliary prediction pack

**Hypothesis.** Two groups of auxiliary heads on the GRU output, weighted 0.25 in total, improve long-horizon progression:
- DFP-style future-measurement prediction: Δcells, Δkeys, Δkills and Δhealth, plus "exit within horizon", at horizons {8, 32, 128};
- memory probes: locked door seen, ego-frame bearing and distance to the last locked door and to the start, current cell visited.

**Metric**
- `keys` and `exit%` on keycard validation maps.
- Decisions until `keys` > 0.5.
- Probe accuracy, which also serves as an interpretability artefact.

**Baseline.** The winner of H4 or H5.

**Expected effect.** 1.5× or better sample efficiency to the same `keys`. UNREAL saw about 10× on Labyrinth [V], but our observations are already compact, so expect much less.

**Budget.** 30M × 2 seeds.

**Kill criterion**
- If the probes reach their accuracy target but `keys` and `exit%` are unchanged at 20M, drop the DFP group and keep the probes only if they cost less than 0.5 ms at inference.
- The heads are dropped at evaluation in any case.

## H7: NGU-lite lifelong novelty on top of the cell count

**Hypothesis.** An RND multiplier clip(α, 1, 5), computed on a normalised ray, entity and key embedding, raises discovery of rare structures (locked doors, key rooms). It should do this without hurting `exit%`.

**Setup.** Two value heads: γ_E = 0.995 and γ_I = 0.99. Advantages are mixed as 2·A_E + 1·A_I. The predictor is trained on 25 % of each batch. These follow RND [V].

**Metric**
- `doors` and `keys` on validation maps.
- Locked-door sightings per episode.
- `exit%`.

**Baseline.** The best configuration so far, which already includes the cell bonus.

**Expected effect.** `keys` +15 %.

**Budget.** 30M × 2 seeds.

**Kill criterion**
- Intrinsic return grows while `cells` falls: this is noisy-TV-like behaviour. Check entity-velocity leakage into the embedding.
- No gain at 20M.

## H8: Curriculum and map diversity

**Hypothesis.** A staged curriculum beats a uniform mix of all maps at an equal budget. The stages are locomotion, then doors, then combat, then mazes, then key chains, then full levels, with about 20 % replay of earlier stages. The validation gap also shrinks as the number of training maps grows (Procgen [V]).

**Metric.** `S_est` on validation maps per tier, and `gap`.

**Arms**
- Curriculum against uniform.
- Number of training maps ∈ {~30, ~300, as many as possible from the generator}.

**Baseline.** A uniform mix.

**Expected effect.** Curriculum gives `S_est` +20 % at an equal budget. `gap` halves from 30 to 300 maps.

**Budget.** 50M per arm, which is the largest experiment. Start it once a map generator exists.

**Kill criterion.** The curriculum arm forgets: an earlier-tier validation score drops by more than 20 % and replay does not recover it within 5M decisions.

## H9: BPTT chunk length and burn-in

**Hypothesis.** Full-rollout sequences (128) are at least as good as 32-step chunks, and burn-in is unnecessary for on-policy PPO. Applying R2D2's burn-in result to PPO is unproven.

**Arms.** Chunk ∈ {32, 128}. For the 32-step chunk, burn-in ∈ {0, 8}, plus an optional per-epoch state refresh.

**Metric.** `S_est`, the "door seen at t, locked door revisited later" probe, and update wall time.

**Baseline.** H1 (chunk 128).

**Expected effect.** Under 5 % difference in score, with chunk 32 updates 1.2× or more faster.

**Budget.** 15M × 3 arms × 1 seed. This is a cheap engineering check.

**Kill criterion.** None. Pick by score per wall-hour.

## H10: Simple hierarchical option head

Run this only if H6 and H8 leave tier 3 unsolved.

**Hypothesis.** A high-level head chooses a latent goal z, from 4–8 discrete values, every k = 8 decisions, and the low-level policy is conditioned on z. This improves backtracking from the key to the locked door to the exit.

**Metric.** `exit%` on keycard validation maps, and option usage entropy (to detect collapse onto one option).

**Baseline.** The best flat agent.

**Expected effect.** `exit%` +15 % on tier 3 and neutral elsewhere.

**Budget.** 40M × 2 seeds.

**Kill criterion.** Option entropy below 0.1 (collapse) or `S_est` below the flat agent at 25M. The head would also be useful for the judges' "hierarchical control" criterion if it is at least neutral.

## H11: ICM vs NGU-lite (low priority)

**Hypothesis.** ICM rewards self-caused stochasticity (door reversals, monster reactions) and underperforms NGU-lite.

**Metric.** Door reversals per episode, `doors` and `keys`.

**Baseline.** H7.

**Expected effect.** ICM gives at least 2× more door reversals and fewer `keys`.

**Budget.** 20M × 1 seed.

**Kill criterion.** Stop at 10M if door reversals exceed 3× baseline.

## H12: Time awareness

**Hypothesis.** Adding a normalised own-step-counter input, plus randomised `episode_ticks` in training, improves `exit%` near the cap and the exit efficiency bonus. It does no harm on maps whose cap is unknown.

**Metric.** `exit%`, and the time remaining at exit.

**Baseline.** The best configuration without the input.

**Expected effect.** `exit%` +5–10 % on long maps.

**Budget.** 15M × 2 seeds.

**Kill criterion.** No gain at 15M.

## H13: Inference mode

**Hypothesis.** Stochastic sampling beats per-head argmax on validation maps because it breaks deterministic loops. A temperature of about 0.7 may be best.

**Metric.** `S_est`, `stuck%`.

**Baseline.** Argmax.

**Budget.** Evaluation only (no training): 3 modes × all validation maps.

**Kill criterion.** None.

---

## Rank, execution order and budget

**H numbers are the priority rank.** Execution order differs only where there are dependencies: H4 needs the ego-map, H8 needs the map generator, and H7 and H11 need the best agent so far.

**Wall-hour assumptions** (ESTIMATE, RL_RESEARCH.md §2):

| Map class | Rate |
| --- | --- |
| S: small or generated multi-room maps | about 15M decisions/h |
| M: mixed curriculum | about 8M decisions/h |
| F: keycard or Freedoom-scale maps | about 4M decisions/h |

Running two experiments at once (4 + 4 workers) halves the rate of each.

| H | Map class | Full design (decisions) | Wall h (full) | **Must-run version** | Wall h (must-run) |
| --- | --- | --- | --- | --- | --- |
| H1 | S | 20M × 2 arms × 2 seeds = 80M | ~5.3 | GRU 2 seeds + MLP 1 seed at 20M = 60M | ~4 |
| H2 | S | 10M × 2 seeds = 20M (the Bernoulli arm is reused from H1) | ~1.3 | same = 20M | ~1.3 |
| H3 | F | 20M × 3 × 2 = 120M | ~30 | 15M × 3 arms × 1 seed = 45M | ~11 |
| H4 | M | 30M × 2 = 60M | ~7.5 | 30M × 1 = 30M | ~4 |
| H5 | M | 30M × 2 = 60M | ~7.5 | *optional* | — |
| H6 | F | 30M × 2 = 60M | ~15 | 30M × 1 = 30M | ~7.5 |
| H7 | M | 30M × 2 = 60M | ~7.5 | *optional* | — |
| H8 | M | 50M × (2 curriculum arms + 2 extra map-count arms) = 200M | ~25 | curriculum vs uniform, 30M × 1 seed = 60M | ~7.5 |
| H9 | S | 15M × 3 = 45M | ~3 | 15M × 2 (chunk 32 vs 128) = 30M | ~2 |
| H10 | F | 40M × 2 = 80M | ~20 | *optional* | — |
| H11 | M | 20M | ~2.5 | *skip* | — |
| H12 | F | 15M × 2 = 30M | ~7.5 | *optional* | — |
| H13 | eval only | — | <1 | same | <1 |
| **Total** | | **~835M** | **~130 h** | **~275M** | **~38 h** |

**What this means**
- The full design does **not** fit the low end of the realistic 50M–500M range.
- The **must-run subset (~275M decisions, ~38 machine-hours)** does fit. It covers H1, H2, H3, H4, H6, H8, H9 and H13.
- Single-seed must-run results are *screening* results. Any claim that goes into the brief needs a 3-seed rerun of the winning configuration (about 3 × 30M more).
- If no deadline appears and time is short, run H1 → H2 → H4 → H6 → H8 and skip the rest. Treat H3 and H9 as tuning that piggybacks on those runs.

**Dependency on rulings:** H5's privileged critic, and the geodesic potential in RL_RESEARCH.md §5, need a main-session ruling before they are built.
